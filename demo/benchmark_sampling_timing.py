#!/usr/bin/env python3
"""Benchmark MAR sampling split: conditioning z vs diffusion sampling."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time
from statistics import mean
from typing import Dict

import numpy as np
import torch

# Support direct script execution: add repo root to import path.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from external.mar.models import mar as mar_models
from external.mar.models.mar import mask_by_order


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def _timed(fn, *, device: torch.device):
    _sync(device)
    start = time.perf_counter()
    out = fn()
    _sync(device)
    return out, time.perf_counter() - start


def _load_state_dict_from_checkpoint(
    checkpoint_path: str,
    state_key: str,
) -> Dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must be a dict payload.")
    if state_key not in payload:
        raise ValueError(
            f"State key '{state_key}' not found in checkpoint. "
            f"Available keys: {sorted(payload.keys())}"
        )
    state_dict = payload[state_key]
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint key '{state_key}' is not a state_dict.")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    if not hasattr(mar_models, args.model):
        raise ValueError(f"Unknown model '{args.model}'.")
    model_ctor = getattr(mar_models, args.model)
    model = model_ctor(
        img_size=args.img_size,
        vae_stride=args.vae_stride,
        patch_size=args.patch_size,
        vae_embed_dim=args.vae_embed_dim,
        class_num=args.class_num,
        attn_dropout=args.attn_dropout,
        proj_dropout=args.proj_dropout,
        buffer_size=args.buffer_size,
        diffloss_d=args.diffloss_d,
        diffloss_w=args.diffloss_w,
        num_sampling_steps=str(args.num_sampling_steps),
        diffusion_batch_mul=args.diffusion_batch_mul,
        grad_checkpointing=args.grad_checkpointing,
    ).to(device)
    model.eval()

    state_dict = _load_state_dict_from_checkpoint(args.checkpoint, args.state_key)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[load] Unexpected keys: {len(unexpected)}")
    return model


def _sample_tokens_with_timing(
    model: torch.nn.Module,
    *,
    bsz: int,
    num_iter: int,
    cfg: float,
    cfg_schedule: str,
    temperature: float,
    labels: torch.Tensor | None,
    device: torch.device,
    autocast_bf16: bool,
) -> Dict[str, float]:
    mask = torch.ones(bsz, model.seq_len, device=device)
    tokens = torch.zeros(bsz, model.seq_len, model.token_embed_dim, device=device)
    orders = model.sample_orders(bsz)

    totals = {
        "conditioning": 0.0,
        "diffusion_sample": 0.0,
        "other": 0.0,
        "total": 0.0,
    }

    autocast_ctx_factory = (
        (lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16))
        if (autocast_bf16 and device.type == "cuda")
        else (lambda: torch.no_grad())
    )
    # Run under no_grad always; optionally nest autocast.
    with torch.no_grad():
        for step in range(num_iter):
            step_start = time.perf_counter()
            cur_tokens = tokens.clone()

            if labels is not None:
                class_embedding = model.class_emb(labels)
            else:
                class_embedding = model.fake_latent.repeat(bsz, 1)

            tokens_in = tokens
            mask_in = mask
            if cfg != 1.0:
                tokens_in = torch.cat([tokens, tokens], dim=0)
                class_embedding = torch.cat(
                    [class_embedding, model.fake_latent.repeat(bsz, 1)],
                    dim=0,
                )
                mask_in = torch.cat([mask, mask], dim=0)

            with autocast_ctx_factory():
                x, conditioning_encoder_t = _timed(
                    lambda: model.forward_mae_encoder(tokens_in, mask_in, class_embedding),
                    device=device,
                )
                z, conditioning_decoder_t = _timed(
                    lambda: model.forward_mae_decoder(x, mask_in),
                    device=device,
                )
            conditioning_t = conditioning_encoder_t + conditioning_decoder_t
            totals["conditioning"] += conditioning_t

            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = torch.tensor([np.floor(model.seq_len * mask_ratio)], device=device)
            mask_len = torch.maximum(
                torch.tensor([1], device=device),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )
            mask_next = mask_by_order(mask_len[0], orders, bsz, model.seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if cfg != 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            z_to_sample = z[mask_to_pred.nonzero(as_tuple=True)]
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (model.seq_len - mask_len[0]) / model.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise ValueError("cfg_schedule must be 'linear' or 'constant'.")

            with autocast_ctx_factory():
                sampled_token_latent, diffusion_t = _timed(
                    lambda: model.diffloss.sample(z_to_sample, temperature, cfg_iter),
                    device=device,
                )
            totals["diffusion_sample"] += diffusion_t

            if cfg != 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

            _sync(device)
            step_total = time.perf_counter() - step_start
            totals["total"] += step_total
            totals["other"] += step_total - (conditioning_t + diffusion_t)

    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--state-key", type=str, default="model_ema")
    parser.add_argument("--model", type=str, default="mar_base")
    parser.add_argument("--img-size", type=int, default=16)
    parser.add_argument("--vae-stride", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=1)
    parser.add_argument("--vae-embed-dim", type=int, default=16)
    parser.add_argument("--class-num", type=int, default=1000)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--proj-dropout", type=float, default=0.1)
    parser.add_argument("--buffer-size", type=int, default=64)
    parser.add_argument("--diffloss-d", type=int, default=6)
    parser.add_argument("--diffloss-w", type=int, default=1024)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--diffusion-batch-mul", type=int, default=4)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-iter", type=int, default=16)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--cfg-schedule", type=str, default="linear")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--random-labels", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measure-runs", type=int, default=3)
    parser.add_argument("--autocast-bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable.")
    device = torch.device(args.device)

    model = _build_model(args, device)
    labels = None
    if args.random_labels:
        labels = torch.randint(
            low=0,
            high=args.class_num,
            size=(args.batch_size,),
            device=device,
            dtype=torch.long,
        )

    all_totals: list[Dict[str, float]] = []
    total_runs = args.warmup_runs + args.measure_runs
    for run_idx in range(total_runs):
        totals = _sample_tokens_with_timing(
            model,
            bsz=args.batch_size,
            num_iter=args.num_iter,
            cfg=float(args.cfg),
            cfg_schedule=str(args.cfg_schedule),
            temperature=float(args.temperature),
            labels=labels,
            device=device,
            autocast_bf16=bool(args.autocast_bf16),
        )
        if run_idx >= args.warmup_runs:
            all_totals.append(totals)

    avg = {k: mean([run[k] for run in all_totals]) for k in all_totals[0].keys()}
    denom_steps = max(1, args.num_iter)
    per_step_ms = {k: 1000.0 * v / denom_steps for k, v in avg.items()}

    print("=== MAR Sampling Timing (averaged) ===")
    print(f"runs={args.measure_runs}, warmup={args.warmup_runs}")
    print(f"batch_size={args.batch_size}, num_iter={args.num_iter}, device={device}")
    print(f"cfg={args.cfg}, cfg_schedule={args.cfg_schedule}, num_sampling_steps={args.num_sampling_steps}")
    print("")
    print(f"conditioning_total_s:      {avg['conditioning']:.6f}")
    print(f"diffusion_sample_total_s:  {avg['diffusion_sample']:.6f}")
    print(f"other_total_s:             {avg['other']:.6f}")
    print(f"total_s:                   {avg['total']:.6f}")
    print("")
    print(f"conditioning_ms_per_step:      {per_step_ms['conditioning']:.3f}")
    print(f"diffusion_sample_ms_per_step:  {per_step_ms['diffusion_sample']:.3f}")
    print(f"other_ms_per_step:             {per_step_ms['other']:.3f}")
    print(f"total_ms_per_step:             {per_step_ms['total']:.3f}")


if __name__ == "__main__":
    main()

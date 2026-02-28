import torch
import torch.nn as nn
from torch.func import functional_call, grad as func_grad, jacfwd, jacrev, vmap
from torch.utils.checkpoint import checkpoint
import math

try:
    # Package import path used from kosmos.
    from external.mar.diffusion import create_diffusion
except Exception:  # pragma: no cover - fallback for original MAR scripts
    from diffusion import create_diffusion


class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, grad_checkpointing=False):
        super(DiffLoss, self).__init__()
        self.in_channels = target_channels
        self.net = SimpleMLPAdaLN(
            in_channels=target_channels,
            model_channels=width,
            out_channels=target_channels * 2,  # for vlb loss
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule="cosine")
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule="cosine")
        self._compiled_sample_with_log_prob = {}
        self._compiled_log_prob = {}

    def forward(self, target, z, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        device = z.device
        if not cfg == 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels, device=device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            noise = torch.randn(z.shape[0], self.in_channels, device=device)
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent

    def _prepare_t(self, t, batch_size, device):
        if isinstance(t, int):
            return torch.full((batch_size,), t, device=device, dtype=torch.long)
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"t must be an int or torch.Tensor, got {type(t)}")

        if t.ndim == 0:
            t = t.repeat(batch_size)
        if t.shape[0] != batch_size:
            raise ValueError(f"Expected t shape[0] == {batch_size}, got {t.shape[0]}")
        t = t.to(device=device)
        if t.is_floating_point():
            if t.numel() > 0 and float(t.min().item()) >= 0.0 and float(t.max().item()) <= 1.0:
                t = torch.round(t * (self.train_diffusion.num_timesteps - 1))
        return t.to(dtype=torch.long)

    @staticmethod
    def _broadcast_timesteps(values, ref):
        while values.ndim < ref.ndim:
            values = values.unsqueeze(-1)
        return values

    def predict_eps(self, x, t, z, cfg=1.0):
        """
        Predict epsilon for noisy sample x at discrete timestep t.
        """
        if x.shape[0] != z.shape[0]:
            raise ValueError(f"Batch size mismatch: x has {x.shape[0]}, z has {z.shape[0]}")

        t = self._prepare_t(t, x.shape[0], x.device)
        if cfg != 1.0:
            if x.shape[0] % 2 != 0:
                raise ValueError("For cfg != 1.0, expected concatenated cond/uncond batch with even batch size.")
            model_out = self.net.forward_with_cfg(x, t, c=z, cfg_scale=cfg)
        else:
            model_out = self.net(x, t, c=z)

        eps = model_out[:, :self.in_channels]
        return eps

    def predict_velocity(self, x, t, z, cfg=1.0, eps=1e-12):
        """
        Predict probability-flow ODE velocity:
            v_t = -0.5 * beta_t * (x_t - eps_theta / sigma_t)
        where beta_t and sigma_t are from the training diffusion schedule.
        """
        eps_pred = self.predict_eps(x=x, t=t, z=z, cfg=cfg)
        t = self._prepare_t(t, x.shape[0], x.device)

        betas = torch.as_tensor(self.train_diffusion.betas, device=x.device, dtype=x.dtype)[t]
        sigmas = torch.as_tensor(
            self.train_diffusion.sqrt_one_minus_alphas_cumprod, device=x.device, dtype=x.dtype
        )[t].clamp_min(eps)
        betas = self._broadcast_timesteps(betas, x)
        sigmas = self._broadcast_timesteps(sigmas, x)

        velocity = -0.5 * betas * (x - (eps_pred / sigmas))
        return velocity

    def _predict_velocity(self, x, t, z):
        return self.predict_velocity(x=x, t=t, z=z, cfg=1.0)

    def _exact_divergence(self, x, t, z, use_jacfwd: bool = True):
        def single_vf(x_single, t_single, z_single):
            x_single = x_single.unsqueeze(0)
            t_single = t_single.view(1)
            z_single = z_single.unsqueeze(0)
            return self._predict_velocity(x_single, t_single, z_single).squeeze(0)

        jac_fn = jacfwd if use_jacfwd else jacrev
        jac = vmap(jac_fn(single_vf), in_dims=(0, 0, 0))(x, t, z)
        return jac.diagonal(dim1=-2, dim2=-1).sum(-1)

    def _hutchinson_divergence(
        self,
        x,
        t,
        z,
        num_samples: int = 1,
        noise_type: str = "rademacher",
    ):
        if noise_type == "rademacher":
            v = torch.randint(
                0, 2, (num_samples, *x.shape), device=x.device, dtype=x.dtype
            ) * 2 - 1
        elif noise_type == "gaussian":
            v = torch.randn((num_samples, *x.shape), device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown noise_type: {noise_type}.")

        x = x.detach()
        t = t.detach()
        z = z.detach()

        params = dict(self.net.named_parameters())
        buffers = dict(self.net.named_buffers())
        params_and_buffers = {**params, **buffers}

        def net_velocity(x_in, t_in, z_in):
            model_out = functional_call(self.net, params_and_buffers, (x_in, t_in, z_in))
            eps = model_out[:, :self.in_channels]
            t_idx = self._prepare_t(t_in, x_in.shape[0], x_in.device)
            betas = torch.as_tensor(self.train_diffusion.betas, device=x_in.device, dtype=x_in.dtype)[t_idx]
            sigmas = torch.as_tensor(
                self.train_diffusion.sqrt_one_minus_alphas_cumprod,
                device=x_in.device,
                dtype=x_in.dtype,
            )[t_idx].clamp_min(1e-12)
            while betas.ndim < x_in.ndim:
                betas = betas.unsqueeze(-1)
                sigmas = sigmas.unsqueeze(-1)
            return -0.5 * betas * (x_in - (eps / sigmas))

        def v_dot_velocity(x_in, t_in, z_in, v_in):
            return (v_in * net_velocity(x_in, t_in, z_in)).sum()

        vjp_fn = func_grad(v_dot_velocity, argnums=0)
        vjp = vmap(vjp_fn, in_dims=(None, None, None, 0))(x, t, z, v)
        div_estimate = (v * vjp).sum(dim=-1)
        return div_estimate.mean(dim=0)

    def sample_with_log_prob(
        self,
        z,
        cfg: float = 1.0,
        use_jacfwd: bool = True,
        divergence_method: str = "exact",
        hutchinson_noise_type: str = "rademacher",
        hutchinson_samples: int = 1,
        noise: torch.Tensor | None = None,
    ):
        if cfg != 1.0:
            raise ValueError("sample_with_log_prob currently supports cfg=1.0 only.")
        device = z.device
        if noise is None:
            x = torch.randn(z.shape[0], self.in_channels, device=device)
        else:
            if noise.shape != (z.shape[0], self.in_channels):
                raise ValueError(
                    "noise must have shape (B, in_channels). "
                    f"Got noise={tuple(noise.shape)}, z={tuple(z.shape)}, "
                    f"in_channels={self.in_channels}."
                )
            x = noise.to(device=device)

        log2pi = math.log(2.0 * math.pi)
        logp = -0.5 * (x.pow(2).sum(dim=-1) + self.in_channels * log2pi)
        t = torch.linspace(1.0, 0.0, self.gen_diffusion.num_timesteps + 1, device=device)
        for i in range(self.gen_diffusion.num_timesteps):
            t_cur, t_next = t[i], t[i + 1]
            t_batch = t_cur.expand(x.shape[0])
            with torch.enable_grad():
                if divergence_method == "exact":
                    trace = self._exact_divergence(x, t_batch, z, use_jacfwd=use_jacfwd)
                elif divergence_method == "hutchinson":
                    trace = self._hutchinson_divergence(
                        x,
                        t_batch,
                        z,
                        num_samples=hutchinson_samples,
                        noise_type=hutchinson_noise_type,
                    )
                else:
                    raise ValueError(f"Unknown divergence_method: {divergence_method}.")
            with torch.no_grad():
                velocity = self._predict_velocity(x, t_batch, z)
            dt = t_next - t_cur
            logp = logp + dt * (-trace)
            x = x + dt * velocity
        return x, logp

    def sample_with_log_prob_compiled(self, *args, **kwargs):
        return self.sample_with_log_prob(*args, **kwargs)

    def log_prob(
        self,
        x,
        z,
        use_jacfwd: bool = True,
        divergence_method: str = "exact",
        hutchinson_noise_type: str = "rademacher",
        hutchinson_samples: int = 1,
    ):
        device = z.device
        x = x.to(device=device).clone()
        t = torch.linspace(0.0, 1.0, self.gen_diffusion.num_timesteps + 1, device=device)
        logp_correction = torch.zeros(x.shape[0], device=device, dtype=x.dtype)
        for i in range(self.gen_diffusion.num_timesteps):
            t_cur, t_next = t[i], t[i + 1]
            t_batch = t_cur.expand(x.shape[0])
            with torch.enable_grad():
                if divergence_method == "exact":
                    trace = self._exact_divergence(x, t_batch, z, use_jacfwd=use_jacfwd)
                elif divergence_method == "hutchinson":
                    trace = self._hutchinson_divergence(
                        x,
                        t_batch,
                        z,
                        num_samples=hutchinson_samples,
                        noise_type=hutchinson_noise_type,
                    )
                else:
                    raise ValueError(f"Unknown divergence_method: {divergence_method}.")
            dt = t_next - t_cur
            logp_correction = logp_correction + dt * trace
            with torch.no_grad():
                velocity = self._predict_velocity(x, t_batch, z)
                x = x + dt * velocity
        log2pi = math.log(2.0 * math.pi)
        logp_noise = -0.5 * (x.pow(2).sum(dim=-1) + self.in_channels * log2pi)
        return logp_noise + logp_correction

    def log_prob_compiled(self, *args, **kwargs):
        return self.log_prob(*args, **kwargs)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

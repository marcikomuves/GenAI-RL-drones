import copy
import os
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional, Sequence


class Normalizer(torch.nn.Module):
    def __init__(
        self,
        dataset: torch.Tensor,
        eps: float = 1e-5,
        skip_dims: Optional[List[int]] = None,
        target_std: float = 1.0,
    ):
        super().__init__()
        self.register_buffer('mean', dataset.mean(dim=0))
        self.register_buffer('std', dataset.std(dim=0) + eps)
        self.skip_dims = [] if skip_dims is None else list(skip_dims)
        if self.skip_dims:
            self.mean[self.skip_dims] = 0.0
            self.std[self.skip_dims] = 1.0
        self.target_std = target_std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std * self.target_std

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.target_std * self.std + self.mean

    def reset(self, dataset: torch.Tensor, eps: float = 1e-5):
        self.mean = dataset.mean(dim=0)
        self.std = dataset.std(dim=0) + eps
        if self.skip_dims:
            self.mean[self.skip_dims] = 0.0
            self.std[self.skip_dims] = 1.0


class SinusoidalPosEmb(torch.nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(torch.nn.Module):

    def __init__(self, dim: int, is_random: bool = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = torch.nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        freqs = x*self.weights.unsqueeze(0) * 2 * np.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


# Residual MLP of the form x_{L+1} = MLP(LN(x_L)) + x_L
class ResidualBlock(torch.nn.Module):

    def __init__(self, dim_in: int, dim_out: int, activation: str = "relu", layer_norm: bool = True):
        super().__init__()
        self.linear = torch.nn.Linear(dim_in, dim_out, bias=True)
        if layer_norm:
            self.ln = torch.nn.LayerNorm(dim_in)
        else:
            self.ln = torch.nn.Identity()
        self.activation = getattr(torch.nn.functional, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.activation(self.ln(x)))


class ResidualMLP(torch.nn.Module):

    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        output_dim: int,
        activation: str = "relu",
        layer_norm: bool = False,
    ):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, width),
            *[ResidualBlock(width, width, activation, layer_norm) for _ in range(depth)],
            torch.nn.LayerNorm(width) if layer_norm else torch.nn.Identity(),
        )

        self.activation = getattr(torch.nn.functional, activation)
        self.final_linear = torch.nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.final_linear(self.activation(self.network(x)))


class ResidualMLPDenoiser(torch.nn.Module):

    def __init__(
        self,
        d_in: int,
        dim_t: int = 128,
        mlp_width: int = 1024,
        num_layers: int = 6,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = True,
        learned_sinusoidal_dim: int = 16,
        activation: str = "relu",
        layer_norm: bool = True,
        cond_dim: Optional[int] = None,
    ):

        super().__init__()
        self.residual_mlp = ResidualMLP(
            input_dim=dim_t,
            width=mlp_width,
            depth=num_layers,
            output_dim=d_in,
            activation=activation,
            layer_norm=layer_norm,
        )
        if cond_dim is not None:
            self.proj = torch.nn.Linear(d_in + cond_dim, dim_t)
            self.conditional = True
        else:
            self.proj = torch.nn.Linear(d_in, dim_t)
            self.conditional = False

        # time embeddings
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim_t)
            fourier_dim = dim_t

        self.time_mlp = torch.nn.Sequential(
            sinu_pos_emb,
            torch.nn.Linear(fourier_dim, dim_t),
            torch.nn.SiLU(),
            torch.nn.Linear(dim_t, dim_t)
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, cond=None,) -> torch.Tensor:
        if self.conditional:
            assert cond is not None
            x = torch.cat((x, cond), dim=-1)
        time_embed = self.time_mlp(timesteps)
        x = self.proj(x) + time_embed
        return self.residual_mlp(x)


class ElucidatedDiffusion(torch.nn.Module):

    def __init__(
        self,
        denoiser_net,
        normalizer,
        event_shape: Sequence[int],  # shape of the input and output
        num_sample_steps: int = 32,  # number of sampling steps
        sigma_min: float = 0.002,  # min noise level
        sigma_max: float = 80,  # max noise level
        sigma_data: float = 1.0,  # standard deviation of data distribution
        rho: float = 7,  # controls the sampling schedule
        P_mean: float = -1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std: float = 1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
        S_churn: float = 80,  # parameters for stochastic sampling - depends on dataset, Table 5 in paper
        S_tmin: float = 0.05,
        S_tmax: float = 50,
        S_noise: float = 1.003,
    ):
        super().__init__()
        assert denoiser_net.random_or_learned_sinusoidal_cond
        self.denoiser_net = denoiser_net
        self.normalizer = normalizer
        self.event_shape = event_shape
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps  # otherwise known as N in the paper
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise

    @property
    def device(self):
        return next(self.denoiser_net.parameters()).device

    # derived preconditioning params - Table 1
    def c_skip(self, sigma):
        return (self.sigma_data ** 2) / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma):
        return sigma * self.sigma_data * (self.sigma_data ** 2 + sigma ** 2) ** -0.5

    def c_in(self, sigma):
        return 1 * (sigma ** 2 + self.sigma_data ** 2) ** -0.5

    def c_noise(self, sigma):
        return torch.log(sigma.clamp(min=1e-20)) * 0.25

    def sample_schedule(self, num_sample_steps=None):
        inv_rho = 1 / self.rho
        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (self.sigma_max ** inv_rho + steps / (num_sample_steps - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)) ** self.rho
        sigmas = torch.nn.functional.pad(sigmas, (0, 1), value=0.)  # last step is sigma value of 0.
        return sigmas

    def noise_distribution(self, batch_size):
        return (self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)).exp()


    # preconditioned network output, equation (7) in the paper
    def preconditioned_forward(
        self, 
        noised_inputs, 
        sigma, 
        clamp=False, 
        cond=None
    ):
        batch, device = noised_inputs.shape[0], noised_inputs.device
        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)
        padded_sigma = sigma.view(batch, *([1] * len(self.event_shape)))
        net_out = self.denoiser_net(
            self.c_in(padded_sigma) * noised_inputs,
            self.c_noise(sigma),
            cond=cond,
        )
        out = self.c_skip(padded_sigma) * noised_inputs + self.c_out(padded_sigma) * net_out
        if clamp:
            out = out.clamp(-1., 1.)
        return out

    def denoise_over_sigma(
        self,
        x, 
        sigma, 
        clamp: bool = False, 
        cond=None,
    ):
        denoised = self.preconditioned_forward(x, sigma, clamp=clamp, cond=cond)
        denoised_over_sigma = (x - denoised) / sigma
        return denoised_over_sigma


    def forward(
        self, 
        inputs, 
        cond=None
    ):
        inputs = self.normalizer.normalize(inputs)
        batch_size, *event_shape = inputs.shape
        assert event_shape == self.event_shape, f'mismatch of event shape, expected {self.event_shape}, got {event_shape}'
        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = sigmas.view(batch_size, *([1] * len(self.event_shape)))
        noise = torch.randn_like(inputs)
        noised_inputs = inputs + padded_sigmas * noise  # alphas are 1. in the paper
        denoised = self.preconditioned_forward(noised_inputs, sigmas, cond=cond)
        losses = torch.nn.functional.mse_loss(denoised, inputs, reduction='none')
        losses = losses.flatten(1).mean(1)
        loss_weight = (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2
        losses = losses * loss_weight
        return losses.mean()


    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 16,
        num_sample_steps: Optional[int] = None,
        clamp: bool = False,
        cond = None,
        trace = False,
        real_buffer_packed = None,
        w = None,
        blocks = None
    ):

        if trace:
            assert w is not None and blocks is not None
            traced_values = []
            idxs = torch.randperm(batch_size, device=self.device)[:min(1024, batch_size)]

        num_sample_steps = num_sample_steps if num_sample_steps is not None else self.num_sample_steps
        shape = (batch_size, *self.event_shape)
        # get the schedule, which is returned as (sigma, gamma) tuple, and pair up with the next sigma and gamma
        sigmas = self.sample_schedule(num_sample_steps)
        gammas = torch.where((sigmas >= self.S_tmin) & (sigmas <= self.S_tmax), min(self.S_churn / num_sample_steps, np.sqrt(2) - 1), 0.)

        # sample from noise
        x = sigmas[0] * torch.randn(shape, device=self.device)

        # denoise
        pbar = tqdm(enumerate(zip(sigmas[:-1], sigmas[1:], gammas[:-1])), desc='[SynthER] Sampling time step', mininterval=1, leave=False)
        for step, (s, s_next, g) in pbar:
            s, s_next, g = s.item(), s_next.item(), g.item()
            eps = self.S_noise * torch.randn_like(x)
            s_hat = s + g * s
            x_hat = x + np.sqrt(s_hat**2 - s**2) * eps
            denoised_over_sigma = self.denoise_over_sigma(x_hat, s_hat, clamp=clamp, cond=cond)
            x_next = x_hat + (s_next - s_hat) * denoised_over_sigma
            if s_next != 0:
                denoised_prime_over_sigma = self.denoise_over_sigma(x_next, s_next, clamp=clamp, cond=cond)
                x_next = x_hat + 0.5 * (s_next - s_hat) * (denoised_over_sigma + denoised_prime_over_sigma)
            x = x_next

            if (step % 8 == 0 or step == num_sample_steps - 1) and trace:
                w.add_scalar("sampling/sigma", s, step)

                x0_hat = x_hat - s_hat * denoised_over_sigma

                for k, sl in blocks.items():
                    w.add_scalar(f"sampling/{k}_mean",      x0_hat.mean(0)[sl].mean(),                step)
                    w.add_scalar(f"sampling/{k}_mean_min",  x0_hat.mean(0)[sl].min(),                 step)
                    w.add_scalar(f"sampling/{k}_mean_max",  x0_hat.mean(0)[sl].max(),                 step)
                    w.add_scalar(f"sampling/{k}_mean_real", real_buffer_packed.mean(0)[sl].mean(),   step)
                    w.add_scalar(f"sampling/{k}_std",       x0_hat.std(0)[sl].mean(),                 step)   # drops the confusing "_mean_mean"
                    w.add_scalar(f"sampling/{k}_std_min",   x0_hat.std(0)[sl].min(),                  step)
                    w.add_scalar(f"sampling/{k}_std_max",   x0_hat.std(0)[sl].max(),                  step)
                    w.add_scalar(f"sampling/{k}_std_real",  real_buffer_packed.std(0)[sl].mean(),    step)

                # append the current noisy iterate and clean prediction tensors to a csv maybe
                traced_values.append((step,s,x[idxs].cpu(),x0_hat[idxs].cpu()))

        if trace:
            np.savez_compressed(
            os.path.join(w.get_logdir(), "diagnostics.npz"),
            real_norm=real_buffer_packed[np.random.choice(len(real_buffer_packed), min(20000, len(real_buffer_packed)), replace=False)], # (N, dim), normalized space
            trace_steps=np.array([t[0] for t in traced_values]),
            trace_sigmas=np.array([t[1] for t in traced_values]),
            trace_x=np.stack([t[2].numpy() for t in traced_values]),    # noisy iterates
            trace_x0=np.stack([t[3].numpy() for t in traced_values]),   # clean estimates
    )
                
        if clamp:
            x = x.clamp(-1., 1.)
        return self.normalizer.unnormalize(x)


class SimpleDiffusionGenerator:

    def __init__(
        self,
        obs_space,
        act_space,
        device,
        terminal_threshold = 0.5,
    ):
        self.obs_dim = obs_space.shape[0]
        self.act_dim = act_space.shape[0]
        self.dim = 2 * self.obs_dim + self.act_dim + 2
        self.device = device
        self.terminal_threshold = terminal_threshold
        self.raw_model = None
        self.ema_model = None

    def _pack(self, d):
        o, a = self.obs_dim, self.act_dim
        return torch.cat([
            d["obs"].reshape(-1, o),
            d["action"].reshape(-1, a),
            d["reward"].reshape(-1, 1),
            (d["next_obs"] - d["obs"]).reshape(-1, o), # diffusion over delta obs
            d["terminated"].reshape(-1, 1).float(),
        ], dim=-1).float()

    def retrain(
        self, 
        real_buffer,
        train_steps,
        batch_size,
        ema_freq,
        ema_after,
        beta,
        power,
        lr,
        trace = False,
        w = None
    ):
        if trace:
            assert w is not None
            trace_batch = self._pack(real_buffer.sample(batch_size)).to(self.device)
        
        model = ElucidatedDiffusion(
            denoiser_net = ResidualMLPDenoiser(
                d_in=self.dim, 
                dim_t=256, 
                mlp_width=1024, 
                num_layers=6,
                learned_sinusoidal_cond=False, 
                random_fourier_features=True,
                learned_sinusoidal_dim=16, 
                activation="relu", 
                layer_norm=True,
            ), 
            normalizer = Normalizer(
                dataset=self._pack(real_buffer.ordered()).cpu(), 
                skip_dims=[self.dim - 1]
            ),  
            event_shape = [self.dim],
        ).to(self.device)
        self.raw_model = model
        self.ema_model = copy.deepcopy(model).requires_grad_(False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_steps)

        model.train()
        pbar = tqdm(range(1, train_steps + 1), desc="[SynthER] Training step", mininterval=1, leave=False)
        for step in pbar:
            data = self._pack(real_buffer.sample(batch_size)).to(self.device)
            loss = model.forward(data)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                if step % ema_freq == 0:
                    if step <= ema_after:
                        self.ema_model.load_state_dict(model.state_dict())
                    else:
                        decay = min(beta, 1 - (1 + step - ema_after - 1) ** -power)
                        for pe, p in zip(self.ema_model.parameters(), model.parameters()):
                            pe.lerp_(p, 1 - decay)
                        for be, b in zip(self.ema_model.buffers(), model.buffers()):
                            be.copy_(b)
            if step % 1000 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                if trace:                   
                    w.add_scalar("train/loss", loss.item(), step)
                    with torch.no_grad():
                        sig = model.noise_distribution(len(trace_batch))
                        x = model.normalizer.normalize(trace_batch)
                        noised = x + sig[:, None] * torch.randn_like(x)
                        den = model.preconditioned_forward(noised, sig)
                        per = ((den - x) ** 2).mean(1)
                        edges = torch.quantile(sig, torch.linspace(0, 1, 6, device=sig.device))
                        for k in range(5):
                            m = (sig >= edges[k]) & (sig <= edges[k + 1])
                            w.add_scalar(f"train/mse_bin{k}", float(per[m].mean().item()), step)                                    

        self.raw_model.eval()
        self.ema_model.eval()


    def warm_retrain(
        self, 
        real_buffer, 
        train_steps, 
        batch_size, 
        lr,
        ema_freq=10, 
        ema_decay=0.99, 
        reset_normalizer=False,
        trace=False, 
        w=None
    ):
        assert self.raw_model is not None
        if trace:
            assert w is not None
            trace_batch = self._pack(real_buffer.sample(batch_size)).to(self.device)

        if reset_normalizer:
            stats = self._pack(real_buffer.ordered()).cpu()
            for m in (self.raw_model, self.ema_model):
                m.normalizer.reset(stats)
                m.normalizer.to(self.device)

        opt = torch.optim.AdamW(self.raw_model.parameters(), lr=lr, betas=(0.9, 0.99))
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, train_steps)
        self.raw_model.train()
        pbar = tqdm(range(1, train_steps + 1), desc="[SynthER] Warm training step", mininterval=1, leave=False)
        for step in pbar:
            data = self._pack(real_buffer.sample(batch_size)).to(self.device)
            loss = self.raw_model.forward(data)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.raw_model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % ema_freq == 0:
                with torch.no_grad():
                    for pe, p in zip(self.ema_model.parameters(), self.raw_model.parameters()):
                        pe.lerp_(p, 1 - ema_decay)
                    for be, b in zip(self.ema_model.buffers(), self.raw_model.buffers()):
                        be.copy_(b)
            if step % 1000 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

                if trace:                   
                    w.add_scalar("train/loss", loss.item(), step)
                    with torch.no_grad():
                        sig = self.raw_model.noise_distribution(len(trace_batch))
                        x = self.raw_model.normalizer.normalize(trace_batch)
                        noised = x + sig[:, None] * torch.randn_like(x)
                        den = self.raw_model.preconditioned_forward(noised, sig)
                        per = ((den - x) ** 2).mean(1)
                        edges = torch.quantile(sig, torch.linspace(0, 1, 6, device=sig.device))
                        for k in range(5):
                            m = (sig >= edges[k]) & (sig <= edges[k + 1])
                            w.add_scalar(f"train/mse_bin{k}", float(per[m].mean().item()), step) 

        self.raw_model.eval()
        self.ema_model.eval()


    @torch.no_grad()
    def sample(
        self, 
        num_samples, 
        num_sample_steps, 
        sample_batch_size,
        trace = False,
        w = None,
        real_buffer = None
    ):
        assert self.ema_model is not None

        o, a, end = self.obs_dim, self.act_dim, self.dim - 1

        if trace:
            assert w is not None and real_buffer is not None
            blocks = {
                "obs": slice(0, o), 
                "act": slice(o, o + a),
                "rew": slice(o + a, o + a + 1), 
                "delta": slice(o + a + 1, end)
            }
            real_buffer_packed = self.ema_model.normalizer.normalize(self._pack(real_buffer.ordered()).to(self.device)).cpu().numpy()
        else:
            blocks = None
            real_buffer_packed = None

        sample_chunks = []
        for i, start in enumerate(range(0, num_samples, sample_batch_size)):
            n = min(sample_batch_size, num_samples - start)
            sample_chunk = self.ema_model.sample(
                batch_size=n, 
                num_sample_steps=num_sample_steps, 
                clamp=False,
                trace=trace and i == 0,
                real_buffer_packed=real_buffer_packed,
                w=w,
                blocks=blocks
            )
            sample_chunks.append(sample_chunk.cpu().numpy())
        s = np.concatenate(sample_chunks, axis=0)

        obs = s[:, :o]
        act = s[:, o:o + a]
        rew = s[:, o + a]
        delta = s[:, o+a+1:2*o+a+1]
        term = s[:, -1]
        next_obs = obs + delta
        term = (term > self.terminal_threshold).astype(np.float32)

        return obs, act, rew, next_obs, term
"""Forward diffusion utilities for the DDPM experiment."""

from __future__ import annotations

import torch


class GaussianDiffusion:
    """DDPM forward process using a linear beta schedule."""

    def __init__(
        self,
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: torch.device | str = "cpu",
    ) -> None:
        self.timesteps = timesteps
        self.device = torch.device(device)

        self.betas = torch.linspace(
            beta_start, beta_end, timesteps, device=self.device, dtype=torch.float32
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, x_shape: tuple[int, ...]) -> torch.Tensor:
        """Select one schedule value per batch item and reshape for broadcasting."""
        selected = values.gather(0, timesteps)
        return selected.reshape(timesteps.shape[0], *((1,) * (len(x_shape) - 1)))

    def add_noise(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample x_t from x_0 using the closed-form forward equation."""
        if noise is None:
            noise = torch.randn_like(clean_images)

        sqrt_alpha_bar = self._extract(
            self.sqrt_alpha_bars, timesteps, clean_images.shape
        )
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alpha_bars, timesteps, clean_images.shape
        )

        noisy_images = sqrt_alpha_bar * clean_images + sqrt_one_minus_alpha_bar * noise
        return noisy_images, noise

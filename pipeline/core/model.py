"""Small U-Net used to predict diffusion noise."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.embedding_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(self.activation(self.norm1(x)))
        x = x + self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        x = self.conv2(self.activation(self.norm2(x)))
        return x + residual


class SmallUNet(nn.Module):
    """A compact U-Net for 64x64 RGB images."""

    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64, num_classes: int | None = None) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.num_classes = num_classes
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_embedding = nn.Embedding(num_classes, time_dim) if num_classes else None

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)

        self.up2 = ResidualBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)
        self.up1 = ResidualBlock(base_channels * 2 + base_channels, base_channels, time_dim)
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels are required for a conditional U-Net")
            time_embedding = time_embedding + self.class_embedding(class_labels)

        x = self.input_conv(x)
        skip1 = self.down1(x, time_embedding)
        x = self.downsample1(skip1)
        skip2 = self.down2(x, time_embedding)
        x = self.downsample2(skip2)
        x = self.mid(x, time_embedding)

        x = F.interpolate(x, size=skip2.shape[-2:], mode="nearest")
        x = self.up2(torch.cat((x, skip2), dim=1), time_embedding)
        x = F.interpolate(x, size=skip1.shape[-2:], mode="nearest")
        x = self.up1(torch.cat((x, skip1), dim=1), time_embedding)
        return self.output_conv(self.activation(self.output_norm(x)))

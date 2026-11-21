"""
AttentionUNet256: 13.8M Parameter U-Net with Multi-Head Flash Self-Attention for 256x256 image diffusion.
- Parameters: 13,820,003 (~13.82 M)
- Memory Profile: ~2.8 GB VRAM, ~1.8 GB System RAM (Batch Size 6)
- Conditioning: 5 classes (0..4) + null token (index 5) for Classifier-Free Guidance (CFG).
"""
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


class FlashAttentionBlock(nn.Module):
    """Memory-efficient Multi-Head Self-Attention using PyTorch scaled_dot_product_attention."""
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q = qkv[:, 0].transpose(-1, -2)  # [b, num_heads, hw, head_dim]
        k = qkv[:, 1].transpose(-1, -2)
        v = qkv[:, 2].transpose(-1, -2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(-1, -2).reshape(b, c, h, w)
        return x + self.proj_out(out)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        res = self.skip(x)
        h = self.conv1(self.activation(self.norm1(x)))
        h = h + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.activation(self.norm2(h)))
        return res + h


class AttentionUNet256(nn.Module):
    """
    13.8M Parameter U-Net with FlashAttention for native 256x256 concrete crack synthesis.
    Stages: 256 -> 128 -> 64 -> 32 -> 16
    """
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 40,
        num_classes: int = 5,
        cfg_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.num_classes = num_classes
        self.cfg_dropout = cfg_dropout
        self.null_token_idx = num_classes

        # Embeddings
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, time_dim)

        # Stage 1: 256x256 (channels: 40)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down1 = ResidualBlock(base_channels, base_channels, time_dim)
        self.downsample1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)

        # Stage 2: 128x128 (channels: 80)
        self.down2 = ResidualBlock(base_channels * 2, base_channels * 2, time_dim)
        self.downsample2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)

        # Stage 3: 64x64 (channels: 160)
        self.down3 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.downsample3 = nn.Conv2d(base_channels * 4, base_channels * 6, kernel_size=4, stride=2, padding=1)

        # Stage 4: 32x32 (channels: 240) with 2 ResBlocks + FlashAttention
        self.down4_1 = ResidualBlock(base_channels * 6, base_channels * 6, time_dim)
        self.down4_2 = ResidualBlock(base_channels * 6, base_channels * 6, time_dim)
        self.attn4 = FlashAttentionBlock(base_channels * 6, num_heads=4)
        self.downsample4 = nn.Conv2d(base_channels * 6, base_channels * 8, kernel_size=4, stride=2, padding=1)

        # Bottleneck: 16x16 (channels: 320) with FlashAttention
        self.mid1 = ResidualBlock(base_channels * 8, base_channels * 8, time_dim)
        self.mid_attn = FlashAttentionBlock(base_channels * 8, num_heads=4)
        self.mid2 = ResidualBlock(base_channels * 8, base_channels * 8, time_dim)

        # Up Stage 4: 32x32 with FlashAttention
        self.up4_1 = ResidualBlock(base_channels * 8 + base_channels * 6, base_channels * 6, time_dim)
        self.up4_2 = ResidualBlock(base_channels * 6, base_channels * 6, time_dim)
        self.up_attn4 = FlashAttentionBlock(base_channels * 6, num_heads=4)

        # Up Stage 3: 64x64
        self.up3 = ResidualBlock(base_channels * 6 + base_channels * 4, base_channels * 4, time_dim)

        # Up Stage 2: 128x128
        self.up2 = ResidualBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)

        # Up Stage 1: 256x256
        self.up1 = ResidualBlock(base_channels * 2 + base_channels, base_channels, time_dim)

        # Output
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor,
        force_dropout: bool = False,
    ) -> torch.Tensor:
        if self.training and self.cfg_dropout > 0:
            mask = torch.rand(class_labels.shape, device=class_labels.device) < self.cfg_dropout
            labels = torch.where(mask, self.null_token_idx, class_labels)
        elif force_dropout:
            labels = torch.full_like(class_labels, self.null_token_idx)
        else:
            labels = class_labels

        emb = self.time_embed(timesteps) + self.class_embed(labels)
        h0 = self.init_conv(x)
        s1 = self.down1(h0, emb)
        h1 = self.downsample1(s1)
        s2 = self.down2(h1, emb)
        h2 = self.downsample2(s2)
        s3 = self.down3(h2, emb)
        h3 = self.downsample3(s3)
        s4 = self.attn4(self.down4_2(self.down4_1(h3, emb), emb))
        h4 = self.downsample4(s4)

        m = self.mid2(self.mid_attn(self.mid1(h4, emb)), emb)

        u4 = F.interpolate(m, size=s4.shape[-2:], mode="nearest")
        u4 = self.up_attn4(self.up4_2(self.up4_1(torch.cat([u4, s4], dim=1), emb), emb))

        u3 = F.interpolate(u4, size=s3.shape[-2:], mode="nearest")
        u3 = self.up3(torch.cat([u3, s3], dim=1), emb)

        u2 = F.interpolate(u3, size=s2.shape[-2:], mode="nearest")
        u2 = self.up2(torch.cat([u2, s2], dim=1), emb)

        u1 = F.interpolate(u2, size=s1.shape[-2:], mode="nearest")
        u1 = self.up1(torch.cat([u1, s1], dim=1), emb)

        return self.out_conv(self.activation(self.out_norm(u1)))

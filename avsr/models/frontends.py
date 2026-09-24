"""Per-modality frontends: mouth-crop video, lip skeleton/colour cues, and stacked-fbank audio (SPEC section 10).

Every frontend maps a padded batch ``[B, T, ...]`` to frame features ``[B, T, out_dim]`` at the same frame rate
(25 fps, no subsampling). ``mask`` is a bool ``[B, T]`` tensor, True on real (non-padded) frames; frontends with a
temporal receptive field zero padded frames first, so padding content never leaks into real frames.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

LM_DIM = 80   # 40 lip landmarks x (x, y), SPEC section 4/6
CUE_DIM = 8   # geometry (4) + colour cues (4)


def _resnet_trunk(name: str) -> tuple[nn.Sequential, int, int]:
    """torchvision ResNet stages 1-4 (conv1/bn1/maxpool/fc removed), their input width and output width."""
    if not name.startswith("resnet") or not hasattr(torchvision.models, name):
        raise ValueError(f"unsupported visual resnet {name!r}; expected e.g. 'resnet18' or 'resnet34'")
    net = getattr(torchvision.models, name)(weights=None)
    trunk = nn.Sequential(net.layer1, net.layer2, net.layer3, net.layer4)
    return trunk, int(net.layer1[0].conv1.in_channels), int(net.fc.in_features)


class VisualFrontend(nn.Module):
    """3D-conv stem + per-frame ResNet trunk + global average pooling: ``[B,T,C,H,W] -> [B,T,out_dim]``.

    After the stem, time is folded into the batch and only real frames (``mask``) go through the 2D trunk,
    in channels_last layout (faster cuDNN kernels under bf16).
    """

    def __init__(self, in_channels: int = 1, out_dim: int = 512, resnet: str = "resnet18") -> None:
        super().__init__()
        self.trunk, stem_channels, trunk_dim = _resnet_trunk(resnet)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, stem_channels, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3),
                      bias=False),
            nn.BatchNorm3d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )
        self.proj: nn.Module = nn.Identity() if trunk_dim == out_dim else nn.Linear(trunk_dim, out_dim)
        self.trunk.to(memory_format=torch.channels_last)
        self.in_channels = in_channels
        self.out_dim = out_dim

    def forward(self, video: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if video.dim() != 5 or video.size(2) != self.in_channels:
            raise ValueError(f"video must be [B,T,{self.in_channels},H,W], got {tuple(video.shape)}")
        b, t = video.shape[:2]
        x = video.transpose(1, 2)                                  # [B, C, T, H, W]
        if mask is not None:
            x = x * mask[:, None, :, None, None].to(x.dtype)       # padded frames -> 0 (like the conv's own padding)
        x = self.stem(x)                                           # [B, 64, T, H/4, W/4]
        c, h, w = x.size(1), x.size(3), x.size(4)
        # [B, C, T, h, w] -> [B*T, C, h, w] with channels_last strides (one copy).
        x = x.permute(0, 2, 3, 4, 1).reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        idx = None
        if mask is not None:
            idx = mask.reshape(-1).nonzero().squeeze(1)
            x = x.index_select(0, idx)
        x = self.trunk(x)                                          # [N, D, h', w']
        x = self.proj(x.mean(dim=(2, 3)))                          # [N, out_dim]
        if idx is not None:
            x = x.new_zeros(b * t, x.size(1)).index_copy(0, idx, x)
        return x.view(b, t, self.out_dim)


class SkeletonFrontend(nn.Module):
    """LayerNorm -> Linear -> GELU -> temporal Conv1d -> GELU -> Linear on ``concat(lm, cue) * valid``."""

    def __init__(self, in_dim: int = LM_DIM + CUE_DIM, hidden: int = 256, out_dim: int = 256,
                 kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to keep the frame count unchanged")
        self.norm = nn.LayerNorm(in_dim)
        self.fc_in = nn.Linear(in_dim, hidden)
        self.conv = nn.Conv1d(hidden, hidden, kernel_size, padding=kernel_size // 2)
        self.fc_out = nn.Linear(hidden, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, lm: torch.Tensor, cue: torch.Tensor, valid: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([lm, cue], dim=-1)                           # [B, T, 88]
        if x.size(-1) != self.in_dim:
            raise ValueError(f"skeleton input dim {x.size(-1)} != expected {self.in_dim} (lm {lm.shape}, cue {cue.shape})")
        x = x * valid.to(x.dtype).unsqueeze(-1)
        x = F.gelu(self.fc_in(self.norm(x)))
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)                 # no leakage from padded frames through the conv
        x = F.gelu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        return self.fc_out(x)


class AudioFrontend(nn.Module):
    """Linear -> LayerNorm -> GELU -> Linear on stacked fbank frames ``[B,T,in_dim] -> [B,T,out_dim]``."""

    def __init__(self, in_dim: int = 320, out_dim: int = 512) -> None:
        super().__init__()
        self.fc_in = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.fc_out = nn.Linear(out_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.size(-1) != self.in_dim:
            raise ValueError(f"audio input dim {audio.size(-1)} != expected {self.in_dim}")
        return self.fc_out(F.gelu(self.norm(self.fc_in(audio))))


__all__ = ["VisualFrontend", "SkeletonFrontend", "AudioFrontend", "LM_DIM", "CUE_DIM"]

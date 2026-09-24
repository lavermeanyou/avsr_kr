"""Modality fusion: per-sample modality dropout, presence embedding, projection and sinusoidal positions (SPEC 10)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

MODES = ("av", "audio", "video", "train")


def sinusoidal_table(length: int, dim: int, device: torch.device | None = None) -> torch.Tensor:
    """Classic Transformer sinusoidal position table ``[length, dim]`` (float32)."""
    half = (dim + 1) // 2
    pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=device) * (-math.log(10000.0) / half))
    table = torch.zeros(length, 2 * half, device=device)
    table[:, 0::2] = torch.sin(pos * freq)
    table[:, 1::2] = torch.cos(pos * freq)
    return table[:, :dim]


class SinusoidalPositionalEncoding(nn.Module):
    """Adds a fixed sinusoidal table to ``[B, T, D]`` inputs (not stored in the state dict)."""

    def __init__(self, dim: int, max_len: int = 2048) -> None:
        super().__init__()
        self.dim = dim
        self.register_buffer("table", sinusoidal_table(max_len, dim), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        table = self.table if t <= self.table.size(0) else sinusoidal_table(t, self.dim, x.device)
        return x + table[:t].to(x.dtype)


def modality_flags(mode: str, batch_size: int, probs: tuple[float, float, float],
                   device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample (audio_on, visual_on) bool flags ``[B]`` for a mode.

    ``train`` samples each sample's condition from ``probs = (p_av, p_audio_only, p_video_only)`` using the global
    CPU generator (deterministic under ``torch.manual_seed``), so one batch mixes the three conditions.
    """
    if mode == "av":
        a = v = torch.ones(batch_size, dtype=torch.bool)
    elif mode == "audio":
        a, v = torch.ones(batch_size, dtype=torch.bool), torch.zeros(batch_size, dtype=torch.bool)
    elif mode == "video":
        a, v = torch.zeros(batch_size, dtype=torch.bool), torch.ones(batch_size, dtype=torch.bool)
    elif mode == "train":
        cond = torch.multinomial(torch.tensor(probs, dtype=torch.float64), batch_size, replacement=True)
        a, v = cond != 2, cond != 1   # 0 = av, 1 = audio only, 2 = video only
    else:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return a.to(device), v.to(device)


POS_ENCODINGS = ("none", "sinusoidal")


class ModalityFusion(nn.Module):
    """``concat(audio, video, skeleton)`` (dropped streams zeroed) -> Linear -> + presence embedding -> LayerNorm
    -> [+ sinusoidal positions] -> dropout. The presence embedding is indexed by ``2 * audio_on + visual_on``.

    ``pos_enc='none'`` (recommended): the Conformer's depthwise convolutions already give relative position. An absolute
    sinusoidal table added at the same scale as the LayerNorm-ed features let the model fit a position-only output
    (identical hypotheses for every input) and kept CTC on its plateau, whereas position-free models left it."""

    def __init__(self, audio_dim: int, video_dim: int, skeleton_dim: int, d_model: int, dropout: float = 0.1,
                 pos_enc: str = "sinusoidal") -> None:
        super().__init__()
        if pos_enc not in POS_ENCODINGS:
            raise ValueError(f"pos_enc must be one of {POS_ENCODINGS}, got {pos_enc!r}")
        self.dims = (audio_dim, video_dim, skeleton_dim)
        self.proj = nn.Linear(audio_dim + video_dim + skeleton_dim, d_model)
        self.presence = nn.Embedding(4, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos: nn.Module = SinusoidalPositionalEncoding(d_model) if pos_enc == "sinusoidal" else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, audio: torch.Tensor | None, video: torch.Tensor | None, skeleton: torch.Tensor | None,
                audio_on: torch.Tensor, visual_on: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Streams are ``[B, T, dim]`` or None (= absent, zeros); flags ``[B]`` bool; ``mask`` ``[B, T]`` bool.
        Returns ``[B, T, d_model]`` with padded frames set to 0."""
        b, t = mask.shape
        ref = next(s for s in (audio, video, skeleton) if s is not None)
        parts = []
        for feat, on, dim in zip((audio, video, skeleton), (audio_on, visual_on, visual_on), self.dims):
            if feat is None:
                parts.append(ref.new_zeros(b, t, dim))
            else:
                parts.append(feat * on.view(b, 1, 1).to(feat.dtype))
        x = self.proj(torch.cat(parts, dim=-1))
        x = x + self.presence(2 * audio_on.long() + visual_on.long()).unsqueeze(1).to(x.dtype)
        x = self.pos(self.norm(x))
        return self.dropout(x) * mask.unsqueeze(-1).to(x.dtype)


__all__ = ["MODES", "ModalityFusion", "SinusoidalPositionalEncoding", "modality_flags", "sinusoidal_table"]

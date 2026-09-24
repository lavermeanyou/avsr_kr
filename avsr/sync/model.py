"""Dual encoder for speaker-lip matching (docs/SYNC_SPEC.md section 3).

Video branch: ``VisualFrontend`` (mouth crops) and/or ``SkeletonFrontend`` (lip landmarks + cues, masked by
``valid``) -> concat -> Linear -> temporal encoder -> Linear -> L2 norm, one embedding per 25-fps frame.
Audio branch: ``AudioFrontend`` (stacked fbank) -> Linear -> temporal encoder -> Linear -> L2 norm.
A window's match score is the time-averaged cosine of same-index frames (:func:`avsr.sync.losses.pair_scores`).

Temporal receptive fields stay local, so a frame's embedding describes that moment's articulation: ``conv`` =
4 residual dilated blocks (kernel 5, dilations 1,2,4,1: 33 frames = +-0.64 s; +-0.72 s in the video branch with the
frontends' own 5-frame convolutions), ``gru`` = 2-layer bidirectional GRU over the window only.

The frontends have the AVSR model's module names and shapes, so :meth:`SyncModel.init_from_avsr` can start them from
a trained AVSR checkpoint (``sync.init_from``). :func:`build_sync_model` never loads weights by itself.
"""
from __future__ import annotations

import logging
import math
import os
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from avsr.dataset import cfg_get
from avsr.models.frontends import CUE_DIM, LM_DIM, AudioFrontend, SkeletonFrontend, VisualFrontend
from avsr.sync.losses import frame_scores, pair_scores, sync_loss
from avsr.utils import load_checkpoint

log = logging.getLogger(__name__)

TEMPORAL_TYPES = ("conv", "gru")
AVSR_FRONTENDS = ("visual_frontend", "skeleton_frontend", "audio_frontend")
_WRAPPER_PREFIXES = ("module.", "_orig_mod.")  # DataParallel / torch.compile key prefixes


class DilatedConvEncoder(nn.Module):
    """Residual blocks ``x + Dropout(GELU(LayerNorm(Conv1d(x))))`` with 'same' padding over time ``[B,T,D]``."""

    def __init__(self, dim: int, kernel_size: int = 5, dilations: tuple[int, ...] = (1, 2, 4, 1),
                 dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to keep the frame count unchanged")
        self.convs = nn.ModuleList(nn.Conv1d(dim, dim, kernel_size, dilation=d, padding=d * (kernel_size // 2))
                                   for d in dilations)
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in dilations)
        self.dropout = nn.Dropout(dropout)
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            y = conv(x.transpose(1, 2)).transpose(1, 2)
            x = x + self.dropout(F.gelu(norm(y)))
        return x


class BiGRUEncoder(nn.Module):
    """2-layer bidirectional GRU, ``dim // 2`` units per direction: ``[B,T,D] -> [B,T,D]``.

    Runs in float32 (like the AVSR BiLSTM: cuDNN RNNs under bf16 autocast are not reliable). The layers are separate
    single-layer GRUs with ``nn.Dropout`` in between (= ``nn.GRU(num_layers=2, dropout=p)``): on this Windows /
    torch 2.11 setup a cuDNN RNN built with inter-layer dropout makes the process exit with 0xC0000409 at shutdown."""

    def __init__(self, dim: int, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError(f"sync.hidden must be even for the bidirectional GRU, got {dim}")
        self.layers = nn.ModuleList(nn.GRU(dim, dim // 2, batch_first=True, bidirectional=True)
                                    for _ in range(num_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            h = x.float()
            for i, rnn in enumerate(self.layers):
                if i:
                    h = self.dropout(h)
                h = rnn(h)[0]
        return h


def _temporal(kind: str, dim: int, dropout: float) -> nn.Module:
    if kind == "conv":
        return DilatedConvEncoder(dim, dropout=dropout)
    if kind == "gru":
        return BiGRUEncoder(dim, dropout=dropout)
    raise ValueError(f"unknown sync.temporal {kind!r}; expected one of {TEMPORAL_TYPES}")


class SyncModel(nn.Module):
    """Video/audio dual encoder with per-frame L2-normalised embeddings and a learnable logit scale."""

    def __init__(self, video_channels: int = 1, audio_in_dim: int = 320, hidden: int = 256, emb_dim: int = 256,
                 temporal: str = "conv", use_cnn: bool = True, use_skeleton: bool = True, visual_dim: int = 512,
                 visual_resnet: str = "resnet18", skeleton_hidden: int = 256, skeleton_dim: int = 256,
                 audio_dim: int = 256, dropout: float = 0.1, logit_scale_init: float = 10.0,
                 logit_scale_max: float = 100.0) -> None:
        super().__init__()
        if not (use_cnn or use_skeleton):
            raise ValueError("at least one of sync.use_cnn / sync.use_skeleton must be true")
        if not 0 < logit_scale_init <= logit_scale_max:
            raise ValueError(f"need 0 < logit scale init ({logit_scale_init}) <= max ({logit_scale_max})")
        self.visual_frontend = VisualFrontend(video_channels, visual_dim, visual_resnet) if use_cnn else None
        self.skeleton_frontend = (SkeletonFrontend(LM_DIM + CUE_DIM, skeleton_hidden, skeleton_dim)
                                  if use_skeleton else None)
        self.audio_frontend = AudioFrontend(audio_in_dim, audio_dim)
        video_in = (visual_dim if use_cnn else 0) + (skeleton_dim if use_skeleton else 0)
        self.video_proj = nn.Linear(video_in, hidden)
        self.video_temporal = _temporal(temporal, hidden, dropout)
        self.video_out = nn.Linear(hidden, emb_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden)
        self.audio_temporal = _temporal(temporal, hidden, dropout)
        self.audio_out = nn.Linear(hidden, emb_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(float(logit_scale_init))))
        self.logit_scale_max = float(logit_scale_max)
        self.video_channels = int(video_channels)
        self.emb_dim = int(emb_dim)
        self.temporal = temporal

    # --- embeddings --------------------------------------------------------------------------------------------------
    def embed_video(self, video: torch.Tensor | None, lm: torch.Tensor, cue: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
        """``video [B,W,C,88,88]`` (may be None without the CNN), ``lm [B,W,80]``, ``cue [B,W,8]``, ``valid [B,W]``
        -> ``[B,W,E]`` float32, L2-normalised per frame."""
        feats = []
        if self.visual_frontend is not None:
            if video is None:
                raise ValueError("this SyncModel uses the mouth-crop CNN (sync.use_cnn) but video is None")
            feats.append(self.visual_frontend(video))
        if self.skeleton_frontend is not None:
            feats.append(self.skeleton_frontend(lm, cue, valid))
        x = self.video_temporal(self.video_proj(torch.cat(feats, dim=-1)))
        return F.normalize(self.video_out(x).float(), dim=-1)

    def embed_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """``audio [B,W,320]`` -> ``[B,W,E]`` float32, L2-normalised per frame."""
        x = self.audio_temporal(self.audio_proj(self.audio_frontend(audio)))
        return F.normalize(self.audio_out(x).float(), dim=-1)

    def forward(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """Embeddings of a :func:`avsr.sync.data.sync_collate` batch: ``v``, ``a`` and (when the batch carries
        ``audio_shift``) ``a_shift``, each ``[B,W,E]``. Audio and shifted audio share one audio-branch pass."""
        v = self.embed_video(batch.get("video"), batch["lm"], batch["cue"], batch["valid"])
        audio = batch["audio"]
        if batch.get("audio_shift") is None:
            return {"v": v, "a": self.embed_audio(audio)}
        a_all = self.embed_audio(torch.cat([audio, batch["audio_shift"]], dim=0))
        a, a_shift = a_all.split(audio.shape[0], dim=0)
        return {"v": v, "a": a, "a_shift": a_shift}

    # --- scores / loss -----------------------------------------------------------------------------------------------
    def logit_scale_value(self) -> torch.Tensor:
        """exp(logit_scale) clamped to ``logit_scale_max`` (float32 scalar, differentiable below the clamp)."""
        return self.logit_scale.float().exp().clamp(max=self.logit_scale_max)

    @staticmethod
    def pair_scores(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """``[B,B']`` mean-over-time cosine of every video/audio window pair (float32)."""
        return pair_scores(v, a)

    @staticmethod
    def frame_scores(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """``[B,W]`` per-frame cosine of matched pairs (float32)."""
        return frame_scores(v, a)

    def compute_loss(self, out: Mapping[str, torch.Tensor], batch: Mapping[str, Any]
                     ) -> tuple[torch.Tensor, dict[str, float]]:
        """Loss of docs/SYNC_SPEC.md section 4 on ``out = model(batch)``; uses ``batch['has_shift']`` and
        ``batch['audio_key_ids']`` when present. See :func:`avsr.sync.losses.sync_loss` for ``parts``."""
        return sync_loss(out["v"], out["a"], self.logit_scale_value(), a_shift=out.get("a_shift"),
                         has_shift=batch.get("has_shift"), audio_key_ids=batch.get("audio_key_ids"))

    # --- initialisation from an AVSR checkpoint ----------------------------------------------------------------------
    def init_from_avsr(self, ckpt_path: str | os.PathLike) -> dict[str, Any]:
        """Copy ``visual_frontend.*``, ``skeleton_frontend.*`` and ``audio_frontend.*`` tensors from an AVSR checkpoint
        (``ckpt['model']``) wherever name and shape match. Returns ``{"ckpt", "copied": [names], "skipped":
        ["name: reason"], "modules": {frontend: {"copied", "skipped", "total"}}}``; nothing else is touched."""
        ckpt = load_checkpoint(ckpt_path, map_location="cpu")
        src = ckpt.get("model", ckpt) if isinstance(ckpt, Mapping) else None
        if not isinstance(src, Mapping) or not src:
            raise ValueError(f"{ckpt_path}: no model state dict (expected ckpt['model'])")
        clean: dict[str, torch.Tensor] = {}
        for key, value in src.items():
            for prefix in _WRAPPER_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
            clean[key] = value
        own = self.state_dict()
        report: dict[str, Any] = {"ckpt": str(ckpt_path), "copied": [], "skipped": [], "modules": {}}
        new_state: dict[str, torch.Tensor] = {}
        for module in AVSR_FRONTENDS:
            if getattr(self, module) is None:
                continue
            keys = [k for k in own if k.startswith(module + ".")]
            stats = {"copied": 0, "skipped": 0, "total": len(keys)}
            for k in keys:
                if k not in clean:
                    report["skipped"].append(f"{k}: not in checkpoint")
                    stats["skipped"] += 1
                elif tuple(clean[k].shape) != tuple(own[k].shape):
                    report["skipped"].append(f"{k}: shape {tuple(clean[k].shape)} != {tuple(own[k].shape)}")
                    stats["skipped"] += 1
                else:
                    new_state[k] = clean[k].to(dtype=own[k].dtype)
                    report["copied"].append(k)
                    stats["copied"] += 1
            report["modules"][module] = stats
        result = self.load_state_dict(new_state, strict=False)
        if result.unexpected_keys:  # cannot happen: every key was taken from our own state dict
            raise RuntimeError(f"unexpected keys while loading AVSR frontends: {result.unexpected_keys}")
        log.info("init_from_avsr %s: %s", ckpt_path,
                 ", ".join(f"{m} {s['copied']}/{s['total']} copied" for m, s in report["modules"].items()))
        return report


def build_sync_model(cfg: Any) -> SyncModel:
    """SyncModel from ``cfg.sync.*`` (hidden, emb_dim, temporal, use_cnn, use_skeleton, dropout, resnet,
    logit_scale_init, logit_scale_max), ``cfg.video.channels`` and ``cfg.audio.n_mels * cfg.audio.stack``.
    Weights are random; call :meth:`SyncModel.init_from_avsr` for ``sync.init_from``."""
    return SyncModel(
        video_channels=int(cfg_get(cfg, "video.channels", 1)),
        audio_in_dim=int(cfg_get(cfg, "audio.n_mels", 80)) * int(cfg_get(cfg, "audio.stack", 4)),
        hidden=int(cfg_get(cfg, "sync.hidden", 256)),
        emb_dim=int(cfg_get(cfg, "sync.emb_dim", 256)),
        temporal=str(cfg_get(cfg, "sync.temporal", "conv")),
        use_cnn=bool(cfg_get(cfg, "sync.use_cnn", True)),
        use_skeleton=bool(cfg_get(cfg, "sync.use_skeleton", True)),
        visual_resnet=str(cfg_get(cfg, "sync.resnet", "resnet18")),
        dropout=float(cfg_get(cfg, "sync.dropout", 0.1)),
        logit_scale_init=float(cfg_get(cfg, "sync.logit_scale_init", 10.0)),
        logit_scale_max=float(cfg_get(cfg, "sync.logit_scale_max", 100.0)),
    )


__all__ = ["SyncModel", "build_sync_model", "DilatedConvEncoder", "BiGRUEncoder", "TEMPORAL_TYPES", "AVSR_FRONTENDS"]

"""AVSRModel: frontends -> modality fusion -> Conformer -> CTC head + attention decoder + SNR head (SPEC section 10).

The encoder runs at the 25 fps input frame rate (no subsampling). ``enc_lengths == batch["lengths"]``.
Jamo targets are ~16 tokens/s, i.e. up to ~0.8 tokens per 25-fps frame: feasible for CTC but so dense that training
from scratch never left the blank plateau (the model memorises a few utterances but does not start to generalise).
``ctc_upsample = U`` splits every encoder frame into U CTC sub-frames (Linear d -> U*d, GELU, LayerNorm) before the CTC
head, so CTC runs at U x 25 Hz (density ~0.3-0.4 at U = 2) while encoder, decoder and fusion stay at 25 Hz.
``ctc_logits`` is ``[B, U*T, V]`` and ``ctc_lengths == U * enc_lengths``.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..text import Tokenizer
from .decoder import AttentionDecoder
from .encoder import ENCODER_TYPES, BLSTMEncoder, ConformerEncoder, lengths_to_padding_mask
from .frontends import CUE_DIM, LM_DIM, AudioFrontend, SkeletonFrontend, VisualFrontend
from .fusion import MODES, ModalityFusion, modality_flags

N_SNR_BUCKETS = 4          # 0: >=20 dB, 1: 10-20, 2: 0-10, 3: <0 (SPEC section 8)
DECODE_METHODS = ("ctc_greedy", "attn_greedy")
_MISSING = object()


def cfg_get(cfg: Any, path: str, default: Any = _MISSING) -> Any:
    """Read a dotted key from a nested dict or attribute-style config; raise KeyError when missing and no default."""
    node = cfg
    for key in path.split("."):
        if isinstance(node, Mapping) and key in node:
            node = node[key]
        elif not isinstance(node, Mapping) and hasattr(node, key):
            node = getattr(node, key)
        else:
            if default is _MISSING:
                raise KeyError(f"config key {path!r} is missing (see SPEC section 9)")
            return default
    return node


def ctc_collapse(best: torch.Tensor, mask: torch.Tensor, drop_ids: tuple[int, ...]) -> list[list[int]]:
    """Greedy CTC post-processing of frame argmax ids ``[B, T]``: merge repeats on the raw sequence (so a blank
    between two equal ids keeps both), then drop ``drop_ids`` (blank and other specials) and padded frames."""
    prev = F.pad(best[:, :-1], (1, 0), value=-1)
    keep = (best != prev) & mask
    for special in drop_ids:
        keep &= best != special
    best, keep = best.cpu(), keep.cpu()
    return [best[i][keep[i]].tolist() for i in range(best.size(0))]


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over time of ``x [B, T, D]`` on frames where ``mask [B, T]`` is True (float32)."""
    m = mask.unsqueeze(-1).float()
    return (x.float() * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


class AVSRModel(nn.Module):
    """Audio-visual CTC/attention speech recogniser with an auxiliary SNR-bucket head."""

    def __init__(self, vocab_size: int, *, video_channels: int = 1, audio_in_dim: int = 320, d_model: int = 512,
                 visual_resnet: str = "resnet18", visual_dim: int = 512, skeleton_hidden: int = 256,
                 skeleton_dim: int = 256, audio_dim: int = 512, enc_layers: int = 8, enc_heads: int = 8,
                 enc_ffn: int = 2048, enc_conv_kernel: int = 31, enc_dropout: float = 0.1, dec_layers: int = 4,
                 dec_heads: int = 8, dec_ffn: int = 2048, dec_dropout: float = 0.1,
                 fusion_probs: tuple[float, float, float] = (0.5, 0.25, 0.25), ctc_upsample: int = 1,
                 pos_enc: str = "sinusoidal", enc_type: str = "conformer", rnn_layers: int = 4,
                 rnn_hidden: int = 320) -> None:
        super().__init__()
        if len(fusion_probs) != 3 or min(fusion_probs) < 0 or sum(fusion_probs) <= 0:
            raise ValueError(f"fusion probabilities must be 3 non-negative numbers with a positive sum: {fusion_probs}")
        if int(ctc_upsample) < 1:
            raise ValueError(f"ctc_upsample must be >= 1, got {ctc_upsample}")
        self.ctc_upsample = int(ctc_upsample)
        self.vocab_size = vocab_size
        self.blank_id, self.pad_id = Tokenizer.blank_id, Tokenizer.pad_id
        self.sos_id, self.eos_id = Tokenizer.sos_id, Tokenizer.eos_id
        self.fusion_probs = tuple(float(p) for p in fusion_probs)
        self.visual_frontend = VisualFrontend(video_channels, visual_dim, visual_resnet)
        self.skeleton_frontend = SkeletonFrontend(LM_DIM + CUE_DIM, skeleton_hidden, skeleton_dim)
        self.audio_frontend = AudioFrontend(audio_in_dim, audio_dim)
        self.fusion = ModalityFusion(audio_dim, visual_dim, skeleton_dim, d_model, enc_dropout, pos_enc)
        if enc_type not in ENCODER_TYPES:
            raise ValueError(f"encoder type must be one of {ENCODER_TYPES}, got {enc_type!r}")
        self.encoder: nn.Module = (
            ConformerEncoder(d_model, enc_layers, enc_heads, enc_ffn, enc_conv_kernel, enc_dropout)
            if enc_type == "conformer" else BLSTMEncoder(d_model, rnn_layers, rnn_hidden, enc_dropout))
        # U > 1: each encoder frame -> U CTC sub-frames (absent when U == 1, so old checkpoints load unchanged)
        self.ctc_up: nn.Module | None = (nn.Sequential(nn.Linear(d_model, d_model * self.ctc_upsample), nn.GELU())
                                         if self.ctc_upsample > 1 else None)
        self.ctc_up_norm: nn.Module | None = nn.LayerNorm(d_model) if self.ctc_upsample > 1 else None
        self.ctc_head = nn.Linear(d_model, vocab_size)
        self.decoder = AttentionDecoder(vocab_size, d_model, dec_layers, dec_heads, dec_ffn, dec_dropout,
                                        blank_id=self.blank_id, pad_id=self.pad_id, sos_id=self.sos_id,
                                        eos_id=self.eos_id)
        self.snr_head = nn.Linear(audio_dim, N_SNR_BUCKETS)

    # ------------------------------------------------------------------------------------------------ encoding
    def encode(self, batch: Mapping[str, Any], mode: str = "av") -> dict[str, torch.Tensor]:
        """Frontends + fusion + encoder. Returns ``enc_out [B,T,D]``, ``enc_lengths [B]``, ``snr_logits [B,4]`` and
        ``mask [B,T]`` (True on real frames). ``audio`` mode skips the visual/skeleton frontends; ``video`` mode feeds
        zeros instead of audio features to the fusion; ``train`` zeroes streams per sample. The SNR head always looks
        at the audio-frontend features, whatever the mode (so ``snr_logits`` is the one output that still depends on
        the audio in ``video`` mode)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        audio = batch["audio"]
        b, t = audio.shape[:2]
        lengths = batch["lengths"].to(device=audio.device, dtype=torch.long)
        lo, hi = torch.stack([lengths.min(), lengths.max()]).tolist() if lengths.numel() else (0, 0)
        if lengths.shape != (b,) or lo < 1 or hi > t:
            raise ValueError(f"lengths {lengths.tolist()} do not fit a batch of shape {tuple(audio.shape)}")
        mask = ~lengths_to_padding_mask(lengths, t)
        audio_on, visual_on = modality_flags(mode, b, self.fusion_probs, audio.device)

        audio_feat = self.audio_frontend(audio)
        snr_logits = self.snr_head(_masked_mean(audio_feat, mask))
        video_feat = skel_feat = None
        # skip the visual frontends when no sample uses them (``audio`` mode, or a ``train`` batch whose modality
        # dropout drew audio-only for every sample, e.g. the audio-only stage of train.fusion_curriculum)
        if mode != "audio" and bool(visual_on.any()):
            video = batch["video"]
            if video.shape[:2] != (b, t):
                raise ValueError(f"video {tuple(video.shape)} and audio {tuple(audio.shape)} must share [B, T]")
            video_feat = self.visual_frontend(video, mask)
            skel_feat = self.skeleton_frontend(batch["lm"], batch["cue"], batch["valid"], mask)
        fused = self.fusion(None if mode == "video" else audio_feat, video_feat, skel_feat, audio_on, visual_on, mask)
        enc_out, enc_lengths = self.encoder(fused, lengths)
        return {"enc_out": enc_out, "enc_lengths": enc_lengths, "snr_logits": snr_logits, "mask": mask}

    def ctc_logits(self, enc_out: torch.Tensor, enc_lengths: torch.Tensor, mask: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CTC logits ``[B, U*T, V]``, their lengths ``U * enc_lengths`` and real-frame mask ``[B, U*T]``."""
        u = self.ctc_upsample
        if u == 1:
            return self.ctc_head(enc_out), enc_lengths, mask
        b, t, d = enc_out.shape
        x = self.ctc_up(enc_out).reshape(b, t * u, d)      # sub-frames of frame i are rows i*u .. i*u+u-1
        x = self.ctc_up_norm(x)
        return self.ctc_head(x), enc_lengths * u, mask.repeat_interleave(u, dim=1)

    def forward(self, batch: Mapping[str, Any], mode: str = "av") -> dict[str, torch.Tensor]:
        """SPEC section 10 output dict; ``att_logits`` is ``[B, L + 1, V]`` (targets ``tokens + [eos]``) and is only
        produced when ``batch`` carries ``tokens`` and ``token_lengths``. ``ctc_logits`` is ``[B, U*T, V]`` with
        lengths ``ctc_lengths`` (``U = ctc_upsample``)."""
        enc = self.encode(batch, mode)
        ctc_logits, ctc_lengths, _ = self.ctc_logits(enc["enc_out"], enc["enc_lengths"], enc["mask"])
        out = {"ctc_logits": ctc_logits, "ctc_lengths": ctc_lengths, "enc_out": enc["enc_out"],
               "enc_lengths": enc["enc_lengths"], "snr_logits": enc["snr_logits"]}
        if batch.get("tokens") is not None and batch.get("token_lengths") is not None:
            dec_in, _ = self.decoder.teacher_forcing(batch["tokens"], batch["token_lengths"])
            out["att_logits"] = self.decoder(dec_in, enc["enc_out"], ~enc["mask"])
        return out

    # ------------------------------------------------------------------------------------------------ loss
    def compute_loss(self, out: Mapping[str, torch.Tensor], batch: Mapping[str, Any],
                     cfg: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """``ctc_weight * ctc + (1 - ctc_weight) * att + snr_head_weight * snr``; all losses in float32 and
        normalised per utterance: CTC and attention CE are summed over tokens and divided by B (so ``ctc_weight``
        balances comparable magnitudes, as in ESPnet), SNR CE is the batch mean. ``parts`` also carries ``att_tok``,
        the attention CE per target token, for readable logs."""
        ctc_w = float(cfg_get(cfg, "model.ctc_weight"))
        snr_w = float(cfg_get(cfg, "model.snr_head_weight"))
        smoothing = float(cfg_get(cfg, "model.label_smoothing"))
        tokens = batch["tokens"]
        token_lengths = batch["token_lengths"].to(device=tokens.device, dtype=torch.long)
        b = tokens.size(0)

        log_probs = out["ctc_logits"].float().log_softmax(dim=-1).transpose(0, 1)       # [U*T, B, V]
        ctc = F.ctc_loss(log_probs, tokens, out.get("ctc_lengths", out["enc_lengths"]), token_lengths,
                         blank=self.blank_id, reduction="sum", zero_infinity=True) / b
        zero = ctc.new_zeros(())
        att = att_tok = zero
        if ctc_w < 1.0:
            if "att_logits" not in out:
                raise KeyError("att_logits missing: run forward with tokens when model.ctc_weight < 1")
            _, target = self.decoder.teacher_forcing(tokens, token_lengths)
            att_sum = F.cross_entropy(out["att_logits"].float().flatten(0, 1), target.flatten(),
                                      ignore_index=self.pad_id, label_smoothing=smoothing, reduction="sum")
            att = att_sum / b
            att_tok = att_sum.detach() / (token_lengths.sum() + b)          # + b: one eos per utterance
        snr = zero
        if snr_w > 0.0:
            snr = F.cross_entropy(out["snr_logits"].float(), batch["snr_bucket"].to(tokens.device).long())
        loss = ctc_w * ctc + (1.0 - ctc_w) * att + snr_w * snr
        values = torch.stack([loss, ctc, att, att_tok, snr]).detach().tolist()
        return loss, dict(zip(("loss", "ctc", "att", "att_tok", "snr"), values))

    # ------------------------------------------------------------------------------------------------ decoding
    @torch.no_grad()
    def decode(self, batch: Mapping[str, Any], mode: str = "av", method: str = "ctc_greedy",
               max_len: int | None = None) -> list[list[int]]:
        """Greedy decoding -> token ids per utterance (no blank/pad/sos/eos). ``mode`` in av/audio/video.
        ``max_len`` bounds ``attn_greedy`` (default: the longest input length in frames). Call ``model.eval()``
        first."""
        if mode not in ("av", "audio", "video"):
            raise ValueError(f"decode mode must be av, audio or video, got {mode!r}")
        if method not in DECODE_METHODS:
            raise ValueError(f"decode method must be one of {DECODE_METHODS}, got {method!r}")
        enc = self.encode(batch, mode)
        if method == "attn_greedy":
            limit = int(enc["enc_lengths"].max()) if max_len is None else int(max_len)
            return self.decoder.greedy(enc["enc_out"], ~enc["mask"], limit)
        logits, _, mask = self.ctc_logits(enc["enc_out"], enc["enc_lengths"], enc["mask"])
        best = logits.argmax(dim=-1)                                                      # [B, U*T]
        return ctc_collapse(best, mask, (self.blank_id, self.pad_id, self.sos_id, self.eos_id))


def build_model(cfg: Any, vocab_size: int) -> AVSRModel:
    """Build :class:`AVSRModel` from the SPEC section 9 config (dict or attribute-style)."""
    fusion = cfg_get(cfg, "model.fusion")
    probs = tuple(float(cfg_get(fusion, k)) for k in ("p_av", "p_audio_only", "p_video_only"))
    return AVSRModel(
        vocab_size,
        video_channels=int(cfg_get(cfg, "video.channels")),
        audio_in_dim=int(cfg_get(cfg, "audio.n_mels")) * int(cfg_get(cfg, "audio.stack")),
        d_model=int(cfg_get(cfg, "model.d_model")),
        visual_resnet=str(cfg_get(cfg, "model.visual.resnet")),
        visual_dim=int(cfg_get(cfg, "model.visual.out_dim")),
        skeleton_hidden=int(cfg_get(cfg, "model.skeleton.hidden")),
        skeleton_dim=int(cfg_get(cfg, "model.skeleton.out_dim")),
        audio_dim=int(cfg_get(cfg, "model.audio.out_dim")),
        enc_layers=int(cfg_get(cfg, "model.encoder.layers")),
        enc_heads=int(cfg_get(cfg, "model.encoder.heads")),
        enc_ffn=int(cfg_get(cfg, "model.encoder.ffn")),
        enc_conv_kernel=int(cfg_get(cfg, "model.encoder.conv_kernel")),
        enc_dropout=float(cfg_get(cfg, "model.encoder.dropout")),
        dec_layers=int(cfg_get(cfg, "model.decoder.layers")),
        dec_heads=int(cfg_get(cfg, "model.decoder.heads")),
        dec_ffn=int(cfg_get(cfg, "model.decoder.ffn")),
        dec_dropout=float(cfg_get(cfg, "model.decoder.dropout")),
        fusion_probs=probs,
        ctc_upsample=int(cfg_get(cfg, "model.ctc_upsample", 1)),
        pos_enc=str(cfg_get(cfg, "model.pos_enc", "sinusoidal")),
        enc_type=str(cfg_get(cfg, "model.encoder.type", "conformer")),
        rnn_layers=int(cfg_get(cfg, "model.encoder.rnn_layers", 4)),
        rnn_hidden=int(cfg_get(cfg, "model.encoder.rnn_hidden", 320)),
    )


__all__ = ["AVSRModel", "build_model", "cfg_get", "ctc_collapse", "DECODE_METHODS", "N_SNR_BUCKETS"]

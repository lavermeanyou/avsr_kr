"""Scores and the symmetric contrastive loss of the speaker-lip matching model (docs/SYNC_SPEC.md section 4).

All scores are computed in float32 with autocast disabled, so the loss is exact under bf16 autocast.
Embeddings are per-frame and L2-normalised: ``v [B,W,E]`` (video), ``a [B',W,E]`` (audio).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pair_scores(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """``[B, B']``: mean over time of cos(v[b, t], a[b', t]) (same time index)."""
    if v.dim() != 3 or a.dim() != 3 or v.shape[1:] != a.shape[1:]:
        raise ValueError(f"pair_scores needs [B,W,E] and [B',W,E] with equal W and E, got {tuple(v.shape)}, "
                         f"{tuple(a.shape)}")
    with torch.autocast(device_type=v.device.type, enabled=False):
        return torch.einsum("btd,ctd->bc", v.float(), a.float()) / v.shape[1]


def frame_scores(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """``[B, W]``: per-frame cosine of matched pairs (row b of ``v`` with row b of ``a``)."""
    if v.shape != a.shape or v.dim() != 3:
        raise ValueError(f"frame_scores needs two [B,W,E] tensors of the same shape, got {tuple(v.shape)}, "
                         f"{tuple(a.shape)}")
    with torch.autocast(device_type=v.device.type, enabled=False):
        return (v.float() * a.float()).sum(dim=-1)


def same_audio_mask(audio_key_ids: torch.Tensor | None, n: int, device: torch.device) -> torch.Tensor:
    """Bool ``[n, n]``: True on OFF-diagonal pairs that share the underlying audio (other camera angles of the same
    recording) - they must not be used as negatives. All False when ``audio_key_ids`` is None."""
    eye = torch.eye(n, dtype=torch.bool, device=device)
    if audio_key_ids is None:
        return torch.zeros(n, n, dtype=torch.bool, device=device)
    ids = audio_key_ids.to(device).view(-1)
    if ids.numel() != n:
        raise ValueError(f"audio_key_ids has {ids.numel()} entries for a batch of {n}")
    return (ids[:, None] == ids[None, :]) & ~eye


def sync_loss(v: torch.Tensor, a: torch.Tensor, scale: torch.Tensor | float, a_shift: torch.Tensor | None = None,
              has_shift: torch.Tensor | None = None, audio_key_ids: torch.Tensor | None = None
              ) -> tuple[torch.Tensor, dict[str, float]]:
    """Symmetric InfoNCE over a batch of B matched windows (``v[b]`` belongs to ``a[b]``).

    ``S = scale * pair_scores(v, a)``; off-diagonal entries of the same recording (``audio_key_ids`` equal) are set
    to -inf. Video->audio: cross-entropy over each row plus, when ``has_shift[b]``, one extra column with the score of
    the same utterance's time-shifted audio ``a_shift[b]`` (hard negative). Audio->video: cross-entropy over each
    column (same mask). ``loss = (l_va + l_av) / 2``.

    ``parts`` (python floats): loss, l_va, l_av, acc_va (row argmax on the diagonal, shift column included), acc_av,
    pos_cos (mean diagonal cosine), neg_cos (mean cosine of the pairs used as negatives), shift_cos (mean cosine of
    the shifted pairs), acc_shift (fraction of shifted rows whose true pair scores higher), scale. neg_cos /
    shift_cos / acc_shift are NaN when the batch has no such pair.
    """
    b = v.shape[0]
    if a.shape[0] != b:
        raise ValueError(f"sync_loss needs matched pairs: {b} video vs {a.shape[0]} audio windows")
    with torch.autocast(device_type=v.device.type, enabled=False):
        scale_t = torch.as_tensor(scale, dtype=torch.float32, device=v.device)
        cos = pair_scores(v, a)                                          # [B, B]
        masked = same_audio_mask(audio_key_ids, b, v.device)
        logits = (scale_t * cos).masked_fill(masked, float("-inf"))
        target = torch.arange(b, device=v.device)

        use_shift = a_shift is not None and has_shift is not None
        if use_shift:
            flags = has_shift.to(device=v.device, dtype=torch.bool).view(-1)
            if flags.numel() != b or a_shift.shape != a.shape:
                raise ValueError(f"a_shift {tuple(a_shift.shape)} / has_shift {tuple(flags.shape)} do not match the "
                                 f"batch ({tuple(a.shape)})")
            shift_cos = frame_scores(v, a_shift).mean(dim=1)             # [B]
            shift_col = (scale_t * shift_cos).masked_fill(~flags, float("-inf"))
            logits_va = torch.cat([logits, shift_col[:, None]], dim=1)
        else:
            logits_va = logits
        l_va = F.cross_entropy(logits_va, target)
        l_av = F.cross_entropy(logits.t(), target)
        loss = 0.5 * (l_va + l_av)

        with torch.no_grad():
            diag = cos.diagonal()
            neg = ~masked & ~torch.eye(b, dtype=torch.bool, device=v.device)
            nan = torch.tensor(float("nan"), device=v.device)
            stats = [
                loss.detach(), l_va.detach(), l_av.detach(),
                (logits_va.argmax(dim=1) == target).float().mean(),
                (logits.t().argmax(dim=1) == target).float().mean(),
                diag.mean(),
                (cos * neg).sum() / neg.sum(),                           # 0/0 = NaN without negatives
                (shift_cos * flags).sum() / flags.sum() if use_shift else nan,
                ((diag > shift_cos) & flags).sum() / flags.sum() if use_shift else nan,
                scale_t.detach(),
            ]
            values = torch.stack([s.float() for s in stats]).tolist()   # one device sync
    names = ("loss", "l_va", "l_av", "acc_va", "acc_av", "pos_cos", "neg_cos", "shift_cos", "acc_shift", "scale")
    return loss, dict(zip(names, (float(x) for x in values)))


__all__ = ["pair_scores", "frame_scores", "same_audio_mask", "sync_loss"]

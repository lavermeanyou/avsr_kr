"""Scenario metrics of the speaker-lip matching model (docs/SYNC_SPEC.md section 6).

Usage (from the project root)::

    & $py -m avsr.sync.evaluate --ckpt work/checkpoints_sync/best.pt --split test
    & $py -m avsr.sync.evaluate --ckpt work/checkpoints_sync/best.pt --split heldout --max-utts 900 --scenes 500
    & $py -m avsr.sync.evaluate --ckpt work/checkpoints_sync/epoch_010.pt --split val --out work/eval/try.json

Held-out speakers only: ``val`` = ``split.val_speakers``, ``test`` = ``split.test_speakers``, ``heldout`` = both
(default; the only split with 3 speakers, so 3-person scenes need no fallback). ``--max-utts`` (default
``sync.eval_max_utts``, 0 = all) takes an evenly spread subset of the split.

Every window is embedded ON ITS OWN, exactly as in training (``SyncWindowDataset`` eval items: centre crop, pixel
z-score over the window, cues normalised over the utterance, audio fbank + CMVN over the window, and the temporal
encoder sees only the window, zero-padded at its edges). The centred window of every utterance is embedded once per
window length and reused by all comparisons; shifted / leaky / scene windows get their own embeddings. So no score
depends on context outside its window (slicing whole-utterance embeddings would not be equivalent: the encoder's
receptive field is 33+ frames, longer than a 1-s window).

Score of a (face, stream) pair = mean over the window of the per-frame cosine (``avsr.sync.losses.pair_scores``).
Sections of the JSON (``<work_dir>/eval/sync_<split>_results.json`` unless ``--out``), per window length:

1. ``nway``: video -> which of N audio streams (own + N-1 distractor streams); ``reverse``: audio -> which of N faces.
2. ``leakage``: the same video -> audio test where every candidate stream contains the other candidates' speech,
   each other speaker ``X`` dB below the stream's own speaker (``sync.eval_leak_db``).
3. ``offset``: scores of the audio shifted by -15..+15 frames; correct = argmax within +-1 frame of 0.
4. ``scenes``: K faces x K streams (+ leakage variants), Hungarian assignment over the whole overlap and per 1-s
   sub-window.
5. ``verification``: true pairs vs pairs with a different speaker's audio: ROC-AUC, EER, threshold at the EER;
   ``offscreen_threshold`` = that threshold (score below it: the face is not the stream's speaker).

Distractors / scene partners are drawn with a fixed seed per query (see :class:`CandidatePool`): one utterance of
each OTHER held-out speaker first (every stream a different person, as in the product scenario); only when the split
has too few speakers, further ones come from recording sessions not used yet (the other speaker again, or the
query's own speaker in another session). The JSON counts every kind and ``notes`` says when the fallback was used.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from torch.utils.data import DataLoader, Dataset

from avsr.audio_feats import add_noise, to_float_wave
from avsr.dataset import (
    _ITEM_ERRORS, CUE_NORMS, VIDEO_NORMS, _audio_key, _limit_worker_threads, assign_split, cfg_get, load_manifests,
    normalize_cue, normalize_pixels, read_frames, resample_indices, resampled_length,
)
from avsr.sync.data import audio_window_batch, audio_window_features, fbank_geometry, num_audio_frames
from avsr.sync.losses import pair_scores
from avsr.sync.model import build_sync_model
from avsr.utils import (Config, autocast_context, close_logger, even_subset, get_logger, load_checkpoint, load_config,
                        safe_console, set_seed)

log = logging.getLogger(__name__)

SPLITS = ("val", "test", "heldout")
OFFSET_MAX = 15              # offset sweep -15..+15 frames (docs/SYNC_SPEC.md section 6.3)
REGION_PAD = 3               # 25-fps frames of waveform kept on each side of a window before mixing (0.12 s)
SCENE_MIN_SEC = 2.0          # scenes: every utterance (and so the common crop) is at least this long
VIDEO_CHUNK_FRAMES = 1600    # frames per video-branch forward pass (64 windows of 25)
AUDIO_CHUNK_FRAMES = 16384
# sync.* keys that define the architecture: always taken from the checkpoint
ARCH_KEYS = ("hidden", "emb_dim", "temporal", "use_cnn", "use_skeleton", "resnet", "dropout", "logit_scale_init",
             "logit_scale_max")
KIND_OTHER = "other_speaker"                   # a held-out speaker not yet in the candidate set
KIND_AGAIN = "other_speaker_other_session"     # fallback: a speaker already in the set, another recording session
KIND_SAME = "same_speaker_other_session"       # fallback: the query's own speaker, another recording session
DISTRACTOR_POLICY = (
    "One window per utterance: its centred window (as the SyncWindowDataset eval items). For every query the "
    "distractors are drawn with a per-query fixed seed: first one utterance of each OTHER held-out speaker (speakers "
    "in random order, any session), so that every candidate belongs to a different person as in the product "
    "scenario; only when the split has fewer speakers than needed, the remaining ones are drawn uniformly among the "
    "(speaker, recording session) pairs not used yet in the candidate set, excluding the query's own session "
    f"(kinds '{KIND_AGAIN}' and '{KIND_SAME}'). A distractor stream/face is that utterance's own centred window "
    "(another person talking at the same moment). N-way candidate sets are nested (N=3 extends N=2); the reverse "
    "direction draws its distractor faces independently with the same rule; the leakage test reuses the video->audio "
    "draws.")
LEAK_RULE = ("stream c = its own speech + every other candidate's speech (the same moment, i.e. their own windows), "
             "each scaled to X dB below stream c (RMS over the window +- 0.12 s, avsr.audio_feats.add_noise); with "
             "N-1 interferers the total interference is X - 10*log10(N-1) dB")


# ----------------------------------------------------------------------------------------------------------------------
# utterances in memory
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class Utterance:
    """One held-out utterance in memory, prepared exactly like :class:`avsr.sync.data.SyncWindowDataset` eval items."""

    utt_id: str
    speaker: str
    session: str
    angle: str
    audio_key: tuple
    n: int                # usable 25-fps frames: min(video, stacked audio)
    n_video: int          # resampled video length from the manifest (the dataset drops rows below window + 2)
    frames: np.ndarray    # uint8 [n, C, crop, crop], centre crop, 25 fps (C=1 gray, C=3 RGB)
    lm: np.ndarray        # float32 [n, 80]
    cue: np.ndarray       # float32 [n, 8], normalised over the n usable frames (video.cue_norm)
    valid: np.ndarray     # float32 [n]
    wave: np.ndarray      # int16 [N] 16 kHz


class UtteranceReader(Dataset):
    """Loads whole utterances (Windows-safe for spawn DataLoader workers). Errors are returned, not raised, so the
    caller counts and logs them (a bad row is skipped like in the training dataset)."""

    def __init__(self, rows: Sequence[dict], cfg: Any) -> None:
        self.rows = list(rows)
        self.work_dir = Path(str(cfg_get(cfg, "work_dir", "work")))
        self.channels = int(cfg_get(cfg, "video.channels", 1))
        self.size = int(cfg_get(cfg, "video.size", 96))
        self.crop = int(cfg_get(cfg, "video.crop", 88))
        self.fps_in = float(cfg_get(cfg, "video.fps_in", 30.0))
        self.fps_out = float(cfg_get(cfg, "video.fps_out", 25.0))
        self.cue_norm = str(cfg_get(cfg, "video.cue_norm", "none"))
        self.sr = int(cfg_get(cfg, "audio.sr", 16000))
        self.stack = int(cfg_get(cfg, "audio.stack", 4))
        if self.channels not in (1, 3):
            raise ValueError(f"video.channels must be 1 or 3, got {self.channels}")
        if not 0 < self.crop <= self.size:
            raise ValueError(f"video.crop ({self.crop}) must be in (0, video.size={self.size}]")
        if self.cue_norm not in CUE_NORMS:
            raise ValueError(f"video.cue_norm must be one of {CUE_NORMS}, got {self.cue_norm!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        try:
            return {"utt": self._load(row)}
        except _ITEM_ERRORS as e:
            return {"utt_id": str(row.get("utt_id")), "error": f"{type(e).__name__}: {e}"}

    def _load(self, row: dict) -> Utterance:
        _limit_worker_threads()
        base = Path(row["work_dir"]) if row.get("work_dir") else self.work_dir
        with np.load(base / str(row["npz"])) as z:
            lm_all = np.asarray(z["lm"])
            cue_all = np.asarray(z["cue"])
            valid_all = np.asarray(z["valid"])
            wave = np.asarray(z["audio"])
            fps = float(z["fps"]) if "fps" in z.files else self.fps_in
            sr = int(z["sr"]) if "sr" in z.files else self.sr
        n_t = int(lm_all.shape[0])
        if lm_all.shape != (n_t, 40, 2) or cue_all.shape != (n_t, 8) or valid_all.shape != (n_t,):
            raise ValueError(f"npz schema mismatch: lm {lm_all.shape}, cue {cue_all.shape}, valid {valid_all.shape}")
        if wave.dtype != np.int16 or wave.ndim != 1:
            raise ValueError(f"npz audio must be 1-D int16, got {wave.dtype} {wave.shape}")
        if sr != self.sr:
            raise ValueError(f"npz sample rate {sr} != audio.sr {self.sr}")
        idx_all = resample_indices(n_t, fps, self.fps_out)
        n = min(int(idx_all.size), num_audio_frames(int(wave.size), self.sr, self.stack))
        if n < 1:
            raise ValueError(f"no usable frames (video {idx_all.size}, audio samples {wave.size})")
        idx = idx_all[:n]
        mp4 = base / str(row["mouth_mp4"])
        frames, n_dec = read_frames(mp4, idx, self.size)
        if frames is None:
            raise ValueError(f"mouth video unreadable: {mp4}")
        if n_dec < int(np.unique(idx).size):
            raise ValueError(f"mouth video {mp4} ends before source frame {int(idx[-1])}")
        m = (self.size - self.crop) // 2
        if self.channels == 1:
            gray = cv2.cvtColor(np.ascontiguousarray(frames).reshape(n * self.size, self.size, 3), cv2.COLOR_BGR2GRAY)
            pix = gray.reshape(n, 1, self.size, self.size)[:, :, m:m + self.crop, m:m + self.crop]
        else:
            pix = frames[:, m:m + self.crop, m:m + self.crop, ::-1].transpose(0, 3, 1, 2)
        n_video = resampled_length(int(row["n_frames"]), float(row.get("fps") or self.fps_in), self.fps_out)
        return Utterance(
            utt_id=str(row["utt_id"]), speaker=str(row.get("speaker", "")), session=str(row.get("session", "")),
            angle=str(row.get("angle", "")), audio_key=_audio_key(row), n=n, n_video=int(n_video),
            frames=np.ascontiguousarray(pix), lm=lm_all[idx].reshape(n, 80).astype(np.float32),
            cue=normalize_cue(cue_all[idx].astype(np.float32), valid_all[idx], self.cue_norm),
            valid=valid_all[idx].astype(np.float32), wave=wave,
        )


def _single(batch: list) -> Any:
    """collate_fn for batch_size=1 loaders: the item itself (top level: picklable for spawn workers)."""
    return batch[0]


class EvalStore:
    """Held-out utterances in memory + the window builders shared by every metric (and by per-epoch validation)."""

    def __init__(self, utts: Sequence[Utterance], cfg: Any) -> None:
        self.utts = list(utts)
        self.norm = str(cfg_get(cfg, "video.norm", "global"))
        if self.norm not in VIDEO_NORMS:
            raise ValueError(f"video.norm must be one of {VIDEO_NORMS}, got {self.norm!r}")
        self.mean = float(cfg_get(cfg, "video.mean", 0.421))
        self.std = float(cfg_get(cfg, "video.std", 0.165))
        self.sr = int(cfg_get(cfg, "audio.sr", 16000))
        self.n_mels = int(cfg_get(cfg, "audio.n_mels", 80))
        self.stack = int(cfg_get(cfg, "audio.stack", 4))
        self.fps_out = float(cfg_get(cfg, "video.fps_out", 25.0))
        self.speakers = np.array([u.speaker for u in self.utts], dtype=object)
        self.sessions = np.array([u.session for u in self.utts], dtype=object)
        self.n_bad = 0
        self.errors: list[str] = []

    @classmethod
    def load(cls, rows: Sequence[dict], cfg: Any, num_workers: int = 0) -> "EvalStore":
        """Decode ``rows`` (spawn workers when ``num_workers`` > 0); unreadable rows are skipped and counted."""
        reader = UtteranceReader(rows, cfg)
        nw = min(max(0, int(num_workers)), math.ceil(len(reader) / 32))
        kwargs: dict[str, Any] = {"multiprocessing_context": "spawn"} if nw > 0 else {}
        loader = DataLoader(reader, batch_size=1, shuffle=False, num_workers=nw, collate_fn=_single, **kwargs)
        utts: list[Utterance] = []
        errors: list[str] = []
        for item in loader:
            if "error" in item:
                errors.append(f"{item['utt_id']}: {item['error']}")
                log.warning("skipping unusable utterance %s", errors[-1])
            else:
                utts.append(item["utt"])
        store = cls(utts, cfg)
        store.n_bad, store.errors = len(errors), errors[:20]
        return store

    def __len__(self) -> int:
        return len(self.utts)

    @property
    def frame_ms(self) -> float:
        return 1000.0 / self.fps_out

    def eligible(self, w: int) -> list[int]:
        """Utterances with a ``w``-frame window (the SyncWindowDataset rule: manifest length >= w + 2, usable >= w)."""
        return [i for i, u in enumerate(self.utts) if u.n_video >= w + 2 and u.n >= w]

    def start(self, i: int, w: int) -> int:
        """Centred window start (as the dataset's eval items)."""
        return (self.utts[i].n - w) // 2

    def video_windows(self, specs: Sequence[tuple[int, int]], w: int) -> dict[str, torch.Tensor]:
        """Video-branch inputs of the windows ``(utterance index, start)``: ``video [B,w,C,crop,crop]`` (pixel norm
        over the window), ``lm [B,w,80]``, ``cue [B,w,8]``, ``valid [B,w]``."""
        vids, lms, cues, valids = [], [], [], []
        for i, s in specs:
            u = self.utts[int(i)]
            s = int(s)
            if s < 0 or s + w > u.n:
                raise ValueError(f"{u.utt_id}: window [{s}, {s + w}) outside the {u.n} usable frames")
            vids.append(normalize_pixels(u.frames[s:s + w], self.norm, self.mean, self.std))
            lms.append(u.lm[s:s + w])
            cues.append(u.cue[s:s + w])
            valids.append(u.valid[s:s + w])
        return {"video": torch.from_numpy(np.stack(vids)), "lm": torch.from_numpy(np.stack(lms)),
                "cue": torch.from_numpy(np.stack(cues)), "valid": torch.from_numpy(np.stack(valids))}

    def audio_windows(self, i: int, starts: Sequence[int], w: int) -> torch.Tensor:
        """Audio features ``[len(starts), w, 320]`` of windows of utterance ``i`` (one fbank call)."""
        return audio_window_batch(self.utts[i].wave, [int(s) for s in starts], w, self.sr, n_mels=self.n_mels,
                                  stack=self.stack)

    def region(self, i: int, start: int, w: int) -> np.ndarray:
        """Float waveform of the window ``[start, start + w)`` plus REGION_PAD frames on each side (zeros outside the
        utterance), sample-aligned with every other region of the same ``w`` - the unit that streams are mixed in."""
        _, win, spf = fbank_geometry(self.sr, self.stack)
        hop = spf // self.stack
        a = (int(start) - REGION_PAD) * spf
        b = (int(start) + w + REGION_PAD) * spf + (win - hop)
        wave = to_float_wave(self.utts[i].wave)
        out = np.zeros(b - a, dtype=np.float32)
        lo, hi = max(a, 0), min(b, int(wave.size))
        if hi > lo:
            out[lo - a:hi - a] = wave[lo:hi]
        return out

    def region_features(self, region: np.ndarray, w: int) -> torch.Tensor:
        """Features ``[w, 320]`` of the window inside a :meth:`region` (== :meth:`audio_windows` of the clean region)."""
        return audio_window_features(region, REGION_PAD, w, self.sr, n_mels=self.n_mels, stack=self.stack)

    def region_subwindows(self, region: np.ndarray, offsets: Sequence[int], w: int) -> torch.Tensor:
        """Features ``[len(offsets), w, 320]`` of sub-windows starting ``offsets`` frames into a :meth:`region`."""
        return audio_window_batch(region, [REGION_PAD + int(t) for t in offsets], w, self.sr, n_mels=self.n_mels,
                                  stack=self.stack)


def mix_leakage(streams: Sequence[np.ndarray], leak_db: float) -> list[np.ndarray]:
    """Imperfect separation: stream c + every other stream, each scaled to ``leak_db`` dB below stream c (RMS over the
    region, :func:`avsr.audio_feats.add_noise`). Silent sources are not mixed (their level is undefined)."""
    out = []
    for c, target in enumerate(streams):
        mixed = np.array(target, dtype=np.float32, copy=True)
        for j, other in enumerate(streams):
            if j == c or float(np.sqrt(np.mean(np.square(other, dtype=np.float64)))) < 1e-7:
                continue
            noisy, _ = add_noise(target, other, float(leak_db))
            mixed += noisy - target
        out.append(mixed)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# embeddings
# ----------------------------------------------------------------------------------------------------------------------
class Embedder:
    """Runs the (eval-mode) model on windows in fixed-size chunks; returns float32 embeddings on ``device``."""

    def __init__(self, model: torch.nn.Module, device: torch.device, amp: Any) -> None:
        self.model = model
        self.device = device
        self.amp = amp
        self.uses_video = getattr(model, "visual_frontend", None) is not None
        self.emb_dim = int(getattr(model, "emb_dim"))

    @torch.inference_mode()
    def video(self, store: EvalStore, specs: Sequence[tuple[int, int]], w: int) -> torch.Tensor:
        specs = list(specs)
        step = max(1, VIDEO_CHUNK_FRAMES // w)
        out = []
        for k in range(0, len(specs), step):
            b = {key: t.to(self.device, non_blocking=True) for key, t in store.video_windows(specs[k:k + step],
                                                                                               w).items()}
            with autocast_context(self.device, self.amp):
                v = self.model.embed_video(b["video"] if self.uses_video else None, b["lm"], b["cue"], b["valid"])
            out.append(v.float())
        return torch.cat(out) if out else torch.zeros(0, w, self.emb_dim, device=self.device)

    @torch.inference_mode()
    def audio(self, feats: torch.Tensor) -> torch.Tensor:
        w = int(feats.shape[1])
        step = max(1, AUDIO_CHUNK_FRAMES // w)
        out = []
        for k in range(0, int(feats.shape[0]), step):
            with autocast_context(self.device, self.amp):
                a = self.model.embed_audio(feats[k:k + step].to(self.device, non_blocking=True))
            out.append(a.float())
        return torch.cat(out) if out else torch.zeros(0, w, self.emb_dim, device=self.device)


# ----------------------------------------------------------------------------------------------------------------------
# candidate drawing and metric primitives
# ----------------------------------------------------------------------------------------------------------------------
def seeded_rng(seed: int, *parts: Any) -> np.random.Generator:
    """Generator from the seed and stable (crc32) hashes of ``parts``: independent of hash randomisation and order."""
    return np.random.default_rng(np.random.SeedSequence(
        [int(seed) & 0xFFFFFFFF] + [zlib.crc32(str(p).encode("utf-8")) for p in parts]))


class CandidatePool:
    """Positions ``0..M-1`` grouped by speaker and by (speaker, recording session); draws distractors (module doc)."""

    def __init__(self, speakers: Sequence[str], sessions: Sequence[str]) -> None:
        self.speakers = [str(s) for s in speakers]
        self.sessions = [str(s) for s in sessions]
        self.by_speaker: dict[str, list[int]] = defaultdict(list)
        self.by_group: dict[tuple[str, str], list[int]] = defaultdict(list)
        for p, (spk, ses) in enumerate(zip(self.speakers, self.sessions)):
            self.by_speaker[spk].append(p)
            self.by_group[(spk, ses)].append(p)
        self.speaker_list = sorted(self.by_speaker)
        self.group_list = sorted(self.by_group)

    def draw(self, speaker: str, session: str, k: int, rng: np.random.Generator) -> tuple[list[int], list[str]]:
        """Up to ``k`` distractor positions for a query of ``(speaker, session)`` and their kinds. Every pick comes
        from a (speaker, session) group not yet in the candidate set (the query's own group included), so no pick
        shares the query's audio or another pick's recording. Fewer than ``k`` when the groups run out."""
        others = [s for s in self.speaker_list if s != speaker]
        used = {(speaker, session)}
        picks: list[int] = []
        kinds: list[str] = []
        for j in rng.permutation(len(others)).tolist():
            if len(picks) >= k:
                break
            members = self.by_speaker[others[j]]
            p = members[int(rng.integers(len(members)))]
            picks.append(p)
            kinds.append(KIND_OTHER)
            used.add((self.speakers[p], self.sessions[p]))
        while len(picks) < k:
            free = [g for g in self.group_list if g not in used]
            if not free:
                break
            g = free[int(rng.integers(len(free)))]
            members = self.by_group[g]
            picks.append(members[int(rng.integers(len(members)))])
            kinds.append(KIND_SAME if g[0] == speaker else KIND_AGAIN)
            used.add(g)
        return picks, kinds


def selection_credit(true: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Per query: 1 when the true candidate scores strictly highest, ``1/(1+t)`` when it ties with ``t`` distractors
    for the highest score (the expected accuracy of a random tie-break), else 0. ``true [Q]``, ``others [Q, N-1]``."""
    true = np.asarray(true, dtype=np.float64)
    others = np.asarray(others, dtype=np.float64).reshape(true.shape[0], -1)
    if others.shape[1] == 0:
        return np.ones_like(true)
    best = others.max(axis=1)
    ties = (others == true[:, None]).sum(axis=1)
    return np.where(true > best, 1.0, np.where(true == best, 1.0 / (1.0 + ties), 0.0))


def assign_streams(scores: np.ndarray) -> np.ndarray:
    """Hungarian assignment maximising the total score: ``out[i]`` = stream given to face ``i`` (``scores[face, stream]``)."""
    scores = np.asarray(scores, dtype=np.float64)
    rows, cols = linear_sum_assignment(scores, maximize=True)
    out = np.full(scores.shape[0], -1, dtype=np.int64)
    out[rows] = cols
    return out


def roc_auc_eer(pos: Sequence[float], neg: Sequence[float]) -> dict[str, float]:
    """ROC-AUC (ties count 1/2), equal error rate and the score threshold at the EER for "accept when score >=
    threshold". The ROC is linearly interpolated between operating points; NaN without positives or negatives."""
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    n_p, n_n = int(pos.size), int(neg.size)
    out = {"auc": math.nan, "eer": math.nan, "threshold": math.nan, "n_pos": n_p, "n_neg": n_n}
    if n_p == 0 or n_n == 0:
        return out
    scores = np.concatenate([pos, neg])
    ranks = rankdata(scores)  # average ranks: ties get 1/2
    out["auc"] = float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2.0) / (n_p * n_n))
    labels = np.concatenate([np.ones(n_p), np.zeros(n_n)])
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    last = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]          # last position of every distinct score
    tpr = np.r_[0.0, np.cumsum(y)[last] / n_p]
    fpr = np.r_[0.0, np.cumsum(1.0 - y)[last] / n_n]
    thr = np.r_[np.inf, s[last]]
    d = (1.0 - tpr) - fpr                                        # FNR - FPR: 1 at +inf, decreasing to -1
    i = int(np.argmax(d <= 0))                                   # first point with FNR <= FPR (i >= 1)
    a = d[i - 1] / (d[i - 1] - d[i])
    out["eer"] = float(fpr[i - 1] + a * (fpr[i] - fpr[i - 1]))
    out["threshold"] = float(thr[i] if i == 1 else thr[i - 1] + a * (thr[i] - thr[i - 1]))
    return out


def _mean(x: Sequence[float] | np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return float(x.mean()) if x.size else math.nan


def _group_mean(values: np.ndarray, keys: Sequence[str]) -> dict[str, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for v, k in zip(values.tolist(), keys):
        groups[str(k)].append(v)
    return {k: round(_mean(v), 4) for k, v in sorted(groups.items())}


# ----------------------------------------------------------------------------------------------------------------------
# evaluator
# ----------------------------------------------------------------------------------------------------------------------
class SyncEvaluator:
    """All metrics of docs/SYNC_SPEC.md section 6 for one model on one :class:`EvalStore` (deterministic)."""

    def __init__(self, store: EvalStore, embedder: Embedder, seed: int, draw_k: int = 3,
                 offset_max: int = OFFSET_MAX) -> None:
        self.store = store
        self.emb = embedder
        self.seed = int(seed)
        self.draw_k = int(draw_k)
        self.offset_max = int(offset_max)
        self._canon: dict[int, dict[str, Any]] = {}
        self._draws: dict[tuple[int, str], list[tuple[list[int], list[str]]]] = {}
        self._offset_pred: dict[int, dict[int, int]] = {}

    # --- canonical (centred) windows, embedded once per window length --------------------------------------------
    def canonical(self, w: int) -> dict[str, Any]:
        """Centred window of every eligible utterance: indices, starts, embeddings ``v``/``a`` and the score matrix
        ``S[face, stream]`` over all of them."""
        if w in self._canon:
            return self._canon[w]
        idx = self.store.eligible(w)
        starts = [self.store.start(i, w) for i in idx]
        v = self.emb.video(self.store, list(zip(idx, starts)), w)
        feats = [self.store.audio_windows(i, [s], w) for i, s in zip(idx, starts)]
        a = self.emb.audio(torch.cat(feats)) if feats else torch.zeros_like(v)
        S = pair_scores(v, a).double().cpu().numpy() if idx else np.zeros((0, 0))
        self._canon[w] = {
            "idx": np.asarray(idx, dtype=np.int64), "starts": np.asarray(starts, dtype=np.int64), "v": v, "a": a,
            "S": S, "pool": CandidatePool(self.store.speakers[idx].tolist(), self.store.sessions[idx].tolist()),
        }
        return self._canon[w]

    def draws(self, w: int, direction: str) -> list[tuple[list[int], list[str]]]:
        """Per canonical position: up to ``draw_k`` distractor positions and kinds (fixed seed per query)."""
        key = (w, direction)
        if key not in self._draws:
            c = self.canonical(w)
            out = []
            for i in c["idx"].tolist():
                u = self.store.utts[i]
                rng = seeded_rng(self.seed, "nway", direction, u.utt_id, w)
                out.append(c["pool"].draw(u.speaker, u.session, self.draw_k, rng))
            self._draws[key] = out
        return self._draws[key]

    # --- 1. N-way selection ----------------------------------------------------------------------------------------
    def nway(self, w: int, n_values: Sequence[int], direction: str = "v2a") -> dict[str, Any]:
        """``v2a``: face -> which of N streams; ``a2v``: stream -> which of N faces. Queries without N-1 possible
        distractors are skipped (counted)."""
        if direction not in ("v2a", "a2v"):
            raise ValueError(f"direction must be 'v2a' or 'a2v', got {direction!r}")
        c = self.canonical(w)
        S = c["S"]
        draws = self.draws(w, direction)
        res: dict[str, Any] = {}
        for n in n_values:
            n = int(n)
            if n - 1 > self.draw_k:
                raise ValueError(f"N={n} needs draw_k >= {n - 1}")
            sel = [(p, d[0][:n - 1], d[1][:n - 1]) for p, d in enumerate(draws) if len(d[0]) >= n - 1]
            entry: dict[str, Any] = {"n": len(sel), "n_skipped": len(draws) - len(sel), "chance": 1.0 / n}
            if not sel:
                entry["acc"] = math.nan
                res[str(n)] = entry
                continue
            q = np.array([p for p, _, _ in sel], dtype=np.int64)
            dist = np.array([d for _, d, _ in sel], dtype=np.int64).reshape(len(sel), n - 1)
            true = S[q, q]
            others = S[q[:, None], dist] if direction == "v2a" else S[dist, q[:, None]]
            credit = selection_credit(true, others)
            kinds = Counter(k for _, _, ks in sel for k in ks)
            utt = [self.store.utts[i] for i in c["idx"][q].tolist()]
            entry.update({
                "acc": _mean(credit),
                "distractors": dict(sorted(kinds.items())),
                "all_other_speakers": _mean([all(k == KIND_OTHER for k in ks) for _, _, ks in sel]),
                "true_score_mean": _mean(true), "distractor_score_mean": _mean(others),
                "by_speaker": _group_mean(credit, [u.speaker for u in utt]),
            })
            if direction == "v2a":
                entry["by_angle"] = _group_mean(credit, [u.angle for u in utt])
            res[str(n)] = entry
        return res

    # --- 2. leakage robustness -------------------------------------------------------------------------------------
    def leakage(self, w: int, n_values: Sequence[int], levels_db: Sequence[float]) -> dict[str, Any]:
        """Video -> audio N-way on the same queries/draws as :meth:`nway` where every candidate stream carries the
        other candidates' speech (:func:`mix_leakage`) at each level. Keys ``leak_<X>dB``; ``clean`` = :meth:`nway`."""
        c = self.canonical(w)
        draws = self.draws(w, "v2a")
        clean = self.nway(w, n_values, "v2a")
        n_values = [int(n) for n in n_values]
        credits: dict[tuple[int, float], list[float]] = defaultdict(list)
        feats: list[torch.Tensor] = []
        meta: list[tuple[int, float, int, int]] = []   # (n, level, canonical position, number of streams)

        def flush() -> None:
            if not feats:
                return
            a = self.emb.audio(torch.stack(feats))
            k = 0
            for n, lv, p, m in meta:
                s = (c["v"][p][None] * a[k:k + m]).sum(-1).mean(-1).double().cpu().numpy()
                credits[(n, lv)].append(float(selection_credit(s[:1], s[None, 1:])[0]))
                k += m
            feats.clear()
            meta.clear()

        per_flush = max(1, AUDIO_CHUNK_FRAMES // w)
        for p, (dist, _) in enumerate(draws):
            ns = [n for n in n_values if len(dist) >= n - 1]
            if not ns:
                continue
            cands = [p] + list(dist[:max(ns) - 1])
            regions = [self.store.region(int(c["idx"][x]), int(c["starts"][x]), w) for x in cands]
            for n in ns:
                for lv in levels_db:
                    for stream in mix_leakage(regions[:n], float(lv)):
                        feats.append(self.store.region_features(stream, w))
                    meta.append((n, float(lv), p, n))
            if len(feats) >= per_flush:
                flush()
        flush()
        out: dict[str, Any] = {"rule": LEAK_RULE}
        for n in n_values:
            accs = {"clean": clean[str(n)]["acc"]}
            accs.update({f"leak_{float(lv):g}dB": _mean(credits[(n, float(lv))]) for lv in levels_db})
            out[str(n)] = {"n": clean[str(n)]["n"], "chance": 1.0 / n, "acc": accs}
        return out

    # --- 3. sync offset --------------------------------------------------------------------------------------------
    def offsets(self, w: int) -> dict[str, Any]:
        """Score the centred video window against the same utterance's audio shifted by -offset_max..+offset_max
        frames (all inside the utterance: shorter utterances are skipped); correct = argmax within +-1 frame."""
        c = self.canonical(w)
        m = self.offset_max
        offs = np.arange(-m, m + 1)
        queries = [p for p, (i, s) in enumerate(zip(c["idx"].tolist(), c["starts"].tolist()))
                   if s - m >= 0 and s + m + w <= self.store.utts[i].n]
        preds: dict[int, int] = {}
        per_chunk = max(1, AUDIO_CHUNK_FRAMES // (w * offs.size))
        for k in range(0, len(queries), per_chunk):
            chunk = queries[k:k + per_chunk]
            feats = torch.cat([self.store.audio_windows(int(c["idx"][p]), (int(c["starts"][p]) + offs).tolist(), w)
                               for p in chunk])
            a = self.emb.audio(feats).view(len(chunk), offs.size, w, -1)
            s = torch.einsum("qwe,qowe->qo", c["v"][chunk], a) / w
            for p, j in zip(chunk, s.argmax(dim=1).cpu().tolist()):
                preds[p] = int(offs[j])
        self._offset_pred[w] = preds
        err = np.array(list(preds.values()), dtype=np.float64)
        hist = Counter(int(x) for x in err.tolist())
        return {
            "n": len(queries), "n_skipped_short": int(c["idx"].size) - len(queries), "range_frames": [-m, m],
            "frame_ms": self.store.frame_ms,
            "acc_pm1": _mean(np.abs(err) <= 1), "acc_exact": _mean(err == 0),
            "median_abs_err_ms": float(np.median(np.abs(err)) * self.store.frame_ms) if err.size else math.nan,
            "mean_abs_err_ms": _mean(np.abs(err)) * self.store.frame_ms,
            "mean_offset_ms": _mean(err) * self.store.frame_ms,
            "histogram": {str(o): hist.get(int(o), 0) for o in offs.tolist()},
        }

    # --- 4. scene assignment ---------------------------------------------------------------------------------------
    def scenes(self, k: int, n_scenes: int, sub_w: int, levels_db: Sequence[float]) -> dict[str, Any]:
        """``n_scenes`` scenes of ``k`` faces / ``k`` streams (one utterance each, cropped to the common length, all
        >= SCENE_MIN_SEC); Hungarian assignment over the whole overlap and per ``sub_w``-frame sub-window, for clean
        streams and each leakage level."""
        st = self.store
        min_len = max(int(round(SCENE_MIN_SEC * st.fps_out)), int(sub_w))
        pool_idx = [i for i, u in enumerate(st.utts) if u.n >= min_len]
        pool = CandidatePool(st.speakers[pool_idx].tolist(), st.sessions[pool_idx].tolist())
        variants = ["clean"] + [f"leak_{float(lv):g}dB" for lv in levels_db]
        whole = {name: {"scene": [], "stream": []} for name in variants}
        sub = {name: {"scene": [], "stream": []} for name in variants}
        kinds: Counter = Counter()
        lengths: list[int] = []
        n_infeasible = 0
        for s_idx in range(int(n_scenes)):
            if not pool.speaker_list:
                n_infeasible += 1
                continue
            rng = seeded_rng(self.seed, "scene", k, s_idx)
            spk0 = pool.speaker_list[int(rng.integers(len(pool.speaker_list)))]
            p0 = pool.by_speaker[spk0][int(rng.integers(len(pool.by_speaker[spk0])))]
            picks, pick_kinds = pool.draw(spk0, pool.sessions[p0], k - 1, rng)
            if len(picks) < k - 1:
                n_infeasible += 1
                continue
            kinds.update(pick_kinds)
            members = [pool_idx[p] for p in [p0] + picks]
            length = min(st.utts[i].n for i in members)
            starts = [(st.utts[i].n - length) // 2 for i in members]
            lengths.append(length)
            t0s = list(range(0, (length // sub_w) * sub_w, sub_w))
            n_sub = len(t0s)
            v = self.emb.video(st, list(zip(members, starts)), length)                        # [k, L, E]
            vs = self.emb.video(st, [(i, s + t) for i, s in zip(members, starts) for t in t0s], sub_w)
            vs = vs.view(k, n_sub, sub_w, -1)                                                 # [k, n_sub, w, E]
            regions = [st.region(i, s, length) for i, s in zip(members, starts)]
            streams = [regions] + [mix_leakage(regions, float(lv)) for lv in levels_db]
            a = self.emb.audio(torch.stack([st.region_features(x, length) for var in streams for x in var]))
            a_sub = self.emb.audio(torch.cat([st.region_subwindows(x, t0s, sub_w) for var in streams for x in var]))
            a = a.view(len(variants), k, length, -1)
            a_sub = a_sub.view(len(variants), k, n_sub, sub_w, -1)
            for vi, name in enumerate(variants):
                s_whole = pair_scores(v, a[vi]).double().cpu().numpy()
                assign = assign_streams(s_whole)
                whole[name]["scene"].append(float(np.all(assign == np.arange(k))))
                whole[name]["stream"].append(float(np.mean(assign == np.arange(k))))
                s_sub = (torch.einsum("itwe,jtwe->tij", vs, a_sub[vi]) / sub_w).double().cpu().numpy()
                for t in range(n_sub):
                    assign = assign_streams(s_sub[t])
                    sub[name]["scene"].append(float(np.all(assign == np.arange(k))))
                    sub[name]["stream"].append(float(np.mean(assign == np.arange(k))))
        return {
            "n_scenes": len(lengths), "n_requested": int(n_scenes), "n_infeasible": n_infeasible,
            "chance_scene": 1.0 / math.factorial(k), "min_frames": min_len, "sub_window_frames": int(sub_w),
            "length_frames": ({"min": int(min(lengths)), "median": float(np.median(lengths)), "max": int(max(lengths))}
                              if lengths else {}),
            "partners": dict(sorted(kinds.items())),
            "whole": {name: {"scene_acc": _mean(d["scene"]), "stream_acc": _mean(d["stream"])}
                      for name, d in whole.items()},
            "subwindows": {name: {"window_acc": _mean(d["scene"]), "stream_acc": _mean(d["stream"]),
                                  "n_windows": len(d["scene"])} for name, d in sub.items()},
        }

    # --- 5. match / no-match verification --------------------------------------------------------------------------
    def verification(self, w: int) -> dict[str, Any]:
        """True pairs (diagonal) vs every pair with a different speaker's audio (single-speaker split: the same
        speaker from another recording session): ROC-AUC, EER, threshold at the EER."""
        c = self.canonical(w)
        S = c["S"]
        spk = self.store.speakers[c["idx"]]
        ses = self.store.sessions[c["idx"]]
        if len(set(spk.tolist())) >= 2:
            mask = spk[:, None] != spk[None, :]
            kind = "different speaker"
        else:
            mask = ses[:, None] != ses[None, :]
            kind = "same speaker, other recording session (single-speaker split)"
        pos, neg = np.diag(S), S[mask]
        out = roc_auc_eer(pos, neg)
        out.update({"negatives": kind, "pos_mean": _mean(pos), "neg_mean": _mean(neg)})
        return out

    # --- examples --------------------------------------------------------------------------------------------------
    def examples(self, w: int, n_examples: int, n_way: int) -> list[dict[str, Any]]:
        """A few evenly spaced video -> audio queries: true score, distractor scores, predicted offset (if computed)."""
        c = self.canonical(w)
        draws = self.draws(w, "v2a")
        m = int(c["idx"].size)
        out = []
        for p in sorted({int(j * m / max(n_examples, 1)) for j in range(n_examples)} if m else set()):
            dist, ks = draws[p][0][:n_way - 1], draws[p][1][:n_way - 1]
            u = self.store.utts[int(c["idx"][p])]
            true = float(c["S"][p, p])
            others = [float(c["S"][p, d]) for d in dist]
            out.append({"utt_id": u.utt_id, "speaker": u.speaker, "true": round(true, 4),
                        "distractors": [round(x, 4) for x in others], "kinds": ks,
                        "correct": bool(len(others) == n_way - 1 and all(true > x for x in others)),
                        "offset_pred": self._offset_pred.get(w, {}).get(p)})
        return out


# ----------------------------------------------------------------------------------------------------------------------
# full evaluation run
# ----------------------------------------------------------------------------------------------------------------------
def _int_list(value: Any, name: str) -> list[int]:
    items = value if isinstance(value, (list, tuple)) else [value]
    out = sorted({int(v) for v in items})
    if not out:
        raise ValueError(f"{name} must not be empty")
    return out


def run_evaluation(ev: SyncEvaluator, cfg: Any, n_scenes: int) -> dict[str, Any]:
    """Every section of docs/SYNC_SPEC.md section 6; returns the JSON body (without run metadata)."""
    windows = _int_list(cfg_get(cfg, "sync.eval_windows", [13, 25, 50]), "sync.eval_windows")
    n_way = _int_list(cfg_get(cfg, "sync.eval_n_way", [2, 3, 4]), "sync.eval_n_way")
    leak_n = _int_list(cfg_get(cfg, "sync.eval_leak_n_way", [2, 4]), "sync.eval_leak_n_way")
    levels = sorted({float(x) for x in (cfg_get(cfg, "sync.eval_leak_db", [20, 10, 5]) or [])}, reverse=True)
    scene_k = _int_list(cfg_get(cfg, "sync.scene_k", [2, 3]), "sync.scene_k")
    base_w = int(cfg_get(cfg, "sync.window_frames", 25))
    if min(n_way + leak_n + scene_k) < 2:
        raise ValueError("N-way sizes and scene sizes must be >= 2")
    timing: dict[str, float] = {}
    res: dict[str, Any] = {"nway": {}, "reverse": {}, "leakage": {}, "offset": {}, "verification": {}, "scenes": {}}
    for w in windows:
        t = time.time()
        ev.canonical(w)
        res["nway"][str(w)] = ev.nway(w, n_way, "v2a")
        res["reverse"][str(w)] = ev.nway(w, n_way, "a2v")
        res["verification"][str(w)] = ev.verification(w)
        timing[f"nway_reverse_verification_w{w}"] = round(time.time() - t, 1)
        t = time.time()
        res["offset"][str(w)] = ev.offsets(w)
        timing[f"offset_w{w}"] = round(time.time() - t, 1)
        t = time.time()
        res["leakage"][str(w)] = ev.leakage(w, leak_n, levels)
        timing[f"leakage_w{w}"] = round(time.time() - t, 1)
        log.info("window %d frames: n-way/reverse/verification/offset/leakage done (%s)", w,
                 ", ".join(f"{k} {v:.0f}s" for k, v in timing.items() if k.endswith(f"w{w}")))
    for k in scene_k:
        t = time.time()
        res["scenes"][str(k)] = ev.scenes(k, n_scenes, base_w, levels)
        timing[f"scenes_k{k}"] = round(time.time() - t, 1)
        log.info("scenes K=%d: %d scenes (%.0f s)", k, res["scenes"][str(k)]["n_scenes"], time.time() - t)
    thr_w = base_w if str(base_w) in res["verification"] else windows[len(windows) // 2]
    res["offscreen_threshold"] = {
        "window_frames": thr_w, "value": res["verification"][str(thr_w)]["threshold"],
        "by_window": {w: v["threshold"] for w, v in res["verification"].items()},
        "rule": "a face whose window score with a stream is below the value is not that stream's speaker; when no "
                "visible face reaches it the stream is an off-screen speaker (threshold at the EER of the "
                "verification test for that window length)",
    }
    res["examples"] = ev.examples(thr_w, 5, max(n_way))
    res["timing_sec"] = timing
    res["settings"] = {"windows": windows, "n_way": n_way, "leak_n_way": leak_n, "leak_db": levels,
                       "scene_k": scene_k, "scenes_requested": int(n_scenes), "offset_max_frames": ev.offset_max,
                       "sub_window_frames": base_w, "seed": ev.seed}
    return res


def fallback_notes(store: EvalStore, res: Mapping[str, Any]) -> list[str]:
    """Human-readable statements about distractor / partner fallbacks actually used."""
    spk = sorted(set(store.speakers.tolist()))
    notes = []
    for w, per_n in res["nway"].items():
        for n, entry in per_n.items():
            kinds = entry.get("distractors") or {}
            other = {k: v for k, v in kinds.items() if k != KIND_OTHER}
            if other:
                notes.append(f"window {w}, N={n}: the split has {len(spk)} held-out speaker(s) {spk}, fewer than N: "
                             f"distractors drawn from other recording sessions of the same speakers ({other}; "
                             f"{kinds.get(KIND_OTHER, 0)} from a different speaker)")
            if entry.get("n_skipped"):
                notes.append(f"window {w}, N={n}: {entry['n_skipped']} queries skipped (not enough distinct "
                             f"recording sessions for N-1 distractors)")
    for k, sc in res["scenes"].items():
        other = {kk: v for kk, v in (sc.get("partners") or {}).items() if kk != KIND_OTHER}
        if other:
            notes.append(f"scenes K={k}: {len(spk)} held-out speaker(s): partners from other recording sessions of "
                         f"the same speakers ({other})")
        if sc.get("n_infeasible"):
            notes.append(f"scenes K={k}: {sc['n_infeasible']} of {sc['n_requested']} scenes could not be drawn")
    for w, v in res["verification"].items():
        if v.get("negatives", "").startswith("same speaker"):
            notes.append(f"verification window {w}: single-speaker split, negatives are the same speaker's other "
                         f"recording sessions")
            break
    return notes


# ----------------------------------------------------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------------------------------------------------
def _num(x: Any) -> float:
    """Float value, NaN for None (JSON-loaded results) and non-numbers."""
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else math.nan


def _p(x: Any) -> str:
    return "   -  " if not math.isfinite(_num(x)) else f"{100 * _num(x):5.1f}%"


def _fmt(x: Any, spec: str, unit: str = "") -> str:
    return "-" if not math.isfinite(_num(x)) else f"{_num(x):{spec}}{unit}"


def format_tables(res: Mapping[str, Any], frame_ms: float) -> str:
    """Plain-text tables of every section (accepts the in-memory result or the JSON-loaded one)."""
    lines: list[str] = []
    n_way = res["settings"]["n_way"]
    head = f"{'window':>8} | " + " ".join(f"{'N=' + str(n):>6}" for n in n_way)
    lines += ["1) N-way selection, video -> which audio stream? (chance 1/N) | reverse: audio -> which face?",
              head + " | " + " ".join(f"{'N=' + str(n):>6}" for n in n_way) + " | n"]
    for w, per_n in res["nway"].items():
        rev = res["reverse"][w]
        lines.append(f"{float(w) * frame_ms / 1000:>6.2f} s | " + " ".join(_p(per_n[str(n)]["acc"]) for n in n_way)
                     + " | " + " ".join(_p(rev[str(n)]["acc"]) for n in n_way)
                     + f" | {per_n[str(n_way[0])]['n']}")
    levels = ["clean"] + [f"leak_{lv:g}dB" for lv in res["settings"]["leak_db"]]
    lines += ["", "2) Leakage robustness, video -> audio (every stream carries the other candidates' speech X dB below)",
              f"{'window':>8} {'N':>3} | " + " ".join(f"{lv.replace('leak_', ''):>7}" for lv in levels)]
    for w, per_n in res["leakage"].items():
        for n, entry in per_n.items():
            if n == "rule":
                continue
            lines.append(f"{float(w) * frame_ms / 1000:>6.2f} s {n:>3} | "
                         + " ".join(f"{_p(entry['acc'][lv]):>7}" for lv in levels))
    lines += ["", f"3) Sync offset (audio shifted -{res['settings']['offset_max_frames']}..+"
                  f"{res['settings']['offset_max_frames']} frames; correct = argmax within +-1 frame)",
              f"{'window':>8} | acc+-1  exact | median|err|  mean offset |    n"]
    for w, o in res["offset"].items():
        lines.append(f"{float(w) * frame_ms / 1000:>6.2f} s | {_p(o['acc_pm1'])} {_p(o['acc_exact'])} | "
                     f"{_fmt(o['median_abs_err_ms'], '.0f', ' ms'):>11} "
                     f"{_fmt(o['mean_offset_ms'], '+.0f', ' ms'):>12} | {o['n']:>4}")
    lines += ["", "4) Scene assignment (Hungarian, K faces x K streams): whole overlap scene acc / stream acc | "
                  "1-s windows: window acc / stream acc"]
    for k, sc in res["scenes"].items():
        for name in sc["whole"]:
            wh, sb = sc["whole"][name], sc["subwindows"][name]
            lines.append(f"  K={k} {name:<12} | {_p(wh['scene_acc'])} / {_p(wh['stream_acc'])} | "
                         f"{_p(sb['window_acc'])} / {_p(sb['stream_acc'])}  ({sc['n_scenes']} scenes, "
                         f"{sb['n_windows']} windows)")
    lines += ["", "5) Match / no-match (true pair vs a different speaker's audio)",
              f"{'window':>8} |   AUC    EER  threshold | pos mean  neg mean |  n_pos   n_neg"]
    for w, v in res["verification"].items():
        lines.append(f"{float(w) * frame_ms / 1000:>6.2f} s | {_p(v['auc'])} {_p(v['eer'])} "
                     f"{_fmt(v['threshold'], '.4f'):>10} | {_fmt(v['pos_mean'], '8.4f'):>8} "
                     f"{_fmt(v['neg_mean'], '9.4f'):>9} | {v['n_pos']:6d} {v['n_neg']:7d}")
    ot = res["offscreen_threshold"]
    lines.append(f"offscreen_threshold ({ot['window_frames']} frames): {_fmt(ot['value'], '.4f')}  (score >= "
                 f"threshold: the face matches the stream)")
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------
def eval_config(config_path: str | Path | None, overrides: Sequence[str], ckpt_cfg: Mapping[str, Any] | None) -> Config:
    """CLI config with the checkpoint's ``video`` / ``audio`` sections and ``sync`` architecture keys, then --set."""
    base = load_config(config_path) if config_path is not None else Config()
    if ckpt_cfg:
        for sec in ("video", "audio"):
            if sec in ckpt_cfg:
                base[sec] = ckpt_cfg[sec]
        if "sync" in ckpt_cfg:
            if "sync" not in base:
                base["sync"] = Config()
            for key in ARCH_KEYS:
                if key in ckpt_cfg["sync"]:
                    base["sync"][key] = ckpt_cfg["sync"][key]
    return load_config(None, overrides, base=base)


def split_rows(splits: Mapping[str, list[dict]], split: str) -> list[dict]:
    if split == "heldout":
        return list(splits.get("val", [])) + list(splits.get("test", []))
    return list(splits.get(split, []))


def _finite(obj: Any) -> Any:
    """NaN/inf -> None (standard JSON), numpy scalars -> python."""
    if isinstance(obj, (np.floating, np.integer)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Mapping):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a speaker-lip matching (sync) checkpoint on held-out speakers.")
    p.add_argument("--config", default="configs/sync.yaml", help="YAML config (architecture/features from the ckpt)")
    p.add_argument("--ckpt", required=True, help="sync checkpoint (e.g. work/checkpoints_sync/best.pt)")
    p.add_argument("--split", choices=SPLITS, default="heldout",
                   help="val, test, or heldout = val + test (default; 3 speakers)")
    p.add_argument("--max-utts", type=int, default=None,
                   help="evenly spread subset of N utterances (default sync.eval_max_utts; 0 = all)")
    p.add_argument("--scenes", type=int, default=None, help="scenes per K (default sync.eval_scenes, 300)")
    p.add_argument("--out", default=None, help="result JSON (default <work_dir>/eval/sync_<split>_results.json)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    p.add_argument("--work-dir", default=None, help="override work_dir (features, manifests, eval output)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    safe_console()
    args = parse_args(argv)
    t0 = time.time()
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    if not isinstance(ckpt, Mapping) or "model" not in ckpt:
        raise ValueError(f"{args.ckpt}: not a training checkpoint (no 'model' entry)")
    if ckpt.get("kind") != "sync":
        raise ValueError(f"{args.ckpt} is not a sync checkpoint (kind={ckpt.get('kind')!r}; an AVSR checkpoint "
                         "is evaluated with avsr.evaluate)")
    overrides = list(args.set) + ([f"work_dir={json.dumps(str(args.work_dir))}"] if args.work_dir else [])
    cfg = eval_config(args.config, overrides, ckpt.get("cfg"))
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    work_dir = Path(str(cfg.work_dir))
    logger = get_logger(work_dir, "sync_evaluate")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        model = build_sync_model(cfg)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        amp = cfg_get(cfg, "train.amp", "bf16")
        logger.info("loaded %s (epoch %s, step %s) on %s, amp %s", args.ckpt, ckpt.get("epoch", "?"),
                    ckpt.get("step", "?"), device, amp)
        ckpt_split = (ckpt.get("cfg") or {}).get("split")
        if ckpt_split and dict(ckpt_split) != dict(cfg.split):
            logger.warning("split config differs from the checkpoint's (%s vs %s): held-out speakers may have been "
                           "seen in training", dict(cfg.split), dict(ckpt_split))

        splits = assign_split(load_manifests(work_dir), cfg)
        rows_all = split_rows(splits, args.split)
        max_utts = args.max_utts if args.max_utts is not None else int(cfg_get(cfg, "sync.eval_max_utts", 600))
        rows = even_subset(rows_all, max_utts if max_utts and max_utts > 0 else None)
        if not rows:
            logger.error("split %r is empty (work_dir=%s, split config=%s): nothing to evaluate", args.split,
                         work_dir, dict(cfg.split))
            return 1
        t = time.time()
        store = EvalStore.load(rows, cfg, int(cfg_get(cfg, "data.num_workers", 0)))
        if len(store) == 0:
            logger.error("no usable utterance among the %d rows of split %r", len(rows), args.split)
            return 1
        spk_counts = dict(sorted(Counter(store.speakers.tolist()).items()))
        logger.info("split %s: %d utterances loaded of %d rows (%d in the split; %d unusable) in %.0f s; speakers %s",
                    args.split, len(store), len(rows), len(rows_all), store.n_bad, time.time() - t, spk_counts)
        n_scenes = int(args.scenes if args.scenes is not None else cfg_get(cfg, "sync.eval_scenes", 300))
        draw_k = max(_int_list(cfg_get(cfg, "sync.eval_n_way", [2, 3, 4]), "sync.eval_n_way")
                     + _int_list(cfg_get(cfg, "sync.eval_leak_n_way", [2, 4]), "sync.eval_leak_n_way")) - 1
        ev = SyncEvaluator(store, Embedder(model, device, amp), seed, draw_k=draw_k)
        res = run_evaluation(ev, cfg, n_scenes)
        notes = fallback_notes(store, res)
        init = ckpt.get("init") or {}
        exposed = sorted(set(init.get("heldout_not_held_out_by_init") or []) & set(spk_counts))
        if exposed:
            notes.append(f"the model's frontends were initialised from {init.get('ckpt')}, whose training did not "
                         f"hold out {exposed}: the results for these speakers are optimistic")
            logger.warning("%s", notes[-1])
        table = format_tables(res, store.frame_ms)
        logger.info("results (%s, %d utterances, speakers %s):\n%s", args.split, len(store), spk_counts, table)
        for note in notes:
            logger.info("note: %s", note)
        for ex in res["examples"]:
            logger.info("example %s (%s): true %.3f vs distractors %s -> %s, offset %s", ex["utt_id"], ex["speaker"],
                        ex["true"], ex["distractors"], "correct" if ex["correct"] else "WRONG", ex["offset_pred"])

        out = {
            "split": args.split, "ckpt": str(args.ckpt), "epoch": ckpt.get("epoch"), "step": ckpt.get("step"),
            "seed": seed, "n_utts": len(store), "n_rows_requested": len(rows), "n_rows_split": len(rows_all),
            "n_unusable": store.n_bad, "unusable_examples": store.errors, "speakers": spk_counts,
            "sessions": {s: len({u.session for u in store.utts if u.speaker == s}) for s in spk_counts},
            "distractor_policy": DISTRACTOR_POLICY,
            "embedding": ("every window embedded on its own (as in training); the centred window of each utterance "
                          "once per window length, shared by n-way / reverse / verification / offset"),
            "notes": notes, **res, "elapsed_sec": round(time.time() - t0, 1),
        }
        out_path = Path(args.out) if args.out else work_dir / "eval" / f"sync_{args.split}_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_finite(out), f, ensure_ascii=False, indent=2)
        tmp.replace(out_path)
        logger.info("wrote %s (%.0f s total)", out_path, time.time() - t0)
        return 0
    finally:
        close_logger()


__all__ = [
    "Utterance", "UtteranceReader", "EvalStore", "Embedder", "CandidatePool", "SyncEvaluator", "mix_leakage",
    "selection_credit", "assign_streams", "roc_auc_eer", "seeded_rng", "run_evaluation", "format_tables",
    "fallback_notes", "eval_config", "split_rows",
]


if __name__ == "__main__":
    sys.exit(main())

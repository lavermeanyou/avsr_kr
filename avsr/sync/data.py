"""Windows of synchronised mouth video + audio for the speaker-lip matching model (docs/SYNC_SPEC.md section 2).

- :func:`audio_window_features` / :func:`audio_window_batch`: stacked-fbank features ``[window, 320]`` of 25-fps
  windows of a waveform (pure functions shared by the dataset and the evaluation).
- :class:`SyncWindowDataset`: one fixed-length window per utterance (random start when training, centred for eval)
  with the mouth video, lip skeleton, cues, the matching audio and a time-shifted audio window of the same utterance
  (hard negative).
- :func:`sync_collate`: stacks items into a batch and numbers the underlying audio recordings / speakers.

Time grid (the AVSR convention): 25-fps frame ``k`` of an utterance is video source frame ``round(k * fps / 25)``
and audio samples ``[k * 640, k * 640 + 880)`` at 16 kHz (fbank frames ``4k .. 4k+3``, 25 ms / 10 ms, stacked x4).
Kaldi fbank frames only see their own 25 ms (``snip_edges``), so a window computed from the window's waveform (plus
0.1 s context that is cut away) is the matching slice of the whole-utterance fbank; CMVN is then re-applied over the
window's own frames, so the features depend on the window's audio only (up to compute_fbank's CMVN epsilon: ~2e-5 on
speech, <= 2e-3 on the near-constant bins of pure tones).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Hashable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from avsr.audio_feats import add_noise, compute_fbank, make_noise, stack_frames, to_float_wave
from avsr.dataset import (
    _ITEM_ERRORS, CUE_NORMS, VIDEO_NORMS, _audio_key, _item_seed, _limit_worker_threads, cfg_get, normalize_cue,
    normalize_pixels, read_frames, resample_indices, resampled_length, spec_augment,
)

log = logging.getLogger(__name__)

CONTEXT_SEC = 0.1          # waveform context on each side of a window, cut away after the fbank
SYNC_NOISE_KINDS = ("white", "pink")
_MAX_RETRIES = 8
_CMVN_EPS = 1e-5           # as avsr.audio_feats.compute_fbank


# ----------------------------------------------------------------------------------------------------------------------
# audio windows
# ----------------------------------------------------------------------------------------------------------------------
def fbank_geometry(sr: int = 16000, stack: int = 4) -> tuple[int, int, int]:
    """(hop, frame length, samples per stacked frame) of :func:`avsr.audio_feats.compute_fbank` at ``sr``."""
    hop = int(round(0.010 * sr))
    win = int(round(0.025 * sr))
    return hop, win, hop * int(stack)


def num_audio_frames(n_samples: int, sr: int = 16000, stack: int = 4) -> int:
    """Stacked (25 Hz) frames that ``stack_frames(compute_fbank(wave))`` yields for ``n_samples`` samples."""
    hop, win, _ = fbank_geometry(sr, stack)
    if n_samples < win:
        return 0
    return (1 + (int(n_samples) - win) // hop) // int(stack)


def _cmvn(fb: torch.Tensor) -> torch.Tensor:
    """Mean/variance normalisation over time (the rule of compute_fbank), on the window's own frames."""
    mean = fb.mean(dim=0, keepdim=True)
    var = fb.var(dim=0, unbiased=False, keepdim=True)
    return (fb - mean) / torch.sqrt(var + _CMVN_EPS)


def _window_fbanks(wave: np.ndarray | torch.Tensor, starts: Sequence[int], n_fb: int, sr: int,
                   n_mels: int) -> list[torch.Tensor]:
    """CMVN'd 100-Hz fbank ``[n_fb, n_mels]`` of each segment whose first fbank frame starts at sample ``starts[i]``
    (multiples of the hop). One fbank call over the union of the segments plus ``CONTEXT_SEC`` of context on each
    side (context only where the waveform has samples); samples a segment needs outside the waveform are zeros."""
    hop, win, _ = fbank_geometry(sr, 1)
    if n_fb < 1:
        raise ValueError(f"window must be at least one frame, got {n_fb} fbank frames")
    if any(int(s) % hop for s in starts):
        raise ValueError(f"segment starts must be multiples of the fbank hop ({hop} samples): {list(starts)}")
    data = wave.detach().cpu().numpy() if isinstance(wave, torch.Tensor) else np.asarray(wave)
    data = data.reshape(-1)
    n = int(data.size)
    span = (n_fb - 1) * hop + win
    ctx = int(round(CONTEXT_SEC * sr / hop)) * hop
    need_lo, need_hi = int(min(starts)), int(max(starts)) + span
    lo = need_lo if need_lo < 0 else max(need_lo - ctx, 0)          # stays on the hop grid (0 and ctx are on it)
    hi = max(min(need_hi + ctx, n), need_hi)
    a, b = max(lo, 0), min(hi, n)
    if lo == a and hi == b:
        chunk = to_float_wave(data[a:b])
    else:  # the requested window reaches outside the waveform: zero samples there
        chunk = np.zeros(hi - lo, dtype=np.float32)
        if b > a:
            chunk[a - lo:b - lo] = to_float_wave(data[a:b])
    fb = compute_fbank(chunk, sr, n_mels=n_mels)
    out = []
    for s in starts:
        off = (int(s) - lo) // hop
        seg = fb[off:off + n_fb]
        if seg.shape[0] != n_fb:  # cannot happen with the arithmetic above; guards against silent truncation
            raise RuntimeError(f"fbank segment has {seg.shape[0]} frames, expected {n_fb}")
        out.append(_cmvn(seg))
    return out


def audio_window_batch(wave: np.ndarray | torch.Tensor, start_frames: Sequence[int], window: int, sr: int = 16000,
                       *, n_mels: int = 80, stack: int = 4) -> torch.Tensor:
    """Features ``[len(start_frames), window, n_mels * stack]`` of several 25-fps windows of one waveform (one fbank
    call). ``wave`` is int16 PCM (native scale) or float in [-1, 1]. Windows may reach outside the waveform (e.g.
    offset sweeps): the missing samples are silence (zeros)."""
    if int(window) < 1:
        raise ValueError(f"window must be >= 1 frame, got {window}")
    if len(start_frames) == 0:
        return torch.zeros(0, int(window), n_mels * stack, dtype=torch.float32)
    _, _, spf = fbank_geometry(sr, stack)
    fbs = _window_fbanks(wave, [int(s) * spf for s in start_frames], int(window) * stack, sr, n_mels)
    return torch.stack([stack_frames(fb, stack) for fb in fbs])


def audio_window_features(wave: np.ndarray | torch.Tensor, start_frame: int, window: int, sr: int = 16000,
                          *, n_mels: int = 80, stack: int = 4) -> torch.Tensor:
    """Stacked-fbank features ``[window, n_mels * stack]`` (default ``[window, 320]``) of the 25-fps frames
    ``start_frame .. start_frame + window - 1`` of a waveform: kaldi fbank of the window's samples (+0.1 s context,
    cut away), per-window CMVN, x4 stacking. Equals the matching rows of the whole-utterance features up to CMVN."""
    return audio_window_batch(wave, [int(start_frame)], window, sr, n_mels=n_mels, stack=stack)[0]


# ----------------------------------------------------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------------------------------------------------
class SyncWindowDataset(Dataset):
    """One ``window``-frame (25 fps) audio-visual window per manifest row (docs/SYNC_SPEC.md section 2).

    Args:
        rows: manifest rows (:func:`avsr.dataset.load_manifests` / :func:`avsr.dataset.assign_split`). Rows whose
            25-fps length is below ``window + 2`` are dropped (count logged, ``n_dropped``).
        cfg: config (``video.*``, ``audio.*``, ``sync.*`` keys; see configs/sync.yaml).
        train: random window start, random crop/flip, leakage/noise mixing and light SpecAugment; otherwise the
            centred window without augmentation and a per-utterance deterministic shifted negative.
        window: frames per window (default ``sync.window_frames``, 25 = 1 s).
        seed: seed of the augmentation generators (per DataLoader worker) and of the eval shifts.
        leak_rows: rows whose audio may leak into a training item (default: ``rows``); only utterances of a
            different speaker are used.
        work_dir: base for rows without a ``work_dir`` key (default ``cfg.work_dir``).

    Video pixels are normalised over the decoded window (``video.norm``), cues over the whole utterance
    (``video.cue_norm``, as the AVSR dataset). A row whose npz/mp4 is unreadable, whose mouth video is shorter than
    the window needs, or whose usable length (min of video and audio frames) is below ``window`` is skipped (the next
    row is returned) and counted in ``n_bad``.
    """

    pool_size = 64            # leakage clips cached per worker
    pool_refresh_every = 16   # items between partial pool refreshes
    pool_refresh_count = 4    # clips replaced per refresh

    def __init__(self, rows: Sequence[dict], cfg: Any, train: bool, window: int | None = None, seed: int = 0,
                 *, leak_rows: Sequence[dict] | None = None, work_dir: str | os.PathLike | None = None) -> None:
        self.train = bool(train)
        self.seed = int(seed)
        self.window = int(window if window is not None else cfg_get(cfg, "sync.window_frames", 25))
        if self.window < 1:
            raise ValueError(f"window must be >= 1 frame, got {self.window}")
        self.work_dir = Path(work_dir if work_dir is not None else cfg_get(cfg, "work_dir", "work"))

        self.channels = int(cfg_get(cfg, "video.channels", 1))
        self.size = int(cfg_get(cfg, "video.size", 96))
        self.crop = int(cfg_get(cfg, "video.crop", 88))
        self.fps_in = float(cfg_get(cfg, "video.fps_in", 30.0))
        self.fps_out = float(cfg_get(cfg, "video.fps_out", 25.0))
        self.mean = float(cfg_get(cfg, "video.mean", 0.421))
        self.std = float(cfg_get(cfg, "video.std", 0.165))
        self.norm = str(cfg_get(cfg, "video.norm", "global"))
        self.cue_norm = str(cfg_get(cfg, "video.cue_norm", "none"))
        self.flip_prob = float(cfg_get(cfg, "video.flip_prob", 0.5))
        if self.norm not in VIDEO_NORMS or self.cue_norm not in CUE_NORMS:
            raise ValueError(f"video.norm must be one of {VIDEO_NORMS} and video.cue_norm one of {CUE_NORMS}; "
                             f"got {self.norm!r}, {self.cue_norm!r}")
        if self.channels not in (1, 3):
            raise ValueError(f"video.channels must be 1 or 3, got {self.channels}")
        if not 0 < self.crop <= self.size:
            raise ValueError(f"video.crop ({self.crop}) must be in (0, video.size={self.size}]")

        self.sr = int(cfg_get(cfg, "audio.sr", 16000))
        self.n_mels = int(cfg_get(cfg, "audio.n_mels", 80))
        self.stack = int(cfg_get(cfg, "audio.stack", 4))

        self.shift_min = int(cfg_get(cfg, "sync.shift_min", 5))
        self.shift_max = int(cfg_get(cfg, "sync.shift_max", 15))
        if self.shift_max > 0 and not 1 <= self.shift_min <= self.shift_max:
            raise ValueError(f"need 1 <= sync.shift_min <= sync.shift_max (shift_max 0 disables shifted negatives); "
                             f"got {self.shift_min}, {self.shift_max}")
        self.leak_prob = float(cfg_get(cfg, "sync.leak_prob", 0.5))
        self.leak_snr = (float(cfg_get(cfg, "sync.leak_snr_min", 5.0)), float(cfg_get(cfg, "sync.leak_snr_max", 30.0)))
        self.noise_prob = float(cfg_get(cfg, "sync.noise_prob", 0.3))
        self.noise_snr = (float(cfg_get(cfg, "sync.noise_snr_min", 5.0)),
                          float(cfg_get(cfg, "sync.noise_snr_max", 30.0)))
        self.noise_kinds = [str(k) for k in (cfg_get(cfg, "sync.noise_kinds", list(SYNC_NOISE_KINDS)) or [])]
        bad = [k for k in self.noise_kinds if k not in SYNC_NOISE_KINDS]
        if bad or (self.noise_prob > 0 and not self.noise_kinds):
            raise ValueError(f"sync.noise_kinds must be a non-empty subset of {SYNC_NOISE_KINDS}, got {self.noise_kinds}")
        for name, (lo, hi) in (("leak", self.leak_snr), ("noise", self.noise_snr)):
            if lo > hi:
                raise ValueError(f"sync.{name}_snr_min ({lo}) > sync.{name}_snr_max ({hi})")
        self.specaug = {"freq_mask": int(cfg_get(cfg, "sync.specaug.freq_mask", 1)),
                        "freq_width": int(cfg_get(cfg, "sync.specaug.freq_width", 8)),
                        "time_mask": int(cfg_get(cfg, "sync.specaug.time_mask", 1)),
                        "time_width": int(cfg_get(cfg, "sync.specaug.time_width", 4))}

        min_len = self.window + 2
        kept = [r for r in rows
                if resampled_length(int(r["n_frames"]), float(r.get("fps") or self.fps_in), self.fps_out) >= min_len]
        self.rows = kept
        self.n_dropped = len(rows) - len(kept)
        log.info("SyncWindowDataset(%s, window %d): %d rows, %d dropped (25-fps length < %d)",
                 "train" if self.train else "eval", self.window, len(kept), self.n_dropped, min_len)
        self.leak_rows = list(leak_rows) if leak_rows is not None else self.rows

        self.n_bad = 0       # unreadable / too short rows skipped in this process
        self.n_no_leak = 0   # leakage drawn but no clip of another speaker available
        self._reset_process_state()

    # --- process-local state (never pickled into workers) ------------------------------------------------------------
    def _reset_process_state(self) -> None:
        self._rng: np.random.Generator | None = None
        self._rng_owner: tuple | None = None
        self._pool: list[tuple[str, str, np.ndarray]] = []
        self._pool_counter = 0

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rng"] = None
        state["_rng_owner"] = None
        state["_pool"] = []
        state["_pool_counter"] = 0
        return state

    def _process_rng(self) -> np.random.Generator:
        """Training generator, (re)created per process / DataLoader worker (reproducible under torch.manual_seed)."""
        info = get_worker_info()
        owner = (os.getpid(), -1 if info is None else info.id, None if info is None else info.seed)
        if self._rng is None or self._rng_owner != owner:
            if info is None:
                ss = np.random.SeedSequence([self.seed & 0xFFFFFFFF, 0x5E17C])
            else:
                ss = np.random.SeedSequence([self.seed & 0xFFFFFFFF, 0x5E17C, info.id,
                                             info.seed & 0xFFFFFFFFFFFFFFFF])
            self._rng = np.random.default_rng(ss)
            self._rng_owner = owner
            self._pool = []
            self._pool_counter = 0
        return self._rng

    # --- io ----------------------------------------------------------------------------------------------------------
    def _path(self, row: dict, key: str) -> Path:
        base = Path(row["work_dir"]) if row.get("work_dir") else self.work_dir
        return base / str(row[key])

    def _load_clip(self, row: dict) -> tuple[str, str, np.ndarray] | None:
        """(utt_id, speaker, int16 audio) of a leakage clip; None when unreadable or empty (counted in n_bad)."""
        try:
            with np.load(self._path(row, "npz")) as z:
                audio = np.asarray(z["audio"], dtype=np.int16)
        except _ITEM_ERRORS as e:
            self.n_bad += 1
            log.warning("leakage clip %s unreadable (%s: %s)", row.get("utt_id"), type(e).__name__, e)
            return None
        if audio.size == 0:
            return None
        return str(row["utt_id"]), str(row.get("speaker", "")), audio

    def _leak_pool(self, rng: np.random.Generator) -> list[tuple[str, str, np.ndarray]]:
        """Per-worker pool: filled on first use, then ``pool_refresh_count`` clips replaced every
        ``pool_refresh_every`` items (an item costs a fraction of an npz read instead of one)."""
        n_rows = len(self.leak_rows)
        if n_rows == 0:
            return []
        if not self._pool:
            picks = rng.choice(n_rows, size=min(self.pool_size, n_rows), replace=False)
            self._pool = [c for c in (self._load_clip(self.leak_rows[int(i)]) for i in picks) if c is not None]
        else:
            self._pool_counter += 1
            if self._pool_counter % self.pool_refresh_every == 0:
                for _ in range(self.pool_refresh_count):
                    clip = self._load_clip(self.leak_rows[int(rng.integers(0, n_rows))])
                    if clip is not None:
                        self._pool[int(rng.integers(0, len(self._pool)))] = clip
        return self._pool

    # --- item construction -------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        if not self.rows:
            raise IndexError("SyncWindowDataset is empty")
        for attempt in range(_MAX_RETRIES):
            j = (int(index) + attempt) % len(self.rows)
            try:
                return self._make_item(self.rows[j])
            except _ITEM_ERRORS as e:
                self.n_bad += 1
                log.warning("skipping bad utterance %s (%s: %s); bad items in this process: %d",
                            self.rows[j].get("utt_id"), type(e).__name__, e, self.n_bad)
        raise RuntimeError(f"{_MAX_RETRIES} consecutive unusable utterances starting at index {index}")

    def _choose_shift(self, start: int, n: int, rng: np.random.Generator) -> int:
        """Signed shift ``s`` (``shift_min <= |s| <= shift_max``) drawn uniformly among those whose window
        ``start + s .. start + s + window`` stays inside the ``n`` usable frames; 0 when none does."""
        if self.shift_max <= 0:
            return 0
        cands = [sign * m for m in range(self.shift_min, self.shift_max + 1) for sign in (-1, 1)
                 if 0 <= start + sign * m and start + sign * m + self.window <= n]
        return int(cands[int(rng.integers(0, len(cands)))]) if cands else 0

    def _augment_audio(self, region: np.ndarray, row: dict, rng: np.random.Generator, aug: dict) -> np.ndarray:
        """Leakage of one other speaker's utterance and/or white/pink noise; every SNR is relative to the clean
        target region (the add_noise rule)."""
        parts: list[tuple[np.ndarray, float]] = []
        if self.leak_prob > 0 and rng.random() < self.leak_prob:
            speaker = str(row.get("speaker", ""))
            eligible = [c for c in self._leak_pool(rng) if c[1] != speaker]
            if eligible:
                utt, _, clip = eligible[int(rng.integers(0, len(eligible)))]
                snr = float(rng.uniform(*self.leak_snr))
                parts.append((make_noise("babble", int(region.size), rng, [clip]), snr))  # one clip, random offset
                aug["leak_utt"], aug["leak_snr"] = utt, snr
            else:
                self.n_no_leak += 1
        if self.noise_prob > 0 and rng.random() < self.noise_prob:
            kind = self.noise_kinds[int(rng.integers(0, len(self.noise_kinds)))]
            snr = float(rng.uniform(*self.noise_snr))
            parts.append((make_noise(kind, int(region.size), rng), snr))
            aug["noise_kind"], aug["noise_snr"] = kind, snr
        if not parts:
            return region
        mixed = region.copy()
        for noise, snr in parts:
            noisy, _ = add_noise(region, noise, snr)
            mixed += noisy - region
        return mixed

    def _make_item(self, row: dict) -> dict:
        _limit_worker_threads()
        utt_id = str(row["utt_id"])
        rng = self._process_rng() if self.train else np.random.default_rng(_item_seed(self.seed, utt_id, "sync"))
        with np.load(self._path(row, "npz")) as z:
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

        w = self.window
        idx_all = resample_indices(n_t, fps, self.fps_out)
        n = min(int(idx_all.size), num_audio_frames(int(wave.size), self.sr, self.stack))
        if n < w:
            raise ValueError(f"usable length {n} frames (video {idx_all.size}, audio "
                             f"{num_audio_frames(int(wave.size), self.sr, self.stack)}) < window {w}")
        start = int(rng.integers(0, n - w + 1)) if self.train else (n - w) // 2
        shift = self._choose_shift(start, n, rng)

        # ---- video: decode only the window's frames, crop/flip, normalise over the window
        idx = idx_all[start:start + w]
        mp4 = self._path(row, "mouth_mp4")
        frames, n_dec = read_frames(mp4, idx, self.size)
        if frames is None:
            raise ValueError(f"mouth video unreadable: {mp4}")
        if n_dec < int(np.unique(idx).size):
            raise ValueError(f"mouth video {mp4} ends before source frame {int(idx[-1])}")
        margin = self.size - self.crop
        if self.train:
            y0, x0 = int(rng.integers(0, margin + 1)), int(rng.integers(0, margin + 1))
            flip = bool(rng.random() < self.flip_prob)
        else:
            y0 = x0 = margin // 2
            flip = False
        y1, x1 = y0 + self.crop, x0 + self.crop
        if self.channels == 1:
            gray = cv2.cvtColor(np.ascontiguousarray(frames).reshape(w * self.size, self.size, 3), cv2.COLOR_BGR2GRAY)
            pix = gray.reshape(w, 1, self.size, self.size)[:, :, y0:y1, x0:x1]
        else:
            pix = frames[:, y0:y1, x0:x1, ::-1].transpose(0, 3, 1, 2)  # BGR -> RGB, [W, 3, H, W]
        if flip:
            pix = pix[..., ::-1]
        video = normalize_pixels(pix, self.norm, self.mean, self.std)

        lm = lm_all[idx].reshape(w, 80).astype(np.float32)
        if flip:
            lm[:, 0::2] *= -1.0
        valid = valid_all[idx].astype(np.float32)
        idx_n = idx_all[:n]
        cue = normalize_cue(cue_all[idx_n].astype(np.float32), valid_all[idx_n], self.cue_norm)[start:start + w]

        # ---- audio: the region covering the window and its shifted copy (+ context), augmented once, then both
        # windows cut from the same (noisy) signal, per-window CMVN, SpecAugment each (train), stack to 25 Hz
        hop, win, spf = fbank_geometry(self.sr, self.stack)
        starts = [start] if shift == 0 else [start, start + shift]
        ctx = int(round(CONTEXT_SEC * self.sr / hop)) * hop
        lo = max(min(starts) * spf - ctx, 0)
        hi = min(max(starts) * spf + (w * self.stack - 1) * hop + win + ctx, int(wave.size))
        region = to_float_wave(wave[lo:hi])
        aug: dict[str, Any] = {"leak_utt": None, "leak_snr": None, "noise_kind": None, "noise_snr": None}
        if self.train:
            region = self._augment_audio(region, row, rng, aug)
        fbs = _window_fbanks(region, [s * spf - lo for s in starts], w * self.stack, self.sr, self.n_mels)
        if self.train:
            for fb in fbs:
                spec_augment(fb, rng, **self.specaug)
        audio = stack_frames(fbs[0], self.stack).contiguous()
        if shift:
            audio_shift = stack_frames(fbs[1], self.stack).contiguous()
        else:
            audio_shift = torch.zeros_like(audio)

        return {
            "video": torch.from_numpy(np.ascontiguousarray(video)),
            "lm": torch.from_numpy(lm),
            "cue": torch.from_numpy(np.ascontiguousarray(cue)),
            "valid": torch.from_numpy(valid),
            "audio": audio,
            "audio_shift": audio_shift,
            "has_shift": bool(shift != 0),
            "shift": int(shift),
            "speaker": str(row.get("speaker", "")),
            "audio_key": _audio_key(row),
            "utt_id": utt_id,
            "start_frame": start,
            "aug": aug,
        }


# ----------------------------------------------------------------------------------------------------------------------
# batching
# ----------------------------------------------------------------------------------------------------------------------
_TENSOR_KEYS = ("video", "lm", "cue", "valid", "audio", "audio_shift")
_LIST_KEYS = ("speaker", "audio_key", "utt_id", "aug")


def _first_seen_ids(keys: Sequence[Hashable]) -> torch.Tensor:
    """Number the distinct keys in order of first appearance: equal keys get equal ids."""
    table: dict[Hashable, int] = {}
    return torch.tensor([table.setdefault(k, len(table)) for k in keys], dtype=torch.long)


def sync_collate(items: list[dict]) -> dict:
    """Stack equal-length window items into a batch.

    Tensors: ``video [B,W,C,88,88]``, ``lm [B,W,80]``, ``cue [B,W,8]``, ``valid [B,W]``, ``audio``/``audio_shift``
    ``[B,W,320]``, ``has_shift`` bool ``[B]``, ``shift`` / ``start_frame`` long ``[B]``, ``audio_key_ids`` /
    ``speaker_ids`` long ``[B]`` (equal ids = same underlying recording / speaker). Lists (same key names as the
    items): ``speaker``, ``audio_key``, ``utt_id``, ``aug``.
    """
    if not items:
        raise ValueError("sync_collate got an empty batch")
    w = int(items[0]["video"].shape[0])
    for it in items:
        for key in _TENSOR_KEYS:
            if int(it[key].shape[0]) != w:
                raise ValueError(f"{it['utt_id']}: {key} has {it[key].shape[0]} frames, expected {w} "
                                 "(all windows of a batch must have the same length)")
    batch: dict[str, Any] = {k: torch.stack([it[k] for it in items]) for k in _TENSOR_KEYS}
    batch["has_shift"] = torch.tensor([bool(it["has_shift"]) for it in items], dtype=torch.bool)
    batch["shift"] = torch.tensor([int(it["shift"]) for it in items], dtype=torch.long)
    batch["start_frame"] = torch.tensor([int(it["start_frame"]) for it in items], dtype=torch.long)
    for key in _LIST_KEYS:
        batch[key] = [it[key] for it in items]
    batch["audio_key_ids"] = _first_seen_ids([tuple(k) if isinstance(k, list) else k for k in batch["audio_key"]])
    batch["speaker_ids"] = _first_seen_ids(batch["speaker"])
    return batch


__all__ = [
    "SyncWindowDataset", "sync_collate", "audio_window_features", "audio_window_batch", "num_audio_frames",
    "fbank_geometry", "CONTEXT_SEC", "SYNC_NOISE_KINDS",
]

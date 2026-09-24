"""Torch dataset over preprocessed utterances (SPEC section 8).

- :func:`load_manifests` reads every ``work_dir/manifests/*.jsonl`` shard.
- :func:`assign_split` splits rows into train/val/test from the config (speaker-independent).
- :class:`AVSRDataset` returns per-utterance tensors: mouth video (25 fps), lip skeleton, colour cues, validity,
  stacked fbank audio (optionally noise-augmented), SNR bucket and target tokens.
- :func:`collate_fn` pads a list of items into a batch; :class:`DurationBatchSampler` builds length-bucketed batches.

Randomness: training augmentation draws from a per-process generator seeded by ``cfg.seed`` and the DataLoader worker
seed (so it is reproducible under ``torch.manual_seed``); evaluation is deterministic, and ``fixed_noise`` uses a
per-utterance generator seeded from ``zlib.crc32(utt_id)`` so SNR sweeps are identical across runs and worker counts.
"""
from __future__ import annotations

import json
import logging
import math
import os
import zipfile
import zlib
from pathlib import Path
from typing import Any, Iterator, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, get_worker_info

from avsr.audio_feats import NOISE_KINDS, add_noise, compute_fbank, estimate_snr_db, make_noise, stack_frames
from avsr.text import Tokenizer

log = logging.getLogger(__name__)

REQUIRED_KEYS = (
    "utt_id", "split_dir", "speaker", "angle", "noise_env", "duration", "n_frames", "text", "has_unk", "mouth_mp4", "npz",
)
_ITEM_ERRORS = (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile, cv2.error)
_MAX_RETRIES = 8
# 25-fps video vs stacked-audio length: rounding alone gives up to 3 frames (source-frame rounding of the span +-1,
# ceil of the 30->25 resampling, and the 25 ms fbank window + floor of the 4x stacking), so only > 4 is reported
_MAX_AV_DESYNC = 4
_worker_threads_limited = False  # per process: OpenCV pinned to one thread inside DataLoader workers


# ----------------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------------
def cfg_get(cfg: Any, dotted: str, default: Any = None) -> Any:
    """Read ``cfg["a"]["b"]`` for ``dotted="a.b"`` from a dict or attribute-style config; ``default`` when absent."""
    node = cfg
    for part in dotted.split("."):
        if isinstance(node, dict):
            if part not in node:
                return default
            node = node[part]
        elif node is not None and hasattr(node, part):
            node = getattr(node, part)
        else:
            return default
    return default if node is None else node


def resampled_length(n_frames: int, fps_in: float = 30.0, fps_out: float = 25.0) -> int:
    """Number of frames after resampling ``n_frames`` from ``fps_in`` to ``fps_out``: ceil(n * fps_out / fps_in)."""
    if n_frames <= 0:
        return 0
    return int(math.ceil(n_frames * float(fps_out) / float(fps_in) - 1e-9))


def resample_indices(n_frames: int, fps_in: float = 30.0, fps_out: float = 25.0) -> np.ndarray:
    """Source frame index for each output frame: round(k * fps_in / fps_out), clipped to the last frame."""
    n_out = resampled_length(n_frames, fps_in, fps_out)
    idx = np.floor(np.arange(n_out) * (float(fps_in) / float(fps_out)) + 0.5).astype(np.int64)
    return np.clip(idx, 0, max(n_frames - 1, 0))


VIDEO_NORMS = ("global", "utterance")
CUE_NORMS = ("none", "utterance")


def normalize_pixels(pix: np.ndarray, mode: str = "global", mean: float = 0.421, std: float = 0.165) -> np.ndarray:
    """uint8 pixels [T, C, H, W] -> float32.

    'global'    : (x / 255 - mean) / std with dataset-wide constants.
    'utterance' : per-utterance, per-channel z-score. Removes brightness/contrast differences between recording
                  setups (in this dataset the expert speakers' crops are ~2x brighter than the others'), so lighting
                  cannot act as a speaker cue. Shared by training (dataset) and inference (infer.py)."""
    video = pix.astype(np.float32) * np.float32(1.0 / 255.0)
    if mode == "utterance":
        m = video.mean(axis=(0, 2, 3), keepdims=True)
        s = video.std(axis=(0, 2, 3), keepdims=True)
        video -= m
        video /= np.maximum(s, np.float32(1e-3))
    elif mode == "global":
        video -= np.float32(mean)
        video /= np.float32(std)
    else:
        raise ValueError(f"unknown video norm {mode!r}; expected one of {VIDEO_NORMS}")
    return video


def normalize_cue(cue: np.ndarray, valid: np.ndarray, mode: str = "none") -> np.ndarray:
    """Per-frame cues [T, 8] -> float32. 'utterance': z-score each channel over the frames with landmarks
    (lighting-dependent colour cues and speaker-specific lip size become relative movements); frames without
    landmarks stay 0. With fewer than 2 valid frames all cues are 0. 'none': unchanged."""
    cue = np.asarray(cue, dtype=np.float32)
    if mode == "none":
        return cue
    if mode != "utterance":
        raise ValueError(f"unknown cue norm {mode!r}; expected one of {CUE_NORMS}")
    ok = np.asarray(valid) > 0.5
    out = np.zeros_like(cue)
    if int(ok.sum()) >= 2:
        mu = cue[ok].mean(axis=0)
        sd = np.maximum(cue[ok].std(axis=0), np.float32(0.01))
        out[ok] = (cue[ok] - mu) / sd
    return out


def snr_to_bucket(snr_db: float | None) -> int:
    """SNR bucket: 0 = clean / >= 20 dB, 1 = [10, 20), 2 = [0, 10), 3 = < 0 dB."""
    if snr_db is None or snr_db >= 20.0:
        return 0
    if snr_db >= 10.0:
        return 1
    if snr_db >= 0.0:
        return 2
    return 3


def _parse_angles(value: Any) -> set[str] | None:
    """``"all"`` / empty → None (keep everything); ``"A,C"`` or ``[A, C]`` → {"A", "C"}."""
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in ("", "all"):
            return None
        items = value.split(",")
    else:
        items = list(value)
    angles = {str(a).strip().upper() for a in items if str(a).strip()}
    return angles or None


def _audio_key(row: dict) -> tuple[str, str, str]:
    """Identity of the recorded audio: all camera angles of one session share it (SPEC section 2)."""
    return str(row.get("speaker", "")), str(row.get("session", "")), str(row.get("sentence_id", row["utt_id"]))


# ----------------------------------------------------------------------------------------------------------------------
# manifests and splits
# ----------------------------------------------------------------------------------------------------------------------
def load_manifests(work_dir: str | os.PathLike) -> list[dict]:
    """Load every manifest shard ``work_dir/manifests/*.jsonl`` (sorted by file name).

    Each row gets an extra ``work_dir`` key (absolute) so its relative ``npz``/``mouth_mp4`` paths resolve anywhere.
    Raises on malformed JSON, missing required keys or duplicate ``utt_id``.
    """
    work = Path(work_dir).resolve()
    man_dir = work / "manifests"
    if not man_dir.is_dir():
        raise FileNotFoundError(f"manifest directory not found: {man_dir} (run avsr.preprocess first)")
    rows: list[dict] = []
    seen: set[str] = set()
    shards = sorted(man_dir.glob("*.jsonl"))
    undone = [s.stem for s in shards if not (man_dir / f"{s.stem}.done").exists()]
    if undone and len(undone) < len(shards):
        log.warning("%d manifest shard(s) have no .done marker (stale or being re-processed?), e.g. %s",
                    len(undone), undone[0])
    for shard in shards:
        with open(shard, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{shard}:{line_no}: invalid JSON ({e})") from e
                missing = [k for k in REQUIRED_KEYS if k not in row]
                if missing:
                    raise ValueError(f"{shard}:{line_no}: manifest row lacks keys {missing}")
                if row["utt_id"] in seen:
                    raise ValueError(f"{shard}:{line_no}: duplicate utt_id {row['utt_id']}")
                seen.add(row["utt_id"])
                row["work_dir"] = str(work)
                rows.append(row)
    return rows


def assign_split(rows: Sequence[dict], cfg: Any) -> dict[str, list[dict]]:
    """Split rows into ``train`` / ``val`` / ``test``.

    Dropped: ``has_unk`` rows, rows with no frames, durations outside [data.min_duration, data.max_duration], angles
    not in ``data.angles``. Then speakers in ``split.test_speakers`` → test, in ``split.val_speakers`` → val (these
    speakers never reach train), remaining rows whose ``split_dir`` is in ``split.train_split_dirs`` → train.
    """
    val_spk = {str(s) for s in (cfg_get(cfg, "split.val_speakers", []) or [])}
    test_spk = {str(s) for s in (cfg_get(cfg, "split.test_speakers", []) or [])}
    both = val_spk & test_spk
    if both:
        raise ValueError(f"speakers listed in both split.val_speakers and split.test_speakers: {sorted(both)}")
    train_dirs = {str(d) for d in (cfg_get(cfg, "split.train_split_dirs", []) or [])}
    min_d = float(cfg_get(cfg, "data.min_duration", 0.0))
    max_d = float(cfg_get(cfg, "data.max_duration", float("inf")))
    angles = _parse_angles(cfg_get(cfg, "data.angles", "all"))

    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    dropped = {"has_unk": 0, "duration": 0, "angle": 0, "no_split": 0}
    for r in rows:
        if bool(r["has_unk"]):
            dropped["has_unk"] += 1
            continue
        if int(r["n_frames"]) <= 0 or not (min_d <= float(r["duration"]) <= max_d):
            dropped["duration"] += 1
            continue
        if angles is not None and str(r["angle"]).upper() not in angles:
            dropped["angle"] += 1
            continue
        spk = str(r["speaker"])
        if spk in test_spk:
            out["test"].append(r)
        elif spk in val_spk:
            out["val"].append(r)
        elif str(r["split_dir"]) in train_dirs:
            out["train"].append(r)
        else:
            dropped["no_split"] += 1
    log.info("assign_split: train %d, val %d, test %d; dropped %s",
             len(out["train"]), len(out["val"]), len(out["test"]), dropped)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# decoding helpers
# ----------------------------------------------------------------------------------------------------------------------
def _open_capture(path: Path) -> cv2.VideoCapture | None:
    """Open a video with the FFmpeg backend single-threaded (fastest for 96×96 clips), any other backend as fallback."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def read_frames(path: str | os.PathLike, idx: np.ndarray, size: int) -> tuple[np.ndarray | None, int]:
    """Decode the frames ``idx`` (non-decreasing source indices) of a video in one sequential pass.

    Returns (frames [len(idx), size, size, 3] uint8 BGR, number of distinct frames actually decoded), or (None, 0)
    when the file cannot be opened/decoded. Frames past the end of a short file repeat the last decoded frame.
    """
    uniq = np.unique(idx)
    if uniq.size == 0:
        return np.zeros((0, size, size, 3), np.uint8), 0
    cap = _open_capture(Path(path))
    if cap is None:
        return None, 0
    wanted = uniq.tolist()
    buf = np.zeros((len(wanted), size, size, 3), np.uint8)
    got = 0
    try:
        frame_no = 0
        last = wanted[-1]
        while frame_no <= last:
            if not cap.grab():
                break
            if frame_no == wanted[got]:
                ok, img = cap.retrieve()
                if not ok or img is None:
                    break
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                if img.shape[0] != size or img.shape[1] != size:
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                buf[got] = img[:, :, :3]
                got += 1
            frame_no += 1
    finally:
        cap.release()
    if got == 0:
        return None, 0
    if got < len(wanted):
        buf[got:] = buf[got - 1]
    if len(wanted) == len(idx):
        return buf, got
    return buf[np.searchsorted(uniq, idx)], got


def spec_augment(fb: torch.Tensor, rng: np.random.Generator, freq_mask: int, freq_width: int,
                 time_mask: int, time_width: int) -> torch.Tensor:
    """SpecAugment on a (CMVN-normalised) fbank [T, F], in place: zero ``freq_mask`` bands of width U[0, freq_width]
    and ``time_mask`` spans of width U[0, time_width] frames."""
    n_t, n_f = fb.shape
    for _ in range(int(freq_mask)):
        w = int(rng.integers(0, min(int(freq_width), n_f) + 1))
        if w > 0:
            f0 = int(rng.integers(0, n_f - w + 1))
            fb[:, f0:f0 + w] = 0.0
    for _ in range(int(time_mask)):
        w = int(rng.integers(0, min(int(time_width), n_t) + 1))
        if w > 0:
            t0 = int(rng.integers(0, n_t - w + 1))
            fb[t0:t0 + w, :] = 0.0
    return fb


def _limit_worker_threads() -> None:
    """Inside a DataLoader worker, keep OpenCV single-threaded (torch already is): N workers each running an
    all-core OpenCV pool would oversubscribe the CPU. The main process is left untouched."""
    global _worker_threads_limited
    if not _worker_threads_limited and get_worker_info() is not None:
        cv2.setNumThreads(1)
        _worker_threads_limited = True


def _item_seed(seed: int, utt_id: str, kind: str) -> np.random.SeedSequence:
    """Stable per-utterance seed (independent of Python's hash randomisation and of the SNR level)."""
    return np.random.SeedSequence([seed & 0xFFFFFFFF, zlib.crc32(utt_id.encode("utf-8")), zlib.crc32(kind.encode("utf-8"))])


# ----------------------------------------------------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------------------------------------------------
class AVSRDataset(Dataset):
    """Audio-visual utterance dataset over manifest rows (see module docstring and SPEC section 8).

    Args:
        rows: manifest rows (from :func:`load_manifests` / :func:`assign_split`).
        cfg: config (dict or attribute-style), keys of SPEC section 9.
        train: enable augmentation (random crop, flip, time-mask, noise, SpecAugment).
        tokenizer: :class:`avsr.text.Tokenizer` (a new one when None).
        fixed_noise: ``(kind, snr_db)`` applied to every item (SNR sweeps); overrides random noise augmentation.
        babble_rows: rows whose audio forms the babble pool (default: ``rows``).
        work_dir: base for relative paths of rows without a ``work_dir`` key (default ``cfg.work_dir``).
    """

    pool_size = 64  # babble clips cached per worker
    pool_refresh_every = 16  # items between partial pool refreshes (train)
    pool_refresh_count = 4  # clips replaced per refresh

    def __init__(self, rows: Sequence[dict], cfg: Any, train: bool, tokenizer: Tokenizer | None = None,
                 fixed_noise: tuple[str, float] | None = None, babble_rows: Sequence[dict] | None = None,
                 work_dir: str | os.PathLike | None = None) -> None:
        self.rows = list(rows)
        self.train = bool(train)
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self.work_dir = Path(work_dir if work_dir is not None else cfg_get(cfg, "work_dir", "work"))
        self.seed = int(cfg_get(cfg, "seed", 0))

        self.channels = int(cfg_get(cfg, "video.channels", 1))
        self.size = int(cfg_get(cfg, "video.size", 96))
        self.crop = int(cfg_get(cfg, "video.crop", 88))
        self.fps_in = float(cfg_get(cfg, "video.fps_in", 30.0))
        self.fps_out = float(cfg_get(cfg, "video.fps_out", 25.0))
        self.mean = float(cfg_get(cfg, "video.mean", 0.421))
        self.std = float(cfg_get(cfg, "video.std", 0.165))
        self.norm = str(cfg_get(cfg, "video.norm", "global"))
        self.cue_norm = str(cfg_get(cfg, "video.cue_norm", "none"))
        if self.norm not in VIDEO_NORMS or self.cue_norm not in CUE_NORMS:
            raise ValueError(f"video.norm must be one of {VIDEO_NORMS} and video.cue_norm one of {CUE_NORMS}; "
                             f"got {self.norm!r}, {self.cue_norm!r}")
        self.flip_prob = float(cfg_get(cfg, "video.flip_prob", 0.5))
        self.time_mask_prob = float(cfg_get(cfg, "video.time_mask_prob", 0.0))
        self.time_mask_max = int(cfg_get(cfg, "video.time_mask_max", 0))
        if self.channels not in (1, 3):
            raise ValueError(f"video.channels must be 1 or 3, got {self.channels}")
        if not 0 < self.crop <= self.size:
            raise ValueError(f"video.crop ({self.crop}) must be in (0, video.size={self.size}]")

        self.sr = int(cfg_get(cfg, "audio.sr", 16000))
        self.n_mels = int(cfg_get(cfg, "audio.n_mels", 80))
        self.stack = int(cfg_get(cfg, "audio.stack", 4))
        self.noise_prob = float(cfg_get(cfg, "audio.noise_prob", 0.0))
        self.noise_kinds = [str(k) for k in (cfg_get(cfg, "audio.noise_kinds", list(NOISE_KINDS)) or [])]
        self.snr_min = float(cfg_get(cfg, "audio.snr_min", -5.0))
        self.snr_max = float(cfg_get(cfg, "audio.snr_max", 20.0))
        self.specaug = {k: int(cfg_get(cfg, f"audio.specaug.{k}", 0))
                        for k in ("freq_mask", "freq_width", "time_mask", "time_width")}
        bad = [k for k in self.noise_kinds if k not in NOISE_KINDS]
        if bad:
            raise ValueError(f"unknown audio.noise_kinds {bad}; expected a subset of {NOISE_KINDS}")
        if self.train and self.noise_prob > 0 and not self.noise_kinds:
            raise ValueError("audio.noise_prob > 0 but audio.noise_kinds is empty")

        if fixed_noise is not None:
            kind, snr = fixed_noise
            if kind not in NOISE_KINDS:
                raise ValueError(f"fixed_noise kind {kind!r} not in {NOISE_KINDS}")
            fixed_noise = (str(kind), float(snr))
        self.fixed_noise = fixed_noise
        self.babble_rows = list(babble_rows) if babble_rows is not None else self.rows

        self.n_bad = 0
        self.n_desync = 0  # items whose video/audio lengths differ by more than _MAX_AV_DESYNC frames
        self._reset_process_state()

    # --- process-local state (never pickled into workers) ------------------------------------------------------------
    def _reset_process_state(self) -> None:
        self._rng: np.random.Generator | None = None
        self._rng_owner: tuple | None = None
        self._pool: list[tuple[tuple[str, str, str], str, str, np.ndarray]] = []
        self._pool_counter = 0
        self._fixed_pool: list[tuple[tuple[str, str, str], str, str, np.ndarray]] | None = None
        self._warned_babble = False

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for key in ("_rng", "_rng_owner", "_fixed_pool"):
            state[key] = None
        state["_pool"] = []
        state["_pool_counter"] = 0
        return state

    def _process_rng(self) -> np.random.Generator:
        """Generator for training augmentation, (re)created per process / DataLoader worker."""
        info = get_worker_info()
        owner = (os.getpid(), -1 if info is None else info.id, None if info is None else info.seed)
        if self._rng is None or self._rng_owner != owner:
            if info is None:
                ss = np.random.SeedSequence([self.seed & 0xFFFFFFFF])
            else:
                ss = np.random.SeedSequence([self.seed & 0xFFFFFFFF, info.id, info.seed & 0xFFFFFFFFFFFFFFFF])
            self._rng = np.random.default_rng(ss)
            self._rng_owner = owner
            self._pool = []
            self._pool_counter = 0
        return self._rng

    # --- paths / io --------------------------------------------------------------------------------------------------
    def _path(self, row: dict, key: str) -> Path:
        base = Path(row["work_dir"]) if row.get("work_dir") else self.work_dir
        return base / str(row[key])

    def _load_clip(self, row: dict) -> tuple[tuple[str, str, str], str, str, np.ndarray] | None:
        """Babble pool entry (audio key, speaker, text, int16 audio) for ``row``; None if unreadable/empty."""
        try:
            with np.load(self._path(row, "npz")) as z:
                audio = np.asarray(z["audio"], dtype=np.int16)
        except _ITEM_ERRORS as e:
            self.n_bad += 1
            log.warning("babble clip %s unreadable (%s: %s)", row.get("utt_id"), type(e).__name__, e)
            return None
        if audio.size == 0:
            return None
        return _audio_key(row), str(row.get("speaker", "")), str(row.get("text", "")), audio

    def _train_pool(self, rng: np.random.Generator) -> list:
        """Per-worker babble pool: filled on first use, then ``pool_refresh_count`` clips replaced every
        ``pool_refresh_every`` items, so an item costs a fraction of an npz read instead of 2-4."""
        n_rows = len(self.babble_rows)
        if n_rows == 0:
            return []
        if not self._pool:
            picks = rng.choice(n_rows, size=min(self.pool_size, n_rows), replace=False)
            self._pool = [c for c in (self._load_clip(self.babble_rows[int(i)]) for i in picks) if c is not None]
        else:
            self._pool_counter += 1
            if self._pool_counter % self.pool_refresh_every == 0:
                for _ in range(self.pool_refresh_count):
                    clip = self._load_clip(self.babble_rows[int(rng.integers(0, n_rows))])
                    if clip is not None:
                        self._pool[int(rng.integers(0, len(self._pool)))] = clip
        return self._pool

    def _eval_pool(self) -> list:
        """Deterministic babble pool (same in every process): ``pool_size`` rows chosen with ``seed``."""
        if self._fixed_pool is None:
            n_rows = len(self.babble_rows)
            rng = np.random.default_rng([self.seed & 0xFFFFFFFF, 0xBABB1E])
            picks = sorted(rng.choice(n_rows, size=min(self.pool_size, n_rows), replace=False).tolist()) if n_rows else []
            self._fixed_pool = [c for c in (self._load_clip(self.babble_rows[i]) for i in picks) if c is not None]
        return self._fixed_pool

    def _babble_clips(self, row: dict, pool: list, rng: np.random.Generator) -> list[np.ndarray]:
        """2-4 pool clips that are not this utterance's audio (other angles share it) nor the same sentence text;
        other speakers are preferred when the pool has at least two."""
        key, spk, text = _audio_key(row), str(row.get("speaker", "")), str(row.get("text", ""))
        eligible = [c for c in pool if c[0] != key and c[2] != text]
        other_spk = [c for c in eligible if c[1] != spk]
        if len(other_spk) >= 2:
            eligible = other_spk
        if not eligible:
            return []
        n = min(int(rng.integers(2, 5)), len(eligible))
        picks = rng.choice(len(eligible), size=n, replace=False)
        return [eligible[int(i)][3] for i in picks]

    def _choose_noise(self, row: dict, rng: np.random.Generator | None
                      ) -> tuple[str | None, float | None, np.random.Generator | None, list]:
        """(kind, snr_db, generator, babble pool) of the noise to add to this item, or all-None for clean audio."""
        if self.fixed_noise is not None:
            kind, snr = self.fixed_noise
            noise_rng = np.random.default_rng(_item_seed(self.seed, str(row["utt_id"]), kind))
            return kind, snr, noise_rng, (self._eval_pool() if kind == "babble" else [])
        if rng is not None and self.noise_prob > 0 and rng.random() < self.noise_prob:
            kind = str(self.noise_kinds[int(rng.integers(0, len(self.noise_kinds)))])
            snr = float(rng.uniform(self.snr_min, self.snr_max))
            return kind, snr, rng, (self._train_pool(rng) if kind == "babble" else [])
        return None, None, None, []

    # --- item construction -------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        for attempt in range(_MAX_RETRIES):
            j = (int(index) + attempt) % len(self.rows)
            try:
                return self._make_item(self.rows[j])
            except _ITEM_ERRORS as e:
                self.n_bad += 1
                log.warning("skipping bad utterance %s (%s: %s); bad items in this process: %d",
                            self.rows[j].get("utt_id"), type(e).__name__, e, self.n_bad)
        raise RuntimeError(f"{_MAX_RETRIES} consecutive unreadable utterances starting at index {index}")

    def _make_item(self, row: dict) -> dict:
        _limit_worker_threads()
        rng = self._process_rng() if self.train else None
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
        if n_t == 0:
            raise ValueError("utterance has no video frames")

        # ---- visual streams: 30 -> 25 fps, crop, flip, normalise, time-mask
        idx = resample_indices(n_t, fps, self.fps_out)
        n_v = int(idx.size)
        mp4 = self._path(row, "mouth_mp4")
        frames, n_dec = read_frames(mp4, idx, self.size)
        if frames is None:
            log.warning("mouth video unreadable, using zeros: %s", mp4)
            frames = np.zeros((n_v, self.size, self.size, 3), np.uint8)
        elif n_dec < int(np.unique(idx).size):
            log.warning("mouth video %s is shorter than its npz (%d frames): last frame repeated", mp4, n_t)

        margin = self.size - self.crop
        if self.train:
            y0, x0 = int(rng.integers(0, margin + 1)), int(rng.integers(0, margin + 1))
            flip = bool(rng.random() < self.flip_prob)
        else:
            y0 = x0 = margin // 2
            flip = False
        y1, x1 = y0 + self.crop, x0 + self.crop
        if self.channels == 1:  # one cvtColor call over the whole (contiguous) stack, then a crop view
            gray = cv2.cvtColor(np.ascontiguousarray(frames).reshape(n_v * self.size, self.size, 3), cv2.COLOR_BGR2GRAY)
            pix = gray.reshape(n_v, 1, self.size, self.size)[:, :, y0:y1, x0:x1]
        else:
            pix = frames[:, y0:y1, x0:x1, ::-1].transpose(0, 3, 1, 2)  # BGR -> RGB, [T, 3, H, W]
        if flip:
            pix = pix[..., ::-1]
        video = normalize_pixels(pix, self.norm, self.mean, self.std)

        lm = lm_all[idx].reshape(n_v, 80).astype(np.float32)
        if flip:
            lm[:, 0::2] *= -1.0
        valid = valid_all[idx].astype(np.float32)
        cue = normalize_cue(cue_all[idx].astype(np.float32), valid, self.cue_norm)
        if self.train and self.time_mask_max > 0 and rng.random() < self.time_mask_prob:
            w = int(rng.integers(1, min(self.time_mask_max, n_v) + 1))
            t0 = int(rng.integers(0, n_v - w + 1))
            video[t0:t0 + w] = 0.0
            lm[t0:t0 + w] = 0.0
            cue[t0:t0 + w] = 0.0
            valid[t0:t0 + w] = 0.0

        # ---- audio: optional noise -> fbank + CMVN -> SpecAugment (train) -> stack to 25 Hz
        kind, snr, noise_rng, pool = self._choose_noise(row, rng)
        signal: np.ndarray = wave
        if kind is not None and snr is not None and noise_rng is not None and wave.size > 0:
            clips: list[np.ndarray] = []
            if kind == "babble":
                clips = self._babble_clips(row, pool, noise_rng)
                if not clips:
                    if not self._warned_babble:
                        log.warning("babble pool has no clip usable for %s; using white noise instead", row["utt_id"])
                        self._warned_babble = True
                    kind = "white"
            noise = make_noise(kind, int(wave.size), noise_rng, clips or None)
            signal, snr = add_noise(wave, noise, snr)
        else:
            kind, snr = None, None

        fb = compute_fbank(signal, self.sr, n_mels=self.n_mels)
        if self.train and fb.shape[0] > 0:
            spec_augment(fb, rng, **self.specaug)
        audio = stack_frames(fb, self.stack)
        if abs(n_v - int(audio.shape[0])) > _MAX_AV_DESYNC:
            self.n_desync += 1
            if self.n_desync <= 5 or self.n_desync % 1000 == 0:
                log.warning("%s: video has %d frames at %g fps but audio %d (> %d apart; trimmed to the shorter); "
                            "desynced items in this process: %d", row["utt_id"], n_v, self.fps_out,
                            int(audio.shape[0]), _MAX_AV_DESYNC, self.n_desync)
        if audio.shape[0] == 0:  # shorter than one stacked frame: keep one silent (mean) frame
            audio = torch.zeros(1, self.stack * self.n_mels, dtype=torch.float32)

        n = min(n_v, int(audio.shape[0]))
        return {
            "video": torch.from_numpy(video[:n]),
            "lm": torch.from_numpy(lm[:n]),
            "cue": torch.from_numpy(cue[:n]),
            "valid": torch.from_numpy(valid[:n]),
            "audio": audio[:n].contiguous(),
            "snr_bucket": torch.tensor(snr_to_bucket(snr), dtype=torch.long),
            "tokens": torch.tensor(self.tokenizer.encode(str(row["text"])), dtype=torch.long),
            "text": str(row["text"]),
            "utt_id": str(row["utt_id"]),
            "meta": {
                "speaker": row.get("speaker"), "angle": row.get("angle"), "noise_env": row.get("noise_env"),
                "noise_kind": kind or "clean", "snr_db": snr,
                # eval only: the DSP SNR estimate inference would see (used to calibrate infer.snr_threshold)
                "snr_dsp": None if self.train else float(estimate_snr_db(signal, self.sr)),
            },
        }


# ----------------------------------------------------------------------------------------------------------------------
# batching
# ----------------------------------------------------------------------------------------------------------------------
def collate_fn(items: list[dict]) -> dict:
    """Pad a list of dataset items into a batch (zeros for features, ``pad_id`` for tokens); see SPEC section 8."""
    if not items:
        raise ValueError("collate_fn got an empty batch")
    bsz = len(items)
    lengths = [int(it["video"].shape[0]) for it in items]
    for it, n in zip(items, lengths):
        for key in ("lm", "cue", "valid", "audio"):
            if int(it[key].shape[0]) != n:
                raise ValueError(f"{it['utt_id']}: {key} has {it[key].shape[0]} frames, video has {n}")
    t_max = max(lengths)
    c, h, w = items[0]["video"].shape[1:]
    d_audio = items[0]["audio"].shape[1]
    video = torch.zeros(bsz, t_max, c, h, w, dtype=torch.float32)
    lm = torch.zeros(bsz, t_max, 80, dtype=torch.float32)
    cue = torch.zeros(bsz, t_max, 8, dtype=torch.float32)
    valid = torch.zeros(bsz, t_max, dtype=torch.float32)
    audio = torch.zeros(bsz, t_max, d_audio, dtype=torch.float32)
    tok_lens = [int(it["tokens"].numel()) for it in items]
    tokens = torch.full((bsz, max(tok_lens)), Tokenizer.pad_id, dtype=torch.long)
    for b, it in enumerate(items):
        n = lengths[b]
        video[b, :n] = it["video"]
        lm[b, :n] = it["lm"]
        cue[b, :n] = it["cue"]
        valid[b, :n] = it["valid"]
        audio[b, :n] = it["audio"]
        tokens[b, :tok_lens[b]] = it["tokens"]
    return {
        "video": video, "lm": lm, "cue": cue, "valid": valid, "audio": audio,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "tokens": tokens, "token_lengths": torch.tensor(tok_lens, dtype=torch.long),
        "snr_bucket": torch.stack([it["snr_bucket"] for it in items]),
        "texts": [it["text"] for it in items],
        "utt_ids": [it["utt_id"] for it in items],
        "metas": [it["meta"] for it in items],
    }


class DurationBatchSampler(Sampler[list[int]]):
    """Length-bucketed batches of dataset indices.

    Rows are sorted by their 25-fps length (ties broken randomly per epoch when shuffling) and greedily packed so the
    padded size ``batch_size * longest`` stays ≤ ``max_frames`` (an over-long single utterance gets its own batch).
    Batch boundaries depend only on the sorted lengths, so ``len()`` is constant; with ``shuffle`` the batch order is
    reshuffled every epoch. Call ``set_epoch(epoch)`` before each epoch (otherwise the epoch auto-increments).
    """

    def __init__(self, rows: Sequence[dict], max_frames: int, shuffle: bool = True, seed: int = 0,
                 fps_in: float = 30.0, fps_out: float = 25.0) -> None:
        if int(max_frames) <= 0:
            raise ValueError("max_frames must be positive")
        self.lengths = np.array([resampled_length(int(r["n_frames"]), float(r.get("fps") or fps_in), fps_out)
                                 for r in rows], dtype=np.int64)
        self.max_frames = int(max_frames)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._bounds = self._pack(np.sort(self.lengths, kind="stable"))

    def _pack(self, sorted_lengths: np.ndarray) -> list[tuple[int, int]]:
        bounds: list[tuple[int, int]] = []
        start = 0
        for pos, length in enumerate(sorted_lengths.tolist()):
            if pos > start and (pos - start + 1) * max(length, 1) > self.max_frames:
                bounds.append((start, pos))
                start = pos
        if start < len(sorted_lengths):
            bounds.append((start, len(sorted_lengths)))
        return bounds

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self._bounds)

    def __iter__(self) -> Iterator[list[int]]:
        epoch = self.epoch
        self.epoch += 1
        if self.shuffle:
            rng = np.random.default_rng([self.seed & 0xFFFFFFFF, epoch & 0xFFFFFFFF])
            perm = rng.permutation(self.lengths.size)
            order = perm[np.argsort(self.lengths[perm], kind="stable")]
            batch_order = rng.permutation(len(self._bounds)).tolist()
        else:
            order = np.argsort(self.lengths, kind="stable")
            batch_order = list(range(len(self._bounds)))
        for b in batch_order:
            s, e = self._bounds[b]
            yield order[s:e].tolist()


__all__ = [
    "AVSRDataset", "DurationBatchSampler", "collate_fn", "load_manifests", "assign_split", "cfg_get",
    "resampled_length", "resample_indices", "snr_to_bucket", "read_frames", "spec_augment",
]

"""Video -> Korean subtitles (.srt + .json) with SNR-gated modality selection (SPEC section 12).

CLI (run from the project root)::

    python -m avsr.infer --config configs/base.yaml --ckpt work/checkpoints/best.pt --video input.mp4
                         [--audio other.wav] [--out out.srt] [--mode auto|av|audio|video]
                         [--snr-threshold 5] [--vad auto|energy|visual] [--set key.sub=value ...]

Python::

    from avsr.infer import transcribe_video
    segments = transcribe_video("input.mp4", "work/checkpoints/best.pt", "configs/base.yaml")

Pipeline: audio (16 kHz mono) -> video frames resampled by time to 30 fps -> MediaPipe lip landmarks (on the frame
downscaled to max side 640), skeleton, colour cues and 96x96 mouth crops (same per-frame logic as
``avsr.preprocess.process_video``) -> segmentation (energy VAD when the audio is trusted, otherwise visual VAD on the
mouth opening) -> per segment: SNR estimate, modality choice (av / video), ``model.decode`` -> .srt + .json.

``avsr.models`` is imported only when a checkpoint is loaded, so the utilities of this file (SRT writer, VADs,
resampling, gating) can be used and tested without it.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
from tqdm import tqdm

from avsr import video_feats as vf
from avsr.audio_feats import compute_fbank, estimate_snr_db, load_audio_16k, stack_frames
from avsr.dataset import cfg_get, normalize_cue, normalize_pixels, resample_indices
from avsr.text import Tokenizer
from avsr.utils import Config, load_config, merge_dicts, safe_console, to_dict

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
DEFAULT_LANDMARKER = PROJECT_ROOT / "assets" / "face_landmarker.task"
MODES = ("auto", "av", "audio", "video")
VAD_KINDS = ("auto", "energy", "visual")
SNR_ESTIMATORS = ("dsp", "model", "hybrid")
#: midpoints (dB) of the SNR-head buckets: 0 = clean/>=20 dB, 1 = 10-20 dB, 2 = 0-10 dB, 3 = <0 dB
SNR_BUCKET_MID = (25.0, 15.0, 5.0, -5.0)
#: config sections that define the network and its input features: always taken from the checkpoint
PROTECTED_CFG_SECTIONS = ("model", "video", "audio")
#: in auto mode a segment whose lips are visible on less than this fraction of frames is not decoded lips-only
MIN_VISUAL_RATIO = 0.2


# ----------------------------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------------------------
def build_run_config(ckpt_cfg: Mapping[str, Any] | None, cli_cfg: Mapping[str, Any] | None = None,
                     overrides: Sequence[str] = ()) -> Config:
    """Run-time config = checkpoint config, overlaid with the CLI config (``--config``: ``infer``, ``eval``, ...),
    with the sections that define the network and its input features (:data:`PROTECTED_CFG_SECTIONS`) restored
    from the checkpoint (the weights only fit them), then ``--set`` overrides last (as in ``avsr.evaluate``)."""
    data: dict = to_dict(ckpt_cfg) if ckpt_cfg else {}
    if cli_cfg:
        data = merge_dicts(data, cli_cfg)
    for section in PROTECTED_CFG_SECTIONS:
        if ckpt_cfg and section in ckpt_cfg:
            data[section] = to_dict(ckpt_cfg[section])
    return load_config(None, list(overrides), base=data)


# ----------------------------------------------------------------------------------------------------------
# Segments and subtitle writers
# ----------------------------------------------------------------------------------------------------------
@dataclass
class Segment:
    """One subtitle cue. Times in seconds; SNR values in dB (None when no audio is available)."""

    index: int
    start: float
    end: float
    text: str = ""
    mode: str = ""
    snr_est: float | None = None
    snr_dsp: float | None = None
    snr_model: float | None = None
    visual_ratio: float | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        def r(x: float | None) -> float | None:
            return None if x is None else round(float(x), 3)

        return {"index": self.index, "start": r(self.start), "end": r(self.end), "text": self.text,
                "mode": self.mode, "snr_est": r(self.snr_est), "snr_dsp": r(self.snr_dsp),
                "snr_model": r(self.snr_model), "visual_ratio": r(self.visual_ratio)}


def format_srt_time(seconds: float) -> str:
    """Seconds -> SRT time ``HH:MM:SS,mmm`` (rounded to the nearest millisecond, negative clamped to 0)."""
    total_ms = int(round(max(float(seconds), 0.0) * 1000.0))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(path: str | Path, segments: Iterable[Segment], skip_empty: bool = True) -> int:
    """Write segments as a SubRip file (UTF-8 without BOM, CRLF line ends on every OS); cues are numbered 1..N.
    Returns the number of cues written."""
    lines: list[str] = []
    n = 0
    for seg in segments:
        text = " ".join((seg.text or "").split())
        if skip_empty and not text:
            continue
        n += 1
        lines += [str(n), f"{format_srt_time(seg.start)} --> {format_srt_time(seg.end)}", text, ""]
    Path(path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8", newline="\r\n")
    return n


def write_json(path: str | Path, segments: Iterable[Segment], meta: dict | None = None) -> None:
    """Write the JSON sidecar: ``meta`` keys first, then ``segments`` (start, end, text, mode, snr_est, ...)."""
    payload = dict(meta or {})
    payload["segments"] = [seg.to_dict() for seg in segments]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def format_summary(segments: Sequence[Segment]) -> str:
    """Human-readable summary table (segment, times, mode, SNR, text)."""
    header = f"{'#':>3}  {'start':>12}  {'end':>12}  {'mode':<5}  {'SNR':>6}  text"
    rows = [header, "-" * len(header)]
    for seg in segments:
        snr = "n/a" if seg.snr_est is None else f"{seg.snr_est:.1f}"
        rows.append(f"{seg.index:>3}  {format_srt_time(seg.start):>12}  {format_srt_time(seg.end):>12}  "
                    f"{seg.mode:<5}  {snr:>6}  {seg.text}")
    return "\n".join(rows)


def output_paths(video_path: str | Path, out: str | Path | None) -> tuple[Path, Path]:
    """(srt_path, json_path) for ``--out``: default ``<video>.srt``; an existing directory, or a path ending with a
    separator (created later), -> ``<dir>/<video stem>.srt``; a ``.json`` path names the sidecar; any other suffix
    gets ``.srt`` appended. The JSON sits next to the SRT."""
    video_path = Path(video_path)
    if out is None:
        srt = video_path.with_suffix(".srt")
    else:
        is_dir = isinstance(out, str) and out.endswith(("/", "\\"))
        out = Path(out)
        if is_dir or out.is_dir():
            srt = out / f"{video_path.stem}.srt"
        elif out.suffix.lower() == ".json":
            return out.with_suffix(".srt"), out
        elif out.suffix.lower() == ".srt":
            srt = out
        else:
            srt = Path(f"{out}.srt")
    return srt, srt.with_suffix(".json")


_WIN_INVALID_CHARS = frozenset('<>:"|?*')


def prepare_output(srt_path: Path) -> None:
    """Fail fast, before the slow pipeline, on an output path Windows cannot create, and create its directory.

    Typical cause: Windows PowerShell 5.1 passes a quoted path that ends with a backslash (``--out "D:\\my dir\\"``)
    so that the backslash escapes the closing quote: the path then swallows ``"`` and the following arguments."""
    if os.name == "nt":
        for part in srt_path.parts[1:] if srt_path.anchor else srt_path.parts:
            if _WIN_INVALID_CHARS & set(part):
                raise ValueError(f"invalid output path {str(srt_path)!r}: {part!r} contains one of <>:\"|?* "
                                 "(in PowerShell, do not end a quoted path with a backslash)")
    srt_path.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------------------------------------
# Frame-rate resampling
# ----------------------------------------------------------------------------------------------------------
_EPS_T = 1e-6  # seconds; target times within this distance of a boundary go to the later source frame


def sane_fps(fps: float | None, fallback: float = 30.0) -> float:
    """``fps`` if it is a plausible frame rate, else ``fallback`` (OpenCV sometimes reports 0 or NaN)."""
    if fps is None or not math.isfinite(fps) or fps < 1.0 or fps > 240.0:
        return float(fallback)
    return float(fps)


class TimeResampler:
    """Streaming nearest-frame resampler from (possibly variable-rate) source timestamps to a ``dst_fps`` grid.

    Call :meth:`push` with the timestamp of every decoded source frame, in order; it returns how many target frames
    (times ``k / dst_fps``) the *previous* source frame is the nearest frame for (0 = the frame is dropped). After the
    last frame, :meth:`finish` returns the last frame's share: targets up to the end of the video (last timestamp +
    last frame interval). Invalid timestamps (NaN, negative, not increasing) are replaced by the previous timestamp
    plus the last frame interval, so a backend without timestamps degrades to constant-rate resampling.
    """

    def __init__(self, dst_fps: float = 30.0, default_interval: float = 1.0 / 30.0) -> None:
        self.dst_fps = float(dst_fps)
        self.interval = float(default_interval)
        self.k = 0  # next target frame index
        self.last_ts: float | None = None
        self.n_fixed = 0  # number of timestamps that had to be repaired

    def _sanitize(self, ts: float | None) -> float:
        prev = self.last_ts
        ok = ts is not None and math.isfinite(ts) and ts >= 0.0
        if ok and (prev is None or ts > prev + _EPS_T):
            return float(ts)
        self.n_fixed += 1
        return 0.0 if prev is None else prev + self.interval

    def _take_until(self, bound: float) -> int:
        """Advance over all targets with time < ``bound`` (minus a tolerance); return how many."""
        n = int(math.ceil((bound - _EPS_T) * self.dst_fps)) - self.k
        n = max(n, 0)
        self.k += n
        return n

    def push(self, ts: float | None) -> int:
        ts = self._sanitize(ts)
        if self.last_ts is None:
            self.last_ts = ts
            return 0
        n = self._take_until(0.5 * (self.last_ts + ts))
        self.interval = ts - self.last_ts
        self.last_ts = ts
        return n

    def finish(self) -> int:
        if self.last_ts is None:
            return 0
        return self._take_until(self.last_ts + self.interval)


# ----------------------------------------------------------------------------------------------------------
# Segmentation: energy VAD / visual VAD -> segments within [segment_min_sec, segment_max_sec]
# ----------------------------------------------------------------------------------------------------------
def frame_rms_db(wave: np.ndarray, sr: int = 16000, win_sec: float = 0.025, hop_sec: float = 0.010) -> np.ndarray:
    """Per-frame RMS level in dB (25 ms window / 10 ms hop -> 100 Hz). int16 input is scaled to [-1, 1]."""
    x = np.asarray(wave)
    x = x.astype(np.float64) / 32768.0 if np.issubdtype(x.dtype, np.integer) else x.astype(np.float64)
    win = max(int(round(win_sec * sr)), 1)
    hop = max(int(round(hop_sec * sr)), 1)
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    csum = np.concatenate([[0.0], np.cumsum(x * x)])
    starts = np.arange(n) * hop
    energy = (csum[starts + win] - csum[starts]) / win
    return (10.0 * np.log10(np.maximum(energy, 0.0) + 1e-10)).astype(np.float32)


def moving_stats(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Centred moving mean and moving standard deviation over ``window`` samples (edge-padded, same length)."""
    window = max(int(window), 1)
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return x.astype(np.float32), x.astype(np.float32)
    offset = float(x.mean())  # centring keeps the cumulative sums well conditioned (constant input -> std 0)
    pad_l = window // 2
    xp = np.pad(x - offset, (pad_l, window - 1 - pad_l), mode="edge")
    c1 = np.concatenate([[0.0], np.cumsum(xp)])
    c2 = np.concatenate([[0.0], np.cumsum(xp * xp)])
    i = np.arange(n)
    mean = (c1[i + window] - c1[i]) / window
    var = (c2[i + window] - c2[i]) / window - mean * mean
    return (mean + offset).astype(np.float32), np.sqrt(np.maximum(var, 0.0)).astype(np.float32)


def fill_invalid(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``x`` where ``valid == 0`` (edges held); all-invalid -> zeros."""
    x = np.asarray(x, dtype=np.float64).copy()
    v = np.asarray(valid).astype(bool)
    if len(x) == 0 or not v.any():
        return np.zeros_like(x)
    if not v.all():
        idx = np.arange(len(x))
        x[~v] = np.interp(idx[~v], idx[v], x[v])
    return x


Span = tuple[float, float]


def _mask_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Runs of True in a boolean mask as ``(start, end_exclusive)`` index pairs."""
    d = np.diff(np.concatenate([[0], np.asarray(mask).astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _merge_close(spans: list[Span], gap: float) -> list[Span]:
    """Merge consecutive spans separated by less than ``gap`` seconds (overlaps always merge)."""
    out: list[Span] = []
    for s, e in spans:
        if out and s - out[-1][1] < gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def _pack(spans: list[Span], max_gap: float, max_len: float) -> list[Span]:
    """Merge neighbouring spans across pauses of at most ``max_gap`` seconds, smallest pause first, as long as the
    merged span stays <= ``max_len``: in-sentence pauses do not produce 1-word fragments, while the longest pauses
    remain the boundaries."""
    spans = list(spans)
    while len(spans) > 1:
        best: tuple[float, int] | None = None
        for i in range(len(spans) - 1):
            gap = spans[i + 1][0] - spans[i][1]
            if gap <= max_gap and spans[i + 1][1] - spans[i][0] <= max_len and (best is None or gap < best[0]):
                best = (gap, i)
        if best is None:
            break
        i = best[1]
        spans[i:i + 2] = [(spans[i][0], spans[i + 1][1])]
    return spans


def _split_long(start: float, end: float, cut_cost: np.ndarray, frame_hz: float, min_sec: float, max_sec: float,
                piece_cost: float = 0.5, balance: float = 0.05) -> list[Span]:
    """Split ``[start, end)`` into pieces of ``[min_sec, max_sec]`` seconds, cutting where the mouth/voice is least
    active. Dynamic programming over cut positions (frame resolution) minimising
    ``sum(cut_cost at the cuts) + piece_cost * n_pieces + balance * sum((piece / max_sec)^2)``, where ``cut_cost``
    is the activity normalised to [0, 1]: deep pauses are preferred, 1-word fragments cost an extra piece each, and
    flat activity is cut into equal pieces."""
    length = end - start
    if length <= max_sec + 1e-9:
        return [(start, end)]
    n_pos = max(int(round(length * frame_hz)), 1)
    step = length / n_pos
    lo = max(int(math.ceil(min_sec / step - 1e-9)), 1)
    hi = int(math.floor(max_sec / step + 1e-9))
    times = start + np.arange(n_pos + 1) * step
    cost = cut_cost[np.clip(np.round(times * frame_hz).astype(np.int64), 0, max(len(cut_cost) - 1, 0))] \
        if len(cut_cost) else np.zeros(n_pos + 1)
    best = np.full(n_pos + 1, np.inf)
    back = np.zeros(n_pos + 1, dtype=np.int64)
    best[0] = 0.0
    for k in range(lo, n_pos + 1):
        a, b = max(k - hi, 0), k - lo
        if b < a:
            continue
        piece = (k - np.arange(a, b + 1)) * (step / max_sec)
        total = best[a:b + 1] + balance * piece * piece
        j = int(np.argmin(total))
        if np.isfinite(total[j]):
            best[k] = total[j] + piece_cost + (0.0 if k == n_pos else float(cost[k]))
            back[k] = a + j
    if not np.isfinite(best[n_pos]):  # no admissible cut sequence (max_sec < 2 * min_sec): equal pieces
        n = int(math.ceil(length / max_sec))
        edges = [start + length * i / n for i in range(n + 1)]
        return list(zip(edges[:-1], edges[1:]))
    cuts = [n_pos]
    while cuts[-1] > 0:
        cuts.append(int(back[cuts[-1]]))
    edges = [end if k == n_pos else float(times[k]) for k in reversed(cuts)]
    return list(zip(edges[:-1], edges[1:]))


def _expand_short(spans: list[Span], min_sec: float, total: float) -> list[Span]:
    """Grow spans shorter than ``min_sec`` around their centre, never into a neighbour or outside [0, total]
    (a span boxed in by close neighbours stays shorter)."""
    out = list(spans)
    for i, (s, e) in enumerate(out):
        need = min_sec - (e - s)
        if need <= 0:
            continue
        room_l = s - (out[i - 1][1] if i > 0 else 0.0)
        room_r = (out[i + 1][0] if i + 1 < len(out) else total) - e
        left = min(0.5 * need, room_l)
        right = min(need - left, room_r)
        left = min(need - right, room_l)
        out[i] = (s - left, e + right)
    return out


def _pad_spans(spans: list[Span], pad: float, total: float) -> list[Span]:
    """Pad each span by up to ``pad`` seconds per side without crossing the midpoint to its neighbours."""
    out: list[Span] = []
    for i, (s, e) in enumerate(spans):
        left = min(pad, 0.5 * (s - spans[i - 1][1])) if i > 0 else min(pad, s)
        right = min(pad, 0.5 * (spans[i + 1][0] - e)) if i + 1 < len(spans) else min(pad, total - e)
        out.append((max(s - max(left, 0.0), 0.0), min(e + max(right, 0.0), total)))
    return out


def activity_to_segments(active: np.ndarray, activity: np.ndarray, frame_hz: float, total_sec: float, *,
                         time_offset: float = 0.0, min_active_sec: float = 0.1, hangover_sec: float = 0.3,
                         merge_gap_sec: float = 0.25, pack_gap_sec: float = 1.0, min_sec: float = 1.0,
                         max_sec: float = 8.0, pad_sec: float = 0.2) -> list[Span]:
    """Per-frame activity mask -> utterance segments ``(start, end)`` in seconds.

    Active runs shorter than ``min_active_sec`` (clicks) are dropped -> hangover after each run -> gaps shorter than
    ``merge_gap_sec`` merged -> neighbours packed across pauses <= ``pack_gap_sec`` while they fit (:func:`_pack`)
    -> spans longer than ``max_sec - 2 * pad_sec`` split at low (0.3 s smoothed) activity (:func:`_split_long`) ->
    spans shorter than ``min_sec`` grown -> padded by ``pad_sec`` without overlapping (every segment <= ``max_sec``).
    Frame ``j`` covers time ``time_offset + j / frame_hz``. Always returns at least one segment (the whole
    timeline, split to ``max_sec``, when nothing is active).
    """
    active = np.asarray(active).astype(bool)
    activity = np.asarray(activity, dtype=np.float32)
    total_sec = float(max(total_sec, 1.0 / frame_hz))
    n = len(active)
    min_run = max(int(round(min_active_sec * frame_hz)), 1)
    hang = int(round(hangover_sec * frame_hz))
    runs = [(s, min(e + hang, n)) for s, e in _mask_runs(active) if e - s >= min_run]
    spans = [(max(time_offset + s / frame_hz, 0.0), min(time_offset + e / frame_hz, total_sec)) for s, e in runs]
    max_core = max(min_sec, max_sec - 2.0 * pad_sec)
    spans = _pack(_merge_close(spans, merge_gap_sec), pack_gap_sec, max_core) or [(0.0, total_sec)]
    cut_cost = np.zeros(0, dtype=np.float32)
    if n:  # activity smoothed over 0.3 s and normalised to [0, 1] (5th .. 95th percentile)
        smooth = moving_stats(activity, int(round(0.3 * frame_hz)))[0]
        p5, p95 = np.percentile(smooth, [5, 95])
        cut_cost = np.clip((smooth - p5) / max(float(p95 - p5), 1e-6), 0.0, 1.0)
    pieces: list[Span] = []
    for s, e in spans:
        pieces += _split_long(s, e, cut_cost, frame_hz, min_sec, max_core)
    return _pad_spans(_expand_short(pieces, min(min_sec, total_sec), total_sec), pad_sec, total_sec)


def energy_vad_segments(wave: np.ndarray, sr: int = 16000, *, margin_db: float = 6.0, min_margin_db: float = 2.0,
                        range_frac: float = 0.35, max_range_db: float = 45.0, **seg_kw: float) -> list[Span]:
    """Energy VAD on frame RMS (dB, 25 ms / 10 ms). Threshold = noise floor (10th percentile) + margin, but never
    more than ``max_range_db`` below the loudest frames (99th percentile), which guards against digital silence
    (-100 dB). The margin is ``margin_db`` (6 dB) as long as the level range ``p90 - p10`` exceeds
    ``margin_db / range_frac`` (~17 dB, any reasonably clean recording); in noisier audio it shrinks to
    ``range_frac * (p90 - p10)`` (at least ``min_margin_db``), because with speech only 5-10 dB above the floor a
    fixed 6 dB margin keeps just the loudest syllables (measured on a real video: 34-76 % of the speech detected at
    2-5 dB white noise, versus >= 96 % with the adaptive margin, clean audio unchanged).
    ``seg_kw`` go to :func:`activity_to_segments` (min_sec, max_sec, pad_sec, ...)."""
    db = frame_rms_db(wave, sr)
    p10, p90, p99 = (float(v) for v in np.percentile(db, [10, 90, 99]))
    margin = float(np.clip(range_frac * (p90 - p10), min_margin_db, margin_db))
    threshold = max(p10 + margin, p99 - max_range_db)
    total = len(np.asarray(wave)) / float(sr)
    return activity_to_segments(db > threshold, db, 100.0, total, time_offset=0.0075, **seg_kw)


def visual_vad_segments(inner_height: np.ndarray, valid: np.ndarray, fps: float = 30.0, *, window_sec: float = 0.5,
                        percentile: float = 40.0, rel_floor: float = 0.15, rel_ceil: float = 0.35,
                        min_activity: float = 1e-4, **seg_kw: float) -> list[Span]:
    """Visual VAD: mouth activity = moving std of the inner lip height (``cue[:, 1]``, in inter-ocular units) over
    ``window_sec``, invalid frames interpolated. Threshold = its ``percentile``-th percentile, kept within
    [``rel_floor``, ``rel_ceil``] x the 95th percentile so that neither a mostly-silent nor a mostly-speaking video
    is mis-thresholded; a mouth that never moves more than ``min_activity`` counts as silent throughout."""
    x = fill_invalid(inner_height, valid)
    mstd = moving_stats(x, int(round(window_sec * fps)))[1]
    p95 = float(np.percentile(mstd, 95)) if len(mstd) else 0.0
    if p95 <= min_activity:
        active = np.zeros(len(mstd), dtype=bool)
    else:
        threshold = float(np.clip(np.percentile(mstd, percentile), rel_floor * p95, rel_ceil * p95))
        active = mstd > threshold
    return activity_to_segments(active, mstd, fps, len(x) / float(fps), **seg_kw)


# ----------------------------------------------------------------------------------------------------------
# SNR estimation and modality gating
# ----------------------------------------------------------------------------------------------------------
def choose_mode(snr_est: float | None, threshold: float, forced: str = "auto",
                visual_ratio: float | None = None) -> str:
    """Modality for one segment. A forced mode wins; otherwise ``av`` when ``snr_est >= threshold``, else
    ``video``. Without any SNR estimate (no audio) the answer is ``video``. Exception (auto mode only): when the
    lips are visible on fewer than :data:`MIN_VISUAL_RATIO` of the segment's frames and audio exists, lip reading
    is impossible, so ``audio`` is used instead of ``video``."""
    if forced not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {forced!r}")
    if forced != "auto":
        return forced
    if snr_est is None:
        return "video"
    if float(snr_est) >= float(threshold):
        return "av"
    if visual_ratio is not None and visual_ratio < MIN_VISUAL_RATIO:
        return "audio"
    return "video"


def combine_snr(snr_dsp: float | None, snr_model: float | None, estimator: str = "hybrid") -> float | None:
    """Combine the DSP and model SNR estimates per ``infer.snr_estimator`` (dsp | model | hybrid = mean);
    falls back to whichever estimate exists, None when neither does."""
    if estimator not in SNR_ESTIMATORS:
        raise ValueError(f"infer.snr_estimator must be one of {SNR_ESTIMATORS}, got {estimator!r}")
    if estimator == "dsp":
        return snr_dsp if snr_dsp is not None else snr_model
    if estimator == "model":
        return snr_model if snr_model is not None else snr_dsp
    both = [float(v) for v in (snr_dsp, snr_model) if v is not None]
    return float(np.mean(both)) if both else None


def snr_from_logits(snr_logits: torch.Tensor) -> float:
    """SNR-head logits [1, 4] -> midpoint (dB) of the argmax bucket (25, 15, 5, -5)."""
    return SNR_BUCKET_MID[int(snr_logits.detach().float().reshape(-1, len(SNR_BUCKET_MID))[0].argmax().item())]


def choose_vad(wave_float: np.ndarray | None, threshold: float, mode: str = "auto", vad: str = "auto",
               sr: int = 16000, visual_ratio: float | None = None) -> tuple[str, float | None]:
    """Pick the segmenter for the whole file: ``energy`` (audio trusted) or ``visual`` (mouth activity).

    Returns ``(vad_kind, global_snr_db)`` where the global SNR is the DSP estimate over the whole waveform.
    ``vad`` other than ``auto`` forces the choice; ``--mode audio|av`` means the user trusts the audio (energy);
    otherwise the audio is trusted when its SNR reaches ``threshold``. When the lips were found on fewer than
    :data:`MIN_VISUAL_RATIO` of the frames (``visual_ratio``), there is no mouth activity to segment on, so audio
    (if any) is segmented with the energy VAD whatever its SNR.
    """
    if vad not in VAD_KINDS:
        raise ValueError(f"vad must be one of {VAD_KINDS}, got {vad!r}")
    snr: float | None = None
    if wave_float is not None and len(wave_float) > 0:
        snr = float(estimate_snr_db(np.asarray(wave_float, dtype=np.float32), sr))
    if vad == "energy" and snr is None:
        raise ValueError("energy VAD requested but no audio is available")
    if vad != "auto":
        return vad, snr
    if snr is None:
        return "visual", None
    if mode in ("audio", "av"):
        return "energy", snr
    if visual_ratio is not None and visual_ratio < MIN_VISUAL_RATIO:
        return "energy", snr
    return ("energy" if snr >= threshold else "visual"), snr


# ----------------------------------------------------------------------------------------------------------
# Visual stream: frames -> landmarks, skeleton, cues, mouth crops at 30 fps
# ----------------------------------------------------------------------------------------------------------
@dataclass
class FrameFeatures:
    crop: np.ndarray  # [S, S, C] uint8, C = 1 (gray) or 3 (RGB)
    lm: np.ndarray  # [40, 2] float32 (zeros when invalid)
    cue: np.ndarray  # [8] float32 (zeros when invalid)
    valid: int
    has_box: bool  # False while no face has been seen yet (crop from the preprocess-style centre fallback)


@dataclass
class VisualStream:
    """Per-frame visual features of a whole video at ``fps`` (30) frames per second."""

    crops: np.ndarray  # [T, S, S, C] uint8, C = 1 (gray) or 3 (RGB)
    lm: np.ndarray  # [T, 40, 2] float32
    cue: np.ndarray  # [T, 8] float32
    valid: np.ndarray  # [T] uint8
    src_index: np.ndarray  # [T] int64 source frame used for each target frame
    fps: float = 30.0
    src_fps: float = 30.0
    n_src: int = 0

    @property
    def n_frames(self) -> int:
        return int(self.crops.shape[0])


def crop_to_channels(crop_bgr: np.ndarray, channels: int) -> np.ndarray:
    """BGR crop -> [S, S, 1] grayscale (``cv2.COLOR_BGR2GRAY``) or [S, S, 3] RGB."""
    if channels == 1:
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)[..., None]
    if channels == 3:
        return np.ascontiguousarray(crop_bgr[..., ::-1])
    raise ValueError(f"video.channels must be 1 or 3, got {channels}")


class MouthTracker:
    """Per-frame visual features with the same logic as ``avsr.preprocess.process_video``: landmarks -> skeleton
    features + colour cues (zeros and ``valid = 0`` without a face) and ``mouth_box -> BoxSmoother -> crop_square``.

    Differences forced by the absence of labels: landmarks are detected on the full frame downscaled so that
    ``max(H, W) <= detect_max_side`` (points mapped back to full-frame pixels), and a frame without a face keeps the
    last smoothed box (``BoxSmoother`` hold-last) instead of the label lip box.
    """

    def __init__(self, landmarker: Any, *, channels: int = 1, out_size: int = 96, detect_max_side: int = 640) -> None:
        self.landmarker = landmarker
        self.channels = int(channels)
        self.out_size = int(out_size)
        self.detect_max_side = int(detect_max_side)
        self.smoother = vf.BoxSmoother()
        self.first_box: tuple[float, float, float] | None = None
        self._zero_lm = np.zeros((40, 2), dtype=np.float32)
        self._zero_cue = np.zeros(8, dtype=np.float32)

    def __call__(self, frame_bgr: np.ndarray, ts_ms: int) -> FrameFeatures:
        small, scale = vf.downscale_for_detection(frame_bgr, self.detect_max_side)
        det = self.landmarker.detect(small, int(ts_ms))
        if det is not None:
            pts = np.asarray(det, dtype=np.float32) / np.float32(scale)
            lm, geom = vf.skeleton_features(pts)
            col = vf.color_cues(frame_bgr, pts)
            box = self.smoother.update(vf.mouth_box(pts))
            cue, valid = np.concatenate([geom, col]).astype(np.float32), 1
        else:
            box = self.smoother.update(None)
            lm, cue, valid = self._zero_lm, self._zero_cue, 0
        has_box = box is not None
        if box is None:  # no face seen yet: same centre fallback as preprocess (replaced later by a backfill)
            box = self.fallback_box(frame_bgr)
        elif self.first_box is None:
            self.first_box = box
        return FrameFeatures(self.crop(frame_bgr, box), lm, cue, valid, has_box)

    def crop(self, frame_bgr: np.ndarray, box: tuple[float, float, float]) -> np.ndarray:
        return crop_to_channels(vf.crop_square(frame_bgr, box[0], box[1], box[2], self.out_size), self.channels)

    @staticmethod
    def fallback_box(frame_bgr: np.ndarray) -> tuple[float, float, float]:
        h, w = frame_bgr.shape[:2]
        return w / 2.0, h / 2.0, float(min(h, w) / 3)


def _open_capture(video_path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    return cap


def _backfill_leading(video_path: Path, tracker: MouthTracker, crops: list[np.ndarray], src_index: list[int],
                      n_lead: int) -> None:
    """Re-crop the first ``n_lead`` target frames (decoded before any face was found) with the first face box, so
    they show the mouth like every other frame. Only the leading source frames are decoded again."""
    box = tracker.first_box
    if box is None or n_lead <= 0:
        return
    last_src = src_index[n_lead - 1]
    cap = _open_capture(video_path)
    try:
        k = 0
        for i in range(last_src + 1):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if k < n_lead and src_index[k] == i:
                crop = tracker.crop(frame, box)
                while k < n_lead and src_index[k] == i:
                    crops[k] = crop
                    k += 1
    finally:
        cap.release()


def extract_visual_stream(video_path: str | Path, landmarker: Any, *, channels: int = 1, out_size: int = 96,
                          detect_max_side: int = 640, dst_fps: float = 30.0, progress: bool = True) -> VisualStream:
    """Decode a video with ``cv2.VideoCapture``, resample it by presentation time to ``dst_fps`` (nearest source
    frame; variable frame rates handled through ``CAP_PROP_POS_MSEC``) and compute the visual features once per used
    source frame (a duplicated frame shares its features, a dropped frame is not analysed)."""
    video_path = Path(video_path)
    cap = _open_capture(video_path)
    tracker = MouthTracker(landmarker, channels=channels, out_size=out_size, detect_max_side=detect_max_side)
    crops: list[np.ndarray] = []
    lms: list[np.ndarray] = []
    cues: list[np.ndarray] = []
    valids: list[int] = []
    src_index: list[int] = []
    n_lead = 0  # target frames produced before the first face box existed

    def emit(frame: np.ndarray, ts: float, i: int, n_rep: int) -> None:
        nonlocal n_lead
        if n_rep <= 0:
            return
        feats = tracker(frame, int(round(ts * 1000.0)))
        if not feats.has_box and len(crops) == n_lead:
            n_lead += n_rep
        crops.extend([feats.crop] * n_rep)
        lms.extend([feats.lm] * n_rep)
        cues.extend([feats.cue] * n_rep)
        valids.extend([feats.valid] * n_rep)
        src_index.extend([i] * n_rep)
        bar.update(n_rep)

    src_fps = sane_fps(cap.get(cv2.CAP_PROP_FPS), dst_fps)
    n_est = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), 0)  # some backends report -1 (unknown)
    resampler = TimeResampler(dst_fps, 1.0 / src_fps)
    bar = tqdm(total=int(round(n_est * dst_fps / src_fps)) or None, unit="fr", desc="landmarks",
               disable=not progress, leave=False)
    n_src = 0
    try:
        prev: np.ndarray | None = None
        prev_ts = 0.0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            n_rep = resampler.push(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            if prev is not None:
                emit(prev, prev_ts, n_src - 1, n_rep)
            prev, prev_ts = frame, float(resampler.last_ts)
            n_src += 1
        if prev is not None:
            emit(prev, prev_ts, n_src - 1, resampler.finish())
    finally:
        cap.release()
        bar.close()
    if not crops:
        raise RuntimeError(f"no frames decoded from {video_path}")
    if resampler.n_fixed:
        log.warning("%d frame timestamps were missing or non-increasing and were interpolated", resampler.n_fixed)
    if n_lead and tracker.first_box is not None:
        _backfill_leading(video_path, tracker, crops, src_index, n_lead)
    stream = VisualStream(
        crops=np.stack(crops),
        # same float16 precision as the preprocessed npz features the model was trained on
        lm=np.stack(lms).astype(np.float16).astype(np.float32),
        cue=np.stack(cues).astype(np.float16).astype(np.float32),
        valid=np.asarray(valids, dtype=np.uint8),
        src_index=np.asarray(src_index, dtype=np.int64),
        fps=float(dst_fps), src_fps=src_fps, n_src=n_src)
    ratio = float(stream.valid.mean())
    log.info("video: %d source frames @ %.3f fps -> %d frames @ %g fps; lips found on %.1f%% of frames",
             n_src, src_fps, stream.n_frames, dst_fps, 100.0 * ratio)
    if ratio < 0.5:
        log.warning("lips were found on only %.1f%% of the frames: the visual stream is unreliable", 100.0 * ratio)
    return stream


# ----------------------------------------------------------------------------------------------------------
# Model inputs per segment
# ----------------------------------------------------------------------------------------------------------
def build_segment_batch(visual: VisualStream | None, wave_int16: np.ndarray | None, start: float, end: float,
                        cfg: Any, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Model input batch (B = 1) for ``[start, end)`` built like an eval item of ``avsr.dataset.AVSRDataset``:
    30 fps frames ``[round(start*30), round(end*30))`` resampled to 25 fps with ``avsr.dataset.resample_indices``,
    centre crop, pixels/cues normalised per ``video.norm`` / ``video.cue_norm`` (shared helpers of avsr.dataset); audio samples ``[round(start*sr), round(end*sr))`` -> fbank -> stacked
    frames; audio and video trimmed to the same length."""
    fps_in = float(cfg_get(cfg, "video.fps_in", 30.0))
    fps_out = float(cfg_get(cfg, "video.fps_out", 25.0))
    size = int(cfg_get(cfg, "video.size", 96))
    crop = int(cfg_get(cfg, "video.crop", 88))
    channels = int(cfg_get(cfg, "video.channels", 1))
    mean = float(cfg_get(cfg, "video.mean", 0.421))
    std = float(cfg_get(cfg, "video.std", 0.165))
    sr = int(cfg_get(cfg, "audio.sr", 16000))
    stack = int(cfg_get(cfg, "audio.stack", 4))
    n_mels = int(cfg_get(cfg, "audio.n_mels", 80))

    f0 = int(round(start * fps_in))
    n_in = max(int(round(end * fps_in)) - f0, 1)
    idx = resample_indices(n_in, fps_in, fps_out) + f0
    n_out = len(idx)
    frames = np.zeros((n_out, size, size, channels), dtype=np.uint8)
    lm = np.zeros((n_out, 80), dtype=np.float32)
    cue = np.zeros((n_out, 8), dtype=np.float32)
    valid = np.zeros(n_out, dtype=np.float32)
    if visual is not None and visual.n_frames > 0:
        if visual.crops.shape[1:] != (size, size, channels):
            raise ValueError(f"visual crops {visual.crops.shape[1:]} do not match the model input "
                             f"({size}, {size}, {channels})")
        ok = idx < visual.n_frames
        src = idx[ok]
        frames[ok] = visual.crops[src]
        lm[ok] = visual.lm[src].reshape(len(src), -1)
        cue[ok] = visual.cue[src]
        valid[ok] = visual.valid[src]
    off = (size - crop) // 2
    pix = frames[:, off:off + crop, off:off + crop].transpose(0, 3, 1, 2)  # [T, C, crop, crop]
    # identical to avsr.dataset: global constants or per-utterance (per-segment) z-score, cue normalisation
    video = normalize_pixels(pix, str(cfg_get(cfg, "video.norm", "global")), mean, std)
    cue = normalize_cue(cue, valid, str(cfg_get(cfg, "video.cue_norm", "none")))

    if wave_int16 is not None and len(wave_int16) > 0:
        s0, s1 = max(int(round(start * sr)), 0), max(int(round(end * sr)), 0)
        piece = np.asarray(wave_int16[s0:s1], dtype=np.int16)
        want = max(s1 - s0, int(0.1 * sr))  # >= 100 ms so that the fbank always has frames
        if len(piece) < want:
            piece = np.pad(piece, (0, want - len(piece)))
        audio = stack_frames(compute_fbank(piece, sr), stack).float()
    else:
        audio = torch.zeros(n_out, stack * n_mels, dtype=torch.float32)
    t = max(min(n_out, int(audio.shape[0])), 1)
    if audio.shape[0] < t:
        audio = torch.cat([audio, torch.zeros(t - audio.shape[0], audio.shape[1])], dim=0)
    batch = {
        "video": torch.from_numpy(np.ascontiguousarray(video[:t]))[None],
        "lm": torch.from_numpy(lm[:t])[None],
        "cue": torch.from_numpy(cue[:t])[None],
        "valid": torch.from_numpy(valid[:t])[None],
        "audio": audio[:t][None],
        "lengths": torch.tensor([t], dtype=torch.long),
    }
    dev = torch.device(device)
    return {key: val.to(dev) for key, val in batch.items()}


# ----------------------------------------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------------------------------------
@dataclass
class ModelBundle:
    model: torch.nn.Module
    cfg: Config
    tokenizer: Tokenizer
    device: torch.device


def autocast_context(device: torch.device | str) -> contextlib.AbstractContextManager:
    """bf16 autocast on CUDA, a no-op elsewhere."""
    if torch.device(device).type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_model_bundle(ckpt_path: str | Path, cli_cfg: Mapping[str, Any] | None = None, overrides: Sequence[str] = (),
                      device: torch.device | str | None = None) -> ModelBundle:
    """``torch.load`` a checkpoint written by ``avsr.train``, build the run config (:func:`build_run_config`) and the
    model (``build_model(cfg, vocab_size)``), load the weights and switch to eval mode."""
    from avsr.models import build_model

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, Mapping) or "model" not in ckpt:
        raise KeyError(f"{ckpt_path} is not an avsr checkpoint (no 'model' weights, SPEC section 11)")
    if not ckpt.get("cfg"):
        log.warning("checkpoint %s carries no config: the CLI config must describe the model", ckpt_path)
    cfg = build_run_config(ckpt.get("cfg"), cli_cfg, overrides)
    tokenizer = Tokenizer()
    vocab = ckpt.get("vocab")
    if vocab is not None and list(vocab) != list(tokenizer.tokens):
        raise ValueError(f"{ckpt_path}: tokenizer vocabulary differs from avsr.text.Tokenizer")
    model = build_model(cfg, tokenizer.vocab_size)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    log.info("model %s: %.1fM params, epoch=%s, step=%s, device=%s", ckpt_path,
             sum(p.numel() for p in model.parameters()) / 1e6, ckpt.get("epoch", "?"), ckpt.get("step", "?"), dev)
    return ModelBundle(model=model, cfg=cfg, tokenizer=tokenizer, device=dev)


# ----------------------------------------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------------------------------------
def segment_timeline(wave_float: np.ndarray | None, visual: VisualStream | None, cfg: Any, vad: str,
                     sr: int = 16000) -> list[Segment]:
    """Segment the timeline with the chosen VAD (``energy`` on the float waveform, ``visual`` on inner_height)."""
    kw = dict(min_sec=float(cfg_get(cfg, "infer.segment_min_sec", 1.0)),
              max_sec=float(cfg_get(cfg, "infer.segment_max_sec", 8.0)),
              pad_sec=float(cfg_get(cfg, "infer.chunk_pad_sec", 0.2)))
    if not 0.0 < kw["min_sec"] <= kw["max_sec"] or kw["pad_sec"] < 0.0:
        raise ValueError(f"need 0 < infer.segment_min_sec <= infer.segment_max_sec and infer.chunk_pad_sec >= 0, "
                         f"got {kw}")
    if vad == "energy":
        if wave_float is None:
            raise ValueError("energy VAD requested without audio")
        spans = energy_vad_segments(wave_float, sr, **kw)
    elif vad == "visual":
        if visual is None:
            raise ValueError("visual VAD requested without a visual stream")
        spans = visual_vad_segments(visual.cue[:, 1], visual.valid, visual.fps, **kw)
    else:
        raise ValueError(f"unknown vad {vad!r}")
    return [Segment(index=i + 1, start=float(s), end=float(e)) for i, (s, e) in enumerate(spans)]


def _segment_visual_ratio(visual: VisualStream | None, start: float, end: float) -> float | None:
    """Fraction of the segment's frames with detected lips (None without a visual stream)."""
    if visual is None or visual.n_frames == 0:
        return None
    f0 = min(max(int(round(start * visual.fps)), 0), visual.n_frames)
    f1 = min(max(int(round(end * visual.fps)), f0), visual.n_frames)
    return float(visual.valid[f0:f1].mean()) if f1 > f0 else 0.0


def decode_segments(model: torch.nn.Module, tokenizer: Tokenizer, cfg: Any, visual: VisualStream | None,
                    wave_int16: np.ndarray | None, segments: list[Segment], *, mode: str = "auto",
                    snr_threshold: float | None = None, device: torch.device | str = "cpu",
                    progress: bool = True) -> list[Segment]:
    """Per segment: estimate the SNR (``infer.snr_estimator``), choose the modality (:func:`choose_mode`) and decode
    the text with ``model.decode`` (method ``eval.decode``). Fills ``text``, ``mode``, ``snr_*`` and
    ``visual_ratio`` of every segment in place."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    threshold = float(cfg_get(cfg, "infer.snr_threshold", 5.0) if snr_threshold is None else snr_threshold)
    estimator = str(cfg_get(cfg, "infer.snr_estimator", "hybrid"))
    if estimator not in SNR_ESTIMATORS:
        raise ValueError(f"infer.snr_estimator must be one of {SNR_ESTIMATORS}, got {estimator!r}")
    method = str(cfg_get(cfg, "eval.decode", "ctc_greedy"))
    sr = int(cfg_get(cfg, "audio.sr", 16000))
    have_audio = wave_int16 is not None and len(wave_int16) > 0
    if not have_audio and mode in ("audio", "av"):
        log.warning("mode %s requested but no audio is available: using video", mode)
        mode = "video"
    wave_f = wave_int16.astype(np.float32) / 32768.0 if have_audio else None
    dev = torch.device(device)
    for seg in tqdm(segments, desc="decode", unit="seg", disable=not progress, leave=False):
        batch = build_segment_batch(visual, wave_int16, seg.start, seg.end, cfg, dev)
        seg.visual_ratio = _segment_visual_ratio(visual, seg.start, seg.end)
        with torch.inference_mode(), autocast_context(dev):
            if wave_f is not None:
                piece = wave_f[int(round(seg.start * sr)):int(round(seg.end * sr))]
                seg.snr_dsp = float(estimate_snr_db(piece, sr)) if len(piece) >= int(0.1 * sr) else None
                if estimator != "dsp":  # the SNR head only looks at the audio: skip the visual frontends
                    seg.snr_model = snr_from_logits(model(batch, mode="audio")["snr_logits"])
            seg.snr_est = combine_snr(seg.snr_dsp, seg.snr_model, estimator)
            seg.mode = choose_mode(seg.snr_est, threshold, mode, seg.visual_ratio)
            length = int(batch["lengths"][0].item())
            hyps = model.decode(batch, seg.mode, method=method, max_len=max(32, 2 * length))
        seg.text = tokenizer.decode([int(t) for t in hyps[0]]).strip() if hyps else ""
    return segments


@dataclass
class TranscriptResult:
    segments: list[Segment]
    meta: dict


def load_wave(source: Path, sr: int = 16000, required: bool = False) -> np.ndarray | None:
    """16 kHz mono int16 audio of ``source`` (``avsr.audio_feats.load_audio_16k``); None when the file has no usable
    audio (no track, undecodable, digital silence) - lip reading still works then. With ``required`` (an explicit
    ``--audio`` file) an undecodable file raises instead. A missing ffmpeg always raises (FileNotFoundError): that is
    a broken installation, not a video without sound."""
    try:
        wave = np.asarray(load_audio_16k(str(source)), dtype=np.int16)
    except RuntimeError as exc:  # ffmpeg ran but found no decodable audio stream
        if required:
            raise
        log.warning("no audio from %s (%s): continuing with video only", source, " ".join(str(exc).split())[:300])
        return None
    if len(wave) < int(0.1 * sr) or int(np.abs(wave.astype(np.int32)).max()) < 4:
        log.warning("audio of %s is empty or silent: continuing with video only", source)
        return None
    return wave


def run_pipeline(bundle: ModelBundle, video_path: str | Path, *, audio_path: str | Path | None = None,
                 mode: str = "auto", snr_threshold: float | None = None, vad: str = "auto", landmarker: Any = None,
                 landmarker_path: str | Path | None = None, progress: bool = True) -> TranscriptResult:
    """Full inference for one video with an already loaded :class:`ModelBundle`. ``landmarker`` may be any object
    with ``detect(frame_bgr, timestamp_ms) -> [478, 2] | None`` (default: a new MediaPipe ``FaceLandmarker``)."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    cfg = bundle.cfg
    threshold = float(cfg_get(cfg, "infer.snr_threshold", 5.0) if snr_threshold is None else snr_threshold)
    sr = int(cfg_get(cfg, "audio.sr", 16000))
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video not found: {video_path}")
    source = Path(audio_path) if audio_path is not None else video_path
    if not source.is_file():
        raise FileNotFoundError(f"audio not found: {source}")
    t0 = time.time()
    wave = load_wave(source, sr, required=audio_path is not None)
    log.info("audio: %s", "none" if wave is None else f"{len(wave) / sr:.1f} s from {source.name}")

    own_landmarker = landmarker is None
    if own_landmarker:
        landmarker = vf.FaceLandmarker(model_path=str(landmarker_path or DEFAULT_LANDMARKER), running_mode="video")
    try:
        visual = extract_visual_stream(video_path, landmarker, channels=int(cfg_get(cfg, "video.channels", 1)),
                                       out_size=int(cfg_get(cfg, "video.size", 96)),
                                       dst_fps=float(cfg_get(cfg, "video.fps_in", 30.0)), progress=progress)
    finally:
        if own_landmarker:
            landmarker.close()
    t_visual = time.time()

    wave_f = wave.astype(np.float32) / 32768.0 if wave is not None else None
    vad_kind, global_snr = choose_vad(wave_f, threshold, mode, vad, sr, visual_ratio=float(visual.valid.mean()))
    log.info("global SNR %s dB, threshold %.1f dB -> %s VAD (mode=%s)",
             "n/a" if global_snr is None else f"{global_snr:.1f}", threshold, vad_kind, mode)
    segments = segment_timeline(wave_f, visual, cfg, vad_kind, sr)
    log.info("%d segments", len(segments))
    decode_segments(bundle.model, bundle.tokenizer, cfg, visual, wave, segments, mode=mode,
                    snr_threshold=threshold, device=bundle.device, progress=progress)
    meta = {
        "video": str(video_path), "audio": None if wave is None else str(source), "mode": mode,
        "snr_threshold": threshold, "snr_estimator": str(cfg_get(cfg, "infer.snr_estimator", "hybrid")),
        "decode": str(cfg_get(cfg, "eval.decode", "ctc_greedy")), "vad": vad_kind,
        "global_snr_db": None if global_snr is None else round(global_snr, 3),
        "video_src_fps": round(visual.src_fps, 3), "video_frames_30fps": visual.n_frames,
        "landmark_valid_ratio": round(float(visual.valid.mean()), 4), "n_segments": len(segments),
        "modes_used": {m: sum(s.mode == m for s in segments) for m in ("av", "audio", "video")},
        "time_visual_sec": round(t_visual - t0, 2), "time_total_sec": round(time.time() - t0, 2),
        "device": str(bundle.device),
    }
    return TranscriptResult(segments=segments, meta=meta)


def transcribe_video(video_path: str | Path, ckpt_path: str | Path | None = None,
                     cfg: str | Path | Mapping[str, Any] | None = None, *, overrides: Sequence[str] = (),
                     audio_path: str | Path | None = None, mode: str = "auto", snr_threshold: float | None = None,
                     vad: str = "auto", device: torch.device | str | None = None,
                     landmarker_path: str | Path | None = None, model_bundle: ModelBundle | None = None,
                     progress: bool = False) -> list[Segment]:
    """Transcribe one video and return its :class:`Segment` list (reusable from Python).

    ``cfg`` is the run-time config (a YAML path or a mapping such as ``avsr.utils.load_config(...)``; None = the
    checkpoint's own config) and ``overrides`` are ``key.sub=value`` strings; both are combined with the checkpoint's
    config by :func:`build_run_config`. Pass ``model_bundle`` (from :func:`load_model_bundle`) to reuse one loaded
    model for many videos; ``ckpt_path``, ``cfg``, ``overrides`` and ``device`` are then unused.
    """
    if model_bundle is None:
        if ckpt_path is None:
            raise ValueError("either ckpt_path or model_bundle is required")
        cli_cfg = load_config(cfg) if isinstance(cfg, (str, Path)) else cfg
        model_bundle = load_model_bundle(ckpt_path, cli_cfg, overrides, device)
    result = run_pipeline(model_bundle, video_path, audio_path=audio_path, mode=mode, snr_threshold=snr_threshold,
                          vad=vad, landmarker_path=landmarker_path, progress=progress)
    return result.segments


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m avsr.infer",
                                description="Korean AVSR inference: video -> .srt/.json subtitles with SNR-gated "
                                            "lip reading")
    p.add_argument("--config", default=str(DEFAULT_CONFIG) if DEFAULT_CONFIG.is_file() else None,
                   help="YAML config merged over the checkpoint config (model/video/audio always come from the "
                        "checkpoint); default: configs/base.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY.SUB=VALUE",
                   help="config override, repeatable (e.g. --set infer.snr_estimator=dsp)")
    p.add_argument("--ckpt", required=True, help="checkpoint, e.g. work/checkpoints/best.pt")
    p.add_argument("--video", required=True, help="input video (any format OpenCV/ffmpeg can read)")
    p.add_argument("--audio", default=None, help="take the audio from this file instead of the video's track")
    p.add_argument("--out", default=None, help="output .srt path or directory (default: next to the video); "
                                               "the .json sidecar is written next to the .srt")
    p.add_argument("--mode", default="auto", choices=MODES,
                   help="auto = av when the estimated SNR >= threshold, else video; or force a modality")
    p.add_argument("--snr-threshold", type=float, default=None, help="override infer.snr_threshold (dB)")
    p.add_argument("--vad", default="auto", choices=VAD_KINDS,
                   help="segmenter: auto (energy when the audio is trusted, else visual), energy or visual")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
    p.add_argument("--landmarker", default=None, help="MediaPipe face_landmarker.task (default: assets/)")
    p.add_argument("--no-progress", action="store_true", help="disable progress bars")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    safe_console()
    os.environ.setdefault("GLOG_minloglevel", "2")  # quieter MediaPipe
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)  # stdout like train/evaluate
    srt_path, json_path = output_paths(args.video, args.out)
    prepare_output(srt_path)
    cli_cfg = load_config(args.config) if args.config else None
    bundle = load_model_bundle(args.ckpt, cli_cfg, args.set, args.device)
    result = run_pipeline(bundle, args.video, audio_path=args.audio, mode=args.mode,
                          snr_threshold=args.snr_threshold, vad=args.vad, landmarker_path=args.landmarker,
                          progress=not args.no_progress)
    n_cues = write_srt(srt_path, result.segments)
    write_json(json_path, result.segments, {"ckpt": str(args.ckpt), **result.meta})
    meta = result.meta
    print(format_summary(result.segments))
    print(f"\n{len(result.segments)} segments ({n_cues} with text) | VAD={meta['vad']} | "
          f"global SNR={meta['global_snr_db']} dB | threshold={meta['snr_threshold']} dB | "
          f"modes={meta['modes_used']} | {meta['time_total_sec']} s")
    print(f"wrote {srt_path}\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

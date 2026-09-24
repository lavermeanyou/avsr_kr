"""Audio features: ffmpeg loading, kaldi fbank + CMVN, frame stacking, noise augmentation, DSP SNR estimate.

See docs/SPEC.md section 7 for the API contract.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchaudio.compliance.kaldi as kaldi
from scipy.fft import next_fast_len

SR = 16000
INT16_SCALE = 32768.0
NOISE_KINDS = ("white", "pink", "babble")

_FFMPEG_CACHE: str | None = None


def _registry_path_entries() -> list[str]:
    """PATH entries from the Windows registry (Machine + User), for shells whose env PATH is stale."""
    if sys.platform != "win32":
        return []
    entries: list[str] = []
    try:
        import winreg  # noqa: WPS433 (Windows only)

        for root, sub in (
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, r"Environment"),
        ):
            try:
                with winreg.OpenKey(root, sub) as key:
                    value, _ = winreg.QueryValueEx(key, "Path")
                entries.extend(os.path.expandvars(p) for p in str(value).split(";") if p)
            except OSError:
                continue
    except ImportError:  # pragma: no cover
        return []
    return entries


def find_ffmpeg() -> str:
    """Locate the ffmpeg executable (env FFMPEG_BINARY, PATH, then the registry PATH on Windows)."""
    global _FFMPEG_CACHE
    if _FFMPEG_CACHE and Path(_FFMPEG_CACHE).exists():
        return _FFMPEG_CACHE
    candidates: list[str | None] = [os.environ.get("FFMPEG_BINARY"), shutil.which("ffmpeg")]
    for entry in _registry_path_entries():
        candidates.append(str(Path(entry) / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")))
    for cand in candidates:
        if cand and Path(cand).is_file():
            _FFMPEG_CACHE = str(Path(cand))
            return _FFMPEG_CACHE
    raise FileNotFoundError("ffmpeg executable not found (set FFMPEG_BINARY or add ffmpeg to PATH)")


def load_audio_16k(path_or_video: str | os.PathLike, start: float | None = None, end: float | None = None) -> np.ndarray:
    """Decode any media file to mono 16 kHz int16 PCM through an ffmpeg pipe.

    ``start``/``end`` (seconds) are placed after ``-i`` so the cut is sample-accurate (output-side trim).
    """
    path = Path(path_or_video)
    if not path.is_file():
        raise FileNotFoundError(f"media file not found: {path}")
    cmd = [find_ffmpeg(), "-v", "error", "-nostdin", "-i", str(path)]
    if start is not None:
        cmd += ["-ss", f"{float(start):.6f}"]
    if end is not None:
        cmd += ["-to", f"{float(end):.6f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}) on {path}: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.int16).copy()


def to_float_wave(wave: np.ndarray | torch.Tensor) -> np.ndarray:
    """int16 → float32 in [-1, 1]; float input is returned as float32 unchanged."""
    if isinstance(wave, torch.Tensor):
        wave = wave.detach().cpu().numpy()
    wave = np.asarray(wave)
    if np.issubdtype(wave.dtype, np.integer):
        return (wave.astype(np.float32) / INT16_SCALE).reshape(-1)
    return wave.astype(np.float32, copy=False).reshape(-1)


def compute_fbank(wave: np.ndarray | torch.Tensor, sr: int = SR, n_mels: int = 80, eps: float = 1e-5) -> torch.Tensor:
    """80-dim kaldi log-mel filterbank (25 ms / 10 ms, no dither) with per-utterance CMVN over time → [n_frames, n_mels].

    Int16 input is used at its native scale; float input is assumed in [-1, 1] and scaled by 32768 as kaldi expects.
    """
    if isinstance(wave, torch.Tensor):
        x = wave.detach().cpu().reshape(-1)
        x = x.to(torch.float32) if not x.is_floating_point() else x.to(torch.float32) * INT16_SCALE
    else:
        wave = np.asarray(wave).reshape(-1)
        if np.issubdtype(wave.dtype, np.integer):
            x = torch.from_numpy(wave.astype(np.float32))
        else:
            x = torch.from_numpy(wave.astype(np.float32)) * INT16_SCALE
    win = int(round(0.025 * sr))
    if x.numel() < win:  # kaldi returns an empty tensor for signals shorter than one window
        return torch.zeros(0, n_mels, dtype=torch.float32)
    fb = kaldi.fbank(
        x.unsqueeze(0),
        num_mel_bins=n_mels,
        frame_length=25.0,
        frame_shift=10.0,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=float(sr),
    )
    if fb.shape[0] == 0:
        return torch.zeros(0, n_mels, dtype=torch.float32)
    mean = fb.mean(dim=0, keepdim=True)
    var = fb.var(dim=0, unbiased=False, keepdim=True)
    return ((fb - mean) / torch.sqrt(var + eps)).to(torch.float32)


def stack_frames(fb: torch.Tensor, stack: int = 4) -> torch.Tensor:
    """Concatenate ``stack`` consecutive frames: [n, d] → [n // stack, d * stack] (remainder dropped)."""
    n = fb.shape[0] // stack
    return fb[: n * stack].reshape(n, stack * fb.shape[1])


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def _fit_length(x: np.ndarray, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Tile (then crop) ``x`` to exactly ``n`` samples; a random circular offset is used when ``rng`` is given."""
    if x.size == 0:
        raise ValueError("cannot fit an empty signal")
    reps = int(np.ceil(n / x.size)) + 1
    tiled = np.tile(x, reps)
    offset = int(rng.integers(0, x.size)) if rng is not None else 0
    return tiled[offset : offset + n]


def add_noise(wave_float: np.ndarray, noise_float: np.ndarray, snr_db: float) -> tuple[np.ndarray, float]:
    """Mix ``noise`` into ``wave`` at exactly ``snr_db`` (RMS over the whole utterance). Returns (noisy float32, snr)."""
    wave = to_float_wave(wave_float)
    if wave.size == 0:
        return wave.astype(np.float32), float(snr_db)
    noise = to_float_wave(noise_float)
    noise = _fit_length(noise, wave.size)
    rms_s = max(_rms(wave), 1e-8)
    rms_n = _rms(noise)
    if rms_n < 1e-12:
        raise ValueError("noise signal is silent")
    scale = rms_s / (rms_n * 10.0 ** (float(snr_db) / 20.0))
    noisy = (wave + scale * noise).astype(np.float32)
    return noisy, float(snr_db)


def _pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """1/f-power noise: white noise whose amplitude spectrum is divided by sqrt(f) (DC removed).

    The FFT runs on a fast (5-smooth) length ≥ n and is cropped: arbitrary utterance lengths can be prime, which
    makes a direct FFT an order of magnitude slower.
    """
    if n < 3:  # too short to carry any non-DC shape
        return rng.standard_normal(n)
    m = next_fast_len(n, real=True)
    spec = np.fft.rfft(rng.standard_normal(m))
    freqs = np.arange(spec.size, dtype=np.float64)
    freqs[0] = 1.0  # avoid division by zero; the DC bin is zeroed below
    spec /= np.sqrt(freqs)
    spec[0] = 0.0
    return np.fft.irfft(spec, n=m)[:n]


def make_noise(
    kind: str,
    n_samples: int,
    rng: np.random.Generator,
    babble_pool: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Unit-RMS float32 noise of ``kind`` ('white' | 'pink' | 'babble') and length ``n_samples``.

    Babble = sum of the pool clips (int16 or float), each tiled/cropped to length with a random offset.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if kind == "white":
        noise = rng.standard_normal(n_samples)
    elif kind == "pink":
        noise = _pink_noise(n_samples, rng)
    elif kind == "babble":
        if not babble_pool:
            raise ValueError("babble noise requires a non-empty babble_pool")
        noise = np.zeros(n_samples, dtype=np.float64)
        for clip in babble_pool:
            c = to_float_wave(clip)
            if c.size == 0:
                continue
            c = _fit_length(c, n_samples, rng).astype(np.float64)
            noise += c / max(_rms(c), 1e-8)
        if _rms(noise) < 1e-8:  # every pool clip silent → fall back to white so SNR stays defined
            noise = rng.standard_normal(n_samples)
    else:
        raise ValueError(f"unknown noise kind {kind!r}; expected one of {NOISE_KINDS}")
    noise = noise / max(_rms(noise), 1e-8)
    return noise.astype(np.float32)


def _frame_energies_db(wave: np.ndarray, sr: int, win_sec: float = 0.025, hop_sec: float = 0.010) -> np.ndarray:
    """Per-frame mean-square energy in dB (25 ms window / 10 ms hop), padding signals shorter than one window.

    Uses a cumulative sum of squares, so memory stays O(N) even for whole-file (minutes long) signals.
    """
    win = max(1, int(round(win_sec * sr)))
    hop = max(1, int(round(hop_sec * sr)))
    if wave.size < win:
        wave = np.pad(wave, (0, win - wave.size))
    n_frames = 1 + (wave.size - win) // hop
    csum = np.concatenate([[0.0], np.cumsum(np.square(wave, dtype=np.float64))])
    starts = hop * np.arange(n_frames)
    energy = np.maximum(csum[starts + win] - csum[starts], 0.0) / win
    return 10.0 * np.log10(energy + 1e-10)


def estimate_snr_db(wave_float: np.ndarray, sr: int = SR) -> float:
    """DSP SNR estimate: 10th-percentile frame energy = noise floor, mean of frames above the 70th percentile = speech.

    Returns speech − noise in dB, clipped to [-10, 40]. Int16 input is scaled to [-1, 1] first.
    """
    wave = to_float_wave(wave_float)
    if wave.size == 0:
        return -10.0
    db = _frame_energies_db(wave, sr)
    noise_floor = float(np.percentile(db, 10))
    thr = float(np.percentile(db, 70))
    above = db[db > thr]
    speech = float(above.mean()) if above.size else float(db.max())
    return float(np.clip(speech - noise_floor, -10.0, 40.0))

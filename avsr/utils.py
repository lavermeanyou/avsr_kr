"""Shared helpers: config loading/overrides, seeding, logging, checkpoint io, metrics (SPEC sections 3, 9, 11)."""
from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import jiwer
import numpy as np
import torch
import yaml

# --------------------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------------------


class Config(dict):
    """Nested dict with attribute access: ``cfg.model.d_model``, ``cfg["model"]["d_model"]`` and ``cfg.get`` all work.

    Picklable (DataLoader workers on Windows receive it through spawn).
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, _wrap(value))

    def __deepcopy__(self, memo: dict) -> "Config":
        return Config.from_dict(to_dict(self))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Config":
        cfg = cls()
        for k, v in d.items():
            cfg[str(k)] = v
        return cfg


def _wrap(value: Any) -> Any:
    """Convert nested mappings (also inside lists) to :class:`Config`."""
    if isinstance(value, Config):
        return value
    if isinstance(value, Mapping):
        return Config.from_dict(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_wrap(v) for v in value)
    return value


def to_dict(cfg: Any) -> Any:
    """Recursively convert a :class:`Config` (or any mapping) into plain dicts/lists (for yaml/json/checkpoints)."""
    if isinstance(cfg, Mapping):
        return {str(k): to_dict(v) for k, v in cfg.items()}
    if isinstance(cfg, (list, tuple)):
        return [to_dict(v) for v in cfg]
    return cfg


_NUMBER_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def parse_value(text: str) -> Any:
    """Infer the type of a CLI override value: int/float/bool/null/list/dict via YAML, plus ``5e-4`` style floats."""
    text = text.strip()
    try:
        value = yaml.safe_load(text) if text else ""
    except yaml.YAMLError:
        return text
    if isinstance(value, str) and _NUMBER_RE.match(value):
        return float(value)
    return value


def _get_path(d: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def apply_override(cfg: Config, override: str) -> str:
    """Apply one ``a.b.c=value`` override in place and return the dotted key."""
    if "=" not in override:
        raise ValueError(f"override must look like 'a.b=value', got {override!r}")
    key, raw = override.split("=", 1)
    key = key.strip()
    if not key or any(not p for p in key.split(".")):
        raise ValueError(f"bad override key in {override!r}")
    parts = key.split(".")
    node: Config = cfg
    for i, part in enumerate(parts[:-1]):
        if part not in node:
            node[part] = Config()
        elif not isinstance(node[part], Mapping):
            raise ValueError(f"cannot set {key!r}: {'.'.join(parts[:i + 1])!r} is not a section")
        node = node[part]
    if parts[-1] not in node:
        warnings.warn(f"config override creates new key {key!r} (not in the config file)", stacklevel=3)
    node[parts[-1]] = parse_value(raw)
    return key


def merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursive merge: values of ``override`` win; nested mappings are merged."""
    out = to_dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = merge_dicts(out[k], v)
        else:
            out[k] = to_dict(v)
    return out


def load_config(path: str | Path | None, overrides: Sequence[str] | None = None,
                base: Mapping[str, Any] | None = None) -> Config:
    """Load a YAML config (optionally merged over ``base``) and apply ``--set a.b=c`` overrides.

    When ``work_dir`` is overridden but ``train.ckpt_dir`` is not, a relative ``ckpt_dir`` that lives under the
    original work_dir (default ``work/checkpoints``) is re-based onto the new work_dir so a run never writes its
    checkpoints into another run's directory.
    """
    data: dict = to_dict(base) if base is not None else {}
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"config {path} must contain a mapping at top level")
        data = merge_dicts(data, loaded)
    cfg = Config.from_dict(data)
    old_work_dir = cfg.get("work_dir")
    touched = {apply_override(cfg, o) for o in (overrides or [])}
    if "work_dir" in touched and "train.ckpt_dir" not in touched and old_work_dir is not None:
        try:
            ckpt_dir = Path(str(_get_path(cfg, "train.ckpt_dir")))
        except KeyError:
            ckpt_dir = None
        if ckpt_dir is not None and not ckpt_dir.is_absolute():
            try:
                rel = ckpt_dir.relative_to(Path(str(old_work_dir)))
            except ValueError:
                rel = None
            if rel is not None:
                cfg.train.ckpt_dir = str(Path(str(cfg.work_dir)) / rel)
    return cfg


# --------------------------------------------------------------------------------------------------------------
# Reproducibility / console
# --------------------------------------------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_console() -> None:
    """Make stdout/stderr never crash on characters the console code page cannot encode."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


class _ConsoleHandler(logging.Handler):
    """Console handler that cooperates with tqdm progress bars and flushes every record."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm

            tqdm.write(self.format(record), file=sys.stdout)
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - logging must never raise
            self.handleError(record)


_PACKAGE_LOGGER = "avsr"


def get_logger(work_dir: str | Path, name: str = "train") -> logging.Logger:
    """Return logger ``avsr.<name>``; the package logger ``avsr`` (so also ``avsr.dataset``, ``avsr.models``...) is
    routed to the console and to ``<work_dir>/logs/<name>.log`` (utf-8, appended). Replaces previous handlers."""
    log_dir = Path(work_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    close_logger()
    pkg = logging.getLogger(_PACKAGE_LOGGER)
    pkg.setLevel(logging.INFO)
    pkg.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = _ConsoleHandler()
    ch.setFormatter(fmt)
    pkg.addHandler(fh)
    pkg.addHandler(ch)
    return logging.getLogger(f"{_PACKAGE_LOGGER}.{name}")


def close_logger() -> None:
    """Detach and close the handlers installed by :func:`get_logger` (releases the log file on Windows)."""
    pkg = logging.getLogger(_PACKAGE_LOGGER)
    for h in list(pkg.handlers):
        pkg.removeHandler(h)
        h.close()


# --------------------------------------------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------------------------------------------


def save_checkpoint(path: str | Path, state: Mapping[str, Any]) -> Path:
    """Atomically write a checkpoint (tmp file + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(dict(state), tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load a checkpoint written by :func:`save_checkpoint` (trusted local file: full unpickling)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------------------


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Corpus character error rate, spaces removed (SPEC section 11). NaN when there is no reference text."""
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references vs {len(hyps)} hypotheses")
    r = [s.replace(" ", "") for s in refs]
    h = [s.replace(" ", "") for s in hyps]
    if sum(len(s) for s in r) == 0:
        return float("nan")
    return float(jiwer.cer(r, h))


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    """Corpus word error rate on space-split words. NaN when there is no reference word."""
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references vs {len(hyps)} hypotheses")
    r = [" ".join(s.split()) for s in refs]
    h = [" ".join(s.split()) for s in hyps]
    if sum(len(s.split()) for s in r) == 0:
        return float("nan")
    return float(jiwer.wer(r, h))


class AverageMeter:
    """Running (weighted) average."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0.0

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0.0

    def update(self, value: float, n: float = 1.0) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else float("nan")


class MetricsWriter:
    """Append-only JSONL metrics file (one JSON object per line, utf-8, flushed per record)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Mapping[str, Any]) -> None:
        rec = {"time": round(time.time(), 3), **{k: _jsonable(v) for k, v in record.items()}}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, Mapping):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file (blank lines ignored)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def even_subset(rows: Sequence[dict], n: int | None, key: str = "utt_id") -> list[dict]:
    """Deterministic, evenly spread subset of at most ``n`` rows (sorted by ``key`` first)."""
    ordered = sorted(rows, key=lambda r: str(r.get(key, "")))
    if n is None or n <= 0 or n >= len(ordered):
        return ordered
    step = len(ordered) / n
    return [ordered[int(i * step)] for i in range(n)]


def format_duration(seconds: float) -> str:
    """``3725`` -> ``1:02:05``."""
    if not math.isfinite(seconds) or seconds < 0:
        return "?"
    s = int(round(seconds))
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def move_to_device(batch: Mapping[str, Any], device: torch.device) -> dict:
    """Copy tensors of a collated batch to ``device`` (non_blocking); strings/lists/dicts stay on the CPU."""
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def make_loader(dataset: torch.utils.data.Dataset, batch_sampler: Any, collate_fn: Any, num_workers: int,
                device: torch.device, persistent: bool) -> torch.utils.data.DataLoader:
    """Windows-safe DataLoader over a batch sampler (spawn workers, pin_memory on CUDA, prefetch_factor 4)."""
    num_workers = max(0, int(num_workers))
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs.update(persistent_workers=persistent, prefetch_factor=4, multiprocessing_context="spawn")
    return torch.utils.data.DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
                                       num_workers=num_workers, pin_memory=device.type == "cuda", **kwargs)


def amp_dtype(amp: Any) -> torch.dtype | None:
    """Map ``train.amp`` (``bf16`` | ``none``/``fp32``/false) to an autocast dtype (None = full precision)."""
    if amp is None or amp is False or str(amp).lower() in ("none", "fp32", "false", "off", "no", ""):
        return None
    if str(amp).lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"unsupported train.amp={amp!r} (use bf16 or none)")


def autocast_context(device: torch.device, amp: Any) -> contextlib.AbstractContextManager:
    """bf16 autocast on CUDA when enabled, otherwise a no-op context."""
    dtype = amp_dtype(amp)
    if dtype is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=dtype)


__all__ = [
    "Config", "to_dict", "parse_value", "apply_override", "merge_dicts", "load_config",
    "set_seed", "safe_console", "get_logger", "close_logger",
    "save_checkpoint", "load_checkpoint",
    "cer", "wer", "AverageMeter", "MetricsWriter", "read_jsonl", "even_subset", "format_duration",
    "move_to_device", "make_loader", "amp_dtype", "autocast_context",
]

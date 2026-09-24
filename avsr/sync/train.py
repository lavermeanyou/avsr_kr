"""Training CLI of the speaker-lip matching (audio-visual sync) model (docs/SYNC_SPEC.md section 5).

Usage (from the project root)::

    & $py -m avsr.sync.train --config configs/sync.yaml
    & $py -m avsr.sync.train --config configs/sync.yaml --limit 256 --set train.epochs=1 `
        --set train.ckpt_dir=work/checkpoints_sync_smoke           # smoke test: keep its checkpoints apart
    & $py -m avsr.sync.train --config configs/sync.yaml --work-dir D:/avsr/work   # data (and logs) elsewhere

One step = ``sync.batch_size`` windows (``SyncWindowDataset``: random 1-s window per utterance, leakage / noise
augmentation, shifted-audio hard negative) from a ``RandomSampler`` over the train rows (``drop_last``), symmetric
InfoNCE (``SyncModel.compute_loss``), AdamW, linear warmup + cosine, bf16 autocast, gradient clipping; non-finite
steps and CUDA OOM batches are skipped. The frontends start from the AVSR checkpoint ``sync.init_from`` (a missing
file is reported and training starts from scratch).

After every ``train.eval_every_epoch`` epochs: 4-way selection accuracy (video -> which of 4 audio streams) at
``sync.window_frames`` and the sync-offset accuracy on the validation speaker(s) (``avsr.sync.evaluate``, same
rules as the full evaluation; the val split has one speaker, so its distractors are that speaker's other recording
sessions). ``best.pt`` = highest 4-way accuracy; ``last.pt`` every epoch; rotating ``epoch_XXX.pt``.
``train.resume: auto`` continues from ``last.pt`` (also mid-epoch after Ctrl+C or a crash); a ``--limit`` run
refuses to resume (or overwrite) the checkpoints of a run with another ``--limit``.
Logs: console, ``<work_dir>/logs/sync_train.log`` and ``sync_metrics.jsonl``; a ``--limit`` (smoke) run writes
``sync_smoke_train.log`` / ``sync_smoke_metrics.jsonl`` instead, so it never mixes into the real run's logs.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

# see avsr.train: keep PyTorch's CUDA cache compact (must be set before torch initialises CUDA)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:256")

import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import BatchSampler, RandomSampler  # noqa: E402
from tqdm import tqdm  # noqa: E402

from avsr.dataset import assign_split, cfg_get, load_manifests  # noqa: E402
from avsr.sync.data import SyncWindowDataset, sync_collate  # noqa: E402
from avsr.sync.evaluate import Embedder, EvalStore, SyncEvaluator  # noqa: E402
from avsr.sync.model import build_sync_model  # noqa: E402
from avsr.train import (LR_FLOOR, MAX_CONSECUTIVE_OOM, GracefulStop, ResumableBatchSampler, lr_factor,  # noqa: E402
                        param_groups)
from avsr.utils import (AverageMeter, Config, MetricsWriter, amp_dtype, autocast_context, close_logger,  # noqa: E402
                        even_subset, format_duration, get_logger, load_checkpoint, load_config, make_loader,
                        move_to_device, safe_console, save_checkpoint, set_seed, to_dict)

VAL_N_WAY = 4            # best.pt criterion: 4-way selection accuracy (docs/SYNC_SPEC.md section 5)
N_EXAMPLES = 3           # example queries printed per validation
CKPT_KIND = "sync"
EPOCH_FILE = re.compile(r"^epoch_(\d+)\.pt$")


def epoch_files(ckpt_dir: Path) -> list[tuple[int, Path]]:
    """``(epoch number, path)`` of the rotating ``epoch_XXX.pt`` files in ``ckpt_dir``, in numeric order."""
    if not ckpt_dir.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in ckpt_dir.glob("epoch_*.pt") if (m := EPOCH_FILE.match(p.name))]
    return sorted(found)


def checkpoint_meta(path: Path) -> Mapping[str, Any] | None:
    """The checkpoint dict at ``path`` (None when it cannot be read)."""
    try:
        ck = load_checkpoint(path, map_location="cpu")
    except Exception:  # noqa: BLE001 - unreadable / foreign file: the caller reports it
        return None
    return ck if isinstance(ck, Mapping) else None


class ShuffledBatchSampler:
    """``BatchSampler(RandomSampler(n), batch_size, drop_last=True)`` whose shuffle is seeded by (seed, epoch), so an
    interrupted epoch replays the same batches (``ResumableBatchSampler`` then skips the consumed ones)."""

    def __init__(self, n: int, batch_size: int, seed: int) -> None:
        if int(batch_size) < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.n = int(n)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        g = torch.Generator()
        g.manual_seed(int(np.random.SeedSequence([self.seed & 0xFFFFFFFF, self.epoch]).generate_state(1)[0]))
        yield from BatchSampler(RandomSampler(range(self.n), generator=g), self.batch_size, drop_last=True)


class _Window:
    """Throughput counters between two log lines."""

    def __init__(self) -> None:
        self.items = 0
        self.steps = 0
        self.data_wait = 0.0
        self.grad_norm = AverageMeter()


def log_prefix(limit: int | None) -> str:
    """Log/metrics file prefix: ``sync`` for real runs, ``sync_smoke`` for ``--limit`` runs."""
    return "sync_smoke" if limit else "sync"


class SyncTrainer:
    """Data, model and optimisation state of one sync training run; :meth:`run` executes the epochs."""

    def __init__(self, cfg: Config, limit: int | None, logger: Any) -> None:
        self.cfg = cfg
        self.logger = logger
        tc = cfg.train
        self.work_dir = Path(str(cfg.work_dir))
        self.ckpt_dir = Path(str(tc.ckpt_dir))
        self.limit = int(limit) if limit else None  # stored in checkpoints: a --limit run never resumes a full one
        self.metrics = MetricsWriter(self.work_dir / "logs" / f"{log_prefix(self.limit)}_metrics.jsonl")
        self.seed = int(cfg.seed)
        set_seed(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True  # fixed window shapes: let cuDNN pick the fastest kernels
            frac = float(tc.get("cuda_mem_fraction", 0.0) or 0.0)
            if 0.0 < frac < 1.0:
                dev_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
                torch.cuda.set_per_process_memory_fraction(frac, dev_idx)
                total = torch.cuda.get_device_properties(dev_idx).total_memory / 2**30
                logger.info("CUDA memory cap: %.2f x %.1f GiB = %.1f GiB (alloc conf %s)", frac, total, frac * total,
                            os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
        self.amp = tc.get("amp", "bf16")
        amp_dtype(self.amp)  # validate early
        self.epochs = int(tc.epochs)
        self.log_every = max(1, int(tc.get("log_every", 50)))
        self.eval_every = max(1, int(tc.get("eval_every_epoch", 1)))
        self.keep_last = int(tc.get("keep_last", 3))
        self.grad_clip = float(tc.get("grad_clip", 0.0))
        self.window = int(cfg_get(cfg, "sync.window_frames", 25))
        self.batch_size = int(cfg_get(cfg, "sync.batch_size", 64))
        n_way = cfg_get(cfg, "sync.eval_n_way", [2, 3, 4]) or []
        self.val_n_way = sorted({int(n) for n in (n_way if isinstance(n_way, (list, tuple)) else [n_way])}
                                | {VAL_N_WAY})

        # ---- data -------------------------------------------------------------------------------------------
        rows = load_manifests(self.work_dir)
        splits = assign_split(rows, cfg)
        train_rows, val_rows = list(splits.get("train", [])), list(splits.get("val", []))
        if self.limit:
            train_rows, val_rows = even_subset(train_rows, self.limit), even_subset(val_rows, self.limit)
        max_val = int(cfg_get(cfg, "sync.val_max_utts", 600) or 0)
        val_rows = even_subset(val_rows, max_val if max_val > 0 else None)
        if not train_rows:
            raise RuntimeError(f"no training utterances (work_dir={self.work_dir}, {len(rows)} manifest rows, "
                               f"split={dict(cfg.split)}); run avsr.preprocess first or check the split config")
        self.train_ds = SyncWindowDataset(train_rows, cfg, True, seed=self.seed)
        if len(self.train_ds) < self.batch_size:
            raise RuntimeError(f"only {len(self.train_ds)} training windows (of {len(train_rows)} rows) for "
                               f"sync.batch_size={self.batch_size}: lower sync.batch_size or use more data")
        base_sampler = ShuffledBatchSampler(len(self.train_ds), self.batch_size, self.seed)
        self.sampler = ResumableBatchSampler(base_sampler)
        self.steps_per_epoch = len(base_sampler)
        nw = int(cfg_get(cfg, "data.num_workers", 0))
        self.train_loader = make_loader(self.train_ds, self.sampler, sync_collate, nw, self.device, persistent=True)
        hours = sum(float(r.get("duration", 0.0)) for r in train_rows) / 3600.0
        logger.info("manifest rows %d -> train %d rows (%.1f h; %d windows of %d frames, %d dropped as too short), "
                    "val %d rows (per-epoch subset), test %d", len(rows), len(train_rows), hours, len(self.train_ds),
                    self.window, self.train_ds.n_dropped, len(val_rows), len(splits.get("test", [])))
        self.val_store: EvalStore | None = None
        if val_rows:
            t = time.time()
            self.val_store = EvalStore.load(val_rows, cfg, nw)
            speakers = sorted(set(self.val_store.speakers.tolist()))
            logger.info("validation: %d utterances of speaker(s) %s loaded in %.0f s (%d unusable)%s",
                        len(self.val_store), speakers, time.time() - t, self.val_store.n_bad,
                        "; one speaker: 4-way distractors are its other recording sessions" if len(speakers) == 1
                        else "")
            if len(self.val_store) == 0:
                self.val_store = None
        if self.val_store is None:
            logger.warning("validation set is empty: per-epoch evaluation and best.pt are skipped")

        # ---- model / optimisation ------------------------------------------------------------------------
        self.model = build_sync_model(cfg).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        lr = float(tc.lr)
        self.optimizer = torch.optim.AdamW(param_groups(self.model, float(tc.weight_decay)), lr=lr)
        self.total_steps = max(1, self.epochs * self.steps_per_epoch)
        warmup = int(tc.warmup_steps)
        eff_warmup = max(1, min(warmup, self.total_steps // 3))
        if eff_warmup != warmup:
            logger.warning("warmup_steps %d > 1/3 of the %d total steps: using %d", warmup, self.total_steps,
                           eff_warmup)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, functools.partial(lr_factor, warmup=eff_warmup, total=self.total_steps, floor=LR_FLOOR))
        logger.info("sync model %.1f M params on %s (amp=%s, temporal=%s, cnn=%s, skeleton=%s) | batch %d windows x "
                    "%d frames | %d steps/epoch x %d epochs | lr %.2e warmup %d | workers %d", n_params / 1e6,
                    self.device, self.amp, cfg_get(cfg, "sync.temporal", "conv"), cfg_get(cfg, "sync.use_cnn", True),
                    cfg_get(cfg, "sync.use_skeleton", True), self.batch_size, self.window, self.steps_per_epoch,
                    self.epochs, lr, eff_warmup, nw)

        self.step = 0
        self.epoch = 0             # completed epochs (index of the epoch in progress)
        self.batch_in_epoch = 0    # batches consumed in the epoch in progress
        self.best: dict[str, Any] = {"acc": -math.inf, "epoch": None, "step": None}
        self.last_val: dict[str, Any] | None = None
        self.init_report: dict[str, Any] | None = None
        self.skipped_steps = 0     # non-finite loss/gradient or OOM batches
        self.consecutive_oom = 0
        if not self._maybe_resume():
            self._init_frontends()

    # ---- initialisation / checkpoints -------------------------------------------------------------------------
    def _init_frontends(self) -> None:
        """Copy the AVSR frontends (``sync.init_from``) into a fresh model; a missing file is reported loudly."""
        src = str(cfg_get(self.cfg, "sync.init_from", "") or "").strip()
        if not src:
            self.logger.info("sync.init_from is empty: all weights from scratch")
            self.init_report = {"ckpt": None, "status": "scratch"}
            return
        path = Path(src)
        if not path.is_file():
            self.logger.warning("sync.init_from=%s not found: training ALL weights from scratch (copy an AVSR "
                                "checkpoint there, or set sync.init_from to its path, to start from its lip/audio "
                                "frontends)", path)
            self.init_report = {"ckpt": str(path), "status": "missing"}
            return
        report = self.model.init_from_avsr(path)
        exposed = self._init_heldout_exposure(path)
        self.init_report = {"ckpt": str(path), "status": "ok", "modules": report["modules"],
                            "skipped": report["skipped"], "heldout_not_held_out_by_init": exposed}
        self.logger.info("frontends initialised from %s: %s%s", path,
                         ", ".join(f"{m} {s['copied']}/{s['total']}" for m, s in report["modules"].items()),
                         f"; skipped {report['skipped']}" if report["skipped"] else "")
        if exposed is None:
            self.logger.warning("%s has no split config: cannot check that it held out this run's val/test speakers "
                                "(if it trained on them, their sync metrics are optimistic)", path)
        elif exposed:
            self.logger.warning("%s was trained WITHOUT holding out the speakers %s, which are val/test speakers "
                                "here: its frontends may have seen them, so their sync metrics will be optimistic "
                                "(use an AVSR checkpoint trained with the same split, or sync.init_from='')", path,
                                exposed)

    def _init_heldout_exposure(self, path: Path) -> list[str] | None:
        """This config's val/test speakers that the init checkpoint's own split did NOT hold out (so they may have
        been in its training data); None when that checkpoint carries no split config."""
        split = ((checkpoint_meta(path) or {}).get("cfg") or {}).get("split")
        if not isinstance(split, Mapping):
            return None
        keys = ("val_speakers", "test_speakers")
        held = {str(s) for k in keys for s in (cfg_get(self.cfg, f"split.{k}", []) or [])}
        init_held = {str(s) for k in keys for s in (split.get(k) or [])}
        return sorted(held - init_held)

    def _state(self) -> dict[str, Any]:
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {
            "kind": CKPT_KIND, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "step": self.step, "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch, "best": dict(self.best), "cfg": to_dict(self.cfg),
            "val": self.last_val, "rng": rng, "limit": self.limit, "init": self.init_report,
        }

    def save_last(self) -> Path:
        path = save_checkpoint(self.ckpt_dir / "last.pt", self._state())
        self.logger.info("saved %s (epoch %d, step %d)", path, self.epoch, self.step)
        return path

    def _copy(self, src: Path, name: str) -> Path:
        dst = self.ckpt_dir / name
        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        return dst

    def _save_epoch_end(self, is_best: bool) -> None:
        last = self.save_last()
        if self.keep_last > 0:
            self._copy(last, f"epoch_{self.epoch:03d}.pt")
            # rotate this run's files only (numbers up to the current epoch, numeric order): higher-numbered files of
            # an earlier run in the same directory are left alone until this run overwrites them
            ours = [p for n, p in epoch_files(self.ckpt_dir) if n <= self.epoch]
            for old in ours[:-self.keep_last]:
                old.unlink(missing_ok=True)
        if is_best:
            self._copy(last, "best.pt")
            self.logger.info("new best val %d-way accuracy %.2f%% at epoch %d -> best.pt", VAL_N_WAY,
                             100 * self.best["acc"], self.epoch)

    def _check_fresh_ckpt_dir(self) -> None:
        """A new run will write last.pt / best.pt / epoch_XXX.pt into ckpt_dir: refuse when that directory holds a
        last.pt / best.pt that is not a sync checkpoint (e.g. the AVSR model's), warn about an earlier sync run."""
        named = [p for p in (self.ckpt_dir / "last.pt", self.ckpt_dir / "best.pt") if p.is_file()]
        old_epochs = epoch_files(self.ckpt_dir)
        if not named and not old_epochs:
            return
        kinds = {p.name: (checkpoint_meta(p) or {"kind": "unreadable"}).get("kind") for p in named}
        foreign = [f"{name} (kind={kind!r})" for name, kind in kinds.items() if kind != CKPT_KIND]
        if foreign:
            raise RuntimeError(f"train.ckpt_dir {self.ckpt_dir} holds checkpoints that are not sync checkpoints "
                               f"({', '.join(foreign)}): a new sync run would overwrite them. Use another "
                               f"train.ckpt_dir (default work/checkpoints_sync)")
        self.logger.warning("%s holds files of an earlier sync run (%s): this new run replaces last.pt, best.pt (at "
                            "its first validation) and each epoch_XXX.pt as it writes them; the earlier run's "
                            "epoch_XXX.pt with a higher number stay until then", self.ckpt_dir,
                            ", ".join([p.name for p in named] + [p.name for _, p in old_epochs]))

    def _maybe_resume(self) -> bool:
        """Resume per ``train.resume`` (auto = ``last.pt`` if present); returns True when a checkpoint was loaded."""
        resume = self.cfg.train.get("resume", "auto")
        text = str(resume).strip().lower() if resume is not None else "none"
        if text in ("none", "no", "off", "false", ""):
            path = None
        elif text in ("auto", "true"):
            path = self.ckpt_dir / "last.pt"
            path = path if path.is_file() else None
        else:
            path = Path(str(resume))
            if not path.is_file():
                raise FileNotFoundError(f"train.resume={resume!r}: checkpoint not found")
        if path is None:
            self._check_fresh_ckpt_dir()
            self.logger.info("starting a new run (checkpoints -> %s)", self.ckpt_dir)
            return False
        ckpt = load_checkpoint(path, map_location="cpu")
        if ckpt.get("kind") != CKPT_KIND:
            raise RuntimeError(f"{path} is not a sync checkpoint (kind={ckpt.get('kind')!r}): use another "
                               f"train.ckpt_dir (the AVSR model's checkpoints live in work/checkpoints)")
        saved_limit = ckpt.get("limit")
        if saved_limit != self.limit:
            was = f"--limit {saved_limit}" if saved_limit else "no --limit"
            now = f"--limit {self.limit}" if self.limit else "no --limit"
            msg = f"{path} comes from a run with {was}, this run has {now}"
            if text in ("auto", "true"):  # never mix a smoke run and a real run in one checkpoint directory
                raise RuntimeError(f"{msg}: refusing to resume it or to overwrite its checkpoints. Give smoke "
                                   f"tests (--limit) their own checkpoint directory (e.g. --set "
                                   f"train.ckpt_dir=work/checkpoints_sync_smoke); to start over in {path.parent} "
                                   f"instead, pass --set train.resume=none (its last.pt, best.pt and epoch_*.pt "
                                   f"will then be overwritten)")
            self.logger.warning("%s (resuming anyway: explicit train.resume path)", msg)
        try:
            self.model.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            raise RuntimeError(f"{path} does not match the current sync model config ({e}); use another "
                               f"train.ckpt_dir or --set train.resume=none") from e
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = int(ckpt["step"])
        self.epoch = int(ckpt["epoch"])
        self.batch_in_epoch = int(ckpt.get("batch_in_epoch", 0))
        self.best = dict(ckpt.get("best") or self.best)
        self.last_val = ckpt.get("val")
        self.init_report = ckpt.get("init")
        rng = ckpt.get("rng") or {}
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available() and len(rng["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng["cuda"])
        best = self.best.get("acc")
        self.logger.info("resumed from %s: epoch %d (+%d batches), step %d, best val %d-way acc %s", path,
                         self.epoch, self.batch_in_epoch, self.step, VAL_N_WAY,
                         f"{100 * best:.2f}%" if best is not None and math.isfinite(best) else "-")
        return True

    # ---- training -----------------------------------------------------------------------------------------
    def _train_step(self, batch: dict, meters: dict[str, AverageMeter]) -> float | None:
        """One optimisation step; returns the gradient norm, or None when the step was skipped (non-finite)."""
        b = move_to_device(batch, self.device)
        with autocast_context(self.device, self.amp):
            out = self.model(b)
            loss, parts = self.model.compute_loss(out, b)
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            self.logger.warning("non-finite loss at step %d (utts %s...): batch skipped", self.step,
                                batch["utt_id"][:2])
            return None
        loss.backward()
        clip = self.grad_clip if self.grad_clip > 0 else math.inf
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip))
        if not math.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            self.logger.warning("non-finite gradient at step %d: batch skipped", self.step)
            return None
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self.consecutive_oom = 0
        n = len(batch["utt_id"])
        for k, v in parts.items():
            if math.isfinite(float(v)):  # neg_cos / shift_cos / acc_shift are NaN when the batch has no such pair
                meters[k].update(float(v), n)
        return grad_norm

    def _handle_oom(self, batch: dict) -> None:
        """Skip a batch that ran out of GPU memory; give up after MAX_CONSECUTIVE_OOM in a row."""
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self.skipped_steps += 1
        self.consecutive_oom += 1
        self.logger.warning("CUDA out of memory on a batch of %d windows: skipped (lower sync.batch_size if this "
                            "repeats)", len(batch["utt_id"]))
        if self.consecutive_oom >= MAX_CONSECUTIVE_OOM:
            raise RuntimeError(f"{MAX_CONSECUTIVE_OOM} consecutive CUDA OOM batches: lower sync.batch_size")

    def _train_epoch(self, epoch: int, stop: GracefulStop) -> None:
        self.model.train()
        self.sampler.set_epoch(epoch)
        skip = self.batch_in_epoch
        if skip:
            self.sampler.skip_next(skip)
            self.logger.info("epoch %d: skipping %d already-trained batches", epoch + 1, skip)
        meters: dict[str, AverageMeter] = defaultdict(AverageMeter)
        epoch_loss = AverageMeter()
        bar = (tqdm(total=self.steps_per_epoch, initial=skip, desc=f"epoch {epoch + 1}", dynamic_ncols=True,
                    leave=False, file=sys.stdout) if sys.stdout.isatty() else None)
        t_epoch = t_log = t_mark = time.time()
        window = _Window()
        steady = _Window()               # epoch totals after the first step (worker start-up excluded)
        t_steady: float | None = None
        first_batch_sec = math.nan
        try:
            for batch in self.train_loader:
                wait = time.time() - t_mark
                window.data_wait += wait
                window.items += len(batch["utt_id"])
                if t_steady is not None:
                    steady.data_wait += wait
                    steady.items += len(batch["utt_id"])
                    steady.steps += 1
                else:
                    first_batch_sec = time.time() - t_epoch
                oom = False
                try:
                    gn = self._train_step(batch, meters)
                except torch.cuda.OutOfMemoryError:
                    oom, gn = True, None
                if oom:
                    self._handle_oom(batch)
                self.batch_in_epoch += 1
                if gn is not None:
                    window.steps += 1
                    window.grad_norm.update(gn)
                if bar is not None:
                    bar.update(1)
                    if meters["loss"].count:
                        bar.set_postfix(loss=f"{meters['loss'].avg:.3f}", acc=f"{meters['acc_va'].avg:.2f}",
                                        step=self.step)
                if gn is not None and self.step % self.log_every == 0:
                    self._log_train(epoch, meters, window, time.time() - t_log)
                    epoch_loss.update(meters["loss"].avg, meters["loss"].count)
                    meters = defaultdict(AverageMeter)
                    window, t_log = _Window(), time.time()
                if stop.requested:
                    raise KeyboardInterrupt
                if t_steady is None:
                    t_steady = time.time()
                t_mark = time.time()
        finally:
            if bar is not None:
                bar.close()
        if meters["loss"].count:
            epoch_loss.update(meters["loss"].avg, meters["loss"].count)
        steady_sec = time.time() - t_steady if t_steady is not None else 0.0
        speed = {"first_batch_sec": round(first_batch_sec, 2)}
        if steady.steps and steady_sec > 0:
            speed.update(sec_per_step=round(steady_sec / steady.steps, 4),
                         items_per_sec=round(steady.items / steady_sec, 1),
                         data_wait_frac=round(min(1.0, steady.data_wait / steady_sec), 3))
        self.logger.info("epoch %d/%d done in %s | mean loss %.4f | step %d | first batch after %.1f s (worker "
                         "start-up)%s%s", epoch + 1, self.epochs, format_duration(time.time() - t_epoch),
                         epoch_loss.avg, self.step, first_batch_sec,
                         (f", then {speed['sec_per_step']:.3f} s/step, {speed['items_per_sec']:.0f} windows/s, data "
                          f"wait {100 * speed['data_wait_frac']:.0f}%" if "sec_per_step" in speed else ""),
                         f" | {self.skipped_steps} skipped steps so far" if self.skipped_steps else "")
        self.metrics.write({"type": "epoch", "epoch": epoch + 1, "step": self.step, "loss": epoch_loss.avg,
                            "time_sec": round(time.time() - t_epoch, 1), "skipped_steps": self.skipped_steps,
                            **speed})

    def _log_train(self, epoch: int, meters: dict[str, AverageMeter], window: _Window, elapsed: float) -> None:
        elapsed = max(elapsed, 1e-6)
        items_per_sec = window.items / elapsed
        sec_per_step = elapsed / max(window.steps, 1)
        data_frac = min(1.0, window.data_wait / elapsed)
        eta_epoch = (self.steps_per_epoch - self.batch_in_epoch) * sec_per_step
        eta_total = (self.total_steps - self.step) * sec_per_step
        lr = self.optimizer.param_groups[0]["lr"]
        shown = ("l_va", "l_av", "acc_va", "acc_av", "acc_shift", "pos_cos", "neg_cos", "shift_cos", "scale")
        parts = " ".join(f"{k} {meters[k].avg:.3f}" for k in shown if meters[k].count)
        mem = (f" | mem {torch.cuda.max_memory_allocated(self.device) / 2**30:.1f}G"
               if self.device.type == "cuda" else "")
        self.logger.info("ep %d/%d it %d/%d step %d | loss %.4f (%s) | lr %.2e | gnorm %.2f | %.2f s/step, %.0f "
                         "windows/s, data wait %.0f%%%s | ETA epoch %s total %s", epoch + 1, self.epochs,
                         self.batch_in_epoch, self.steps_per_epoch, self.step, meters["loss"].avg, parts, lr,
                         window.grad_norm.avg, sec_per_step, items_per_sec, 100 * data_frac, mem,
                         format_duration(eta_epoch), format_duration(eta_total))
        self.metrics.write({"type": "train", "epoch": epoch + 1, "step": self.step, "lr": lr,
                            **{k: m.avg for k, m in meters.items()}, "grad_norm": window.grad_norm.avg,
                            "items_per_sec": round(items_per_sec, 1), "sec_per_step": round(sec_per_step, 4),
                            "data_wait_frac": round(data_frac, 3)})

    # ---- validation ---------------------------------------------------------------------------------------
    def validate(self) -> dict[str, Any]:
        """N-way selection (video -> audio) and sync offset at ``sync.window_frames`` on the val subset."""
        assert self.val_store is not None
        t0 = time.time()
        w = self.window
        was_training = self.model.training
        self.model.eval()
        try:
            ev = SyncEvaluator(self.val_store, Embedder(self.model, self.device, self.amp), self.seed,
                               draw_k=max(self.val_n_way) - 1)
            nway = ev.nway(w, self.val_n_way, "v2a")
            off = ev.offsets(w)
            examples = ev.examples(w, N_EXAMPLES, VAL_N_WAY)
        finally:
            self.model.train(was_training)
        crit = nway[str(VAL_N_WAY)]
        scores = {f"acc{n}": nway[str(n)]["acc"] for n in self.val_n_way}
        scores.update({"n": crit["n"], "true_cos": crit.get("true_score_mean", math.nan),
                       "distractor_cos": crit.get("distractor_score_mean", math.nan), "offset_acc": off["acc_pm1"],
                       "offset_median_ms": off["median_abs_err_ms"], "offset_mean_ms": off["mean_offset_ms"],
                       "offset_n": off["n"], "distractors": crit.get("distractors", {})})

        def pct(x: float) -> str:
            return f"{100 * x:.1f}%" if math.isfinite(x) else "-"

        self.logger.info("val epoch %d (%d queries, %.0f s): %s | offset acc(+-1) %s, median |err| %s ms, mean "
                         "offset %s ms (%d queries) | cos true %.3f vs distractors %.3f", self.epoch, crit["n"],
                         time.time() - t0, " ".join(f"{n}-way {pct(scores[f'acc{n}'])}" for n in self.val_n_way),
                         pct(off["acc_pm1"]), f"{off['median_abs_err_ms']:.0f}" if off["n"] else "-",
                         f"{off['mean_offset_ms']:+.0f}" if off["n"] else "-", off["n"], scores["true_cos"],
                         scores["distractor_cos"])
        for ex in examples:
            self.logger.info("  example %s: true %.3f vs distractors %s -> %s; best audio offset %s frames",
                             ex["utt_id"], ex["true"], ex["distractors"], "correct" if ex["correct"] else "wrong",
                             ex["offset_pred"] if ex["offset_pred"] is not None else "-")
        self.metrics.write({"type": "val", "epoch": self.epoch, "step": self.step, "window": w,
                            **{k: v for k, v in scores.items() if k != "distractors"}})
        if not math.isfinite(scores[f"acc{VAL_N_WAY}"]):
            self.logger.warning("val %d-way accuracy undefined (%s): not enough distinct recording sessions in the "
                                "val subset", VAL_N_WAY, crit)
        return scores

    # ---- main loop ----------------------------------------------------------------------------------------
    def run(self) -> int:
        if self.epoch >= self.epochs:
            self.logger.info("already trained %d/%d epochs (raise train.epochs to continue)", self.epoch, self.epochs)
            return 0
        self.metrics.write({"type": "start", "epoch": self.epoch, "step": self.step, "total_steps": self.total_steps,
                            "steps_per_epoch": self.steps_per_epoch, "batch_size": self.batch_size})
        stop = GracefulStop(self.logger)
        stop.install()
        try:
            for epoch in range(self.epoch, self.epochs):
                self.epoch = epoch
                self._train_epoch(epoch, stop)
                self.epoch, self.batch_in_epoch = epoch + 1, 0
                is_best = False
                if self.val_store is not None and (self.epoch % self.eval_every == 0 or self.epoch == self.epochs):
                    scores = self.validate()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()  # drop inference-shaped cache blocks before training resumes
                    self.last_val = scores
                    crit = float(scores[f"acc{VAL_N_WAY}"])
                    if math.isfinite(crit) and crit > self.best["acc"]:
                        self.best = {"acc": crit, "epoch": self.epoch, "step": self.step}
                        is_best = True
                self._save_epoch_end(is_best)
                if stop.requested:
                    self.logger.warning("stopped after epoch %d on request", self.epoch)
                    return 130
        except BaseException as e:  # noqa: BLE001 - Ctrl+C (or a worker killed by it): save and exit cleanly
            if isinstance(e, KeyboardInterrupt) or stop.requested:
                self.logger.warning("interrupted at epoch %d (+%d batches), step %d: saving last.pt",
                                    self.epoch, self.batch_in_epoch, self.step)
                self.save_last()
                return 130
            self.logger.error("training failed at epoch %d (+%d batches), step %d: %s: %s", self.epoch,
                              self.batch_in_epoch, self.step, type(e).__name__, e)
            if self.step > 0:
                try:
                    self.save_last()
                except Exception as save_error:  # noqa: BLE001 - the original error matters more
                    self.logger.error("could not save last.pt: %s", save_error)
            raise
        finally:
            stop.uninstall()
        if math.isfinite(self.best["acc"]):
            self.logger.info("training finished: best val %d-way accuracy %.2f%% at epoch %s (%s)", VAL_N_WAY,
                             100 * self.best["acc"], self.best["epoch"], self.ckpt_dir / "best.pt")
        else:
            self.logger.info("training finished (no validation score; last checkpoint %s)", self.ckpt_dir / "last.pt")
        return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the speaker-lip matching (audio-visual sync) model.")
    p.add_argument("--config", default="configs/sync.yaml", help="YAML config")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, repeatable (e.g. --set train.epochs=1)")
    p.add_argument("--limit", type=int, default=None, help="use only N train and N val utterances (smoke tests)")
    p.add_argument("--work-dir", default=None, help="override work_dir (checkpoints follow unless train.ckpt_dir set)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    safe_console()
    args = parse_args(argv)
    # JSON-quoted so any path survives the YAML value parsing of overrides
    overrides = list(args.set) + ([f"work_dir={json.dumps(str(args.work_dir))}"] if args.work_dir else [])
    cfg = load_config(args.config, overrides)
    prefix = log_prefix(args.limit)
    logger = get_logger(Path(str(cfg.work_dir)), f"{prefix}_train")
    try:
        logger.info("config %s overrides %s%s", args.config, overrides, f" limit {args.limit}" if args.limit else "")
        cfg_dump = Path(str(cfg.work_dir)) / "logs" / f"{prefix}_train_config.yaml"
        with open(cfg_dump, "w", encoding="utf-8") as f:
            yaml.safe_dump(to_dict(cfg), f, allow_unicode=True, sort_keys=False)
        return SyncTrainer(cfg, args.limit, logger).run()
    finally:
        close_logger()


if __name__ == "__main__":
    sys.exit(main())

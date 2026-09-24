"""Training CLI (SPEC section 11).

Usage (from the project root)::

    & $py -m avsr.train --config configs/base.yaml
    & $py -m avsr.train --config configs/base.yaml --set train.epochs=1 --set data.num_workers=2 --limit 64 `
        --set train.ckpt_dir=work/checkpoints_smoke     # smoke test: keep its checkpoints apart
    & $py -m avsr.train --work-dir D:/runs/exp1          # features/manifests/logs/checkpoints under that dir

Per epoch: train (modality dropout, bf16 autocast, grad clipping, warmup + cosine LR), then greedy CTC decode of a fixed
validation subset under ``av`` / ``audio`` / ``video`` on clean audio plus ``av`` / ``audio`` at 0 dB
``eval.noise_kind`` noise; saves ``last.pt`` (every epoch), ``best.pt`` (lowest val CER(av, clean)) and rotating
``epoch_XXX.pt``. ``train.resume: auto`` continues from ``last.pt`` (also mid-epoch after Ctrl+C or a crash); a
``--limit`` run refuses to resume (or overwrite) the checkpoints of a run with another ``--limit``.
Logs: console, ``<work_dir>/logs/train.log`` and ``<work_dir>/logs/metrics.jsonl``.
"""
from __future__ import annotations

import argparse
import copy
import functools
import json
import math
import os
import random
import shutil
import signal
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

# Variable-length batches fragment PyTorch's CUDA cache; on Windows (WDDM) a cache that outgrows the dedicated VRAM is
# silently backed by system RAM ("sysmem fallback") and training slows ~2.5x. Reclaim cached blocks early and do not
# split big blocks (must be set before torch initialises CUDA). See also train.cuda_mem_fraction.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:256")

import torch  # noqa: E402
import yaml  # noqa: E402
from tqdm import tqdm  # noqa: E402

from avsr.dataset import AVSRDataset, DurationBatchSampler, assign_split, collate_fn, load_manifests
from avsr.evaluate import ALL_MODES, AUDIO_MODES, build_eval_loader, decode_loader, parse_conditions, score
from avsr.models import build_model
from avsr.text import Tokenizer
from avsr.utils import (AverageMeter, Config, MetricsWriter, amp_dtype, autocast_context, close_logger, even_subset,
                        format_duration, get_logger, load_checkpoint, load_config, make_loader, move_to_device,
                        safe_console, save_checkpoint, set_seed, to_dict)

LR_FLOOR = 0.02          # cosine decays to LR_FLOOR * lr
VAL_NOISE_SNR = 0.0      # extra validation pass at this SNR (dB) to track the benefit of the lips under noise
DEFAULT_MAX_VAL_UTTS = 400
VAL_DECODE = "ctc_greedy"  # per-epoch validation is always fast greedy CTC (evaluate.py honours eval.decode)
N_EXAMPLES = 2           # example hypotheses printed per validation
MAX_CONSECUTIVE_OOM = 5  # abort after this many CUDA OOM batches in a row


def lr_factor(step: int, warmup: int, total: int, floor: float) -> float:
    """Linear warmup to 1 over ``warmup`` steps, then cosine from 1 to ``floor`` at ``total`` steps."""
    if step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = min(1.0, (step - warmup) / (total - warmup))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


class ResumableBatchSampler:
    """Wraps the duration batch sampler: forwards ``set_epoch`` and can skip the first k batches of the next pass.

    Iteration happens in the main process, so this works with persistent DataLoader workers.
    """

    def __init__(self, base: Any) -> None:
        self.base = base
        self._skip = 0

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def skip_next(self, n: int) -> None:
        self._skip = max(0, int(n))

    def __iter__(self) -> Iterator[list[int]]:
        skip, self._skip = self._skip, 0
        for i, batch in enumerate(self.base):
            if i >= skip:
                yield batch

    def __len__(self) -> int:
        return max(0, len(self.base) - self._skip)


class GracefulStop:
    """First Ctrl+C: finish the current step, save ``last.pt`` and exit. Second Ctrl+C: abort immediately."""

    def __init__(self, logger: Any) -> None:
        self.logger = logger
        self.requested = False
        self._prev: Any = None

    def install(self) -> None:
        if threading.current_thread() is threading.main_thread():
            self._prev = signal.signal(signal.SIGINT, self._handle)

    def uninstall(self) -> None:
        if self._prev is not None:
            signal.signal(signal.SIGINT, self._prev)
            self._prev = None

    def _handle(self, signum: int, frame: Any) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        self.logger.warning("Ctrl+C received: saving last.pt and stopping (press Ctrl+C again to abort now)")


class _Window:
    """Throughput counters between two log lines."""

    def __init__(self) -> None:
        self.frames = 0
        self.steps = 0
        self.data_wait = 0.0
        self.grad_norm = AverageMeter()


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """AdamW groups: weight decay on matrices/kernels, none on biases and norm/scale vectors."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


class Trainer:
    """Holds data, model, optimisation state; :meth:`run` executes the epochs."""

    def __init__(self, cfg: Config, limit: int | None, logger: Any) -> None:
        self.cfg = cfg
        self.logger = logger
        tc = cfg.train
        self.work_dir = Path(str(cfg.work_dir))
        self.ckpt_dir = Path(str(tc.ckpt_dir))
        self.metrics = MetricsWriter(self.work_dir / "logs" / "metrics.jsonl")
        set_seed(int(cfg.seed))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            frac = float(tc.get("cuda_mem_fraction", 0.0) or 0.0)
            if 0.0 < frac < 1.0:  # keep the caching allocator inside dedicated VRAM (see PYTORCH_CUDA_ALLOC_CONF above)
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
        self.method = VAL_DECODE
        self.conditions = parse_conditions(cfg.eval.get("conditions", list(ALL_MODES)))
        self.limit = int(limit) if limit else None  # stored in checkpoints: a --limit run never resumes a full one
        self.tok = Tokenizer()

        # ---- data -------------------------------------------------------------------------------------------
        rows = load_manifests(self.work_dir)
        splits = assign_split(rows, cfg)
        train_rows, val_rows = list(splits.get("train", [])), list(splits.get("val", []))
        if self.limit:
            train_rows, val_rows = even_subset(train_rows, self.limit), even_subset(val_rows, self.limit)
        max_val = cfg.eval.get("max_val_utts", DEFAULT_MAX_VAL_UTTS)
        val_rows = even_subset(val_rows, int(max_val) if max_val else None)
        if not train_rows:
            raise RuntimeError(f"no training utterances (work_dir={self.work_dir}, {len(rows)} manifest rows, "
                               f"split={dict(cfg.split)}); run avsr.preprocess first or check the split config")
        hours = sum(float(r.get("duration", 0.0)) for r in train_rows) / 3600.0
        logger.info("manifest rows %d -> train %d (%.1f h), val %d (decode subset), test %d", len(rows),
                    len(train_rows), hours, len(val_rows), len(splits.get("test", [])))
        nw = int(cfg.data.get("num_workers", 0))
        train_ds = AVSRDataset(train_rows, cfg, True, self.tok)
        base_sampler = DurationBatchSampler(list(getattr(train_ds, "rows", train_rows)),
                                            max_frames=int(tc.max_frames_per_batch), shuffle=True, seed=int(cfg.seed))
        self.sampler = ResumableBatchSampler(base_sampler)
        self.steps_per_epoch = len(base_sampler)
        self._train_rows, self._num_workers = train_rows, nw
        # train.aug_start_epoch: epochs before it train on clean audio (no noise mixing, no SpecAugment) so the CTC
        # alignment is learnt first; the loader is rebuilt with full augmentation when that epoch starts.
        self.aug_start_epoch = int(tc.get("aug_start_epoch", 1) or 1)
        self._train_aug: bool | None = None
        self.train_loader: torch.utils.data.DataLoader | None = None
        self._full_aug_ds = train_ds
        self.noise_kind = str(cfg.eval.get("noise_kind", "babble"))
        self.val_loaders: dict[str, torch.utils.data.DataLoader] = {}
        if val_rows:
            self.val_loaders["clean"] = build_eval_loader(val_rows, cfg, self.tok, None, nw, self.device, True)
            if any(c in AUDIO_MODES for c in self.conditions):
                self.val_loaders["noisy"] = build_eval_loader(val_rows, cfg, self.tok,
                                                              (self.noise_kind, VAL_NOISE_SNR), nw, self.device, True,
                                                              babble_rows=train_rows)
        else:
            logger.warning("validation set is empty: per-epoch evaluation and best.pt are skipped")

        # ---- model / optimisation ------------------------------------------------------------------------
        self.model = build_model(cfg, self.tok.vocab_size).to(self.device)
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
        logger.info("model %.1f M params on %s (amp=%s) | %d steps/epoch x %d epochs | lr %.2e warmup %d | "
                    "workers %d", n_params / 1e6, self.device, self.amp, self.steps_per_epoch, self.epochs, lr,
                    eff_warmup, nw)

        self.step = 0
        self.epoch = 0             # completed epochs (index of the epoch in progress)
        self.batch_in_epoch = 0    # batches consumed in the epoch in progress
        self.best: dict[str, Any] = {"cer": math.inf, "epoch": None, "step": None}
        self.last_val: dict[str, Any] | None = None
        self.skipped_steps = 0     # non-finite loss/gradient or OOM batches
        self.consecutive_oom = 0
        self._maybe_resume()

    # ---- checkpoints --------------------------------------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(), "step": self.step, "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch, "sampler_epoch": self.epoch, "best": dict(self.best),
            "cfg": to_dict(self.cfg), "vocab": list(self.tok.tokens), "val": self.last_val, "rng": rng,
            "limit": self.limit,
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
            # numeric order, and only files up to the current epoch: a restarted run (resume=none) in a used dir must
            # rotate its own files, not delete them while keeping a previous run's higher-numbered epochs
            olds = sorted((int(p.stem.split("_")[1]), p) for p in self.ckpt_dir.glob("epoch_*.pt")
                          if p.stem.split("_")[1].isdigit() and int(p.stem.split("_")[1]) <= self.epoch)
            for _, old in olds[:-self.keep_last]:
                old.unlink(missing_ok=True)
        if is_best:
            self._copy(last, "best.pt")
            self.logger.info("new best val CER(av) %.2f%% at epoch %d -> best.pt", 100 * self.best["cer"], self.epoch)

    def _maybe_resume(self) -> None:
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
            self.logger.info("starting from scratch (checkpoints -> %s)", self.ckpt_dir)
            return
        ckpt = load_checkpoint(path, map_location="cpu")
        saved_limit = ckpt.get("limit")
        if saved_limit != self.limit:
            was = f"--limit {saved_limit}" if saved_limit else "no --limit"
            now = f"--limit {self.limit}" if self.limit else "no --limit"
            msg = f"{path} comes from a run with {was}, this run has {now}"
            if text in ("auto", "true"):  # never mix a smoke run and a real run in one checkpoint directory
                raise RuntimeError(f"{msg}: refusing to resume it or to overwrite its checkpoints. Give smoke "
                                   f"tests (--limit) their own checkpoint directory (e.g. --set "
                                   f"train.ckpt_dir=work/checkpoints_smoke); to start over in {path.parent} "
                                   f"instead, pass --set train.resume=none (its last.pt, best.pt and epoch_*.pt "
                                   f"will then be overwritten)")
            self.logger.warning("%s (resuming anyway: explicit train.resume path)", msg)
        if list(ckpt.get("vocab", self.tok.tokens)) != list(self.tok.tokens):
            raise ValueError(f"{path}: tokenizer vocabulary differs from avsr.text.Tokenizer")
        try:
            self.model.load_state_dict(ckpt["model"])
        except RuntimeError as e:
            raise RuntimeError(f"{path} does not match the current model config ({e}); use another work_dir / "
                               f"train.ckpt_dir or --set train.resume=none") from e
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = int(ckpt["step"])
        self.epoch = int(ckpt["epoch"])
        self.batch_in_epoch = int(ckpt.get("batch_in_epoch", 0))
        self.best = dict(ckpt.get("best") or self.best)
        self.last_val = ckpt.get("val")
        rng = ckpt.get("rng") or {}
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available() and len(rng["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng["cuda"])
        self.logger.info("resumed from %s: epoch %d (+%d batches), step %d, best CER(av) %s", path, self.epoch,
                         self.batch_in_epoch, self.step,
                         "-" if not math.isfinite(self.best["cer"]) else f"{100 * self.best['cer']:.2f}%")

    # ---- training -----------------------------------------------------------------------------------------
    def _train_step(self, batch: dict, meters: dict[str, AverageMeter]) -> float | None:
        """One optimisation step; returns the gradient norm, or None when the step was skipped (non-finite)."""
        b = move_to_device(batch, self.device)
        with autocast_context(self.device, self.amp):
            out = self.model(b, mode="train")
            loss, parts = self.model.compute_loss(out, b, self.cfg)
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped_steps += 1
            self.logger.warning("non-finite loss at step %d (utts %s...): batch skipped", self.step,
                                batch["utt_ids"][:2])
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
        n = len(batch["utt_ids"])
        meters["loss"].update(float(loss.item()), n)
        for k, v in parts.items():
            if k != "loss":
                meters[k].update(float(v), n)
        return grad_norm

    def _handle_oom(self, batch: dict) -> None:
        """Skip a batch that ran out of GPU memory; give up after MAX_CONSECUTIVE_OOM in a row."""
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        self.skipped_steps += 1
        self.consecutive_oom += 1
        self.logger.warning("CUDA out of memory on a batch of %d utts / %d frames: skipped (lower "
                            "train.max_frames_per_batch if this repeats)", len(batch["utt_ids"]),
                            int(batch["lengths"].max()) * len(batch["utt_ids"]))
        if self.consecutive_oom >= MAX_CONSECUTIVE_OOM:
            raise RuntimeError(f"{MAX_CONSECUTIVE_OOM} consecutive CUDA OOM batches: lower train.max_frames_per_batch")

    def _fusion_probs_for(self, epoch: int) -> tuple[float, float, float]:
        """Modality-dropout probabilities (p_av, p_audio_only, p_video_only) for 0-based ``epoch``.

        ``train.fusion_curriculum`` is a list of stages ``{until_epoch: N, p_av, p_audio_only, p_video_only}``; the first
        stage with ``epoch + 1 <= until_epoch`` applies, otherwise ``model.fusion``. Starting without lips-only samples
        lets the encoder learn the CTC alignment from the audio first: from scratch, 25 % lips-only samples (almost
        unlearnable early on) pull every output towards the text prior and the CTC collapses to an input-independent
        output (observed: identical hypotheses for av/audio/video and a flat CTC loss for 5k steps)."""
        fc = self.cfg.model.fusion
        probs = (float(fc.p_av), float(fc.p_audio_only), float(fc.p_video_only))
        for stage in self.cfg.train.get("fusion_curriculum") or []:
            if epoch + 1 <= int(stage["until_epoch"]):
                probs = (float(stage["p_av"]), float(stage["p_audio_only"]), float(stage["p_video_only"]))
                break
        if min(probs) < 0 or sum(probs) <= 0:
            raise ValueError(f"invalid modality-dropout probabilities {probs} for epoch {epoch + 1}")
        return probs

    def _ensure_train_loader(self, epoch: int) -> None:
        """(Re)build the train DataLoader so its audio augmentation matches ``train.aug_start_epoch``."""
        want_aug = epoch + 1 >= self.aug_start_epoch
        if self.train_loader is not None and self._train_aug == want_aug:
            return
        if want_aug:
            ds = self._full_aug_ds
        else:
            clean_cfg = copy.deepcopy(self.cfg)
            clean_cfg["audio"]["noise_prob"] = 0.0
            for k in ("freq_mask", "time_mask"):
                clean_cfg["audio"]["specaug"][k] = 0
            ds = AVSRDataset(self._train_rows, clean_cfg, True, self.tok)
        self.train_loader = None  # release the previous loader (and its persistent workers) first
        self.train_loader = make_loader(ds, self.sampler, collate_fn, self._num_workers, self.device,
                                        persistent=want_aug)
        self._train_aug = want_aug
        self.logger.info("epoch %d: train audio augmentation %s (noise mixing + SpecAugment)", epoch + 1,
                         "ON" if want_aug else f"OFF until epoch {self.aug_start_epoch}")

    def _train_epoch(self, epoch: int, stop: GracefulStop) -> None:
        self._ensure_train_loader(epoch)
        probs = self._fusion_probs_for(epoch)
        if tuple(getattr(self.model, "fusion_probs", ())) != probs:
            self.logger.info("epoch %d: modality dropout (p_av, p_audio_only, p_video_only) = %s", epoch + 1, probs)
            self.model.fusion_probs = probs
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
        try:
            for batch in self.train_loader:
                window.data_wait += time.time() - t_mark
                window.frames += int(batch["lengths"].sum().item())
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
                        bar.set_postfix(loss=f"{meters['loss'].avg:.3f}", step=self.step)
                if gn is not None and self.step % self.log_every == 0:
                    self._log_train(epoch, meters, window, time.time() - t_log)
                    epoch_loss.update(meters["loss"].avg, meters["loss"].count)
                    meters = defaultdict(AverageMeter)
                    window, t_log = _Window(), time.time()
                if stop.requested:
                    raise KeyboardInterrupt
                t_mark = time.time()
        finally:
            if bar is not None:
                bar.close()
        if meters["loss"].count:
            epoch_loss.update(meters["loss"].avg, meters["loss"].count)
        self.logger.info("epoch %d/%d done in %s | mean loss %.4f | step %d%s", epoch + 1, self.epochs,
                         format_duration(time.time() - t_epoch), epoch_loss.avg, self.step,
                         f" | {self.skipped_steps} skipped non-finite steps so far" if self.skipped_steps else "")
        self.metrics.write({"type": "epoch", "epoch": epoch + 1, "step": self.step, "loss": epoch_loss.avg,
                            "time_sec": round(time.time() - t_epoch, 1), "skipped_steps": self.skipped_steps})

    def _log_train(self, epoch: int, meters: dict[str, AverageMeter], window: "_Window", elapsed: float) -> None:
        elapsed = max(elapsed, 1e-6)
        fps = window.frames / elapsed
        sec_per_step = elapsed / max(window.steps, 1)
        data_frac = min(1.0, window.data_wait / elapsed)
        eta_epoch = (self.steps_per_epoch - self.batch_in_epoch) * sec_per_step
        eta_total = (self.total_steps - self.step) * sec_per_step
        lr = self.optimizer.param_groups[0]["lr"]
        parts = " ".join(f"{k} {m.avg:.3f}" for k, m in meters.items() if k != "loss")
        mem = (f" | mem {torch.cuda.max_memory_allocated(self.device) / 2**30:.1f}G"
               if self.device.type == "cuda" else "")
        self.logger.info("ep %d/%d it %d/%d step %d | loss %.4f (%s) | lr %.2e | gnorm %.2f | %.0f fr/s (%.1fx RT, "
                         "data wait %.0f%%)%s | ETA epoch %s total %s", epoch + 1, self.epochs, self.batch_in_epoch,
                         self.steps_per_epoch, self.step, meters["loss"].avg, parts, lr, window.grad_norm.avg, fps,
                         fps / 25.0, 100 * data_frac, mem, format_duration(eta_epoch), format_duration(eta_total))
        self.metrics.write({"type": "train", "epoch": epoch + 1, "step": self.step, "lr": lr,
                            **{k: m.avg for k, m in meters.items()}, "grad_norm": window.grad_norm.avg,
                            "frames_per_sec": round(fps, 1), "sec_per_step": round(sec_per_step, 4),
                            "data_wait_frac": round(data_frac, 3)})

    # ---- validation ---------------------------------------------------------------------------------------
    def validate(self) -> dict[str, dict[str, float]]:
        """Greedy decode of the val subset: conditions on clean audio + audio conditions at VAL_NOISE_SNR."""
        t0 = time.time()
        clean = decode_loader(self.model, self.val_loaders["clean"], self.conditions, self.tok, self.device,
                              self.amp, self.method)
        scores = score(clean)
        noisy_tag = f"{self.noise_kind}{VAL_NOISE_SNR:g}dB"
        noisy: dict[str, Any] = {}
        if "noisy" in self.val_loaders:
            audio_conds = [c for c in self.conditions if c in AUDIO_MODES]
            noisy = decode_loader(self.model, self.val_loaders["noisy"], audio_conds, self.tok, self.device,
                                  self.amp, self.method)
            scores.update({f"{m}@{noisy_tag}": s for m, s in score(noisy).items()})

        def pct(x: float) -> str:
            return f"{100 * x:.2f}%" if math.isfinite(x) else "-"

        n = next(iter(scores.values()))["n"] if scores else 0
        summary = " | ".join(f"{k} CER {pct(s['cer'])} WER {pct(s['wer'])}" for k, s in scores.items())
        self.logger.info("val epoch %d (%d utts, %.0f s): %s", self.epoch, n, time.time() - t0, summary)
        av_n, au_n = scores.get(f"av@{noisy_tag}"), scores.get(f"audio@{noisy_tag}")
        if av_n and au_n:
            gain = 100 * (au_n["cer"] - av_n["cer"])
            self.logger.info("lips under noise (%s): CER audio %s -> av %s (%+.2f pt %s)", noisy_tag,
                             pct(au_n["cer"]), pct(av_n["cer"]), -gain, "better" if gain > 0 else "not better")
        base = next(iter(clean.values()))
        for i in range(min(N_EXAMPLES, len(base["refs"]))):
            lines = [f"  example {base['utt_ids'][i]}", f"    {'REF':<16}: {base['refs'][i]}"]
            lines += [f"    {m:<16}: {d['hyps'][i]}" for m, d in clean.items()]
            lines += [f"    {m + '@' + noisy_tag:<16}: {d['hyps'][i]}" for m, d in noisy.items()]
            self.logger.info("\n".join(lines))
        self.metrics.write({"type": "val", "epoch": self.epoch, "step": self.step, "n_utts": n,
                            **{f"{k}/cer": s["cer"] for k, s in scores.items()},
                            **{f"{k}/wer": s["wer"] for k, s in scores.items()}})
        return scores

    def _criterion(self, scores: dict[str, dict[str, float]]) -> float:
        key = "av" if "av" in scores else next(iter(scores))
        return float(scores[key]["cer"])

    # ---- main loop ----------------------------------------------------------------------------------------
    def run(self) -> int:
        if self.epoch >= self.epochs:
            self.logger.info("already trained %d/%d epochs (raise train.epochs to continue)", self.epoch, self.epochs)
            return 0
        self.metrics.write({"type": "start", "epoch": self.epoch, "step": self.step, "total_steps": self.total_steps,
                            "steps_per_epoch": self.steps_per_epoch})
        stop = GracefulStop(self.logger)
        stop.install()
        try:
            for epoch in range(self.epoch, self.epochs):
                self.epoch = epoch
                self._train_epoch(epoch, stop)
                self.epoch, self.batch_in_epoch = epoch + 1, 0
                is_best = False
                if self.val_loaders and (self.epoch % self.eval_every == 0 or self.epoch == self.epochs):
                    scores = self.validate()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()  # drop inference-shaped cache blocks before training resumes
                    self.last_val = scores
                    crit = self._criterion(scores)
                    if math.isfinite(crit) and crit < self.best["cer"]:
                        self.best = {"cer": crit, "epoch": self.epoch, "step": self.step}
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
            # a crash (data error, repeated OOM, failed validation...): keep the progress made, then re-raise
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
        if math.isfinite(self.best["cer"]):
            self.logger.info("training finished: best val CER(av) %.2f%% at epoch %s (%s)", 100 * self.best["cer"],
                             self.best["epoch"], self.ckpt_dir / "best.pt")
        else:
            self.logger.info("training finished (no validation score; last checkpoint %s)", self.ckpt_dir / "last.pt")
        return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the Korean AVSR model.")
    p.add_argument("--config", default="configs/base.yaml", help="YAML config")
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
    logger = get_logger(Path(str(cfg.work_dir)), "train")
    try:
        logger.info("config %s overrides %s%s", args.config, overrides,
                    f" limit {args.limit}" if args.limit else "")
        cfg_dump = Path(str(cfg.work_dir)) / "logs" / "train_config.yaml"
        with open(cfg_dump, "w", encoding="utf-8") as f:
            yaml.safe_dump(to_dict(cfg), f, allow_unicode=True, sort_keys=False)
        return Trainer(cfg, args.limit, logger).run()
    finally:
        close_logger()


if __name__ == "__main__":
    sys.exit(main())

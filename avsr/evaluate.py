"""Evaluate a checkpoint: CER/WER per condition (av / audio / video) x SNR sweep -> suggested SNR threshold.

Usage (from the project root)::

    & $py -m avsr.evaluate --config configs/base.yaml --ckpt work/checkpoints/best.pt --split test
    & $py -m avsr.evaluate --ckpt work/checkpoints/best.pt --split val --max-utts 300 --set eval.snr_sweep=[10,0]

The audio of every utterance is corrupted with ``eval.noise_kind`` at each SNR of ``eval.snr_sweep``; conditions
``av`` and ``audio`` are decoded at every SNR (and on clean audio), ``video`` once (it does not see the audio).
Suggested threshold (for ``infer.snr_threshold``) = highest SNR at which CER(av) > CER(video) (SPEC section 11).
Results are printed as a table and written to ``<work_dir>/eval/{split}_results.json``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from avsr.dataset import AVSRDataset, DurationBatchSampler, assign_split, collate_fn, load_manifests
from avsr.models import build_model
from avsr.text import Tokenizer
from avsr.utils import (Config, autocast_context, cer, close_logger, even_subset, get_logger, load_checkpoint,
                        load_config, make_loader, move_to_device, safe_console, set_seed, wer)

# Config sections that define the model input/architecture: always taken from the checkpoint.
PROTECTED_SECTIONS = ("model", "video", "audio")
AUDIO_MODES = ("av", "audio")
ALL_MODES = ("av", "audio", "video")

Decoded = dict[str, list[str]]  # keys utt_ids / refs / hyps


# --------------------------------------------------------------------------------------------------------------
# Shared helpers (also used by avsr.train)
# --------------------------------------------------------------------------------------------------------------


def parse_conditions(value: Any) -> list[str]:
    """``eval.conditions`` as a de-duplicated list; unknown names raise instead of being dropped silently."""
    items = value if isinstance(value, (list, tuple)) else [value]
    conds = list(dict.fromkeys(str(c) for c in items if c is not None))
    bad = [c for c in conds if c not in ALL_MODES]
    if bad or not conds:
        raise ValueError(f"eval.conditions={value!r}: expected a non-empty subset of {list(ALL_MODES)}")
    return conds


def build_eval_loader(rows: Sequence[dict], cfg: Config, tok: Tokenizer, fixed_noise: tuple[str, float] | None,
                      num_workers: int, device: torch.device, persistent: bool = False,
                      babble_rows: Sequence[dict] | None = None) -> torch.utils.data.DataLoader:
    """Eval DataLoader (no augmentation, deterministic duration batches); ``fixed_noise=(kind, snr_db)`` or clean.

    ``babble_rows``: utterances whose audio forms the babble noise (pass the TRAIN rows so val/test babble consists of
    other speakers, as in training; default = the eval rows themselves)."""
    kwargs: dict[str, Any] = {"fixed_noise": fixed_noise} if fixed_noise is not None else {}
    if fixed_noise is not None and babble_rows:
        kwargs["babble_rows"] = list(babble_rows)
    ds = AVSRDataset(list(rows), cfg, False, tok, **kwargs)
    sampler = DurationBatchSampler(list(getattr(ds, "rows", rows)), max_frames=int(cfg.train.max_frames_per_batch),
                                   shuffle=False, seed=int(cfg.seed))
    return make_loader(ds, sampler, collate_fn, num_workers, device, persistent)


#: SNR-head bucket midpoints (dB); must equal avsr.infer.SNR_BUCKET_MID (buckets: >=20, 10-20, 0-10, <0)
SNR_BUCKET_MID = (25.0, 15.0, 5.0, -5.0)


@torch.inference_mode()
def decode_loader(model: torch.nn.Module, loader: torch.utils.data.DataLoader, modes: Sequence[str],
                  tok: Tokenizer, device: torch.device, amp: Any, method: str,
                  snr_estimates: list[tuple[float | None, float]] | None = None) -> dict[str, Decoded]:
    """Decode every batch of ``loader`` under each modality ``mode``; returns ``{mode: {utt_ids, refs, hyps}}``.

    References are ``tokenizer.decode(tokenizer.encode(text))`` so refs and hyps live in the same text space.
    ``snr_estimates``: if a list is given, one ``(dsp_estimate, model_estimate)`` pair per utterance is appended —
    the same two SNR estimates inference uses for its modality gate.
    """
    was_training = model.training
    model.eval()
    out: dict[str, Decoded] = {m: {"utt_ids": [], "refs": [], "hyps": []} for m in modes}
    mids = torch.tensor(SNR_BUCKET_MID)
    try:
        for batch in loader:
            refs = [tok.decode(tok.encode(t)) for t in batch["texts"]]
            gpu_batch = move_to_device(batch, device)
            max_len = int(gpu_batch["lengths"].max().item())
            for mode in modes:
                with autocast_context(device, amp):
                    hyp_ids = model.decode(gpu_batch, mode, method=method, max_len=max_len)
                out[mode]["utt_ids"].extend(batch["utt_ids"])
                out[mode]["refs"].extend(refs)
                out[mode]["hyps"].extend(tok.decode(h) for h in hyp_ids)
            if snr_estimates is not None:
                with autocast_context(device, amp):
                    logits = model(gpu_batch, mode="audio")["snr_logits"]
                model_est = mids[logits.float().argmax(dim=-1).cpu()].tolist()
                for meta, m_est in zip(batch["metas"], model_est):
                    snr_estimates.append((meta.get("snr_dsp"), float(m_est)))
    finally:
        model.train(was_training)
    return out


def summarize_estimates(pairs: Sequence[tuple[float | None, float]]) -> dict[str, float | None]:
    """Median of the dsp / model / hybrid (= mean of both, as in infer) SNR estimates over utterances."""
    dsp = [float(d) for d, _ in pairs if d is not None]
    mdl = [float(m) for _, m in pairs]
    hyb = [(float(d) + float(m)) / 2.0 if d is not None else float(m) for d, m in pairs]
    med = lambda v: float(np.median(v)) if v else None  # noqa: E731
    return {"dsp": med(dsp), "model": med(mdl), "hybrid": med(hyb), "n": len(pairs)}


def threshold_in_estimator_units(true_thr: float | None, status: str, levels: Sequence[str],
                                 est: Mapping[str, Mapping[str, float | None]]) -> dict[str, float | None]:
    """Translate the true-SNR switching point into each estimator's units (what ``infer --snr-threshold`` compares
    against). ``levels`` = row keys from cleanest to noisiest ('clean', '20', '10', ...).

    ok                  : midway between the estimate at the threshold level and at the next cleaner level
    video_never_better  : 2 dB below the estimate at the noisiest level (the gate practically never switches)
    video_always_better : 1 dB above the clean estimate (always lips)"""
    out: dict[str, float | None] = {}
    for name in ("hybrid", "dsp", "model"):
        vals = {k: est.get(k, {}).get(name) for k in levels}
        if any(v is None for v in vals.values()):
            out[name] = None
            continue
        if status == "ok" and true_thr is not None:
            key = f"{true_thr:g}"
            i = list(levels).index(key)
            out[name] = round((vals[key] + vals[levels[i - 1]]) / 2.0, 2) if i > 0 else round(vals[key], 2)
        elif status == "video_never_better":
            out[name] = round(vals[levels[-1]] - 2.0, 2)
        elif status == "video_always_better":
            out[name] = round(vals[levels[0]] + 1.0, 2)
        else:
            out[name] = None
    return out


def score(decoded: Mapping[str, Decoded]) -> dict[str, dict[str, float]]:
    """``{mode: {cer, wer, n}}`` from :func:`decode_loader` output."""
    return {m: {"cer": cer(d["refs"], d["hyps"]), "wer": wer(d["refs"], d["hyps"]), "n": len(d["refs"])}
            for m, d in decoded.items()}


def suggest_threshold(av_cer: Mapping[float, float], video_cer: float,
                      av_clean_cer: float | None = None) -> tuple[float | None, str, str]:
    """Highest SNR where CER(av) > CER(video). Returns (threshold or None, status, human message).

    status: ``ok`` | ``video_never_better`` | ``video_always_better``.
    """
    snrs = sorted(av_cer, reverse=True)
    worse = [s for s in snrs if av_cer[s] > video_cer]
    if not snrs:
        return None, "no_data", "no SNR points evaluated"
    if not worse:
        return (None, "video_never_better",
                f"video-only never beats AV in the tested range (down to {snrs[-1]:g} dB): keep AV at all SNRs "
                f"(set infer.snr_threshold below {snrs[-1]:g}, e.g. {snrs[-1] - 5:g}).")
    thr = worse[0]
    if thr == snrs[0]:
        clean_note = ""
        if av_clean_cer is not None and av_clean_cer > video_cer:
            clean_note = " and even on clean audio"
        return (thr, "video_always_better",
                f"video-only beats AV already at the highest tested SNR ({thr:g} dB){clean_note}: the audio stream "
                f"does not help (under-trained model or bad audio?). Suggested threshold >= {thr:g} dB.")
    return thr, "ok", f"suggested infer.snr_threshold = {thr:g} dB (highest SNR where CER(av) > CER(video))."


def _pct(x: float | None) -> str:
    return "   -  " if x is None or not math.isfinite(x) else f"{100.0 * x:5.1f}%"


def format_table(results: Mapping[str, Mapping[str, Mapping[str, float]]], row_keys: Sequence[str],
                 conditions: Sequence[str], video: Mapping[str, float] | None) -> str:
    """Plain-text table: rows = SNR (clean first), columns = conditions (CER / WER) + AV-vs-audio verdict."""
    head = f"{'SNR':>8} | " + " | ".join(f"{c + ' CER/WER':^15}" for c in conditions)
    if "av" in conditions and "audio" in conditions:
        head += " | AV beats audio (CER gain)"
    lines = [head, "-" * len(head)]
    for key in row_keys:
        row = results[key]
        label = "clean" if key == "clean" else f"{float(key):g} dB"
        cells = []
        for c in conditions:
            m = video if (c == "video" and video is not None) else row.get(c)
            cells.append(f"{_pct(m['cer'] if m else None)} / {_pct(m['wer'] if m else None)}".center(15))
        line = f"{label:>8} | " + " | ".join(cells)
        if "av" in row and "audio" in row:
            d = row["audio"]["cer"] - row["av"]["cer"]
            line += f" | {'yes' if d > 0 else 'no':>3} ({100 * d:+.1f} pt)"
        lines.append(line)
    if video is not None and "video" in conditions:
        lines.append("(video is decoded once; it does not depend on the audio SNR)")
    return "\n".join(lines)


def eval_config(config_path: str | Path | None, overrides: Sequence[str], ckpt_cfg: Mapping[str, Any] | None) -> Config:
    """CLI config with the checkpoint's model/video/audio sections (architecture + input features), then --set."""
    base = load_config(config_path) if config_path is not None else Config()
    for sec in PROTECTED_SECTIONS:
        if ckpt_cfg and sec in ckpt_cfg:
            base[sec] = ckpt_cfg[sec]
    return load_config(None, overrides, base=base)


# --------------------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an AVSR checkpoint over conditions x SNR sweep.")
    p.add_argument("--config", default="configs/base.yaml", help="YAML config (model/video/audio come from the ckpt)")
    p.add_argument("--ckpt", required=True, help="checkpoint path (e.g. work/checkpoints/best.pt)")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    p.add_argument("--max-utts", type=int, default=None, help="evaluate a fixed evenly-spread subset of N utterances")
    p.add_argument("--work-dir", default=None, help="override work_dir (features, manifests, eval output)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    safe_console()
    args = parse_args(argv)
    t0 = time.time()
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    overrides = list(args.set) + ([f"work_dir={json.dumps(str(args.work_dir))}"] if args.work_dir else [])
    cfg = eval_config(args.config, overrides, ckpt.get("cfg"))
    set_seed(int(cfg.get("seed", 0)))
    work_dir = Path(str(cfg.work_dir))
    logger = get_logger(work_dir, "evaluate")
    try:
        tok = Tokenizer()
        vocab = ckpt.get("vocab")
        if vocab is not None and list(vocab) != list(tok.tokens):
            raise ValueError("checkpoint vocabulary differs from avsr.text.Tokenizer; cannot evaluate")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg, tok.vocab_size)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        logger.info("loaded %s (epoch %s, step %s) on %s", args.ckpt, ckpt.get("epoch", "?"), ckpt.get("step", "?"),
                    device)

        splits = assign_split(load_manifests(work_dir), cfg)
        rows = even_subset(splits.get(args.split, []), args.max_utts)
        babble_rows = splits.get("train") or None  # babble from training speakers, as during training
        if not rows:
            logger.error("split %r is empty (work_dir=%s, split config=%s): nothing to evaluate", args.split,
                         work_dir, dict(cfg.split))
            return 1
        ev = cfg.eval
        conditions = parse_conditions(ev.get("conditions", list(ALL_MODES)))
        audio_conds = [c for c in conditions if c in AUDIO_MODES]
        method = str(ev.get("decode", "ctc_greedy"))
        kind = str(ev.get("noise_kind", "babble"))
        sweep = ev.get("snr_sweep") or []
        sweep = sweep if isinstance(sweep, (list, tuple)) else [sweep]
        # the sweep only concerns conditions that hear the audio; highest SNR first, duplicates removed
        snrs = sorted({float(s) for s in sweep}, reverse=True) if audio_conds else []
        if sweep and not audio_conds:
            logger.info("eval.snr_sweep ignored: conditions %s do not use the audio", conditions)
        amp = cfg.train.get("amp", "bf16")
        nw = int(cfg.data.get("num_workers", 0))
        logger.info("split=%s utts=%d conditions=%s decode=%s noise=%s snr_sweep=%s", args.split, len(rows),
                    conditions, method, kind, snrs)

        results: dict[str, dict[str, dict[str, float]]] = {}
        examples: dict[str, Decoded] = {}
        t = time.time()
        estimates: dict[str, dict[str, float | None]] = {}
        pairs: list[tuple[float | None, float]] = []
        clean = decode_loader(model, build_eval_loader(rows, cfg, tok, None, nw, device), conditions, tok, device,
                              amp, method, snr_estimates=pairs)
        results["clean"] = score(clean)
        estimates["clean"] = summarize_estimates(pairs)
        examples.update(clean)
        logger.info("clean: %s (%.0f s)", _fmt_scores(results["clean"]), time.time() - t)
        for snr in snrs:
            t = time.time()
            pairs = []
            noisy = decode_loader(model, build_eval_loader(rows, cfg, tok, (kind, snr), nw, device,
                                                           babble_rows=babble_rows), audio_conds,
                                  tok, device, amp, method, snr_estimates=pairs)
            results[f"{snr:g}"] = score(noisy)
            estimates[f"{snr:g}"] = summarize_estimates(pairs)
            if "av" in noisy:
                examples[f"av@{snr:g}dB"] = noisy["av"]
            logger.info("%g dB %s: %s (%.0f s)", snr, kind, _fmt_scores(results[f"{snr:g}"]), time.time() - t)
        logger.info("SNR estimates inference would see (median dsp / model / hybrid):\n%s",
                    "\n".join(f"{k:>8}: {_db(v['dsp'])} / {_db(v['model'])} / {_db(v['hybrid'])}"
                              for k, v in estimates.items()))

        video = results["clean"].get("video")
        row_keys = ["clean"] + [f"{s:g}" for s in snrs]
        table = format_table(results, row_keys, conditions, video)
        logger.info("results (%s, %d utts, noise=%s, decode=%s):\n%s", args.split, len(rows), kind, method, table)

        suggestion: dict[str, Any] = {"threshold": None, "status": "not_applicable",
                                      "message": "needs both 'av' and 'video' conditions and a non-empty snr_sweep"}
        if video is not None and "av" in conditions and snrs:
            av_by_snr = {s: results[f"{s:g}"]["av"]["cer"] for s in snrs}
            thr, status, msg = suggest_threshold(av_by_snr, video["cer"], results["clean"]["av"]["cer"])
            infer_thr = threshold_in_estimator_units(thr, status, row_keys, estimates)
            suggestion = {"threshold": thr, "status": status, "message": msg, "infer_threshold": infer_thr}
        logger.info("SNR threshold: %s", suggestion["message"])
        if suggestion.get("infer_threshold"):
            it = suggestion["infer_threshold"]
            logger.info("-> in estimator units (what infer compares): hybrid %s, dsp %s, model %s  "
                        "(e.g. infer --snr-threshold %s with infer.snr_estimator=hybrid)",
                        it.get("hybrid"), it.get("dsp"), it.get("model"), it.get("hybrid"))
        av_beats_audio = {k: bool(r["av"]["cer"] < r["audio"]["cer"]) for k, r in results.items()
                          if "av" in r and "audio" in r}
        if av_beats_audio:
            logger.info("AV beats audio-only (CER): %s",
                        ", ".join(f"{k if k == 'clean' else k + ' dB'}={'yes' if v else 'no'}"
                                  for k, v in av_beats_audio.items()))

        out = {
            "split": args.split, "ckpt": str(args.ckpt), "epoch": ckpt.get("epoch"), "step": ckpt.get("step"),
            "n_utts": len(rows), "decode": method, "noise_kind": kind, "snr_sweep": snrs, "conditions": conditions,
            "results": results, "video": video, "suggested_snr_threshold": suggestion, "snr_estimates": estimates,
            "av_beats_audio": av_beats_audio, "examples": _examples(examples, n=10),
            "elapsed_sec": round(time.time() - t0, 1),
        }
        out_path = work_dir / "eval" / f"{args.split}_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(_finite(out), f, ensure_ascii=False, indent=2)
        logger.info("wrote %s (%.0f s total)", out_path, time.time() - t0)
        return 0
    finally:
        close_logger()


def _db(v: float | None) -> str:
    return "  -  " if v is None else f"{v:5.1f}"


def _fmt_scores(scores: Mapping[str, Mapping[str, float]]) -> str:
    return "  ".join(f"{m}: CER {_pct(s['cer'])} WER {_pct(s['wer'])}" for m, s in scores.items())


def _examples(examples: Mapping[str, Decoded], n: int) -> list[dict[str, str]]:
    """First ``n`` utterances with the reference + the hypothesis of every collected decode (aligned by utt_id)."""
    if not examples:
        return []
    base = next(iter(examples.values()))
    index = {name: {u: j for j, u in enumerate(dec["utt_ids"])} for name, dec in examples.items()}
    out = []
    for i, uid in enumerate(base["utt_ids"][:n]):
        item = {"utt_id": uid, "ref": base["refs"][i]}
        for name, dec in examples.items():
            j = index[name].get(uid)
            if j is not None:
                item[name] = dec["hyps"][j]
        out.append(item)
    return out


def _finite(obj: Any) -> Any:
    """Replace NaN/inf floats by None so the JSON stays standard."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Mapping):
        return {str(k): _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    return obj


if __name__ == "__main__":
    sys.exit(main())

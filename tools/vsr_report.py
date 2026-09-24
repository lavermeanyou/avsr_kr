"""Lip-reading (video-only) diagnostic report for a checkpoint (read-only; writes a JSON next to the logs).

- CER per speaker x camera angle (val + test speakers), CTC greedy vs attention greedy
- jamo-level analysis from an edit-distance alignment of reference vs hypothesis token ids:
  accuracy per jamo class (bilabial / alveolar / velar ... initial consonants, rounded / open / spread vowels, finals),
  most frequent substitutions, deletion / insertion rates
Usage: $py -m tools.vsr_report --ckpt work/checkpoints/best.pt [--per-cell 40] [--mode video] [--out work/eval/vsr_report.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from avsr.dataset import assign_split, load_manifests
from avsr.evaluate import build_eval_loader, eval_config
from avsr.models import build_model
from avsr.text import Tokenizer
from avsr.utils import autocast_context, cer, load_checkpoint, move_to_device

L_CHARS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
V_CHARS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
# visual classes (place of articulation for consonants, lip shape for vowels)
ONSET_CLASS = {**{c: "onset bilabial ㅁㅂㅃㅍ" for c in "ㅁㅂㅃㅍ"},
               **{c: "onset alveolar ㄴㄷㄸㄹㅌ" for c in "ㄴㄷㄸㄹㅌ"},
               **{c: "onset sibilant ㅅㅆㅈㅉㅊ" for c in "ㅅㅆㅈㅉㅊ"},
               **{c: "onset velar ㄱㄲㅋ" for c in "ㄱㄲㅋ"},
               "ㅇ": "onset silent ㅇ", "ㅎ": "onset glottal ㅎ"}
VOWEL_CLASS = {**{c: "vowel rounded ㅗㅜㅛㅠㅘㅝㅚㅟㅙㅞ" for c in "ㅗㅜㅛㅠㅘㅝㅚㅟㅙㅞ"},
               **{c: "vowel open ㅏㅑㅓㅕ" for c in "ㅏㅑㅓㅕ"},
               **{c: "vowel spread ㅣㅔㅐㅖㅒㅡㅢ" for c in "ㅣㅔㅐㅖㅒㅡㅢ"}}


def token_class(tok: Tokenizer, i: int) -> str:
    name = tok.id_to_token(i)
    if name.startswith("L:"):
        return ONSET_CLASS.get(name[2:], "onset other")
    if name.startswith("V:"):
        return VOWEL_CLASS.get(name[2:], "vowel other")
    if name.startswith("T:"):
        return "final consonant"
    if i == tok.space_id:
        return "space"
    return "punct/other"


def align(ref: list[int], hyp: list[int]) -> list[tuple[int | None, int | None]]:
    """Levenshtein alignment -> list of (ref_id | None, hyp_id | None) pairs."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            out.append((ref[i - 1], hyp[j - 1])); i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            out.append((ref[i - 1], None)); i -= 1
        else:
            out.append((None, hyp[j - 1])); j -= 1
    return out[::-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="work/checkpoints/best.pt")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--mode", default="video")
    ap.add_argument("--per-cell", type=int, default=40, help="utterances per speaker x angle")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default="work/eval/vsr_report.json")
    args = ap.parse_args()
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    cfg = eval_config(args.config, args.set, ckpt.get("cfg"))
    tok = Tokenizer()
    dev = torch.device("cuda")
    model = build_model(cfg, tok.vocab_size)
    model.load_state_dict(ckpt["model"])
    model.to(dev).eval()
    splits = assign_split(load_manifests(cfg.work_dir), cfg)
    cells: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for split in ("val", "test"):
        for r in sorted(splits.get(split, []), key=lambda r: r["utt_id"]):
            cells[(split, r["speaker"], r["angle"])].append(r)
    rows = []
    for key, rs in sorted(cells.items()):
        step = max(1, len(rs) // args.per_cell)
        rows += rs[::step][: args.per_cell]
    loader = build_eval_loader(rows, cfg, tok, None, 6, dev)
    res = defaultdict(lambda: {"refs": [], "ctc": [], "att": []})
    pairs_ctc: list[tuple[list[int], list[int]]] = []
    with torch.inference_mode():
        for batch in loader:
            g = move_to_device(batch, dev)
            with autocast_context(dev, cfg.train.get("amp", "bf16")):
                ctc_ids = model.decode(g, args.mode, method="ctc_greedy")
                att_ids = model.decode(g, args.mode, method="attn_greedy", max_len=int(g["lengths"].max()))
            toks = batch["tokens"]
            for i, meta in enumerate(batch["metas"]):
                ref_ids = toks[i, : int(batch["token_lengths"][i])].tolist()
                key = (meta["speaker"], meta["angle"])
                res[key]["refs"].append(tok.decode(ref_ids))
                res[key]["ctc"].append(tok.decode(ctc_ids[i]))
                res[key]["att"].append(tok.decode(att_ids[i]))
                pairs_ctc.append((ref_ids, ctc_ids[i]))
    speakers = sorted({k[0] for k in res})
    angles = sorted({k[1] for k in res})
    print(f"lip-reading CER (mode={args.mode}, CTC greedy / attention greedy), {len(rows)} utterances")
    print("speaker | " + " | ".join(f"{a:^13}" for a in angles) + " | all")
    report: dict = {"ckpt": args.ckpt, "mode": args.mode, "cells": {}}
    for s in speakers:
        line, allr, allc, alla = [], [], [], []
        for a in angles:
            d = res.get((s, a))
            if not d:
                line.append(f"{'-':^13}")
                continue
            c, t = cer(d["refs"], d["ctc"]), cer(d["refs"], d["att"])
            report["cells"][f"{s}/{a}"] = {"n": len(d["refs"]), "cer_ctc": c, "cer_att": t}
            line.append(f"{100 * c:5.1f}/{100 * t:5.1f}")
            allr += d["refs"]; allc += d["ctc"]; alla += d["att"]
        print(f"{s:7s} | " + " | ".join(line) + f" | {100 * cer(allr, allc):5.1f}/{100 * cer(allr, alla):5.1f}")
    for a in angles:
        r = [x for s in speakers for x in res.get((s, a), {"refs": []})["refs"]]
        c = [x for s in speakers for x in res.get((s, a), {"ctc": []})["ctc"]]
        report.setdefault("per_angle", {})[a] = cer(r, c) if r else None
    print("per angle (CTC):", {a: round(100 * v, 1) for a, v in report["per_angle"].items() if v is not None})
    # jamo analysis
    cls_tot, cls_ok, subs = Counter(), Counter(), Counter()
    n_del = n_ins = n_ref = 0
    for ref_ids, hyp_ids in pairs_ctc:
        for r, h in align(ref_ids, hyp_ids):
            if r is None:
                n_ins += 1
                continue
            n_ref += 1
            c = token_class(tok, r)
            cls_tot[c] += 1
            if h is None:
                n_del += 1
            elif h == r:
                cls_ok[c] += 1
            else:
                subs[(tok.id_to_token(r), tok.id_to_token(h))] += 1
    print(f"\njamo tokens: {n_ref}, deletion rate {100 * n_del / n_ref:.1f}%, insertion rate {100 * n_ins / n_ref:.1f}%")
    print("accuracy per visual class (correct / reference):")
    for c in sorted(cls_tot, key=lambda c: -cls_tot[c]):
        print(f"  {c:36s} {100 * cls_ok[c] / cls_tot[c]:5.1f}%  (n={cls_tot[c]})")
    print("top substitutions (ref -> hyp):", ", ".join(f"{a[2:] or a}->{b[2:] or b} {n}" for (a, b), n in subs.most_common(20)))
    report.update({"deletion_rate": n_del / n_ref, "insertion_rate": n_ins / n_ref,
                   "class_accuracy": {c: cls_ok[c] / cls_tot[c] for c in cls_tot},
                   "top_substitutions": [[a, b, n] for (a, b), n in subs.most_common(50)]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

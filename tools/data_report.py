"""Summarise preprocessed data (read-only): utterances / hours per split, speaker, angle; landmark coverage; shifts.

Usage: $py tools/data_report.py [--config configs/base.yaml] [--work-dir work]
Split rules mirror avsr.dataset.assign_split (val/test speakers from the config; has_unk rows dropped).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    work = Path(args.work_dir or cfg.get("work_dir", "work"))
    val = set(cfg["split"]["val_speakers"])
    test = set(cfg["split"]["test_speakers"])
    dmin, dmax = cfg["data"]["min_duration"], cfg["data"]["max_duration"]
    rows = []
    shards = sorted((work / "manifests").glob("*.jsonl"))
    for p in shards:
        if not (p.parent / (p.stem + ".done")).exists():
            continue
        rows += [json.loads(l) for l in open(p, encoding="utf-8")]
    print(f"videos done: {len(shards)}  utterances: {len(rows)}")
    if not rows:
        return
    split_of = lambda r: "val" if r["speaker"] in val else ("test" if r["speaker"] in test else "train")  # noqa: E731
    agg = defaultdict(lambda: Counter())
    for r in rows:
        s = split_of(r)
        keep = (not r["has_unk"]) and dmin <= r["duration"] <= dmax
        agg[s]["utts_all"] += 1
        agg[s]["utts"] += int(keep)
        agg[s]["sec"] += r["duration"] if keep else 0
        agg[s]["sec_unique_audio"] += r["duration"] if keep and r["angle"] == "A" else 0
    print("\nsplit   utts(kept/all)   hours(all angles)   hours(angle A = unique audio)")
    for s in ("train", "val", "test"):
        a = agg[s]
        print(f"{s:6s} {a['utts']:7d}/{a['utts_all']:<7d}   {a['sec'] / 3600:8.2f}            {a['sec_unique_audio'] / 3600:8.2f}")
    spk = defaultdict(lambda: Counter())
    for r in rows:
        spk[r["speaker"]][r["angle"]] += 1
    print("\nspeaker split  noise gender  angles(utts)")
    for s in sorted(spk):
        r0 = next(r for r in rows if r["speaker"] == s)
        print(f"{s:6s}  {split_of(r0):5s}  {r0['noise_env']}      {r0['gender']}     {dict(sorted(spk[s].items()))}")
    lv = [r["lm_valid_ratio"] for r in rows]
    print(f"\nlandmark valid ratio: mean {sum(lv) / len(lv):.4f}, utts < 0.9: {sum(v < 0.9 for v in lv)}")
    shifts = Counter()
    seen = set()
    for r in rows:
        if r["video_stem"] not in seen:
            seen.add(r["video_stem"])
            shifts[round(float(r.get("time_shift", 0.0)), 2)] += 1
    print("label time shift per video (s: count):", dict(sorted(shifts.items())))
    print("has_unk utterances:", sum(r["has_unk"] for r in rows))


if __name__ == "__main__":
    main()

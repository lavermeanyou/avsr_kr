"""Training progress at a glance (read-only): per-epoch validation CER table + latest train step.

Usage: $py tools/train_status.py [--work-dir work]
Columns: av / audio / video = clean validation CER; @0dB = with babble noise at 0 dB SNR.
"lips gain" = CER(audio@0dB) - CER(av@0dB): how much the lips help when the audio is noisy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    args = ap.parse_args()
    path = Path(args.work_dir) / "logs" / "metrics.jsonl"
    if not path.exists():
        print(f"no metrics yet: {path}")
        return
    vals, last_train = [], None
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == "val":
            vals.append(r)
        elif r.get("type") == "train":
            last_train = r
    pct = lambda v: "   -  " if v is None else f"{100 * v:5.1f}%"  # noqa: E731
    print("epoch |   av    audio   video | av@0dB audio@0dB | lips gain @0dB")
    print("-" * 66)
    best = None
    for r in vals:
        av, au, vi = r.get("av/cer"), r.get("audio/cer"), r.get("video/cer")
        av0, au0 = r.get("av@babble0dB/cer"), r.get("audio@babble0dB/cer")
        gain = f"{100 * (au0 - av0):+5.1f} pt" if av0 is not None and au0 is not None else "   -"
        print(f"{r['epoch']:5d} | {pct(av)} {pct(au)} {pct(vi)} | {pct(av0)} {pct(au0)}   | {gain}")
        if av is not None and (best is None or av < best[1]):
            best = (r["epoch"], av)
    if best:
        print(f"best clean av CER {100 * best[1]:.2f}% at epoch {best[0]}")
    if last_train:
        t = dt.datetime.fromtimestamp(last_train["time"]).strftime("%H:%M:%S")
        print(f"last train log {t}: epoch {last_train['epoch']} step {last_train['step']} loss {last_train['loss']:.1f} "
              f"(ctc {last_train.get('ctc', 0):.1f}) lr {last_train.get('lr', 0):.2e} "
              f"{last_train.get('frames_per_sec', 0):.0f} fr/s")


if __name__ == "__main__":
    main()

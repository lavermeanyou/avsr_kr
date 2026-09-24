"""Print the CTC-loss trend from one or more training console logs (read-only).

Usage: $py tools/ctc_trend.py work/logs/train.out.log [more logs ...] [--last 14]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PAT = re.compile(r"step (\d+) \| loss [\d.]+ \(ctc ([\d.]+).*?\| lr ([\d.e+-]+) \| gnorm ([\d.]+) \| (\d+) fr/s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--last", type=int, default=14)
    args = ap.parse_args()
    for log in args.logs:
        p = Path(log)
        print(f"=== {p.name}")
        if not p.exists():
            print("  (missing)")
            continue
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        rows = [m.groups() for m in (PAT.search(l) for l in lines) if m]
        for step, ctc, lr, gn, fps in rows[-args.last:]:
            print(f"  step {int(step):6d}  ctc {float(ctc):7.1f}  lr {float(lr):.2e}  gnorm {float(gn):7.1f}  {fps} fr/s")
        errs = [l for l in lines if "ERROR" in l or "OOM" in l or "out of memory" in l]
        for l in errs[-3:]:
            print("  !", l[-160:])


if __name__ == "__main__":
    main()

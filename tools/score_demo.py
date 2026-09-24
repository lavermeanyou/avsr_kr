"""Score demo subtitles (.json from avsr.infer) against the label sentences of the clip, per half (clean / noisy).

Usage: $py -m tools.score_demo work/demo/<stem>_60s_reference.json work/demo/out_auto.json [more .json ...]
CER is computed on the concatenated text of each half (subtitle segments are not sentence-aligned).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from avsr.text import Tokenizer
from avsr.utils import cer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("outputs", nargs="+")
    args = ap.parse_args()
    tok = Tokenizer()
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    split = float(ref["clean_until_sec"])
    norm = lambda s: tok.decode(tok.encode(s))  # noqa: E731
    ref_halves = [" ".join(norm(s["text"]) for s in ref["sentences"] if (s["start"] + s["end"]) / 2 < split),
                  " ".join(norm(s["text"]) for s in ref["sentences"] if (s["start"] + s["end"]) / 2 >= split)]
    print(f"{'output':28s} | clean half CER | noisy half CER ({ref['noise']}) | modes used")
    for path in args.outputs:
        out = json.loads(Path(path).read_text(encoding="utf-8"))
        segs = out["segments"]
        hyp_halves = [" ".join(s["text"] for s in segs if (s["start"] + s["end"]) / 2 < split),
                      " ".join(s["text"] for s in segs if (s["start"] + s["end"]) / 2 >= split)]
        modes = [sorted({s["mode"] for s in segs if ((s["start"] + s["end"]) / 2 < split) == first})
                 for first in (True, False)]
        c = [cer([r], [h]) for r, h in zip(ref_halves, hyp_halves)]
        print(f"{Path(path).name:28s} | {100 * c[0]:13.1f}% | {100 * c[1]:27.1f}% | clean {modes[0]} noisy {modes[1]}")


if __name__ == "__main__":
    main()

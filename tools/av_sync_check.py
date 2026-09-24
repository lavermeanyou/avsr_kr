"""Audio-visual sync check on preprocessed data (read-only).

For every finished video: cross-correlate the per-frame mouth opening (cue inner_height) with the audio loudness
envelope (dB, one value per video frame) over lags of +/- 12 frames, summed over all its utterances.
Reports the lag with maximal correlation per video.
  lag > 0  => audio is LATE relative to the lips by lag frames (audio envelope matches the lip signal from `lag` frames earlier)
Usage: $py tools/av_sync_check.py [--work-dir work] [--out work/logs/av_sync.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MAX_LAG = 12


def envelope(audio: np.ndarray, sr: int, fps: float, n: int) -> np.ndarray:
    hop = sr / fps
    x = audio.astype(np.float32)
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        a, b = int(round(i * hop)), int(round((i + 1) * hop))
        seg = x[a:b]
        out[i] = 10 * np.log10(float((seg * seg).mean()) + 1.0) if seg.size else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    work = Path(args.work_dir)
    shards = sorted(p for p in (work / "manifests").glob("*.jsonl") if (work / "manifests" / (p.stem + ".done")).exists())
    if args.max_videos:
        shards = shards[: args.max_videos]
    results = {}
    lags = np.arange(-MAX_LAG, MAX_LAG + 1)
    for shard in shards:
        rows = [json.loads(l) for l in open(shard, encoding="utf-8")]
        acc = np.zeros(len(lags))
        cnt = np.zeros(len(lags))
        for r in rows:
            d = np.load(work / r["npz"])
            valid = d["valid"].astype(bool)
            if valid.mean() < 0.9 or len(valid) < 3 * MAX_LAG:
                continue
            v = d["cue"][:, 1].astype(np.float32)
            a = envelope(d["audio"], int(d["sr"]), float(d["fps"]), len(v))
            a = np.maximum(a, np.percentile(a, 20))  # floor the silence so noise does not dominate
            k = np.ones(3, np.float32) / 3
            # smoothed first differences emphasise articulation events (opening/closing vs onsets/offsets)
            v = np.diff(np.convolve(v, k, mode="same"), prepend=v[0])
            a = np.diff(np.convolve(a, k, mode="same"), prepend=a[0])
            v = (v - v.mean()) / (v.std() + 1e-6)
            a = (a - a.mean()) / (a.std() + 1e-6)
            n = len(v)
            for j, L in enumerate(lags):
                # audio late by L frames: a[t + L] ~ v[t]
                if L >= 0:
                    x, y = v[: n - L], a[L:]
                else:
                    x, y = v[-L:], a[: n + L]
                acc[j] += float((x * y).sum())
                cnt[j] += len(x)
        if cnt.sum() == 0:
            continue
        corr = acc / np.maximum(cnt, 1)
        best = int(lags[int(np.argmax(corr))])
        r0 = rows[0]
        results[r0["video_stem"]] = {"speaker": r0["speaker"], "angle": r0["angle"], "fps": r0.get("fps"),
                                     "time_shift": r0.get("time_shift"), "best_lag_frames": best,
                                     "best_corr": round(float(corr.max()), 3), "corr_at_0": round(float(corr[MAX_LAG]), 3),
                                     "sharpness": round(float(corr.max() - np.median(corr)), 3)}
    by_spk = defaultdict(list)
    for k, v in results.items():
        by_spk[v["speaker"]].append(v["best_lag_frames"])
    print(f"videos analysed: {len(results)}")
    print("speaker  n  lag(frames) median [min..max]   lag_ms median")
    for spk, ls in sorted(by_spk.items()):
        fps = next(v["fps"] for v in results.values() if v["speaker"] == spk) or 30.0
        print(f"{spk:6s} {len(ls):3d}  {np.median(ls):+5.1f} [{min(ls):+d}..{max(ls):+d}]   {1000 * np.median(ls) / fps:+6.0f} ms")
    for k, v in sorted(results.items())[:60]:
        print(k, v)
    if args.out:
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

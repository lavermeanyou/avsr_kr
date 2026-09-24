"""Build a demo clip for the noise-gated subtitle feature (read-only on the dataset; writes into --out-dir).

Takes the first N seconds of a TEST-speaker video from the source tars, keeps the video, and writes an audio file whose
first half is the clean original and whose second half has babble noise (other speakers' utterances) mixed in at a low
SNR. Running infer with --audio <noisy.wav> should then use `av` in the clean half and switch to `video` (lips) in the
noisy half. Also writes the ground-truth sentences inside the clip (from the label) as reference.

Usage: $py -m tools.make_noise_demo --data-root "<dataset>" --stem lip_J_1_M_03_E014_A_001 --seconds 60 --snr -5
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from avsr.aihub import build_media_index, discover_sources, extract_media, iter_label_refs
from avsr.audio_feats import add_noise, load_audio_16k, make_noise
from avsr.dataset import assign_split, load_manifests
from avsr.utils import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--stem", default="lip_J_1_M_03_E014_A_001")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--snr", type=float, default=-5.0)
    ap.add_argument("--out-dir", default="work/demo")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sources = discover_sources(args.data_root)
    index = build_media_index(sources, cache_path=Path("work") / "media_index.json", log=lambda m: None)
    label = next(r for r in iter_label_refs(sources) if r.stem == args.stem).load()
    # cut right after the last sentence that ends within the requested length (no half sentence at the end) and start
    # the noise in the pause between sentences closest to the middle (no sentence straddles clean/noisy)
    sents = [s for s in label.sentences if s.end <= args.seconds]
    if len(sents) < 2:
        raise SystemExit(f"{args.stem}: fewer than 2 complete sentences in the first {args.seconds} s")
    clip_len = min(args.seconds, sents[-1].end + 0.3)
    gaps = [((a.end + b.start) / 2.0) for a, b in zip(sents[:-1], sents[1:])]
    noise_start = min(gaps, key=lambda g: abs(g - clip_len / 2.0))
    media = extract_media(index[f"{args.stem}.mp4"], out / "tmp")
    clip = out / f"{args.stem}_{int(args.seconds)}s.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(media), "-t", f"{clip_len:.3f}", "-c:v", "libx264",
                    "-crf", "18", "-preset", "veryfast", "-c:a", "aac", str(clip)], check=True)
    media.unlink()
    wave = load_audio_16k(str(clip)).astype(np.float32) / 32768.0
    half = int(round(noise_start * 16000))
    cfg = load_config("configs/base.yaml")
    train_rows = assign_split(load_manifests(cfg.work_dir), cfg)["train"]
    rng = np.random.default_rng(0)
    pool = []
    for i in rng.choice(len(train_rows), 12, replace=False):
        with np.load(Path(cfg.work_dir) / train_rows[int(i)]["npz"]) as z:
            pool.append(z["audio"])
    noise = make_noise("babble", len(wave) - half, rng, pool)
    noisy_tail, _ = add_noise(wave[half:], noise, args.snr)
    noisy = np.concatenate([wave[:half], noisy_tail]).astype(np.float32)
    noisy /= max(1.0, float(np.abs(noisy).max()) / 0.99)
    wav_path = out / f"{args.stem}_{int(args.seconds)}s_clean-then-babble{int(args.snr)}dB.wav"
    sf.write(str(wav_path), noisy, 16000, subtype="PCM_16")
    ref_sents = [{"start": s.start, "end": s.end, "text": s.text} for s in sents]
    (out / f"{args.stem}_{int(args.seconds)}s_reference.json").write_text(
        json.dumps({"clean_until_sec": half / 16000.0, "clip_sec": clip_len, "noise": f"babble {args.snr} dB",
                    "sentences": ref_sents}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"video: {clip}\naudio: {wav_path} (clean 0-{half / 16000:.1f}s, babble {args.snr} dB after)\n"
          f"reference: {len(sents)} sentences")


if __name__ == "__main__":
    main()

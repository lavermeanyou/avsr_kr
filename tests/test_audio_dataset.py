"""Tests for avsr.audio_feats and avsr.dataset. Run: $py tests\\test_audio_dataset.py (from the project root).

Synthetic checks always run (3 utterances written to a temp dir: mouth video, npz per SPEC section 4, manifest shard).
If real manifest shards exist under work/manifests, 3 real rows are also run through the dataset (read-only) and
50 __getitem__ calls are timed.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsr.audio_feats import (  # noqa: E402
    add_noise, compute_fbank, estimate_snr_db, load_audio_16k, make_noise, stack_frames,
)
from avsr.dataset import (  # noqa: E402
    AVSRDataset, DurationBatchSampler, assign_split, collate_fn, load_manifests, read_frames, resample_indices,
    resampled_length, snr_to_bucket,
)
from avsr.text import Tokenizer  # noqa: E402

SR = 16000
FPS = 30
TEXTS = ["안녕하세요 반갑습니다.", "오늘 날씨가 참 좋네요?", "음성 인식 모델을 학습합니다,"]
SPEAKERS = ["S001", "S002", "S003"]
N_FRAMES = [45, 61, 94]


# ----------------------------------------------------------------------------------------------------------------------
# synthetic data
# ----------------------------------------------------------------------------------------------------------------------
def base_cfg(work_dir: Path) -> dict:
    with open(ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["work_dir"] = str(work_dir)
    cfg["split"] = {"val_speakers": ["V001"], "test_speakers": ["T001"], "train_split_dirs": ["2.Validation"]}
    # the pixel-exact checks below use the global constants; utterance normalisation is tested separately
    cfg["video"]["norm"] = "global"
    cfg["video"]["cue_norm"] = "none"
    return cfg


def speech_like(n: int, rng: np.random.Generator, pause_every: float = 0.6) -> np.ndarray:
    """Harmonic 'voiced' signal with a 4 Hz syllabic envelope and silent pauses (float in [-1, 1])."""
    t = np.arange(n) / SR
    f0 = 120 + 20 * np.sin(2 * np.pi * 0.7 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voiced = sum(np.sin(k * phase) / k for k in range(1, 12))
    env = np.clip(np.sin(2 * np.pi * 4.0 * t), 0, None) ** 0.5
    env *= ((t % pause_every) < pause_every * 0.7)  # pauses
    x = 0.3 * voiced * env + 1e-4 * rng.standard_normal(n)
    return (x / np.max(np.abs(x)) * 0.5).astype(np.float32)


def write_video(path_mp4: Path, frames: np.ndarray) -> Path:
    """Write BGR frames; try mp4v, avc1 (.mp4) then MJPG (.avi). Returns the path that re-reads correctly."""
    for fourcc, path in (("mp4v", path_mp4), ("avc1", path_mp4), ("MJPG", path_mp4.with_suffix(".avi"))):
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), float(FPS), (frames.shape[2], frames.shape[1]))
        if not wr.isOpened():
            continue
        for f in frames:
            wr.write(f)
        wr.release()
        cap = cv2.VideoCapture(str(path))
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        if n == len(frames):
            return path
    raise RuntimeError("no working VideoWriter codec (mp4v/avc1/MJPG)")


def make_synthetic(work: Path) -> list[dict]:
    rng = np.random.default_rng(0)
    stem_rows = []
    for u, (spk, text, n_t) in enumerate(zip(SPEAKERS, TEXTS, N_FRAMES)):
        stem = f"lip_J_1_F_03_{spk}_A_001"
        utt_id = f"{stem}__{u + 1:03d}"
        out_dir = work / "feats" / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = rng.integers(0, 256, size=(n_t, 96, 96, 3), dtype=np.uint8)
        frames = cv2.GaussianBlur(frames.reshape(n_t * 96, 96, 3), (5, 5), 0).reshape(n_t, 96, 96, 3)
        vid = write_video(out_dir / f"{utt_id}.mp4", frames)
        n_samples = int(round(n_t / FPS * SR))
        audio = (speech_like(n_samples, rng) * 32767).astype(np.int16)
        valid = np.ones(n_t, np.uint8)
        valid[::7] = 0
        lm = rng.standard_normal((n_t, 40, 2)).astype(np.float16) * valid[:, None, None]
        cue = rng.random((n_t, 8)).astype(np.float16) * valid[:, None]
        np.savez_compressed(out_dir / f"{utt_id}.npz", lm=lm.astype(np.float16), cue=cue.astype(np.float16), valid=valid,
                            audio=audio, fps=np.float64(30.0), sr=np.int64(SR))
        stem_rows.append({
            "utt_id": utt_id, "video_stem": stem, "split_dir": "2.Validation", "speaker": spk, "gender": "F", "age": 3,
            "specificity": "C", "angle": "A", "session": "001", "noise_env": 1, "topic": "test", "sentence_id": u + 1,
            "start": 0.0, "end": n_t / FPS, "duration": n_t / FPS, "n_frames": n_t, "n_samples": n_samples,
            "text": text, "text_raw": text, "has_unk": False, "lm_valid_ratio": float(valid.mean()),
            "mouth_mp4": vid.relative_to(work).as_posix(), "npz": f"feats/{stem}/{utt_id}.npz",
        })
    (work / "manifests").mkdir(parents=True, exist_ok=True)
    with open(work / "manifests" / "synthetic.jsonl", "w", encoding="utf-8") as f:
        for r in stem_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return stem_rows


def decode_all(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return np.stack(out)


# ----------------------------------------------------------------------------------------------------------------------
# audio_feats
# ----------------------------------------------------------------------------------------------------------------------
def measured_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    c = clean.astype(np.float64)
    n = noisy.astype(np.float64) - c
    return 20 * math.log10(np.sqrt(np.mean(c ** 2)) / np.sqrt(np.mean(n ** 2)))


def test_audio_feats(tmp: Path) -> None:
    rng = np.random.default_rng(1)
    clean = speech_like(3 * SR, rng)
    clean_i16 = (clean * 32767).astype(np.int16)

    # fbank + CMVN
    fb = compute_fbank(clean_i16)
    assert fb.dtype == torch.float32 and fb.shape[1] == 80 and fb.shape[0] == 1 + (clean.size - 400) // 160, fb.shape
    assert fb.mean(0).abs().max() < 1e-3 and (fb.std(0, unbiased=False) - 1).abs().max() < 1e-2
    fb_f = compute_fbank(clean_i16.astype(np.float32) / 32768.0)
    fb_t = compute_fbank(torch.from_numpy(clean_i16))
    assert torch.allclose(fb, fb_f, atol=1e-3) and torch.allclose(fb, fb_t, atol=1e-5)
    st = stack_frames(fb, 4)
    assert st.shape == (fb.shape[0] // 4, 320) and torch.equal(st[1, 80:160], fb[5])
    assert compute_fbank(np.zeros(100, np.int16)).shape == (0, 80)

    # noise generation + exact SNR mixing
    pool = [(speech_like(SR * 2, np.random.default_rng(s)) * 20000).astype(np.int16) for s in (10, 11, 12)]
    for kind in ("white", "pink", "babble"):
        noise = make_noise(kind, clean.size, rng, pool if kind == "babble" else None)
        assert noise.dtype == np.float32 and noise.shape == clean.shape
        assert abs(float(np.sqrt(np.mean(noise.astype(np.float64) ** 2))) - 1.0) < 1e-3
        for snr in (-5.0, 0.0, 7.5, 20.0):
            noisy, snr_out = add_noise(clean, noise, snr)
            got = measured_snr(clean, noisy)
            assert noisy.dtype == np.float32 and snr_out == snr and abs(got - snr) < 0.5, (kind, snr, got)
        noisy_i, _ = add_noise(clean_i16, noise, 3.0)  # int16 input is scaled to [-1, 1]
        assert abs(measured_snr(clean_i16 / 32768.0, noisy_i) - 3.0) < 0.5
    # pink: power spectrum slope ~ -1 in log-log (1/f)
    pink = make_noise("pink", 2 ** 18, np.random.default_rng(3))
    spec = np.abs(np.fft.rfft(pink)) ** 2
    freqs = np.fft.rfftfreq(pink.size, 1 / SR)
    sel = (freqs > 50) & (freqs < 6000)
    slope = np.polyfit(np.log10(freqs[sel]), np.log10(spec[sel]), 1)[0]
    assert -1.2 < slope < -0.8, slope

    # DSP SNR estimate ordering
    white = make_noise("white", clean.size, rng)
    est_clean = estimate_snr_db(clean)
    est_10 = estimate_snr_db(add_noise(clean, white, 10.0)[0])
    est_0 = estimate_snr_db(add_noise(clean, white, 0.0)[0])
    est_m5 = estimate_snr_db(add_noise(clean, white, -5.0)[0])
    assert est_clean > est_10 > est_0 >= est_m5, (est_clean, est_10, est_0, est_m5)
    assert est_clean > 30 and est_0 < 15, (est_clean, est_0)
    assert -10.0 <= estimate_snr_db(white) <= 40.0 and estimate_snr_db(np.zeros(SR, np.float32)) <= 40.0
    assert abs(estimate_snr_db(clean_i16) - est_clean) < 0.1  # int16 input is scaled like float

    # ffmpeg loading with accurate output-side seeking (48 kHz stereo wav -> 16 kHz mono int16)
    t = np.arange(48000 * 3) / 48000
    tone = 0.3 * np.sin(2 * np.pi * 440 * t)
    wav = tmp / "tone48k.wav"
    sf.write(str(wav), np.stack([tone, tone], 1), 48000, subtype="PCM_16")
    full = load_audio_16k(str(wav))
    assert full.dtype == np.int16 and abs(full.size - 3 * SR) <= 2, full.size
    part = load_audio_16k(str(wav), start=0.5, end=1.75)
    assert abs(part.size - int(1.25 * SR)) <= 2, part.size
    assert abs(np.sqrt(np.mean((part / 32768.0) ** 2)) - 0.3 / np.sqrt(2)) < 0.01
    print(f"audio_feats: OK (DSP SNR est clean {est_clean:.1f} dB, 10 dB -> {est_10:.1f}, 0 dB -> {est_0:.1f}, "
          f"-5 dB -> {est_m5:.1f}; pink slope {slope:.2f})")


# ----------------------------------------------------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------------------------------------------------
def check_item(item: dict, n_t: int, channels: int, tok: Tokenizer, text: str) -> int:
    n_v = resampled_length(n_t)
    n = item["video"].shape[0]
    assert item["video"].shape == (n, channels, 88, 88) and item["video"].dtype == torch.float32
    assert item["lm"].shape == (n, 80) and item["cue"].shape == (n, 8) and item["valid"].shape == (n,)
    assert item["audio"].shape == (n, 320) and item["audio"].dtype == torch.float32
    assert all(item[k].dtype == torch.float32 for k in ("lm", "cue", "valid"))
    assert n_v - 2 <= n <= n_v, (n, n_v)
    assert item["snr_bucket"].dtype == torch.long and item["snr_bucket"].ndim == 0
    assert item["tokens"].dtype == torch.long and item["tokens"].tolist() == tok.encode(text)
    assert item["text"] == text and set(item["meta"]) >= {"speaker", "angle", "noise_env"}
    assert torch.isfinite(item["video"]).all() and torch.isfinite(item["audio"]).all()
    return n


def test_dataset(work: Path, rows_written: list[dict]) -> None:
    tok = Tokenizer()
    cfg = base_cfg(work)
    rows = load_manifests(work)
    assert [r["utt_id"] for r in rows] == [r["utt_id"] for r in rows_written]
    assert all(Path(r["work_dir"]) == work.resolve() for r in rows)

    # splits
    extra = [dict(rows[0], utt_id="unk", has_unk=True), dict(rows[0], utt_id="v", speaker="V001"),
             dict(rows[0], utt_id="t", speaker="T001"), dict(rows[0], utt_id="long", duration=30.0),
             dict(rows[0], utt_id="ang", angle="C"), dict(rows[0], utt_id="other", split_dir="1.Training")]
    sp = assign_split(rows + extra, cfg)
    assert [r["utt_id"] for r in sp["train"]] == [r["utt_id"] for r in rows] + ["ang"]
    assert [r["utt_id"] for r in sp["val"]] == ["v"] and [r["utt_id"] for r in sp["test"]] == ["t"]
    cfg_ang = copy.deepcopy(cfg)
    cfg_ang["data"]["angles"] = ["A"]
    assert "ang" not in [r["utt_id"] for r in assign_split(rows + extra, cfg_ang)["train"]]

    # resampling helpers
    assert resampled_length(240) == 200 and resampled_length(5) == 5 and resampled_length(1) == 1
    idx = resample_indices(240)
    assert idx[0] == 0 and idx[1] == 1 and idx[5] == 6 and idx[-1] == 239 and np.all(np.diff(idx) >= 1)
    assert resample_indices(5).max() == 4
    assert [snr_to_bucket(s) for s in (None, 25, 20, 19.9, 10, 5, 0, -0.1, -5)] == [0, 0, 0, 1, 1, 2, 2, 3, 3]

    # eval mode: shapes, determinism, exact frame/lm selection
    ev = AVSRDataset(rows, cfg, train=False, tokenizer=tok)
    for i, r in enumerate(rows):
        a, b = ev[i], ev[i]
        n = check_item(a, r["n_frames"], 1, tok, r["text"])
        assert int(a["snr_bucket"]) == 0 and a["meta"]["noise_kind"] == "clean"
        for k in ("video", "lm", "cue", "valid", "audio"):
            assert torch.equal(a[k], b[k]), k
        src = decode_all(work / r["mouth_mp4"])
        sel = resample_indices(r["n_frames"])[:n]
        gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in src[sel]])[:, 4:92, 4:92]
        mean, std = float(cfg["video"]["mean"]), float(cfg["video"]["std"])
        exp = (gray.astype(np.float32) / 255.0 - mean) / std
        assert np.allclose(a["video"][:, 0].numpy(), exp, atol=1e-5)
        with np.load(work / r["npz"]) as z:
            assert np.allclose(a["lm"].numpy(), z["lm"][sel].reshape(n, 80).astype(np.float32))
            assert np.array_equal(a["valid"].numpy(), z["valid"][sel].astype(np.float32))
    frames, n_dec = read_frames(work / rows[0]["mouth_mp4"], resample_indices(45), 96)
    assert frames is not None and frames.shape == (38, 96, 96, 3) and n_dec == 38

    # RGB variant
    cfg_rgb = copy.deepcopy(cfg)
    cfg_rgb["video"]["channels"] = 3
    check_item(AVSRDataset(rows, cfg_rgb, train=False, tokenizer=tok)[1], rows[1]["n_frames"], 3, tok, rows[1]["text"])

    # train mode: forced flip -> mirrored crop + negated lm x; time-mask; noise buckets
    cfg_tr = copy.deepcopy(cfg)
    cfg_tr["video"].update(flip_prob=1.0, time_mask_prob=0.0)
    cfg_tr["audio"]["noise_prob"] = 0.0
    tr = AVSRDataset(rows, cfg_tr, train=True, tokenizer=tok)
    it, ref = tr[2], ev[2]
    n = check_item(it, rows[2]["n_frames"], 1, tok, rows[2]["text"])
    assert torch.allclose(it["lm"][:, 0::2], -ref["lm"][:, 0::2]) and torch.equal(it["lm"][:, 1::2], ref["lm"][:, 1::2])
    src = decode_all(work / rows[2]["mouth_mp4"])
    mean, std = float(cfg["video"]["mean"]), float(cfg["video"]["std"])
    g0 = (cv2.cvtColor(src[0], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 - mean) / std
    matches = [(y, x) for y in range(9) for x in range(9)
               if np.allclose(it["video"][0, 0].numpy(), g0[y:y + 88, x:x + 88][:, ::-1], atol=1e-5)]
    assert len(matches) == 1, matches
    assert not torch.equal(it["audio"], ref["audio"])  # SpecAugment changed the fbank

    cfg_tm = copy.deepcopy(cfg_tr)
    cfg_tm["video"].update(time_mask_prob=1.0, time_mask_max=8, flip_prob=0.0)
    it = AVSRDataset(rows, cfg_tm, train=True, tokenizer=tok)[2]
    masked = (it["video"].flatten(1).abs().amax(1) == 0)
    assert 1 <= int(masked.sum()) <= 8 and torch.all(it["valid"][masked] == 0) and torch.all(it["lm"][masked] == 0)

    cfg_nz = copy.deepcopy(cfg)
    cfg_nz["audio"]["noise_prob"] = 1.0
    trn = AVSRDataset(rows, cfg_nz, train=True, tokenizer=tok)
    kinds = set()
    for _ in range(4):
        for i in range(len(rows)):
            it = trn[i]
            snr = it["meta"]["snr_db"]
            kinds.add(it["meta"]["noise_kind"])
            assert -5.0 <= snr <= 20.0 and int(it["snr_bucket"]) == snr_to_bucket(snr)
    assert kinds == {"babble", "white", "pink"}, kinds
    assert trn._pool and all(c[3].dtype == np.int16 for c in trn._pool)

    # fixed noise for SNR sweeps: deterministic across instances, bucket from the requested SNR
    for kind, snr, bucket in (("babble", 0.0, 2), ("white", -5.0, 3), ("pink", 15.0, 1)):
        d1 = AVSRDataset(rows, cfg, train=False, tokenizer=tok, fixed_noise=(kind, snr))
        d2 = AVSRDataset(rows, cfg, train=False, tokenizer=tok, fixed_noise=(kind, snr))
        a, b = d1[1], d2[1]
        assert torch.equal(a["audio"], b["audio"]) and int(a["snr_bucket"]) == bucket
        assert a["meta"]["noise_kind"] == kind and a["meta"]["snr_db"] == snr
        assert not torch.equal(a["audio"], ev[1]["audio"])
        assert torch.equal(a["video"], ev[1]["video"])

    # bad rows: missing npz is skipped (next index), unreadable video -> zeros
    bad = [dict(rows[0], utt_id="missing", npz="feats/nope.npz"), rows[1]]
    dsb = AVSRDataset(bad, cfg, train=False, tokenizer=tok)
    assert dsb[0]["utt_id"] == rows[1]["utt_id"] and dsb.n_bad == 1
    (work / "garbage.mp4").write_bytes(b"not a video")
    it = AVSRDataset([dict(rows[0], mouth_mp4="garbage.mp4")], cfg, train=False, tokenizer=tok)[0]
    assert torch.allclose(it["video"], torch.full_like(it["video"], -float(cfg["video"]["mean"]) / float(cfg["video"]["std"])))
    # audio much shorter than the video: trimmed to the audio length and counted/logged as desynced
    with np.load(work / rows[2]["npz"]) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["audio"] = arrays["audio"][: arrays["audio"].size // 2]
    np.savez_compressed(work / "half_audio.npz", **arrays)
    dsd = AVSRDataset([dict(rows[2], npz="half_audio.npz")], cfg, train=False, tokenizer=tok)
    it = dsd[0]
    n_half = compute_fbank(arrays["audio"]).shape[0] // 4
    assert it["video"].shape[0] == it["audio"].shape[0] == n_half and dsd.n_desync == 1, (it["video"].shape, n_half)
    assert ev.n_desync == 0

    # collate
    items = [ev[i] for i in range(len(rows))]
    batch = collate_fn(items)
    lens = [x["video"].shape[0] for x in items]
    t_max = max(lens)
    assert batch["video"].shape == (3, t_max, 1, 88, 88) and batch["audio"].shape == (3, t_max, 320)
    assert batch["lm"].shape == (3, t_max, 80) and batch["cue"].shape == (3, t_max, 8) and batch["valid"].shape == (3, t_max)
    assert batch["lengths"].tolist() == lens and batch["snr_bucket"].shape == (3,)
    tl = [len(tok.encode(r["text"])) for r in rows]
    assert batch["token_lengths"].tolist() == tl and batch["tokens"].shape == (3, max(tl))
    for b in range(3):
        assert torch.all(batch["tokens"][b, tl[b]:] == tok.pad_id)
        assert torch.all(batch["video"][b, lens[b]:] == 0) and torch.all(batch["audio"][b, lens[b]:] == 0)
        assert torch.equal(batch["audio"][b, :lens[b]], items[b]["audio"])
    assert batch["utt_ids"] == [r["utt_id"] for r in rows] and len(batch["metas"]) == 3 and batch["texts"][0] == TEXTS[0]

    # DataLoader with spawn workers + DurationBatchSampler
    sampler = DurationBatchSampler(rows, max_frames=100, shuffle=True, seed=1)
    dl = torch.utils.data.DataLoader(trn, batch_sampler=sampler, collate_fn=collate_fn, num_workers=2,
                                     persistent_workers=False)
    seen = []
    for batch in dl:
        assert batch["video"].shape[0] == batch["lengths"].numel()
        seen += batch["utt_ids"]
    assert sorted(seen) == sorted(r["utt_id"] for r in rows)
    print(f"dataset: OK (synthetic video codec: {Path(rows[0]['mouth_mp4']).suffix}, lengths {lens})")


def test_sampler() -> None:
    rng = np.random.default_rng(5)
    rows = [{"n_frames": int(n)} for n in rng.integers(20, 480, size=1500)]
    rows += [{"n_frames": 5000}]  # longer than max_frames alone -> its own batch
    s = DurationBatchSampler(rows, max_frames=3200, shuffle=True, seed=7)
    lens = np.array([resampled_length(r["n_frames"]) for r in rows])
    epochs = []
    for ep in range(3):
        s.set_epoch(ep)
        batches = list(s)
        assert len(batches) == len(s)
        flat = sorted(i for b in batches for i in b)
        assert flat == list(range(len(rows))), "every index exactly once per epoch"
        for b in batches:
            assert len(b) == 1 or len(b) * lens[b].max() <= 3200
        epochs.append(batches)
    assert epochs[0] != epochs[1]
    s.set_epoch(1)
    assert list(s) == epochs[1], "set_epoch must be reproducible"
    fill = np.mean([lens[b].sum() for b in epochs[0]]) / 3200
    s_eval = DurationBatchSampler(rows, max_frames=3200, shuffle=False)
    assert list(s_eval) == list(s_eval)
    print(f"sampler: OK ({len(s)} batches, mean fill {fill:.2f})")


# ----------------------------------------------------------------------------------------------------------------------
# real data (read-only)
# ----------------------------------------------------------------------------------------------------------------------
def test_real() -> None:
    work = ROOT / "work"
    if not any((work / "manifests").glob("*.jsonl")):
        print("real: skipped (no manifest shard under work/manifests)")
        return
    cfg = base_cfg(work)
    loaded = load_manifests(work)
    # a shard without its .done marker may be stale while preprocess is re-running that video: prefer finished ones
    done = {p.stem for p in (work / "manifests").glob("*.done")}
    all_rows = [r for r in loaded if not r["has_unk"] and (r["video_stem"] in done or not done)]
    rows = all_rows[:3]
    tok = Tokenizer()
    for train in (False, True):
        ds = AVSRDataset(rows, cfg, train=train, tokenizer=tok, babble_rows=all_rows)
        for i, r in enumerate(rows):
            it = ds[i]
            with np.load(work / r["npz"]) as z:
                n_t, n_a = int(z["lm"].shape[0]), compute_fbank(z["audio"]).shape[0] // 4
            check_item(it, n_t, 1, tok, r["text"])
            assert n_t == r["n_frames"], (r["utt_id"], n_t, r["n_frames"])
            assert abs(resampled_length(n_t) - n_a) <= 2, (r["utt_id"], n_t, n_a)
        batch = collate_fn([ds[i] for i in range(len(rows))])
        assert batch["video"].shape[0] == 3
    ev = AVSRDataset(rows, cfg, train=False, tokenizer=tok)
    v = ev[0]["video"]
    print(f"real: {len(loaded)} manifest rows, {len(all_rows)} usable from finished shards; "
          f"eval video mean {float(v.mean()):.3f} std {float(v.std()):.3f}")

    # time like a DataLoader worker does it (torch/OpenCV single-threaded)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    pool_rows = all_rows[: min(len(all_rows), 400)]
    for train in (True, False):
        ds = AVSRDataset(pool_rows, cfg, train=train, tokenizer=tok)
        ds[0]  # warm-up (fills the babble pool when training)
        t0 = time.perf_counter()
        for k in range(50):
            ds[(k * 7) % len(pool_rows)]
        dt = (time.perf_counter() - t0) / 50 * 1000
        print(f"real: {'train' if train else 'eval '} __getitem__ {dt:.1f} ms/item (50 items, single process)")
    ds = AVSRDataset(pool_rows, cfg, train=False, tokenizer=tok, fixed_noise=("babble", 0.0))
    t0 = time.perf_counter()
    items = [ds[k] for k in range(10)]
    print(f"real: eval+babble 0 dB {(time.perf_counter() - t0) / 10 * 1000:.1f} ms/item (incl. pool load); "
          f"buckets {[int(x['snr_bucket']) for x in items][:5]}")


def test_utterance_norm(work: Path, rows_written: list[dict]) -> None:
    """video.norm / cue_norm = utterance: per item zero-mean unit-std video; cues z-scored over valid frames only;
    a globally brighter copy of the same crops gives the same normalised video (lighting invariance)."""
    from avsr.dataset import normalize_pixels
    cfg = base_cfg(work)
    cfg["video"]["norm"] = "utterance"
    cfg["video"]["cue_norm"] = "utterance"
    rows = [r for r in rows_written if not r["has_unk"]]
    ds = AVSRDataset(rows, cfg, train=False)
    for i in range(len(ds)):
        it = ds[i]
        v, cue, valid = it["video"].numpy(), it["cue"].numpy(), it["valid"].numpy() > 0.5
        # statistics are taken before the item is trimmed to the audio length (<= 2 frames), hence the tolerances
        assert abs(float(v.mean())) < 0.05 and abs(float(v.std()) - 1.0) < 0.05, (v.mean(), v.std())
        assert np.all(cue[~valid] == 0.0)
        if valid.sum() >= 8:
            assert np.abs(cue[valid].mean(axis=0)).max() < 0.25, cue[valid].mean(axis=0)
            assert np.abs(cue[valid].std(axis=0) - 1.0).max() < 0.25, cue[valid].std(axis=0)
    rng = np.random.default_rng(3)
    pix = rng.integers(20, 120, (12, 1, 88, 88)).astype(np.uint8)
    brighter = np.clip(pix.astype(np.int32) * 2, 0, 255).astype(np.uint8)
    assert np.allclose(normalize_pixels(pix, "utterance"), normalize_pixels(brighter, "utterance"), atol=1e-4)
    print("utterance norm: OK")


def main() -> None:
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="avsr_test_", ignore_cleanup_errors=True) as d:
        tmp = Path(d)
        test_audio_feats(tmp)
        work = tmp / "work"
        rows = make_synthetic(work)
        test_dataset(work, rows)
        test_utterance_norm(work, rows)
    test_sampler()
    test_real()
    print("test_audio_dataset: OK")


if __name__ == "__main__":
    main()

"""Tests for the speaker-lip matching data, model and loss (docs/SYNC_SPEC.md sections 2-4).

Run from the project root:  $py -m tests.test_sync   (or  $py tests\\test_sync.py)

Synthetic checks (mp4 + npz + manifest written to a temp dir) need no dataset. Real-data checks run read-only when
work/manifests (and work/checkpoints/best.pt) exist; CUDA checks (bf16, overfit sanity) when a GPU is available.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsr.audio_feats import compute_fbank, stack_frames  # noqa: E402
from avsr.dataset import (  # noqa: E402
    AVSRDataset, _audio_key, assign_split, load_manifests, resample_indices, resampled_length,
)
from avsr.sync.data import (  # noqa: E402
    SyncWindowDataset, audio_window_batch, audio_window_features, num_audio_frames, sync_collate,
)
from avsr.sync.losses import frame_scores, pair_scores, sync_loss  # noqa: E402
from avsr.sync.model import SyncModel, build_sync_model  # noqa: E402
from avsr.utils import even_subset, load_checkpoint  # noqa: E402
from tests.test_audio_dataset import speech_like, write_video  # noqa: E402

SR = 16000
FPS = 30
W = 25
HAS_CUDA = torch.cuda.is_available()
# the same window cut from differently sized chunks differs only through the CMVN epsilon of compute_fbank
# (<= ~1.5e-3 on the low-variance bins of a pure chirp, ~2e-5 on real speech)
CHUNK_TOL = 3e-3
AVSR_CKPT = ROOT / "work" / "checkpoints" / "best.pt"
SYNC_DEFAULTS = {
    "window_frames": 25, "shift_min": 5, "shift_max": 15, "hidden": 256, "emb_dim": 256, "temporal": "conv",
    "use_cnn": True, "use_skeleton": True, "leak_prob": 0.5, "leak_snr_min": 5, "leak_snr_max": 30,
    "noise_prob": 0.3, "noise_snr_min": 5, "noise_snr_max": 30, "batch_size": 64, "init_from": "",
}


# ----------------------------------------------------------------------------------------------------------------------
# config + synthetic data
# ----------------------------------------------------------------------------------------------------------------------
def make_cfg(work_dir: Path, **sync: Any) -> dict:
    """configs/base.yaml + a sync section; global pixel norm / raw cues so pixel values can be checked exactly."""
    with open(ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["work_dir"] = str(work_dir)
    cfg["split"] = {"val_speakers": ["V001"], "test_speakers": ["T001"], "train_split_dirs": ["2.Validation"]}
    cfg["video"]["norm"] = "global"
    cfg["video"]["cue_norm"] = "none"
    cfg["sync"] = {**SYNC_DEFAULTS, **sync}
    return cfg


def chirp(n: int, f0: float, f1: float, rng: np.random.Generator) -> np.ndarray:
    """Linear chirp f0 -> f1 Hz with a slow amplitude wobble: every 40 ms frame has a distinct spectrum."""
    t = np.arange(n) / SR
    dur = n / SR
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / dur * t ** 2)
    x = np.sin(phase) * (0.6 + 0.4 * np.sin(2 * np.pi * 1.3 * t)) + 1e-3 * rng.standard_normal(n)
    return (x / np.max(np.abs(x)) * 0.5).astype(np.float32)


# (speaker, session, sentence_id, angle, n_frames@30fps, audio kind); utterances 0 and 1 are two camera angles of
# ONE recording (identical audio), 4 has no room for a shifted window, 5 is shorter than window + 2 (dropped)
UTTS = [
    ("S001", "001", 1, "A", 90, "chirp_up"),
    ("S001", "001", 1, "B", 90, "chirp_up"),
    ("S002", "001", 1, "A", 75, "chirp_down"),
    ("S003", "002", 2, "A", 66, "speech"),
    ("S004", "001", 3, "A", 32, "speech"),
    ("S005", "001", 4, "A", 30, "speech"),
]


def make_synthetic(work: Path) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    audio_cache: dict[tuple, np.ndarray] = {}
    for spk, session, sent, angle, n_t, kind in UTTS:
        stem = f"lip_J_1_F_03_{spk}_{angle}_{session}"
        utt_id = f"{stem}__{sent:03d}"
        out_dir = work / "feats" / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        frames = rng.integers(0, 256, size=(n_t, 96, 96, 3), dtype=np.uint8)
        frames = cv2.GaussianBlur(frames.reshape(n_t * 96, 96, 3), (5, 5), 0).reshape(n_t, 96, 96, 3)
        vid = write_video(out_dir / f"{utt_id}.mp4", frames)
        n_samples = int(round(n_t / FPS * SR))
        key = (spk, session, sent)
        if key not in audio_cache:
            if kind == "chirp_up":
                wave = chirp(n_samples, 200.0, 6000.0, rng)
            elif kind == "chirp_down":
                wave = chirp(n_samples, 5000.0, 300.0, rng)
            else:
                wave = speech_like(n_samples, rng)
            audio_cache[key] = (wave * 32767).astype(np.int16)
        audio = audio_cache[key]
        valid = np.ones(n_t, np.uint8)
        valid[::9] = 0
        lm = (rng.standard_normal((n_t, 40, 2)) * valid[:, None, None]).astype(np.float16)
        cue = (rng.random((n_t, 8)) * valid[:, None]).astype(np.float16)
        np.savez_compressed(out_dir / f"{utt_id}.npz", lm=lm, cue=cue, valid=valid, audio=audio,
                            fps=np.float64(FPS), sr=np.int64(SR))
        rows.append({
            "utt_id": utt_id, "video_stem": stem, "split_dir": "2.Validation", "speaker": spk, "gender": "F",
            "age": 3, "specificity": "C", "angle": angle, "session": session, "noise_env": 1, "topic": "test",
            "sentence_id": sent, "start": 0.0, "end": n_t / FPS, "duration": n_t / FPS, "n_frames": n_t,
            "n_samples": int(audio.size), "fps": float(FPS), "text": "테스트 문장", "text_raw": "테스트 문장",
            "has_unk": False, "lm_valid_ratio": float(valid.mean()),
            "mouth_mp4": vid.relative_to(work).as_posix(), "npz": f"feats/{stem}/{utt_id}.npz",
        })
    (work / "manifests").mkdir(parents=True, exist_ok=True)
    with open(work / "manifests" / "synthetic.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def decode_all(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(frame)
    cap.release()
    return np.stack(out)


def npz_arrays(work: Path, row: dict) -> dict[str, np.ndarray]:
    with np.load(work / row["npz"]) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def usable_len(row: dict, arrays: dict[str, np.ndarray]) -> int:
    return min(resampled_length(int(arrays["lm"].shape[0]), float(arrays["fps"])),
               num_audio_frames(int(arrays["audio"].size)))


def cmvn(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(0, keepdim=True)) / torch.sqrt(x.var(0, unbiased=False, keepdim=True) + 1e-5)


def best_lag(x: torch.Tensor, y: torch.Tensor, max_lag: int, min_overlap: int = 8) -> int:
    """Lag L maximising the mean frame cosine between x[t + L] and y[t] (cross-correlation over time)."""
    best, best_score = 0, -math.inf
    for lag in range(-max_lag, max_lag + 1):
        t0, t1 = max(0, -lag), min(y.shape[0], x.shape[0] - lag)
        if t1 - t0 < min_overlap:
            continue
        score = float(F.cosine_similarity(x[t0 + lag:t1 + lag], y[t0:t1], dim=1).mean())
        if score > best_score:
            best, best_score = lag, score
    return best


# ----------------------------------------------------------------------------------------------------------------------
# audio windows
# ----------------------------------------------------------------------------------------------------------------------
def test_audio_windows(work: Path, rows: list[dict]) -> None:
    arrays = npz_arrays(work, rows[0])
    wave = arrays["audio"]
    n_a = num_audio_frames(wave.size)
    assert n_a == stack_frames(compute_fbank(wave), 4).shape[0]
    full_fb = compute_fbank(wave)  # whole-utterance fbank (with whole-utterance CMVN)
    for start in (0, 1, 17, n_a - W):
        feat = audio_window_features(wave, start, W)
        assert feat.shape == (W, 320) and feat.dtype == torch.float32 and torch.isfinite(feat).all()
        # the window equals the matching slice of the whole-utterance fbank, re-normalised over the window
        ref = stack_frames(cmvn(full_fb[4 * start:4 * (start + W)]), 4)
        err = float((feat - ref).abs().max())
        assert err < 2e-3, (start, err)
        fb100 = feat.reshape(W * 4, 80)  # un-stacked 100-Hz frames: CMVN over the window's own frames
        assert float(fb100.mean(0).abs().max()) < 1e-4 and float((fb100.std(0, unbiased=False) - 1).abs().max()) < 1e-2
    # float input in [-1, 1] and torch input give the same features as int16
    f_i = audio_window_features(wave, 10, W)
    f_f = audio_window_features(wave.astype(np.float32) / 32768.0, 10, W)
    f_t = audio_window_features(torch.from_numpy(wave), 10, W)
    assert torch.allclose(f_i, f_f, atol=1e-3) and torch.allclose(f_i, f_t, atol=1e-5)
    # several windows in one call == one call per window
    starts = [0, 5, 12, n_a - W]
    batch = audio_window_batch(wave, starts, W)
    assert batch.shape == (4, W, 320)
    for s, b in zip(starts, batch):
        assert torch.allclose(b, audio_window_features(wave, s, W), atol=CHUNK_TOL)
    # windows reaching outside the waveform (offset sweeps) are zero-padded, never an error
    for s in (-3, n_a - W + 4):
        out = audio_window_features(wave, s, W)
        assert out.shape == (W, 320) and torch.isfinite(out).all()
    assert audio_window_batch(wave, [], W).shape == (0, W, 320)
    try:
        audio_window_features(wave, 0, 0)
        raise AssertionError("window 0 must raise")
    except ValueError:
        pass
    print(f"  audio windows: == whole-utterance fbank slice (max err {err:.1e}); int16/float/torch/batch agree")


# ----------------------------------------------------------------------------------------------------------------------
# dataset
# ----------------------------------------------------------------------------------------------------------------------
def check_item(item: dict, channels: int = 1) -> None:
    assert item["video"].shape == (W, channels, 88, 88) and item["video"].dtype == torch.float32
    assert item["lm"].shape == (W, 80) and item["cue"].shape == (W, 8) and item["valid"].shape == (W,)
    assert item["audio"].shape == (W, 320) and item["audio_shift"].shape == (W, 320)
    for k in ("lm", "cue", "valid", "audio", "audio_shift"):
        assert item[k].dtype == torch.float32 and torch.isfinite(item[k]).all(), k
    assert torch.isfinite(item["video"]).all()
    assert isinstance(item["has_shift"], bool) and isinstance(item["shift"], int)
    assert isinstance(item["speaker"], str) and isinstance(item["utt_id"], str)
    assert isinstance(item["audio_key"], tuple) and isinstance(item["start_frame"], int)
    if item["has_shift"]:
        assert 5 <= abs(item["shift"]) <= 15
    else:
        assert item["shift"] == 0 and torch.all(item["audio_shift"] == 0)


def test_dataset_eval(work: Path, rows: list[dict]) -> None:
    cfg = make_cfg(work)
    ds = SyncWindowDataset(rows, cfg, train=False, seed=3)
    assert len(ds) == len(rows) - 1 and ds.n_dropped == 1 and rows[5]["utt_id"] not in [r["utt_id"] for r in ds.rows]
    ds2 = SyncWindowDataset(rows, cfg, train=False, seed=3)
    mean, std = float(cfg["video"]["mean"]), float(cfg["video"]["std"])
    for i, row in enumerate(ds.rows):
        a, b = ds[i], ds2[i]
        check_item(a)
        for k in ("video", "lm", "cue", "valid", "audio", "audio_shift"):  # deterministic across instances
            assert torch.equal(a[k], b[k]), (row["utt_id"], k)
        assert (a["shift"], a["start_frame"], a["has_shift"]) == (b["shift"], b["start_frame"], b["has_shift"])
        arrays = npz_arrays(work, row)
        n = usable_len(row, arrays)
        start = a["start_frame"]
        assert start == (n - W) // 2, (start, n)  # centred window
        assert a["audio_key"] == _audio_key(row) and a["speaker"] == row["speaker"] and a["utt_id"] == row["utt_id"]
        # video / lm / valid / cue: exactly the window's frames of the utterance
        idx = resample_indices(int(arrays["lm"].shape[0]), float(arrays["fps"]))[start:start + W]
        src = decode_all(work / row["mouth_mp4"])
        gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in src[idx]])[:, 4:92, 4:92]
        assert np.allclose(a["video"][:, 0].numpy(), (gray.astype(np.float32) / 255.0 - mean) / std, atol=1e-5)
        assert np.array_equal(a["lm"].numpy(), arrays["lm"][idx].reshape(W, 80).astype(np.float32))
        assert np.array_equal(a["valid"].numpy(), arrays["valid"][idx].astype(np.float32))
        assert np.array_equal(a["cue"].numpy(), arrays["cue"][idx].astype(np.float32))
        # audio: the matching window; shifted negative: same utterance, inside it, offset = shift
        assert torch.allclose(a["audio"], audio_window_features(arrays["audio"], start, W), atol=CHUNK_TOL)
        if a["has_shift"]:
            s = start + a["shift"]
            assert 0 <= s and s + W <= n, (s, n)
            assert torch.allclose(a["audio_shift"], audio_window_features(arrays["audio"], s, W), atol=CHUNK_TOL)
    short = ds[4]
    assert short["utt_id"] == rows[4]["utt_id"] and not short["has_shift"], "no room for a shift -> no negative"

    # shift offset verified by cross-correlation on the chirp utterances (every frame has a distinct spectrum)
    for i in (0, 2):
        it = ds[i]
        arrays = npz_arrays(work, ds.rows[i])
        n = usable_len(ds.rows[i], arrays)
        whole = audio_window_features(arrays["audio"], 0, n)  # the unshifted features of the whole utterance
        assert it["has_shift"]
        lag_audio = best_lag(it["audio"], it["audio_shift"], 15)
        lag_whole = best_lag(whole, it["audio_shift"], n)
        lag_self = best_lag(whole, it["audio"], n)
        assert lag_audio == it["shift"], (lag_audio, it["shift"])
        assert lag_whole == it["start_frame"] + it["shift"] and lag_self == it["start_frame"], (lag_whole, lag_self)

    # other windows lengths and RGB
    ds13 = SyncWindowDataset(rows, cfg, train=False, window=13)
    assert len(ds13) == len(rows) and ds13[0]["video"].shape[0] == 13 and ds13[0]["audio"].shape == (13, 320)
    cfg_rgb = copy.deepcopy(cfg)
    cfg_rgb["video"]["channels"] = 3
    check_item(SyncWindowDataset(rows, cfg_rgb, train=False)[1], channels=3)
    # utterance pixel norm is taken over the decoded window
    cfg_u = copy.deepcopy(cfg)
    cfg_u["video"]["norm"] = "utterance"
    cfg_u["video"]["cue_norm"] = "utterance"
    v = SyncWindowDataset(rows, cfg_u, train=False)[0]["video"]
    assert abs(float(v.mean())) < 1e-3 and abs(float(v.std()) - 1.0) < 1e-2

    # bad rows are skipped (next row returned) and counted
    bad = [dict(rows[0], utt_id="missing", npz="feats/nope.npz"), rows[2]]
    dsb = SyncWindowDataset(bad, cfg, train=False)
    assert dsb[0]["utt_id"] == rows[2]["utt_id"] and dsb.n_bad == 1
    (work / "garbage.mp4").write_bytes(b"not a video")
    dsg = SyncWindowDataset([dict(rows[0], mouth_mp4="garbage.mp4"), rows[2]], cfg, train=False)
    assert dsg[0]["utt_id"] == rows[2]["utt_id"] and dsg.n_bad == 1
    print(f"  eval items: shapes/dtypes, centred, deterministic, exact frames, shifts "
          f"{[ds[i]['shift'] for i in range(len(ds))]} verified by cross-correlation")


def test_dataset_train(work: Path, rows: list[dict]) -> None:
    kept = rows[:5]
    # geometry/flip: no audio augmentation, forced flip
    cfg = make_cfg(work, leak_prob=0.0, noise_prob=0.0)
    cfg["video"]["flip_prob"] = 1.0
    ds = SyncWindowDataset(kept, cfg, train=True, seed=1)
    mean, std = float(cfg["video"]["mean"]), float(cfg["video"]["std"])
    starts, shifts = set(), set()
    for _ in range(12):
        it = ds[0]
        check_item(it)
        arrays = npz_arrays(work, kept[0])
        n = usable_len(kept[0], arrays)
        start = it["start_frame"]
        assert 0 <= start <= n - W
        starts.add(start)
        if it["has_shift"]:
            shifts.add(it["shift"])
            assert 0 <= start + it["shift"] <= n - W
        idx = resample_indices(int(arrays["lm"].shape[0]), float(arrays["fps"]))[start:start + W]
        lm = arrays["lm"][idx].reshape(W, 80).astype(np.float32)
        assert np.array_equal(it["lm"][:, 1::2].numpy(), lm[:, 1::2]) and np.array_equal(it["lm"][:, 0::2].numpy(),
                                                                                         -lm[:, 0::2])
        assert np.array_equal(it["valid"].numpy(), arrays["valid"][idx].astype(np.float32))  # no time masking
    assert len(starts) >= 3 and any(s < 0 for s in shifts) and any(s > 0 for s in shifts), (starts, shifts)
    src = decode_all(work / kept[0]["mouth_mp4"])
    g0 = (cv2.cvtColor(src[idx[0]], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 - mean) / std
    matches = [(y, x) for y in range(9) for x in range(9)
               if np.allclose(it["video"][0, 0].numpy(), g0[y:y + 88, x:x + 88][:, ::-1], atol=1e-5)]
    assert len(matches) == 1, matches  # a random 88 crop of the mirrored frame

    # light SpecAugment only (no leakage/noise here): a zeroed mel band in some items, the rest unchanged
    zero_band = changed = 0
    for _ in range(10):
        it = ds[2]
        clean = audio_window_features(npz_arrays(work, kept[2])["audio"], it["start_frame"], W)
        band = (it["audio"].abs().amax(0) == 0).view(4, 80).all(0)       # bin zero in all 4 stacked copies
        zero_band += int(band.any())
        diff = (it["audio"] - clean).abs() > CHUNK_TOL
        changed += int(diff.any())
        assert torch.all(it["audio"][diff] == 0), "SpecAugment may only zero values"
    assert zero_band >= 1 and changed >= 5, (zero_band, changed)

    # leakage: one other utterance of a DIFFERENT speaker, at the drawn SNR relative to the target
    no_spec = {"freq_mask": 0, "freq_width": 0, "time_mask": 0, "time_width": 0}
    cfg_l = make_cfg(work, leak_prob=1.0, noise_prob=0.0, specaug=no_spec)
    speakers = {r["utt_id"]: r["speaker"] for r in rows}
    dsl = SyncWindowDataset(kept, cfg_l, train=True, seed=2)
    for i in range(len(kept)):
        it = dsl[i]
        aug = it["aug"]
        assert aug["leak_utt"] is not None and speakers[aug["leak_utt"]] != it["speaker"], aug
        assert 5.0 <= aug["leak_snr"] <= 30.0 and aug["noise_kind"] is None
        clean = audio_window_features(npz_arrays(work, kept[i])["audio"], it["start_frame"], W)
        assert float((it["audio"] - clean).abs().max()) > 1e-3
    diffs = {}
    for snr in (40.0, 20.0, 0.0):  # same seed -> same leaked clip and offset; only its level changes
        cfg_s = make_cfg(work, leak_prob=1.0, noise_prob=0.0, leak_snr_min=snr, leak_snr_max=snr, specaug=no_spec)
        it = SyncWindowDataset(kept, cfg_s, train=True, seed=2)[3]
        clean = audio_window_features(npz_arrays(work, kept[3])["audio"], it["start_frame"], W)
        diffs[snr] = float((it["audio"] - clean).abs().mean())
    assert diffs[40.0] < diffs[20.0] < diffs[0.0] and diffs[40.0] < 0.5 * diffs[0.0], diffs
    same_spk = SyncWindowDataset(kept[:2], cfg_l, train=True, seed=2)  # only speaker S001: nothing may leak
    it = same_spk[0]
    assert it["aug"]["leak_utt"] is None and same_spk.n_no_leak >= 1
    # noise
    cfg_n = make_cfg(work, leak_prob=0.0, noise_prob=1.0, specaug=no_spec)
    dsn = SyncWindowDataset(kept, cfg_n, train=True, seed=4)
    kinds = set()
    for k in range(12):
        aug = dsn[k % len(kept)]["aug"]
        kinds.add(aug["noise_kind"])
        assert 5.0 <= aug["noise_snr"] <= 30.0 and aug["leak_utt"] is None
    assert kinds == {"white", "pink"}, kinds
    print(f"  train items: random starts {sorted(starts)}, shifts {sorted(shifts)}, flip+crop, leakage from other "
          f"speakers (mean |feature diff| 40/20/0 dB: {diffs[40.0]:.3f}/{diffs[20.0]:.3f}/{diffs[0.0]:.3f}), "
          f"noise {sorted(kinds)}")


def test_collate_and_loader(work: Path, rows: list[dict]) -> None:
    cfg = make_cfg(work)
    ds = SyncWindowDataset(rows, cfg, train=False)
    items = [ds[i] for i in range(len(ds))]
    batch = sync_collate(items)
    b = len(items)
    assert batch["video"].shape == (b, W, 1, 88, 88) and batch["audio"].shape == (b, W, 320)
    assert batch["audio_shift"].shape == (b, W, 320) and batch["lm"].shape == (b, W, 80)
    assert batch["cue"].shape == (b, W, 8) and batch["valid"].shape == (b, W)
    assert batch["has_shift"].dtype == torch.bool and batch["shift"].dtype == torch.long
    assert batch["start_frame"].tolist() == [it["start_frame"] for it in items]
    ids = batch["audio_key_ids"].tolist()
    assert batch["audio_key_ids"].dtype == torch.long and ids[0] == ids[1] and len(set(ids)) == b - 1, ids
    spk = batch["speaker_ids"].tolist()
    assert spk[0] == spk[1] and len(set(spk)) == b - 1
    assert batch["utt_id"] == [it["utt_id"] for it in items] and batch["speaker"][2] == "S002"
    assert batch["audio_key"][0] == batch["audio_key"][1] and len(batch["aug"]) == b
    try:
        sync_collate([items[0], SyncWindowDataset(rows, cfg, train=False, window=13)[0]])
        raise AssertionError("mixed window lengths must raise")
    except ValueError:
        pass

    # Windows spawn workers: pickled dataset, per-worker generators, batches of equal windows
    train_ds = SyncWindowDataset(rows, make_cfg(work, leak_prob=1.0), train=True, seed=5)
    dl = torch.utils.data.DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, collate_fn=sync_collate,
                                     drop_last=True, multiprocessing_context="spawn", persistent_workers=False)
    seen = []
    for batch in dl:
        assert batch["video"].shape == (2, W, 1, 88, 88) and batch["audio_key_ids"].shape == (2,)
        seen += batch["utt_id"]
    assert len(seen) == 4 and len(set(seen)) == 4
    print(f"  collate: audio_key_ids {ids}, speaker_ids {spk}; spawn DataLoader OK")


# ----------------------------------------------------------------------------------------------------------------------
# model + loss
# ----------------------------------------------------------------------------------------------------------------------
def random_batch(b: int, w: int, device: torch.device, seed: int = 0, channels: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    valid = (torch.rand(b, w, generator=g) > 0.1).float()
    batch = {
        "video": torch.randn(b, w, channels, 88, 88, generator=g),
        "lm": torch.randn(b, w, 80, generator=g) * 0.3 * valid[..., None],
        "cue": torch.randn(b, w, 8, generator=g) * valid[..., None],
        "valid": valid,
        "audio": torch.randn(b, w, 320, generator=g),
        "audio_shift": torch.randn(b, w, 320, generator=g),
        "has_shift": torch.tensor([i % 3 != 2 for i in range(b)]),
        "audio_key_ids": torch.tensor([i // 2 for i in range(b)]),
    }
    return {k: v.to(device) for k, v in batch.items()}


def test_model_cpu() -> None:
    torch.manual_seed(0)
    cfg = {"video": {"channels": 1}, "audio": {"n_mels": 80, "stack": 4}, "sync": dict(SYNC_DEFAULTS)}
    model = build_sync_model(cfg).eval()
    batch = random_batch(3, W, torch.device("cpu"))
    with torch.no_grad():
        out = model(batch)
        v, a, a_shift = out["v"], out["a"], out["a_shift"]
        assert v.shape == a.shape == a_shift.shape == (3, W, 256) and v.dtype == torch.float32
        assert torch.allclose(v.norm(dim=-1), torch.ones(3, W), atol=1e-5)
        assert torch.allclose(a.norm(dim=-1), torch.ones(3, W), atol=1e-5)
        assert torch.allclose(a, model.embed_audio(batch["audio"]), atol=1e-5)         # shared pass == separate
        assert torch.allclose(a_shift, model.embed_audio(batch["audio_shift"]), atol=1e-5)
        s = model.pair_scores(v, a)
        manual = torch.stack([torch.stack([F.cosine_similarity(v[i], a[j], dim=-1).mean() for j in range(3)])
                              for i in range(3)])
        assert s.shape == (3, 3) and torch.allclose(s, manual, atol=1e-5)
        fs = model.frame_scores(v, a)
        assert fs.shape == (3, W) and torch.allclose(fs.mean(1), s.diagonal(), atol=1e-5)
        assert model.pair_scores(v, a[:2]).shape == (3, 2)
        # locality of the conv temporal encoder: frame 0 influences only frames <= 16 (audio) / <= 18 (video)
        long = random_batch(2, 40, torch.device("cpu"), seed=1)
        pert = {k: t.clone() for k, t in long.items()}
        pert["audio"][:, 0] += 5.0
        pert["lm"][:, 0] += 5.0
        pert["video"][:, 0] += 5.0
        a0, a1 = model.embed_audio(long["audio"]), model.embed_audio(pert["audio"])
        v0 = model.embed_video(long["video"], long["lm"], long["cue"], long["valid"])
        v1 = model.embed_video(pert["video"], pert["lm"], pert["cue"], pert["valid"])
        da, dv = (a0 - a1).abs().amax(dim=(0, 2)), (v0 - v1).abs().amax(dim=(0, 2))
        assert float(da[17:].max()) == 0.0 and float(da[:17].max()) > 0, da
        assert float(dv[19:].max()) == 0.0 and float(dv[:19].max()) > 0, dv
    rf = model.audio_temporal.receptive_field

    # variants: GRU encoder, skeleton-only (no video tensor needed), CNN-only; invalid configs raise
    for sync in ({"temporal": "gru"}, {"use_cnn": False}, {"use_skeleton": False}):
        m = build_sync_model({**cfg, "sync": {**SYNC_DEFAULTS, **sync}}).eval()
        bt = dict(batch)
        if sync.get("use_cnn") is False:
            bt["video"] = None
        with torch.no_grad():
            o = m(bt)
        assert o["v"].shape == (3, W, 256) and o["a"].shape == (3, W, 256), sync
    for sync in ({"use_cnn": False, "use_skeleton": False}, {"temporal": "transformer"}):
        try:
            build_sync_model({**cfg, "sync": {**SYNC_DEFAULTS, **sync}})
            raise AssertionError(f"{sync} must raise")
        except ValueError:
            pass

    # logit scale: init 10, clamped at 100, differentiable below the clamp
    with torch.no_grad():
        assert abs(float(model.logit_scale_value()) - 10.0) < 1e-4
        model.logit_scale.fill_(math.log(1000.0))
        assert abs(float(model.logit_scale_value()) - 100.0) < 1e-4

    # compute_loss end to end on CPU, gradients reach every branch and the scale
    model = build_sync_model(cfg).train()
    out = model(batch)
    loss, parts = model.compute_loss(out, batch)
    loss.backward()
    assert math.isfinite(float(loss.detach())) and set(parts) >= {"loss", "l_va", "l_av", "acc_va", "pos_cos",
                                                                  "neg_cos", "shift_cos"}
    for name in ("logit_scale", "video_proj.weight", "audio_proj.weight", "visual_frontend.stem.0.weight",
                 "skeleton_frontend.fc_in.weight", "audio_frontend.fc_in.weight"):
        p = dict(model.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum()) > 0, name
    params = sum(p.numel() for p in model.parameters())
    print(f"  model: shapes/L2 norm/scores OK, conv receptive field {rf} frames, variants OK, "
          f"{params / 1e6:.1f} M params; loss {parts['loss']:.3f}")


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def ref_loss(v: torch.Tensor, a: torch.Tensor, scale: float, ids: torch.Tensor, a_shift: torch.Tensor,
             has_shift: torch.Tensor) -> torch.Tensor:
    """Loop implementation of SYNC_SPEC section 4: candidates of row/column i = itself + other recordings only."""
    b = v.shape[0]
    cos = [[F.cosine_similarity(v[i], a[j], dim=-1).mean() for j in range(b)] for i in range(b)]
    l_va, l_av = [], []
    for i in range(b):
        cands = [j for j in range(b) if j == i or int(ids[j]) != int(ids[i])]
        row = [scale * cos[i][j] for j in cands]
        if bool(has_shift[i]):
            row.append(scale * F.cosine_similarity(v[i], a_shift[i], dim=-1).mean())
        l_va.append(-torch.log_softmax(torch.stack(row), 0)[cands.index(i)])
        col = [scale * cos[j][i] for j in cands]
        l_av.append(-torch.log_softmax(torch.stack(col), 0)[cands.index(i)])
    return 0.5 * (torch.stack(l_va).mean() + torch.stack(l_av).mean())


def test_loss() -> None:
    g = torch.Generator().manual_seed(0)
    b, w, e = 4, 6, 16
    # orthogonal per-sample directions (constant over time) -> cos = 1 on the diagonal, 0 elsewhere
    basis = torch.linalg.qr(torch.randn(e, e, generator=g))[0][:b]
    v = basis[:, None, :].expand(b, w, e).clone()
    a = v.clone()
    loss, parts = sync_loss(v, a, 50.0)
    assert parts["acc_va"] == 1.0 and parts["acc_av"] == 1.0 and float(loss) < 1e-6
    assert abs(parts["pos_cos"] - 1.0) < 1e-5 and abs(parts["neg_cos"]) < 1e-5 and math.isnan(parts["shift_cos"])
    _, parts = sync_loss(v, a[[1, 2, 3, 0]], 50.0)
    assert parts["acc_va"] == 0.0 and parts["acc_av"] == 0.0

    # same-audio masking: items 0 and 1 are two camera angles of ONE recording (identical embeddings)
    v2, a2 = v.clone(), a.clone()
    v2[1], a2[1] = v2[0], a2[0]
    ids = torch.tensor([0, 0, 1, 2])
    loss_m, parts_m = sync_loss(v2, a2, 50.0, audio_key_ids=ids)
    loss_u, parts_u = sync_loss(v2, a2, 50.0)
    assert float(loss_m) < 1e-5 and parts_m["acc_va"] == 1.0, parts_m          # not negatives of each other
    assert abs(float(loss_u) - math.log(2) / 2) < 1e-3, float(loss_u)         # unmasked: rows 0/1 tie -> log 2
    assert abs(parts_m["neg_cos"]) < 1e-5 and parts_u["neg_cos"] > 0.1         # masked pairs are not negatives
    # exact agreement with a loop implementation of SYNC_SPEC section 4 (random ids, random shift flags)
    b6 = 6
    raw6 = torch.randn(b6, w, e, generator=g).requires_grad_(True)  # the model L2-normalises its embeddings
    a6 = unit(torch.randn(b6, w, e, generator=g))
    s6 = unit(torch.randn(b6, w, e, generator=g))
    ids6 = torch.tensor([0, 0, 1, 2, 2, 3])
    flags6 = torch.tensor([True, False, True, True, False, True])
    got, _ = sync_loss(unit(raw6), a6, 7.0, a_shift=s6, has_shift=flags6, audio_key_ids=ids6)
    want = ref_loss(unit(raw6), a6, 7.0, ids6, s6, flags6)
    assert torch.allclose(got, want, atol=1e-5), (float(got), float(want))
    g_got = torch.autograd.grad(got, raw6)[0]
    g_want = torch.autograd.grad(want, raw6)[0]
    assert torch.allclose(g_got, g_want, atol=1e-5), float((g_got - g_want).abs().max())
    # all off-diagonal masked (one recording): finite, NaN-free loss and gradients
    vv = unit(torch.randn(b, w, e, generator=g)).requires_grad_(True)
    aa = unit(torch.randn(b, w, e, generator=g)).requires_grad_(True)
    l_all, p_all = sync_loss(vv, aa, 10.0, audio_key_ids=torch.zeros(b, dtype=torch.long))
    l_all.backward()
    assert math.isfinite(p_all["loss"]) and p_all["loss"] < 1e-6 and math.isnan(p_all["neg_cos"])
    assert torch.isfinite(vv.grad).all() and torch.isfinite(aa.grad).all()

    # shifted hard negative: a shifted copy identical to the positive ties it -> log 2 on the rows that have one
    for flags, expect in (([True] * 4, math.log(2)), ([False] * 4, 0.0), ([True, False, False, False],
                                                                           math.log(2) / 4)):
        loss_s, parts_s = sync_loss(v, a, 50.0, a_shift=a.clone(), has_shift=torch.tensor(flags))
        assert abs(parts_s["l_va"] - expect) < 1e-3 and parts_s["l_av"] < 1e-6, (flags, parts_s)
        if any(flags):  # (acc_shift is a numerical tie here)
            assert abs(parts_s["shift_cos"] - 1.0) < 1e-5
    a_sh = unit(torch.randn(b, w, e, generator=g))
    _, parts_s = sync_loss(v, a, 50.0, a_shift=a_sh, has_shift=torch.ones(b, dtype=torch.bool))
    assert parts_s["acc_shift"] == 1.0 and parts_s["acc_va"] == 1.0
    # the scale is learnable through the loss
    scale = torch.tensor(10.0, requires_grad=True)
    l_sc, _ = sync_loss(unit(torch.randn(b, w, e, generator=g)), unit(torch.randn(b, w, e, generator=g)), scale)
    l_sc.backward()
    assert scale.grad is not None and float(scale.grad.abs()) > 0
    print(f"  loss: same-audio masking (masked {float(loss_m):.1e} vs unmasked {float(loss_u):.3f}), shift column, "
          f"-inf safety OK")


def test_cuda_bf16() -> None:
    dev = torch.device("cuda")
    torch.manual_seed(0)
    cfg = {"video": {"channels": 1}, "audio": {"n_mels": 80, "stack": 4}, "sync": dict(SYNC_DEFAULTS)}
    for temporal in ("conv", "gru"):
        model = build_sync_model({**cfg, "sync": {**SYNC_DEFAULTS, "temporal": temporal}}).to(dev).train()
        batch = random_batch(8, W, dev, seed=2)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(batch)
            loss, parts = model.compute_loss(out, batch)
        assert out["v"].dtype == torch.float32 and loss.dtype == torch.float32
        loss.backward()
        assert all(math.isfinite(parts[k]) for k in ("loss", "l_va", "l_av", "pos_cos", "neg_cos"))
        bad = [n for n, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
        assert not bad, bad
        print(f"  cuda bf16 ({temporal}): loss {parts['loss']:.3f}, all gradients finite")


# ----------------------------------------------------------------------------------------------------------------------
# real data / checkpoint (read-only)
# ----------------------------------------------------------------------------------------------------------------------
def test_init_from_avsr() -> None:
    if not AVSR_CKPT.is_file():
        print(f"  init_from_avsr: skipped ({AVSR_CKPT} not found)")
        return
    cfg = {"video": {"channels": 1}, "audio": {"n_mels": 80, "stack": 4}, "sync": dict(SYNC_DEFAULTS)}
    model = build_sync_model(cfg)
    before = {k: v.clone() for k, v in model.state_dict().items() if not k.split(".")[0].endswith("frontend")}
    report = model.init_from_avsr(AVSR_CKPT)
    src = load_checkpoint(AVSR_CKPT)["model"]
    for module in ("visual_frontend", "skeleton_frontend", "audio_frontend"):
        st = report["modules"][module]
        assert st["total"] > 0 and st["copied"] == st["total"] and st["skipped"] == 0, (module, st)
    assert not report["skipped"]
    own = model.state_dict()
    for k in report["copied"]:
        assert torch.equal(own[k], src[k]), k
    for k, v in before.items():  # nothing outside the frontends is touched
        assert torch.equal(own[k], v), k
    # partial matches: RGB stem differs in shape (reported, not copied); skeleton-only model has no visual entry
    rgb = build_sync_model({**cfg, "video": {"channels": 3}})
    rep_rgb = rgb.init_from_avsr(AVSR_CKPT)
    assert rep_rgb["skipped"] == ["visual_frontend.stem.0.weight: shape (64, 1, 5, 7, 7) != (64, 3, 5, 7, 7)"], \
        rep_rgb["skipped"]
    skel = build_sync_model({**cfg, "sync": {**SYNC_DEFAULTS, "use_cnn": False}})
    assert "visual_frontend" not in skel.init_from_avsr(AVSR_CKPT)["modules"]
    counts = {m: f"{s['copied']}/{s['total']}" for m, s in report["modules"].items()}
    print(f"  init_from_avsr (real {AVSR_CKPT.name}): copied {counts}; RGB stem skipped with reason")


def real_rows() -> tuple[dict, list[dict]] | None:
    work = ROOT / "work"
    if not any((work / "manifests").glob("*.jsonl")):
        return None
    cfg = make_cfg(work)
    with open(ROOT / "configs" / "base.yaml", "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    cfg["split"] = base["split"]
    cfg["video"]["norm"] = base["video"]["norm"]
    cfg["video"]["cue_norm"] = base["video"]["cue_norm"]
    return cfg, assign_split(load_manifests(work), cfg)["train"]


def test_real_data() -> None:
    loaded = real_rows()
    if loaded is None:
        print("  real data: skipped (no manifest shard under work/manifests)")
        return
    cfg, train_rows = loaded
    rows = even_subset(train_rows, 300)
    # the window grid equals the AVSR dataset's: same frames, same audio (up to the per-window CMVN)
    cfg_g = copy.deepcopy(cfg)
    cfg_g["video"]["norm"] = "global"
    cfg_g["video"]["cue_norm"] = "none"
    sync_ds = SyncWindowDataset(rows[:6], cfg_g, train=False)
    avsr_ds = AVSRDataset(rows[:6], cfg_g, train=False)
    for i in range(len(sync_ds)):
        it, ref = sync_ds[i], avsr_ds[i]
        s = it["start_frame"]
        assert ref["video"].shape[0] >= s + W
        assert torch.allclose(it["video"], ref["video"][s:s + W], atol=1e-5)
        assert torch.equal(it["lm"], ref["lm"][s:s + W]) and torch.equal(it["cue"], ref["cue"][s:s + W])
        ref_audio = stack_frames(cmvn(ref["audio"][s:s + W].reshape(W * 4, 80)), 4)
        assert torch.allclose(it["audio"], ref_audio, atol=2e-3), float((it["audio"] - ref_audio).abs().max())
    # train/eval items on real rows are well-formed; timing like a DataLoader worker (1 thread)
    for train in (False, True):
        ds = SyncWindowDataset(rows, cfg, train=train, seed=0)
        for i in range(0, 40, 4):
            check_item(ds[i])
    threads = (torch.get_num_threads(), cv2.getNumThreads())
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    timing = {}
    try:
        for train in (True, False):
            ds = SyncWindowDataset(rows, cfg, train=train, seed=0)
            for k in range(len(rows)):  # warm-up: leakage pool + OS file cache of these rows
                ds[k]
            t0 = time.perf_counter()
            for k in range(len(rows)):
                ds[k]
            timing["train" if train else "eval"] = (time.perf_counter() - t0) / len(rows) * 1000
            assert ds.n_bad == 0, ds.n_bad
    finally:
        torch.set_num_threads(threads[0])
        cv2.setNumThreads(threads[1])
    assert timing["train"] < 60 and timing["eval"] < 60, timing  # target < 15 ms on an idle CPU (loose bound here)
    print(f"  real data: {len(train_rows)} train rows; windows == AVSR dataset frames/audio; __getitem__ "
          f"train {timing['train']:.1f} ms, eval {timing['eval']:.1f} ms per item ({len(rows)} rows, 1 thread, "
          f"warm cache)")


def test_overfit_real() -> None:
    """32 real TRAIN windows (distinct recordings): 150 steps must reach > 90 % 32-way video->audio accuracy."""
    loaded = real_rows()
    if loaded is None or not HAS_CUDA:
        print("  overfit: skipped (needs real manifests and CUDA)")
        return
    cfg, train_rows = loaded
    by_key: dict[tuple, dict] = {}
    for r in sorted(train_rows, key=lambda r: r["utt_id"]):
        by_key.setdefault(_audio_key(r), r)
    rows = even_subset(list(by_key.values()), 32)
    ds = SyncWindowDataset(rows, cfg, train=False, seed=0)
    batch = sync_collate([ds[i] for i in range(32)])
    assert len(set(batch["audio_key_ids"].tolist())) == 32
    dev = torch.device("cuda")
    gpu = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    torch.manual_seed(0)
    model = build_sync_model(cfg).to(dev)
    init = "scratch"
    if AVSR_CKPT.is_file():
        model.init_from_avsr(AVSR_CKPT)
        init = "AVSR frontends"

    def accuracy() -> float:
        model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(gpu)
        s = pair_scores(out["v"], out["a"])
        model.train()
        return float((s.argmax(dim=1) == torch.arange(32, device=dev)).float().mean())

    acc0 = accuracy()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    t0 = time.perf_counter()
    first = last = None
    for step in range(150):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(gpu)
            loss, parts = model.compute_loss(out, gpu)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        first = parts if first is None else first
        last = parts
    dt = time.perf_counter() - t0
    acc = accuracy()
    print(f"  overfit (32 real train windows, {init}): loss {first['loss']:.3f} -> {last['loss']:.3f}, "
          f"32-way v->a accuracy {acc0:.2f} -> {acc:.2f} (eval mode), shift acc {last['acc_shift']:.2f}, "
          f"{dt:.1f} s")
    assert acc > 0.9, acc


def main() -> None:
    torch.manual_seed(0)
    t_all = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="avsr_sync_test_", ignore_cleanup_errors=True) as d:
        work = Path(d) / "work"
        rows = make_synthetic(work)
        for fn in (test_audio_windows, test_dataset_eval, test_dataset_train, test_collate_and_loader):
            t0 = time.perf_counter()
            fn(work, rows)
            print(f"{fn.__name__}: OK ({time.perf_counter() - t0:.1f} s)")
    tests = [test_model_cpu, test_loss, test_init_from_avsr, test_real_data]
    if HAS_CUDA:
        tests += [test_cuda_bf16, test_overfit_real]
    else:
        print("CUDA not available: skipping bf16 and overfit tests")
    for fn in tests:
        t0 = time.perf_counter()
        fn()
        print(f"{fn.__name__}: OK ({time.perf_counter() - t0:.1f} s)")
    print(f"test_sync: ALL OK ({time.perf_counter() - t_all:.0f} s)")


if __name__ == "__main__":
    main()

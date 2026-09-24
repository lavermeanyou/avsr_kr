"""Tests for the sync training CLI and the scenario evaluation (docs/SYNC_SPEC.md sections 5-7).

Run from the project root:  $py -m tests.test_sync_train   (or  $py tests\\test_sync_train.py [--keep])

- metric primitives: Hungarian assignment on a hand-made matrix, ROC-AUC / EER / threshold on synthetic score
  distributions, tie-aware selection credit, leakage mixing levels, the distractor policy;
- N-way pipeline with stand-in embeddings: an oracle (face == its own audio) scores 1.0, a random model ~1/N;
- the in-memory evaluation windows equal SyncWindowDataset eval items;
- synthetic end-to-end smoke (temp work_dir, the real work/ is never touched): 2 epochs with a tiny model
  (last.pt / best.pt / resume / Ctrl+C mid-epoch resume / --limit refusal / missing init_from / fresh run in a used
  or foreign checkpoint directory / init checkpoint that did not hold out the eval speakers), then evaluate on
  ``val`` (one speaker: fallback distractors) and ``heldout`` (3 speakers) -> JSON with every section, deterministic.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import sys
import tempfile
import time
import warnings
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsr.sync import evaluate as sev  # noqa: E402
from avsr.sync import train as strain  # noqa: E402
from avsr.sync.data import SyncWindowDataset, audio_window_features  # noqa: E402
from avsr.dataset import assign_split, load_manifests  # noqa: E402
from avsr.utils import close_logger, get_logger, load_config, read_jsonl  # noqa: E402

CONFIG = ROOT / "configs" / "sync.yaml"
SR = 16000
FPS = 30
CHUNK_TOL = 3e-3  # same window from differently sized chunks: differs only through compute_fbank's CMVN epsilon
SPLIT_SETS = ["split.val_speakers=[V001]", "split.test_speakers=[T001, T002]", "split.train_split_dirs=[2.Validation]"]
TINY = ["sync.hidden=32", "sync.emb_dim=16", "sync.batch_size=8", "sync.val_max_utts=0", "train.log_every=1",
        "train.warmup_steps=2", "train.keep_last=1", "train.lr=1e-3"]


# ----------------------------------------------------------------------------------------------------------------------
# synthetic data: mouth opening drives the lips in the video, the landmarks/cues and the audio envelope
# ----------------------------------------------------------------------------------------------------------------------
def write_utterance(work: Path, speaker: str, session: str, sentence: int, angle: str, n_t: int,
                    opening: np.ndarray, f0: float, rng: np.random.Generator) -> dict:
    stem = f"lip_J_1_M_03_{speaker}_{angle}_{session}"
    utt_id = f"{stem}__{sentence:03d}"
    out = work / "feats" / stem
    out.mkdir(parents=True, exist_ok=True)
    mp4 = out / f"{utt_id}.mp4"
    wr = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(FPS), (96, 96))
    if not wr.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open an mp4v writer")
    skin = (110 + 10 * (zlib.crc32(speaker.encode("utf-8")) % 5), 140, 185)
    for t in range(n_t):
        frame = np.full((96, 96, 3), skin, np.uint8)
        cv2.ellipse(frame, (48, 54), (24, int(4 + 14 * opening[t])), 0, 0, 360, (60, 40, 160), -1)
        cv2.ellipse(frame, (48, 54), (17, int(1 + 10 * opening[t])), 0, 0, 360, (25, 15, 35), -1)
        wr.write(frame)
    wr.release()
    n_samples = int(round(n_t / FPS * SR))
    t_a = np.arange(n_samples) / SR
    env = np.interp(t_a, np.arange(n_t) / FPS, opening) ** 1.5
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.05 * np.sin(2 * np.pi * 0.8 * t_a))) / SR
    voice = sum(np.sin(k * phase) / k for k in range(1, 9))
    wave = 0.12 * env * voice + 0.002 * rng.standard_normal(n_samples)
    audio = np.clip(wave * 32767, -32768, 32767).astype(np.int16)
    ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    outer = np.stack([0.5 * np.cos(ang)[None].repeat(n_t, 0), (0.08 + 0.25 * opening)[:, None] * np.sin(ang)], -1)
    inner = np.stack([0.35 * np.cos(ang)[None].repeat(n_t, 0), (0.01 + 0.2 * opening)[:, None] * np.sin(ang)], -1)
    lm = (np.concatenate([outer, inner], 1) + 0.01 * rng.standard_normal((n_t, 40, 2))).astype(np.float16)
    cue = np.stack([np.full(n_t, 1.0), 0.2 * opening, 0.3 * opening, 0.05 * opening, opening, 0.3 * opening,
                    np.full(n_t, 0.5), np.full(n_t, 0.4)], 1).astype(np.float16)
    valid = np.ones(n_t, np.uint8)
    np.savez_compressed(out / f"{utt_id}.npz", lm=lm, cue=cue, valid=valid, audio=audio, fps=np.float64(FPS),
                        sr=np.int64(SR))
    return {
        "utt_id": utt_id, "video_stem": stem, "split_dir": "2.Validation", "speaker": speaker, "gender": "M",
        "age": 3, "specificity": speaker[0], "angle": angle, "session": session, "noise_env": 1, "topic": "test",
        "sentence_id": sentence, "start": 0.0, "end": n_t / FPS, "duration": n_t / FPS, "n_frames": n_t,
        "n_samples": n_samples, "fps": float(FPS), "text": "테스트", "text_raw": "테스트", "has_unk": False,
        "lm_valid_ratio": 1.0, "mouth_mp4": f"feats/{stem}/{utt_id}.mp4", "npz": f"feats/{stem}/{utt_id}.npz",
    }


def make_synthetic_work(work: Path) -> list[dict]:
    """Train S001-S004 (2 sessions x 3 sentences, S001 session 001 also filmed from angle B), val V001 (4 sessions x
    2 sentences x angles A/B), test T001/T002 (3 sessions x 2 sentences). 3.6-4.4 s per utterance."""
    rng = np.random.default_rng(0)
    plan = []
    for spk in ("S001", "S002", "S003", "S004"):
        for ses in ("001", "002"):
            for sent in (1, 2, 3):
                plan.append((spk, ses, sent, ["A", "B"] if (spk, ses) == ("S001", "001") else ["A"]))
    for ses in ("001", "002", "003", "004"):
        for sent in (1, 2):
            plan.append(("V001", ses, sent, ["A", "B"]))
    for spk in ("T001", "T002"):
        for ses in ("001", "002", "003"):
            for sent in (1, 2):
                plan.append((spk, ses, sent, ["A"]))
    rows = []
    f0 = {s: 100 + 25 * i for i, s in enumerate(["S001", "S002", "S003", "S004", "V001", "T001", "T002"])}
    for spk, ses, sent, angles in plan:
        n_t = int(rng.integers(108, 133))
        t = np.arange(n_t) / FPS
        x = sum(rng.uniform(0.4, 1.0) * np.sin(2 * np.pi * rng.uniform(1.5, 5.0) * t + rng.uniform(0, 2 * np.pi))
                for _ in range(3))
        opening = np.clip(0.5 + 0.3 * x, 0.0, 1.0)
        for angle in angles:  # camera angles of one recording share the audio (and the mouth movement)
            rows.append(write_utterance(work, spk, ses, sent, angle, n_t, opening, f0[spk],
                                        np.random.default_rng([sent, int(ses), zlib.crc32(spk.encode("utf-8"))])))
    (work / "manifests").mkdir(parents=True, exist_ok=True)
    by_stem: dict[str, list[dict]] = {}
    for r in rows:
        by_stem.setdefault(r["video_stem"], []).append(r)
    for stem, rs in by_stem.items():
        with open(work / "manifests" / f"{stem}.jsonl", "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (work / "manifests" / f"{stem}.done").write_text("ok", encoding="utf-8")
    return rows


def synthetic_cfg(work: Path, extra: list[str] | None = None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_config(CONFIG, SPLIT_SETS + TINY + [f"work_dir={json.dumps(str(work))}"] + (extra or []))


# ----------------------------------------------------------------------------------------------------------------------
# metric primitives
# ----------------------------------------------------------------------------------------------------------------------
def test_hungarian() -> None:
    # row-wise argmax would give faces 0 and 1 the same stream 0; the optimum (total 2.25) is [1, 0, 2]
    s = np.array([[0.90, 0.80, 0.10],
                  [0.85, 0.20, 0.30],
                  [0.10, 0.70, 0.60]])
    assert sev.assign_streams(s).tolist() == [1, 0, 2]
    assert sev.assign_streams(np.eye(3) + 0.1).tolist() == [0, 1, 2]
    assert sev.assign_streams(np.array([[0.2, 0.9], [0.8, 0.1]])).tolist() == [1, 0]
    print("  hungarian: hand-made 3x3 -> [1, 0, 2] (total 2.25 beats greedy), identity and swap OK")


def test_roc_eer() -> None:
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1.0, 1.0, 200_000), rng.normal(-1.0, 1.0, 200_000)
    r = sev.roc_auc_eer(pos, neg)
    # analytic: AUC = Phi(2 / sqrt 2) = 0.9214, EER = Phi(-1) = 0.1587 at threshold 0
    assert abs(r["auc"] - 0.92135) < 0.003 and abs(r["eer"] - 0.15866) < 0.003 and abs(r["threshold"]) < 0.02, r
    r2 = sev.roc_auc_eer(rng.normal(0.6, 0.1, 50_000), rng.normal(0.2, 0.2, 100_000))
    # unequal spreads: EER at t with (0.6 - t) / 0.1 = (t - 0.2) / 0.2 -> t = 0.4667, EER = Phi(-1.333) = 0.0912
    assert abs(r2["threshold"] - 0.4667) < 0.01 and abs(r2["eer"] - 0.0912) < 0.004, r2
    # exact small case: 8 of 9 (pos, neg) pairs ordered correctly
    r3 = sev.roc_auc_eer([0.9, 0.8, 0.4], [0.5, 0.3, 0.1])
    assert abs(r3["auc"] - 8 / 9) < 1e-12 and abs(r3["eer"] - 1 / 3) < 1e-12, r3
    assert 0.4 <= r3["threshold"] <= 0.5, r3
    # perfect separation: EER 0, AUC 1, the threshold accepts every positive and no negative
    r4 = sev.roc_auc_eer([0.7, 0.8, 0.9], [0.1, 0.2])
    assert r4["auc"] == 1.0 and r4["eer"] == 0.0 and 0.2 < r4["threshold"] <= 0.7, r4
    # ties count 1/2; empty input -> NaN
    assert sev.roc_auc_eer([0.5], [0.5])["auc"] == 0.5
    assert math.isnan(sev.roc_auc_eer([], [0.1])["auc"])
    print(f"  roc/eer: gaussian AUC {r['auc']:.4f} EER {r['eer']:.4f} thr {r['threshold']:+.4f}; "
          f"unequal spreads thr {r2['threshold']:.4f} EER {r2['eer']:.4f}; exact/perfect/ties OK")


def test_selection_credit_and_mixing() -> None:
    true = np.array([0.9, 0.5, 0.5, 0.2])
    others = np.array([[0.1, 0.2], [0.5, 0.1], [0.5, 0.5], [0.3, 0.1]])
    assert sev.selection_credit(true, others).tolist() == [1.0, 0.5, 1 / 3, 0.0]
    # a constant-score model gets exactly chance (1/N) through the tie rule
    assert np.allclose(sev.selection_credit(np.zeros(5), np.zeros((5, 3))), 0.25)
    rng = np.random.default_rng(1)
    a = rng.standard_normal(16000).astype(np.float32)
    b = 0.3 * rng.standard_normal(16000).astype(np.float32)
    silent = np.zeros(16000, np.float32)
    ma, mb, ms = sev.mix_leakage([a, b, silent], 10.0)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))  # noqa: E731
    assert abs(20 * math.log10(rms(a) / rms(ma - a)) - 10.0) < 0.05          # b leaked 10 dB below a
    assert abs(20 * math.log10(rms(b) / rms(mb - b)) - 10.0) < 0.05          # a leaked 10 dB below b
    assert np.array_equal(ms, silent) or rms(ms) < 1e-6                     # nothing leaks into a silent stream
    print("  selection credit (ties -> 1/(1+t)), leakage mixing at the requested level: OK")


def test_candidate_pool() -> None:
    speakers = ["A"] * 6 + ["B"] * 4 + ["C"] * 4
    sessions = ["1", "1", "2", "2", "3", "3", "1", "1", "2", "2", "1", "1", "2", "2"]
    pool = sev.CandidatePool(speakers, sessions)
    for q in range(len(speakers)):
        picks, kinds = pool.draw(speakers[q], sessions[q], 3, sev.seeded_rng(0, "t", q))
        assert len(picks) == 3 and kinds[:2] == [sev.KIND_OTHER, sev.KIND_OTHER], kinds
        assert {speakers[p] for p in picks[:2]} == {"A", "B", "C"} - {speakers[q]}      # distinct other speakers
        groups = [(speakers[q], sessions[q])] + [(speakers[p], sessions[p]) for p in picks]
        assert len(set(groups)) == 4, groups                                            # all different recordings
        assert kinds[2] in (sev.KIND_AGAIN, sev.KIND_SAME)
        again = pool.draw(speakers[q], sessions[q], 3, sev.seeded_rng(0, "t", q))
        assert again == (picks, kinds)                                                   # deterministic
        prefix = pool.draw(speakers[q], sessions[q], 1, sev.seeded_rng(0, "t", q))
        assert prefix == (picks[:1], kinds[:1])                                          # nested N-way sets
    one = sev.CandidatePool(["A"] * 6, ["1", "1", "2", "2", "3", "3"])
    picks, kinds = one.draw("A", "1", 3, sev.seeded_rng(0, "one"))
    assert len(picks) == 2 and kinds == [sev.KIND_SAME] * 2 and {sessions[p] for p in picks} == {"2", "3"}
    print("  candidate pool: distinct other speakers first, fallback only to unused sessions, deterministic, nested")


class CodeEmbedder:
    """Stand-in for :class:`avsr.sync.evaluate.Embedder`: utterance i's face -> ``codes_v[i]``, its audio (recognised
    from the features, see ``tagged_audio``) -> ``codes_a[i]``, constant over the window."""

    def __init__(self, codes_v: torch.Tensor, codes_a: torch.Tensor) -> None:
        self.codes_v, self.codes_a = codes_v, codes_a
        self.emb_dim = codes_v.shape[1]

    def video(self, store, specs, w):
        return torch.stack([self.codes_v[int(i)] for i, _ in specs])[:, None].expand(-1, w, -1).contiguous()

    def audio(self, feats):
        idx = feats[:, 0, 0].round().long()
        return self.codes_a[idx][:, None].expand(-1, feats.shape[1], -1).contiguous()


def fake_store(m: int) -> sev.EvalStore:
    utts = []
    for i in range(m):
        spk = f"P{i % 4}"
        ses = f"{(i // 4) % 6:03d}"
        utts.append(sev.Utterance(
            utt_id=f"u{i:04d}", speaker=spk, session=ses, angle="A", audio_key=(spk, ses, str(i)), n=60, n_video=60,
            frames=np.zeros((0, 1, 88, 88), np.uint8), lm=np.zeros((0, 80), np.float32),
            cue=np.zeros((0, 8), np.float32), valid=np.zeros(0, np.float32), wave=np.zeros(0, np.int16)))
    store = sev.EvalStore(utts, {"video": {"norm": "utterance"}})
    store.audio_windows = lambda i, starts, w: torch.full((len(starts), w, 320), float(i))  # tag = utterance index
    return store


def test_nway_oracle_and_random() -> None:
    m, e, w = 400, 64, 25
    store = fake_store(m)
    g = torch.Generator().manual_seed(0)
    unit = lambda x: torch.nn.functional.normalize(x, dim=-1)  # noqa: E731
    codes = unit(torch.randn(m, e, generator=g))
    oracle = sev.SyncEvaluator(store, CodeEmbedder(codes, codes), seed=1, draw_k=3)
    rand = sev.SyncEvaluator(store, CodeEmbedder(unit(torch.randn(m, e, generator=g)),
                                                 unit(torch.randn(m, e, generator=g))), seed=1, draw_k=3)
    accs = {}
    for direction in ("v2a", "a2v"):
        res_o = oracle.nway(w, [2, 3, 4], direction)
        res_r = rand.nway(w, [2, 3, 4], direction)
        for n in (2, 3, 4):
            assert res_o[str(n)]["acc"] == 1.0 and res_o[str(n)]["n"] == m, res_o[str(n)]
            acc = res_r[str(n)]["acc"]
            assert abs(acc - 1.0 / n) < 0.07, (direction, n, acc)
            accs[(direction, n)] = acc
            kinds = res_o[str(n)]["distractors"]
            assert kinds == {sev.KIND_OTHER: m * (n - 1)}, kinds  # 4 speakers: never a fallback for N <= 4
    ver = oracle.verification(w)
    assert ver["auc"] == 1.0 and ver["eer"] == 0.0 and ver["negatives"] == "different speaker", ver
    ver_r = rand.verification(w)
    assert abs(ver_r["auc"] - 0.5) < 0.05, ver_r
    print("  n-way with stand-in embeddings: oracle 1.00 (both directions), random "
          + ", ".join(f"{d} N={n} {a:.2f}" for (d, n), a in accs.items()) + f"; verification AUC oracle 1.0, "
          f"random {ver_r['auc']:.2f}")


def test_store_equivalence(work: Path) -> None:
    """EvalStore windows == SyncWindowDataset eval items (the model sees identical inputs in both)."""
    for norm in ("utterance", "global"):
        cfg = synthetic_cfg(work, [f"video.norm={norm}", f"video.cue_norm={'utterance' if norm == 'utterance' else 'none'}"])
        rows = assign_split(load_manifests(work), cfg)["val"][:6] + assign_split(load_manifests(work), cfg)["test"][:3]
        store = sev.EvalStore.load(rows, cfg, 0)
        assert len(store) == len(rows) and store.n_bad == 0
        for w in (13, 25, 50):
            ds = SyncWindowDataset(rows, cfg, train=False, window=w, seed=int(cfg.seed))
            assert [r["utt_id"] for r in ds.rows] == [store.utts[i].utt_id for i in store.eligible(w)]
            for j, i in enumerate(store.eligible(w)):
                item = ds[j]
                s = store.start(i, w)
                assert s == item["start_frame"]
                win = store.video_windows([(i, s)], w)
                for k in ("video", "lm", "cue", "valid"):
                    assert torch.allclose(win[k][0], item[k], atol=1e-5), (norm, w, k)
                aud = store.audio_windows(i, [s], w)[0]
                assert torch.allclose(aud, item["audio"], atol=CHUNK_TOL), float((aud - item["audio"]).abs().max())
                reg = store.region_features(store.region(i, s, w), w)
                assert torch.allclose(reg, aud, atol=CHUNK_TOL), float((reg - aud).abs().max())
                sub = store.region_subwindows(store.region(i, s, w), [0, 3], 10)
                ref = torch.stack([audio_window_features(store.utts[i].wave, s + t, 10) for t in (0, 3)])
                assert torch.allclose(sub, ref, atol=CHUNK_TOL)
    print("  eval store: windows == SyncWindowDataset eval items (video/lm/cue/valid atol 1e-5, audio and mixing "
          "regions within the CMVN-epsilon tolerance) for windows 13/25/50, both pixel norms")


class RandomEmbedder:
    """An information-free model: a fresh random unit vector for every frame of every window."""

    def __init__(self, dim: int = 16, seed: int = 0) -> None:
        self.emb_dim = dim
        self.g = torch.Generator().manual_seed(seed)

    def _rand(self, b: int, w: int) -> torch.Tensor:
        return torch.nn.functional.normalize(torch.randn(b, w, self.emb_dim, generator=self.g), dim=-1)

    def video(self, store, specs, w):
        return self._rand(len(list(specs)), w)

    def audio(self, feats):
        return self._rand(int(feats.shape[0]), int(feats.shape[1]))


def test_chance_level_pipeline(work: Path) -> None:
    """Scenes / leakage / offset / reverse run the real window and mixing code with an information-free embedder:
    every accuracy must sit at chance (a structural leak, e.g. comparing a stream with itself, would show up here)."""
    cfg = synthetic_cfg(work)
    splits = assign_split(load_manifests(work), cfg)
    store = sev.EvalStore.load(splits["val"] + splits["test"], cfg, 0)
    ev = sev.SyncEvaluator(store, RandomEmbedder(), seed=3, draw_k=3)
    sc2 = ev.scenes(2, 400, 25, [10.0])
    sc3 = ev.scenes(3, 400, 25, [10.0])
    for sc, chance in ((sc2, 0.5), (sc3, 1 / 6)):
        for part, key in (("whole", "scene_acc"), ("subwindows", "window_acc")):
            for cond in ("clean", "leak_10dB"):
                acc = sc[part][cond][key]
                assert abs(acc - chance) < 0.08, (len(sc["partners"]), part, cond, acc, chance)
    leak = ev.leakage(25, [2, 4], [10.0])
    off = ev.offsets(25)
    rev = ev.nway(25, [2], "a2v")["2"]["acc"]
    assert abs(off["acc_pm1"] - 3 / 31) < 0.2 and 0.1 < rev < 0.9, (off["acc_pm1"], rev)
    assert all(0.0 <= leak[n]["acc"]["leak_10dB"] <= 1.0 for n in ("2", "4"))
    print(f"  chance level with random embeddings: scenes K=2 {sc2['whole']['clean']['scene_acc']:.2f} "
          f"(chance 0.50), K=3 {sc3['whole']['clean']['scene_acc']:.2f} (0.17), 1-s windows "
          f"{sc2['subwindows']['clean']['window_acc']:.2f}/{sc3['subwindows']['clean']['window_acc']:.2f}, "
          f"offset +-1 {off['acc_pm1']:.2f} (0.10), leak 2-way {leak['2']['acc']['leak_10dB']:.2f}")


# ----------------------------------------------------------------------------------------------------------------------
# end-to-end smoke
# ----------------------------------------------------------------------------------------------------------------------
def set_args(sets: list[str]) -> list[str]:
    return [a for s in sets for a in ("--set", s)]


def run_train_smoke(work: Path) -> Path:
    common = ["--config", str(CONFIG), "--work-dir", str(work)] + set_args(SPLIT_SETS + TINY)
    missing = work / "no_such_avsr.pt"
    t0 = time.time()
    rc = strain.main(common + set_args(["train.epochs=1", "data.num_workers=2", f"sync.init_from={missing.as_posix()}"]))
    assert rc == 0, f"train returned {rc}"
    ckpt_dir = work / "checkpoints_sync"
    for name in ("last.pt", "best.pt", "epoch_001.pt"):
        assert (ckpt_dir / name).is_file(), f"missing {ckpt_dir / name}"
    ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    for key in ("kind", "model", "optimizer", "scheduler", "step", "epoch", "cfg", "best", "val", "limit", "init"):
        assert key in ck, f"checkpoint lacks {key!r}"
    assert ck["kind"] == "sync" and ck["epoch"] == 1 and ck["step"] == 3 and ck["limit"] is None, (ck["epoch"],
                                                                                                    ck["step"])
    assert ck["cfg"]["sync"]["hidden"] == 32 and ck["init"]["status"] == "missing"
    log_text = (work / "logs" / "sync_train.log").read_text(encoding="utf-8")
    assert "not found: training ALL weights from scratch" in log_text
    t1 = time.time()

    # epoch 2 resumes from last.pt (train.resume: auto)
    rc = strain.main(common + set_args(["train.epochs=2", "data.num_workers=0"]))
    assert rc == 0, f"resumed train returned {rc}"
    ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    assert ck["epoch"] == 2 and ck["step"] == 6, (ck["epoch"], ck["step"])
    assert (ckpt_dir / "epoch_002.pt").is_file() and not (ckpt_dir / "epoch_001.pt").exists()  # keep_last=1
    log_text = (work / "logs" / "sync_train.log").read_text(encoding="utf-8")
    assert "resumed from" in log_text, "second run did not resume"
    metrics = read_jsonl(work / "logs" / "sync_metrics.jsonl")
    kinds = {m["type"] for m in metrics}
    assert {"start", "train", "val", "epoch"} <= kinds, kinds
    val = [m for m in metrics if m["type"] == "val"]
    assert len(val) == 2 and {"acc2", "acc3", "acc4", "offset_acc", "offset_median_ms"} <= set(val[-1]), val[-1]
    assert val[-1]["acc4"] is not None and val[-1]["n"] == 16, val[-1]
    train_rec = [m for m in metrics if m["type"] == "train"][-1]
    assert {"loss", "acc_va", "items_per_sec", "sec_per_step", "data_wait_frac"} <= set(train_rec), train_rec
    best = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    assert best["best"]["acc"] == max(v["acc4"] for v in val)

    # a --limit (smoke) run refuses to resume or overwrite the checkpoints of a full run; its logs are separate
    try:
        strain.main(common + set_args(["train.epochs=3", "data.num_workers=0"]) + ["--limit", "20"])
    except RuntimeError as e:
        assert "--limit" in str(e), e
    else:
        raise AssertionError("a --limit run resumed the last.pt of a full run")
    assert (work / "logs" / "sync_smoke_train.log").is_file()
    print(f"  train: 2 epochs x 3 steps (spawn workers {t1 - t0:.0f} s, resume {time.time() - t1:.0f} s); last/best/"
          f"epoch_002 written, resume + --limit refusal + missing init_from OK; val acc4 "
          f"{[round(v['acc4'], 3) for v in val]}")
    return ckpt_dir / "best.pt"


def run_interrupt(work: Path) -> None:
    """Ctrl+C during the first step -> last.pt saved mid-epoch (rc 130); a resumed run skips the consumed batch."""
    ckpt_dir = work / "ckpt_interrupt"
    cfg = synthetic_cfg(work, [f"train.ckpt_dir={json.dumps(str(ckpt_dir))}", "train.epochs=1", "sync.init_from=",
                               "data.num_workers=0"])
    logger = get_logger(work, "sync_train_interrupt")
    try:
        trainer = strain.SyncTrainer(cfg, None, logger)
        original = trainer._train_step

        def interrupted_step(batch: dict, meters: dict):
            signal.raise_signal(signal.SIGINT)  # handled by GracefulStop: finish this step, then save and stop
            return original(batch, meters)

        trainer._train_step = interrupted_step
        assert trainer.run() == 130, "interrupted run should return 130"
        ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        assert (ck["epoch"], ck["batch_in_epoch"], ck["step"]) == (0, 1, 1), (ck["epoch"], ck["batch_in_epoch"])
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, "SIGINT handler not restored"
        resumed = strain.SyncTrainer(cfg, None, logger)
        assert (resumed.epoch, resumed.batch_in_epoch, resumed.step) == (0, 1, 1)
        assert resumed.run() == 0
        ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        assert (ck["epoch"], ck["batch_in_epoch"], ck["step"]) == (1, 0, 3), (ck["epoch"], ck["step"])
        assert (ckpt_dir / "best.pt").is_file()
    finally:
        close_logger()
    print("  interrupt: Ctrl+C mid-epoch -> rc 130, last.pt at (epoch 0, batch 1); resume finished the epoch")


def run_ckpt_dir_guards(work: Path) -> None:
    """Fresh runs in a used checkpoint directory and the init checkpoint's split:
    - an earlier run's higher-numbered epoch_XXX.pt never make this run's own epoch files rotate away;
    - a directory whose best.pt is not a sync checkpoint (e.g. the AVSR model's) is refused, the file untouched;
    - an init checkpoint that did not hold out this split's val/test speakers is reported (train log, checkpoint
      'init', evaluate notes)."""
    fresh = ["train.epochs=2", "data.num_workers=0", "train.keep_last=1", "train.resume=none"]
    ck = work / "ckpt_rotation"
    ck.mkdir(parents=True)
    for n in (28, 29, 30):  # the rotating files an earlier 30-epoch run left behind
        torch.save({"kind": "sync", "epoch": n, "marker": "earlier run"}, ck / f"epoch_{n:03d}.pt")
    fake_avsr = work / "fake_avsr.pt"  # an "AVSR" checkpoint trained with only V001 held out
    torch.save({"model": {"visual_frontend.unused": torch.zeros(1)},
                "cfg": {"split": {"val_speakers": ["V001"], "test_speakers": []}}}, fake_avsr)
    cfg = synthetic_cfg(work, [f"train.ckpt_dir={json.dumps(str(ck))}",
                               f"sync.init_from={json.dumps(str(fake_avsr))}"] + fresh)
    logger = get_logger(work, "sync_train_guards")
    try:
        assert strain.SyncTrainer(cfg, None, logger).run() == 0
    finally:
        close_logger()
    names = sorted(p.name for p in ck.glob("epoch_*.pt"))
    assert names == ["epoch_002.pt", "epoch_028.pt", "epoch_029.pt", "epoch_030.pt"], names
    assert "marker" not in torch.load(ck / "epoch_002.pt", map_location="cpu", weights_only=False)
    last = torch.load(ck / "last.pt", map_location="cpu", weights_only=False)
    assert last["init"]["heldout_not_held_out_by_init"] == ["T001", "T002"], last["init"]
    log_text = (work / "logs" / "sync_train_guards.log").read_text(encoding="utf-8")
    assert "holds files of an earlier sync run" in log_text
    assert "WITHOUT holding out the speakers ['T001', 'T002']" in log_text

    out = work / "eval" / "sync_guard.json"
    assert sev.main(["--ckpt", str(ck / "best.pt"), "--config", str(CONFIG), "--work-dir", str(work), "--split", "test",
                     "--max-utts", "8", "--scenes", "4", "--out", str(out)]
                    + set_args(SPLIT_SETS + ["sync.eval_windows=[25]", "data.num_workers=0"])) == 0
    notes = json.loads(out.read_text(encoding="utf-8"))["notes"]
    assert any("optimistic" in n and "T001" in n and "T002" in n for n in notes), notes

    foreign = work / "ckpt_foreign"
    foreign.mkdir()
    torch.save({"model": {}, "epoch": 26, "step": 36712}, foreign / "best.pt")  # AVSR-style: no 'kind', no last.pt
    before = (foreign / "best.pt").read_bytes()
    for resume in ("auto", "none"):
        cfg = synthetic_cfg(work, [f"train.ckpt_dir={json.dumps(str(foreign))}", "sync.init_from=",
                                   f"train.resume={resume}", "train.epochs=1", "data.num_workers=0"])
        logger = get_logger(work, "sync_train_guards")
        try:
            strain.SyncTrainer(cfg, None, logger)
        except RuntimeError as e:
            assert "not sync checkpoints" in str(e), e
        else:
            raise AssertionError(f"resume={resume}: a new run accepted a directory holding a non-sync best.pt")
        finally:
            close_logger()
    assert (foreign / "best.pt").read_bytes() == before and not (foreign / "last.pt").exists()
    print("  checkpoint dir guards: earlier run's epoch_028..030 kept while this run rotates its own (epoch_002 "
          "kept, epoch_001 rotated); non-sync best.pt refused and untouched; init split exposure logged + noted")


def strip_volatile(res: dict) -> dict:
    return {k: v for k, v in res.items() if k not in ("timing_sec", "elapsed_sec")}


def run_eval_smoke(work: Path, ckpt: Path) -> None:
    common = ["--ckpt", str(ckpt), "--config", str(CONFIG), "--work-dir", str(work), "--scenes", "40"] + \
        set_args(SPLIT_SETS)
    t0 = time.time()
    rc = sev.main(common + ["--split", "val"] + set_args(["data.num_workers=2"]))
    assert rc == 0, f"evaluate returned {rc}"
    path = work / "eval" / "sync_val_results.json"
    res = json.loads(path.read_text(encoding="utf-8"))
    for key in ("nway", "reverse", "leakage", "offset", "scenes", "verification", "offscreen_threshold", "notes",
                "distractor_policy", "examples", "settings"):
        assert key in res, f"result JSON lacks {key!r}"
    assert res["n_utts"] == 16 and res["speakers"] == {"V001": 16}
    for w in ("13", "25", "50"):
        for n in ("2", "3", "4"):
            e = res["nway"][w][n]
            assert e["n"] == 16 and 0.0 <= e["acc"] <= 1.0 and set(e["distractors"]) == {sev.KIND_SAME}, e
            assert 0.0 <= res["reverse"][w][n]["acc"] <= 1.0
        for n in ("2", "4"):
            accs = res["leakage"][w][n]["acc"]
            assert set(accs) == {"clean", "leak_20dB", "leak_10dB", "leak_5dB"} and all(0 <= a <= 1 for a in
                                                                                         accs.values()), accs
        off = res["offset"][w]
        assert off["n"] == 16 and 0 <= off["acc_pm1"] <= 1 and sum(off["histogram"].values()) == 16, off
        v = res["verification"][w]
        assert v["n_pos"] == 16 and v["n_neg"] > 0 and 0 <= v["auc"] <= 1 and v["threshold"] is not None, v
        assert v["negatives"].startswith("same speaker")
    for k in ("2", "3"):
        sc = res["scenes"][k]
        assert sc["n_scenes"] == 40 and set(sc["partners"]) == {sev.KIND_SAME}, sc
        assert set(sc["whole"]) == {"clean", "leak_20dB", "leak_10dB", "leak_5dB"}
        assert sc["subwindows"]["clean"]["n_windows"] >= 40 * 3
    ot = res["offscreen_threshold"]
    assert ot["window_frames"] == 25 and ot["value"] == res["verification"]["25"]["threshold"], ot
    assert any("fewer than N" in n for n in res["notes"]) and any("scenes K=3" in n for n in res["notes"]), res["notes"]
    t1 = time.time()

    # heldout = val + test: 3 speakers -> N <= 3 and K = 3 without fallback; determinism across worker counts
    out1, out2 = work / "eval" / "sync_heldout_a.json", work / "eval" / "sync_heldout_b.json"
    assert sev.main(common + ["--split", "heldout", "--out", str(out1)] + set_args(["data.num_workers=2"])) == 0
    assert sev.main(common + ["--split", "heldout", "--out", str(out2)] + set_args(["data.num_workers=0"])) == 0
    r1 = json.loads(out1.read_text(encoding="utf-8"))
    r2 = json.loads(out2.read_text(encoding="utf-8"))
    assert strip_volatile(r1) == strip_volatile(r2), "evaluation is not deterministic"
    assert r1["speakers"] == {"T001": 6, "T002": 6, "V001": 16}
    for w in ("13", "25", "50"):
        assert r1["nway"][w]["2"]["all_other_speakers"] == 1.0 and r1["nway"][w]["3"]["all_other_speakers"] == 1.0
        assert r1["nway"][w]["4"]["all_other_speakers"] == 0.0  # 3 speakers: the 3rd distractor is a fallback
        assert r1["verification"][w]["negatives"] == "different speaker"
    assert set(r1["scenes"]["3"]["partners"]) == {sev.KIND_OTHER}
    # an empty split -> exit code 1, no crash
    assert sev.main(common + ["--split", "val", "--out", str(work / "eval" / "x.json")]
                    + set_args(["split.val_speakers=[NOBODY]"])) == 1
    table = sev.format_tables(r1, 40.0)
    assert "N-way selection" in table and "offscreen_threshold" in table
    print(f"  evaluate: val (1 speaker, fallback noted) {t1 - t0:.0f} s; heldout x2 identical (workers 2 vs 0) "
          f"{time.time() - t1:.0f} s; val 4-way@1s {res['nway']['25']['4']['acc']:.2f}, heldout 2-way@1s "
          f"{r1['nway']['25']['2']['acc']:.2f}, offscreen_threshold {ot['value']:.3f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--keep", action="store_true", help="keep the temporary work_dir")
    args = p.parse_args(argv)
    t_all = time.time()
    for fn in (test_hungarian, test_roc_eer, test_selection_credit_and_mixing, test_candidate_pool,
               test_nway_oracle_and_random):
        t0 = time.time()
        fn()
        print(f"{fn.__name__}: OK ({time.time() - t0:.1f} s)")
    tmp = Path(tempfile.mkdtemp(prefix="avsr_sync_train_"))
    try:
        work = tmp / "work"
        t0 = time.time()
        rows = make_synthetic_work(work)
        print(f"synthetic work dir: {len(rows)} utterances ({time.time() - t0:.1f} s)")
        t0 = time.time()
        test_store_equivalence(work)
        print(f"test_store_equivalence: OK ({time.time() - t0:.1f} s)")
        t0 = time.time()
        test_chance_level_pipeline(work)
        print(f"test_chance_level_pipeline: OK ({time.time() - t0:.1f} s)")
        t0 = time.time()
        best = run_train_smoke(work)
        print(f"run_train_smoke: OK ({time.time() - t0:.1f} s)")
        t0 = time.time()
        run_interrupt(work)
        print(f"run_interrupt: OK ({time.time() - t0:.1f} s)")
        t0 = time.time()
        run_ckpt_dir_guards(work)
        print(f"run_ckpt_dir_guards: OK ({time.time() - t0:.1f} s)")
        t0 = time.time()
        run_eval_smoke(work, best)
        print(f"run_eval_smoke: OK ({time.time() - t0:.1f} s)")
    finally:
        close_logger()
        if args.keep:
            print("  kept work_dir:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_sync_train: ALL OK ({time.time() - t_all:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Smoke test for avsr.utils / avsr.train / avsr.evaluate on 4 synthetic utterances (no real data needed).

Run from the project root:
    & $py -m tests.test_train_smoke            # real avsr.dataset + avsr.models (skips if they are missing)
    & $py -m tests.test_train_smoke --stubs    # tiny in-file stand-ins for dataset/models (tests train/eval logic only)

Everything is written into a temporary work_dir; the real ``work/`` directory is never touched.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import signal
import sys
import tempfile
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from avsr.text import Tokenizer  # noqa: E402
from avsr.utils import (AverageMeter, cer, close_logger, even_subset, get_logger, load_config,  # noqa: E402
                        read_jsonl, to_dict, wer)

CONFIG = ROOT / "configs" / "base.yaml"
# (speaker, text, duration s); default split: C313 -> val, others -> train.
UTTS = [
    ("E220", "안녕하세요", 2.0),
    ("C084", "감사합니다.", 2.2),
    ("E205", "좋은 아침이에요", 2.4),
    ("C313", "네 알겠어요?", 2.1),
]
TINY_MODEL = [
    "model.d_model=64", "model.encoder.layers=1", "model.encoder.heads=2", "model.encoder.ffn=128",
    "model.encoder.conv_kernel=7", "model.decoder.layers=1", "model.decoder.heads=2", "model.decoder.ffn=128",
]


# --------------------------------------------------------------------------------------------------------------
# Synthetic data (SPEC section 4 layout)
# --------------------------------------------------------------------------------------------------------------


def write_synthetic_utterance(work_dir: Path, video_stem: str, sentence_id: int, speaker: str, text: str,
                              duration: float, rng: np.random.Generator) -> dict:
    """Write mouth mp4 + npz for one utterance and return its manifest row."""
    utt_id = f"{video_stem}__{sentence_id:03d}"
    feat_dir = work_dir / "feats" / video_stem
    feat_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(duration * 30))
    n_samples = int(round(duration * 16000))

    # mouth video: dark ellipse whose opening follows a ~4 Hz syllable rhythm
    writer = cv2.VideoWriter(str(feat_dir / f"{utt_id}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (96, 96))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open an mp4v writer")
    opening = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * np.arange(n_frames) / 30.0 + rng.uniform(0, np.pi))
    for t in range(n_frames):
        frame = np.full((96, 96, 3), (120, 140, 190), np.uint8)
        cv2.ellipse(frame, (48, 52), (22, int(3 + 12 * opening[t])), 0, 0, 360, (40, 30, 150), -1)
        cv2.ellipse(frame, (48, 52), (16, int(1 + 8 * opening[t])), 0, 0, 360, (20, 15, 30), -1)
        writer.write(frame)
    writer.release()

    # audio: harmonic "voice" with the same syllable envelope + a little noise
    t_a = np.arange(n_samples) / 16000.0
    env = np.interp(t_a, np.arange(n_frames) / 30.0, opening)
    f0 = rng.uniform(110, 220)
    voice = sum(np.sin(2 * np.pi * f0 * k * t_a) / k for k in range(1, 6))
    wave = 0.15 * env * voice + 0.003 * rng.standard_normal(n_samples)
    audio = np.clip(wave * 32767, -32768, 32767).astype(np.int16)

    lm = (0.3 * rng.standard_normal((n_frames, 40, 2))).astype(np.float16)
    cue = np.stack([opening] * 8, axis=1).astype(np.float16)
    np.savez_compressed(feat_dir / f"{utt_id}.npz", lm=lm, cue=cue, valid=np.ones(n_frames, np.uint8),
                        audio=audio, fps=np.float64(30.0), sr=np.int64(16000))
    parts = video_stem.split("_")
    return {
        "utt_id": utt_id, "video_stem": video_stem, "split_dir": "2.Validation", "speaker": speaker,
        "gender": parts[3], "age": int(parts[4]), "specificity": speaker[0], "angle": parts[6],
        "session": parts[7], "noise_env": 1, "topic": "test", "sentence_id": sentence_id,
        "start": 0.0, "end": duration, "duration": duration, "n_frames": n_frames, "n_samples": n_samples,
        "text": text, "text_raw": text, "has_unk": False, "lm_valid_ratio": 1.0,
        "mouth_mp4": f"feats/{video_stem}/{utt_id}.mp4", "npz": f"feats/{video_stem}/{utt_id}.npz",
    }


def make_synthetic_work_dir(work_dir: Path) -> list[dict]:
    """4 utterances (one video stem per speaker), manifests written as shards + .done markers."""
    rng = np.random.default_rng(0)
    (work_dir / "manifests").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (speaker, text, dur) in enumerate(UTTS):
        stem = f"lip_J_1_F_03_{speaker}_A_{i + 1:03d}"
        row = write_synthetic_utterance(work_dir, stem, 1, speaker, text, dur, rng)
        with open(work_dir / "manifests" / f"{stem}.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (work_dir / "manifests" / f"{stem}.done").write_text("ok", encoding="utf-8")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------------------------
# Optional stand-ins for avsr.dataset / avsr.models (only with --stubs; SPEC sections 8 and 10 APIs)
# --------------------------------------------------------------------------------------------------------------


def _stub_modules() -> tuple[types.ModuleType, types.ModuleType]:
    import torch.nn as nn
    import torch.nn.functional as F

    def load_manifests(work_dir):
        rows = []
        for p in sorted(Path(work_dir).glob("manifests/*.jsonl")):
            rows += read_jsonl(p)
        for r in rows:
            r["_work_dir"] = str(work_dir)
        return rows

    def assign_split(rows, cfg):
        out = {"train": [], "val": [], "test": []}
        for r in rows:
            if r["has_unk"]:
                continue
            key = ("val" if r["speaker"] in cfg.split.val_speakers else
                   "test" if r["speaker"] in cfg.split.test_speakers else "train")
            out[key].append(r)
        return out

    class AVSRDataset(torch.utils.data.Dataset):
        def __init__(self, rows, cfg, train, tokenizer, fixed_noise=None):
            self.rows, self.cfg, self.tok, self.fixed_noise = list(rows), cfg, tokenizer, fixed_noise

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            wd = Path(r["_work_dir"])
            z = np.load(wd / r["npz"])
            cap = cv2.VideoCapture(str(wd / r["mouth_mp4"]))
            frames = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[4:92, 4:92])
            cap.release()
            idx = [min(round(k * 30 / 25), len(frames) - 1) for k in range(int(len(frames) * 25 / 30))]
            video = (torch.from_numpy(np.stack(frames)[idx]).float() / 255 - 0.421) / 0.165
            wave = z["audio"].astype(np.float32) / 32768.0
            if self.fixed_noise is not None:
                noise = np.random.default_rng(i).standard_normal(len(wave)).astype(np.float32)
                scale = np.sqrt((wave ** 2).mean() / (noise ** 2).mean() / 10 ** (self.fixed_noise[1] / 10))
                wave = wave + scale * noise
            n = len(wave) // 640
            spec = np.log1p(np.abs(np.fft.rfft(wave[: n * 640].reshape(n, 640), axis=1))[:, :320])
            t = min(len(idx), n)
            lm = torch.from_numpy(z["lm"].astype(np.float32).reshape(len(frames), 80)[idx])
            return {"video": video[:t, None], "lm": lm[:t], "cue": torch.from_numpy(z["cue"].astype(np.float32)[idx])[:t],
                    "valid": torch.ones(t), "audio": torch.from_numpy(spec[:t]).float(),
                    "snr_bucket": torch.tensor(0 if self.fixed_noise is None else 2),
                    "tokens": torch.tensor(self.tok.encode(r["text"])), "text": r["text"], "utt_id": r["utt_id"],
                    "meta": {"speaker": r["speaker"]}}

    def collate_fn(items):
        t = max(it["audio"].shape[0] for it in items)
        l_max = max(len(it["tokens"]) for it in items)

        def pad(key):
            x = items[0][key]
            out = torch.zeros((len(items), t) + tuple(x.shape[1:]))
            for j, it in enumerate(items):
                out[j, : it[key].shape[0]] = it[key]
            return out

        tokens = torch.full((len(items), l_max), 1, dtype=torch.long)
        for j, it in enumerate(items):
            tokens[j, : len(it["tokens"])] = it["tokens"]
        return {"video": pad("video"), "lm": pad("lm"), "cue": pad("cue"), "valid": pad("valid"),
                "audio": pad("audio"), "lengths": torch.tensor([it["audio"].shape[0] for it in items]),
                "tokens": tokens, "token_lengths": torch.tensor([len(it["tokens"]) for it in items]),
                "snr_bucket": torch.stack([it["snr_bucket"] for it in items]),
                "texts": [it["text"] for it in items], "utt_ids": [it["utt_id"] for it in items],
                "metas": [it["meta"] for it in items]}

    class DurationBatchSampler:
        def __init__(self, rows, max_frames, shuffle=True, seed=0):
            self.rows, self.max_frames, self.shuffle, self.seed, self.epoch = rows, max_frames, shuffle, seed, 0

        def set_epoch(self, epoch):
            self.epoch = epoch

        def _batches(self):
            order = list(range(len(self.rows)))
            if self.shuffle:
                np.random.default_rng(self.seed + self.epoch).shuffle(order)
            batches, cur, frames = [], [], 0
            for i in order:
                f = self.rows[i]["n_frames"] * 25 // 30
                if cur and frames + f > self.max_frames:
                    batches.append(cur)
                    cur, frames = [], 0
                cur.append(i)
                frames += f
            return batches + ([cur] if cur else [])

        def __iter__(self):
            return iter(self._batches())

        def __len__(self):
            return len(self._batches())

    class StubModel(nn.Module):
        def __init__(self, cfg, vocab_size):
            super().__init__()
            d = int(cfg.model.d_model)
            self.a, self.v, self.s = nn.Linear(320, d), nn.Linear(64, d), nn.Linear(88, d)
            self.pool = nn.AdaptiveAvgPool2d(8)
            self.ctc, self.snr = nn.Linear(d, vocab_size), nn.Linear(d, 4)

        def forward(self, batch, mode="av"):
            b, t = batch["video"].shape[:2]
            v = self.v(self.pool(batch["video"].flatten(0, 1)).reshape(b, t, 64))
            v = v + self.s(torch.cat([batch["lm"], batch["cue"]], -1))
            a = self.a(batch["audio"])
            x = a if mode == "audio" else v if mode == "video" else a + v
            return {"ctc_logits": self.ctc(x), "enc_out": x, "enc_lengths": batch["lengths"],
                    "snr_logits": self.snr(a.mean(1))}

        def compute_loss(self, out, batch, cfg):
            lp = out["ctc_logits"].float().log_softmax(-1).transpose(0, 1)
            ctc = F.ctc_loss(lp, batch["tokens"], batch["lengths"], batch["token_lengths"], blank=0,
                             zero_infinity=True)
            snr = F.cross_entropy(out["snr_logits"].float(), batch["snr_bucket"])
            loss = ctc + float(cfg.model.snr_head_weight) * snr
            return loss, {"ctc": float(ctc.detach()), "snr": float(snr.detach())}

        def decode(self, batch, mode, method="ctc_greedy", max_len=None):
            best = self(batch, mode)["ctc_logits"].argmax(-1).cpu()
            hyps = []
            for row, n in zip(best.tolist(), batch["lengths"].tolist()):
                ids, prev = [], -1
                for i in row[:n]:
                    if i != prev and (i >= 5):
                        ids.append(i)
                    prev = i
                hyps.append(ids)
            return hyps

    def build_model(cfg, vocab_size):
        return StubModel(cfg, vocab_size)

    ds_mod = types.ModuleType("avsr.dataset")
    ds_mod.__dict__.update(load_manifests=load_manifests, assign_split=assign_split, AVSRDataset=AVSRDataset,
                           collate_fn=collate_fn, DurationBatchSampler=DurationBatchSampler)
    md_mod = types.ModuleType("avsr.models")
    md_mod.__dict__.update(build_model=build_model, AVSRModel=StubModel)
    return ds_mod, md_mod


# --------------------------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------------------------


def test_utils() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = load_config(CONFIG, ["train.epochs=1", "train.lr=3e-4", "eval.snr_sweep=[20, 0]",
                                   "data.angles=[A, B]", "work_dir=C:/tmp/run1", "eval.max_val_utts=10"])
    assert cfg.train.epochs == 1 and cfg["train"]["epochs"] == 1
    assert isinstance(cfg.train.lr, float) and abs(cfg.train.lr - 3e-4) < 1e-12
    assert cfg.eval.snr_sweep == [20, 0] and cfg.data.angles == ["A", "B"]
    assert Path(cfg.train.ckpt_dir) == Path("C:/tmp/run1/checkpoints"), cfg.train.ckpt_dir  # re-based
    assert cfg.eval.max_val_utts == 10 and cfg.get("missing", 7) == 7 and getattr(cfg, "missing", None) is None
    assert cfg.split.train_split_dirs == ["2.Validation", "1.Training"]
    plain = to_dict(cfg)
    assert type(plain) is dict and type(plain["model"]["encoder"]) is dict
    assert load_config(CONFIG).train.ckpt_dir == "work/checkpoints"
    assert abs(cer(["안녕 하세요"], ["안녕하세요"])) < 1e-9          # spaces ignored by CER
    assert abs(wer(["안녕 하세요", "감사"], ["안녕 하세요", ""]) - 1 / 3) < 1e-9
    assert math.isnan(cer([""], ["x"]))
    subset = even_subset([{"utt_id": f"{i:02d}"} for i in range(10)], 3)
    assert [r["utt_id"] for r in subset] == ["00", "03", "06"]
    m = AverageMeter()
    m.update(1.0, 2)
    m.update(4.0, 1)
    assert abs(m.avg - 2.0) < 1e-9


def test_helpers(train_mod: types.ModuleType, eval_mod: types.ModuleType) -> None:
    f = [train_mod.lr_factor(s, warmup=10, total=100, floor=0.02) for s in range(101)]
    assert abs(f[9] - 1.0) < 1e-9 and f[0] < f[5] < f[9]
    assert abs(f[100] - 0.02) < 1e-9 and all(a >= b for a, b in zip(f[10:], f[11:]))
    thr, status, _ = eval_mod.suggest_threshold({20: 0.1, 10: 0.2, 5: 0.35, 0: 0.6}, video_cer=0.3)
    assert (thr, status) == (5, "ok")
    thr, status, _ = eval_mod.suggest_threshold({20: 0.1, 0: 0.2}, video_cer=0.5)
    assert (thr, status) == (None, "video_never_better")
    thr, status, _ = eval_mod.suggest_threshold({20: 0.6, 0: 0.7}, video_cer=0.5)
    assert (thr, status) == (20, "video_always_better")


def run_smoke(work_dir: Path, train_mod: types.ModuleType, eval_mod: types.ModuleType, num_workers: int) -> None:
    make_synthetic_work_dir(work_dir)
    common = ["--config", str(CONFIG), "--work-dir", str(work_dir)]
    sets = TINY_MODEL + [f"data.num_workers={num_workers}", "train.log_every=1", "train.warmup_steps=4",
                         "train.keep_last=1"]
    set_args = [a for s in sets for a in ("--set", s)]

    # epoch 1 from scratch
    rc = train_mod.main(common + set_args + ["--set", "train.epochs=1"])
    assert rc == 0, f"train returned {rc}"
    ckpt_dir = work_dir / "checkpoints"
    for name in ("last.pt", "best.pt", "epoch_001.pt"):
        assert (ckpt_dir / name).is_file(), f"missing {ckpt_dir / name}"
    ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "scheduler", "step", "epoch", "cfg", "vocab", "best"):
        assert key in ck, f"checkpoint lacks {key!r}"
    assert ck["epoch"] == 1 and ck["step"] >= 1 and ck["vocab"] == Tokenizer().tokens
    assert ck["cfg"]["model"]["d_model"] == 64

    # epoch 2 resumes from last.pt (train.resume: auto)
    rc = train_mod.main(common + set_args + ["--set", "train.epochs=2"])
    assert rc == 0, f"resumed train returned {rc}"
    ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
    assert ck["epoch"] == 2, ck["epoch"]
    assert (ckpt_dir / "epoch_002.pt").is_file() and not (ckpt_dir / "epoch_001.pt").exists()  # keep_last=1
    log_text = (work_dir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "resumed from" in log_text, "second run did not resume"

    metrics = read_jsonl(work_dir / "logs" / "metrics.jsonl")
    kinds = {m["type"] for m in metrics}
    assert {"train", "val", "epoch"} <= kinds, kinds
    val = [m for m in metrics if m["type"] == "val"]
    assert len(val) == 2 and "av/cer" in val[-1] and "video/cer" in val[-1] and "av@babble0dB/cer" in val[-1], val[-1]

    # evaluate best.pt on val with a 2-point SNR sweep (model config comes from the checkpoint)
    rc = eval_mod.main(["--config", str(CONFIG), "--ckpt", str(ckpt_dir / "best.pt"), "--split", "val",
                        "--work-dir", str(work_dir), "--set", "eval.snr_sweep=[20, 0]",
                        "--set", f"data.num_workers={num_workers}"])
    assert rc == 0, f"evaluate returned {rc}"
    res_path = work_dir / "eval" / "val_results.json"
    assert res_path.is_file(), res_path
    res = json.loads(res_path.read_text(encoding="utf-8"))
    assert set(res["results"]) == {"clean", "20", "0"}, res["results"].keys()
    assert {"av", "audio", "video"} <= set(res["results"]["clean"]) and set(res["results"]["0"]) == {"av", "audio"}
    assert res["n_utts"] == 1 and "status" in res["suggested_snr_threshold"]
    print("  eval:", res["suggested_snr_threshold"]["message"])

    # a test split with no speakers -> clean failure code, no crash
    rc = eval_mod.main(["--config", str(CONFIG), "--ckpt", str(ckpt_dir / "best.pt"), "--split", "test",
                        "--work-dir", str(work_dir), "--set", "data.num_workers=0"])
    assert rc == 1, "empty split should return 1"

    # conditions without audio: the SNR sweep is skipped (no noisy rows), no threshold, no crash
    rc = eval_mod.main(["--config", str(CONFIG), "--ckpt", str(ckpt_dir / "best.pt"), "--split", "val",
                        "--work-dir", str(work_dir), "--set", "eval.conditions=[video]", "--set", "data.num_workers=0"])
    assert rc == 0, f"video-only evaluate returned {rc}"
    res = json.loads(res_path.read_text(encoding="utf-8"))
    assert set(res["results"]) == {"clean"} and res["suggested_snr_threshold"]["status"] == "not_applicable", res
    try:
        eval_mod.parse_conditions(["av", "vidoe"])
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown eval condition was accepted")

    # a --limit (smoke) run must refuse to resume or overwrite the checkpoints of a full run
    assert torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)["limit"] is None
    try:
        train_mod.main(common + set_args + ["--set", "train.epochs=3", "--limit", "2"])
    except RuntimeError as e:
        assert "--limit" in str(e), e
    else:
        raise AssertionError("a --limit run resumed the last.pt of a full run")

    run_interrupt(work_dir, train_mod, sets)
    run_crash(work_dir, train_mod, sets)


def run_interrupt(work_dir: Path, train_mod: types.ModuleType, sets: list[str]) -> None:
    """Ctrl+C during the first step -> last.pt saved mid-epoch (rc 130); a resumed run skips the consumed batch."""
    ckpt_dir = work_dir / "ckpt_interrupt"
    cfg = load_config(CONFIG, sets + [f"work_dir={work_dir}", f"train.ckpt_dir={ckpt_dir}", "train.epochs=1"])
    logger = get_logger(work_dir, "train_interrupt")
    try:
        trainer = train_mod.Trainer(cfg, None, logger)
        original_step = trainer._train_step

        def interrupted_step(batch: dict, meters: dict) -> float | None:
            signal.raise_signal(signal.SIGINT)  # handled by GracefulStop: finish this step, then save and stop
            return original_step(batch, meters)

        trainer._train_step = interrupted_step
        assert trainer.run() == 130, "interrupted run should return 130"
        ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        assert (ck["epoch"], ck["batch_in_epoch"], ck["step"]) == (0, 1, 1), (ck["epoch"], ck["batch_in_epoch"])
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, "SIGINT handler not restored"

        resumed = train_mod.Trainer(cfg, None, logger)
        assert (resumed.epoch, resumed.batch_in_epoch, resumed.step) == (0, 1, 1)
        assert resumed.run() == 0
        ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        assert (ck["epoch"], ck["batch_in_epoch"], ck["step"]) == (1, 0, 1), (ck["epoch"], ck["step"])
        assert (ckpt_dir / "best.pt").is_file()
    finally:
        close_logger()


def run_crash(work_dir: Path, train_mod: types.ModuleType, sets: list[str]) -> None:
    """An exception during training is re-raised, but last.pt keeps the progress made before it."""
    ckpt_dir = work_dir / "ckpt_crash"
    cfg = load_config(CONFIG, sets + [f"work_dir={work_dir}", f"train.ckpt_dir={ckpt_dir}", "train.epochs=2",
                                      "train.eval_every_epoch=5"])
    logger = get_logger(work_dir, "train_crash")
    try:
        trainer = train_mod.Trainer(cfg, None, logger)
        original_step = trainer._train_step
        calls: list[int] = []

        def failing_step(batch: dict, meters: dict) -> float | None:
            calls.append(1)
            if len(calls) == 2:  # first step of epoch 2
                raise RuntimeError("simulated crash")
            return original_step(batch, meters)

        trainer._train_step = failing_step
        try:
            trainer.run()
        except RuntimeError as e:
            assert "simulated crash" in str(e), e
        else:
            raise AssertionError("the training crash was swallowed")
        ck = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        assert (ck["epoch"], ck["batch_in_epoch"], ck["step"]) == (1, 0, 1), (ck["epoch"], ck["step"])
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler, "SIGINT handler not restored"
    finally:
        close_logger()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stubs", action="store_true", help="use in-file stand-ins for avsr.dataset / avsr.models")
    p.add_argument("--workers", type=int, default=None, help="DataLoader workers (default: 1 real, 0 stubs)")
    p.add_argument("--keep", action="store_true", help="keep the temporary work_dir")
    args = p.parse_args(argv)

    test_utils()
    print("test_train_smoke: utils OK")
    if args.stubs:
        ds_mod, md_mod = _stub_modules()
        sys.modules["avsr.dataset"], sys.modules["avsr.models"] = ds_mod, md_mod
    try:
        import avsr.evaluate as eval_mod
        import avsr.train as train_mod
    except ImportError as e:
        print(f"test_train_smoke: SKIPPED train/evaluate smoke test - required module not available yet ({e}). "
              f"Run with --stubs to test the training loop against in-file stand-ins.")
        return 0
    test_helpers(train_mod, eval_mod)
    workers = args.workers if args.workers is not None else (0 if args.stubs else 1)
    tmp = tempfile.mkdtemp(prefix="avsr_smoke_")
    try:
        run_smoke(Path(tmp), train_mod, eval_mod, workers)
    finally:
        if args.keep:
            print("  kept work_dir:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_train_smoke: OK ({'stubs' if args.stubs else 'real modules'}, workers={workers})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

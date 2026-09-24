"""Tests for avsr.infer (SPEC section 12).

Run from the project root (no real dataset needed; everything is synthetic and written to a temp dir)::

    & $py -m unittest tests.test_infer -v        # or:  & $py tests\\test_infer.py

Covered: SRT/JSON writers, energy VAD, visual VAD, 30 fps time resampling (constant and variable frame rate),
SNR gating, model-input batches, run-config merging, the whole pipeline with a stub model + stub landmarker, and an
end-to-end CLI run with a tiny random-initialised checkpoint and the real MediaPipe landmarker. Tests that need
ffmpeg, avsr.models or the MediaPipe model file are skipped with a message when those are unavailable.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsr import infer  # noqa: E402
from avsr import video_feats as vf  # noqa: E402
from avsr.audio_feats import find_ffmpeg  # noqa: E402
from avsr.dataset import resample_indices  # noqa: E402
from avsr.text import Tokenizer  # noqa: E402
from avsr.utils import load_config, to_dict  # noqa: E402

SR = 16000
BASE_CFG = ROOT / "configs" / "base.yaml"
LANDMARKER = ROOT / "assets" / "face_landmarker.task"
TINY_MODEL = [
    "model.d_model=64", "model.visual.out_dim=64", "model.skeleton.hidden=32", "model.skeleton.out_dim=32",
    "model.audio.out_dim=64", "model.encoder.layers=1", "model.encoder.heads=4", "model.encoder.ffn=128",
    "model.encoder.conv_kernel=7", "model.decoder.layers=1", "model.decoder.heads=4", "model.decoder.ffn=128",
]


# --------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------
def tone(sec: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(round(sec * SR))) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(sec: float, rng: np.random.Generator, amp: float = 0.001) -> np.ndarray:
    return (amp * rng.standard_normal(int(round(sec * SR)))).astype(np.float32)


def fake_face_points(width: int = 1280, height: int = 720) -> np.ndarray:
    """478 points in full-frame pixels: eyes at (cx -/+ 100, cy - 60), lip ellipses around (cx, cy + 140)."""
    cx, cy = width / 2.0, height / 2.0 - 40
    pts = np.tile(np.array([cx, cy], np.float32), (478, 1))
    pts[33] = (cx - 100, cy - 60)
    pts[263] = (cx + 100, cy - 60)
    mx, my = cx, cy + 140
    for k, idx in enumerate(vf.OUTER_LIP_IDX):
        a = np.pi - 2 * np.pi * k / 20
        pts[idx] = (mx + 60 * np.cos(a), my - 25 * np.sin(a))
    for k, idx in enumerate(vf.INNER_LIP_IDX):
        a = np.pi - 2 * np.pi * k / 20
        pts[idx] = (mx + 45 * np.cos(a), my - 12 * np.sin(a))
    return pts


class FakeLandmarker:
    """Stands in for vf.FaceLandmarker: returns ``full_pts`` scaled to the frame it is given, or None on the
    call numbers in ``none_calls`` (all calls when ``always_none``)."""

    def __init__(self, full_pts: np.ndarray, full_width: int, none_calls: set[int] | None = None,
                 always_none: bool = False) -> None:
        self.full_pts = full_pts
        self.full_width = full_width
        self.none_calls = none_calls or set()
        self.always_none = always_none
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> np.ndarray | None:
        n = len(self.calls)
        self.calls.append((frame_bgr.shape, int(timestamp_ms)))
        if self.always_none or n in self.none_calls:
            return None
        return self.full_pts * np.float32(frame_bgr.shape[1] / self.full_width)

    def close(self) -> None:
        pass


class StubModel(torch.nn.Module):
    """Minimal AVSR model: SNR head = fixed bucket, decode = fixed text; records the batches it receives."""

    def __init__(self, text: str, bucket: int = 0) -> None:
        super().__init__()
        self.ids = Tokenizer().encode(text)
        self.bucket = bucket
        self.calls: list[tuple[str, str, dict]] = []

    def forward(self, batch: dict, mode: str = "av") -> dict:
        self.calls.append(("forward", mode, {k: tuple(v.shape) for k, v in batch.items()}))
        logits = torch.zeros(batch["audio"].shape[0], 4)
        logits[:, self.bucket] = 1.0
        return {"snr_logits": logits}

    def decode(self, batch: dict, mode: str = "av", method: str = "ctc_greedy", max_len: int | None = None):
        self.calls.append(("decode", mode, {k: tuple(v.shape) for k, v in batch.items()}))
        return [list(self.ids)]


def write_video(path: Path, n_frames: int, fps: float, size: tuple[int, int] = (1280, 720)) -> bool:
    """Synthetic mp4 (mp4v): a diagonal gradient that shifts with the frame index. False if no encoder."""
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return False
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n_frames):
        img = ((xx + yy) // 4 + 7 * i) % 256
        writer.write(np.stack([img, (img * 2) % 256, 255 - img], axis=-1).astype(np.uint8))
    writer.release()
    return path.is_file() and path.stat().st_size > 0


def decode_all(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def ffmpeg_or_none() -> str | None:
    try:
        return find_ffmpeg()
    except FileNotFoundError:
        return None


def mux_audio(ffmpeg: str, video: Path, wave: np.ndarray, out: Path) -> None:
    wav = out.with_suffix(".wav")
    sf.write(str(wav), wave, SR, subtype="PCM_16")
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(video), "-i", str(wav), "-c:v", "copy", "-c:a", "aac",
                    "-shortest", str(out)], check=True)


def speech_like_wave(rng: np.random.Generator) -> np.ndarray:
    """4 s: tone bursts at 0.5-1.5 s and 2.0-3.5 s over a -60 dBFS noise floor."""
    wave = noise(4.0, rng)
    for a, b in ((0.5, 1.5), (2.0, 3.5)):
        s0, s1 = int(a * SR), int(b * SR)
        wave[s0:s1] += tone(b - a)
    return wave


def run_resampler(ts: list[float | None], dst: float = 30.0, default: float = 1 / 30):
    rs = infer.TimeResampler(dst, default)
    out: list[int] = []
    for i, t in enumerate(ts):
        out += [i - 1] * rs.push(t)
    out += [len(ts) - 1] * rs.finish()
    return np.asarray(out), rs


def assert_nearest(tc: unittest.TestCase, ts: np.ndarray, src: np.ndarray, dst: float) -> None:
    """Every target frame k uses a source frame nearest in time to k / dst (ties go to the later frame)."""
    for k, i in enumerate(src):
        d = np.abs(ts - k / dst)
        tc.assertLessEqual(d[i], d.min() + 1e-6, f"target {k}: source {i} is not the nearest")
        near = np.flatnonzero(d <= d.min() + 1e-6)
        tc.assertEqual(i, near.max(), f"target {k}: tie not resolved to the later frame")


# --------------------------------------------------------------------------------------------------------------
# SRT / JSON
# --------------------------------------------------------------------------------------------------------------
class TestSubtitles(unittest.TestCase):
    def test_format_srt_time(self) -> None:
        self.assertEqual(infer.format_srt_time(0), "00:00:00,000")
        self.assertEqual(infer.format_srt_time(1.5), "00:00:01,500")
        self.assertEqual(infer.format_srt_time(3661.25), "01:01:01,250")
        self.assertEqual(infer.format_srt_time(59.9996), "00:01:00,000")  # rounds to the nearest ms
        self.assertEqual(infer.format_srt_time(-0.3), "00:00:00,000")
        self.assertEqual(infer.format_srt_time(36000.0), "10:00:00,000")

    def test_write_srt_and_json(self) -> None:
        segs = [infer.Segment(1, 0.5, 2.25, "안녕하세요.", "av", 20.0, 18.0, 25.0, 1.0),
                infer.Segment(2, 3.0, 4.0, "  ", "video", 1.0),
                infer.Segment(3, 4.5, 7.125, "반갑습니다  여러분", "video", -2.5, None, None, 0.9)]
        with tempfile.TemporaryDirectory() as td:
            srt = Path(td) / "out.srt"
            n = infer.write_srt(srt, segs)
            self.assertEqual(n, 2)  # the empty cue is skipped and numbering stays contiguous
            # SubRip: UTF-8 without BOM, CRLF line ends (byte-exact, independent of the OS)
            self.assertEqual(srt.read_bytes(),
                             ("1\r\n00:00:00,500 --> 00:00:02,250\r\n안녕하세요.\r\n\r\n"
                              "2\r\n00:00:04,500 --> 00:00:07,125\r\n반갑습니다 여러분\r\n\r\n").encode("utf-8"))
            js = Path(td) / "out.json"
            infer.write_json(js, segs, {"video": "x.mp4", "snr_threshold": 5.0})
            raw = js.read_text(encoding="utf-8")
            self.assertIn("안녕하세요", raw)  # readable UTF-8, not \\u escapes
            data = json.loads(raw)
            self.assertEqual(data["video"], "x.mp4")
            self.assertEqual(len(data["segments"]), 3)
            self.assertEqual(data["segments"][0], {"index": 1, "start": 0.5, "end": 2.25, "text": "안녕하세요.",
                                                   "mode": "av", "snr_est": 20.0, "snr_dsp": 18.0,
                                                   "snr_model": 25.0, "visual_ratio": 1.0})
            self.assertIsNone(data["segments"][2]["snr_dsp"])
        table = infer.format_summary(segs)
        self.assertIn("안녕하세요.", table)
        self.assertIn("-2.5", table)

    def test_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            video = Path(td) / "clip.mp4"
            self.assertEqual(infer.output_paths(video, None), (Path(td) / "clip.srt", Path(td) / "clip.json"))
            self.assertEqual(infer.output_paths(video, Path(td) / "a.srt"), (Path(td) / "a.srt", Path(td) / "a.json"))
            self.assertEqual(infer.output_paths(video, Path(td) / "b.json"), (Path(td) / "b.srt", Path(td) / "b.json"))
            self.assertEqual(infer.output_paths(video, Path(td) / "c.v1"),
                             (Path(td) / "c.v1.srt", Path(td) / "c.v1.json"))
            self.assertEqual(infer.output_paths(video, td), (Path(td) / "clip.srt", Path(td) / "clip.json"))
            for sep in ("\\", "/"):  # not created yet: a trailing separator marks a directory
                self.assertEqual(infer.output_paths(video, str(Path(td) / "subs") + sep),
                                 (Path(td) / "subs" / "clip.srt", Path(td) / "subs" / "clip.json"))
            srt, _ = infer.output_paths(video, str(Path(td) / "new dir" / "x" / "y.srt"))
            infer.prepare_output(srt)  # creates the missing directories
            self.assertTrue((Path(td) / "new dir" / "x").is_dir())
            if sys.platform == "win32":
                # PowerShell 5.1 turns  --out "D:\my dir\" --no-progress  into one argument with a quote in it
                srt, _ = infer.output_paths(video, str(Path(td) / 'my dir" --no-progress'))
                with self.assertRaises(ValueError):
                    infer.prepare_output(srt)


# --------------------------------------------------------------------------------------------------------------
# VADs
# --------------------------------------------------------------------------------------------------------------
def check_segments(tc: unittest.TestCase, segs: list[tuple[float, float]], total: float, min_sec: float = 1.0,
                   max_sec: float = 8.0) -> None:
    tc.assertGreaterEqual(len(segs), 1)
    for (s, e), nxt in zip(segs, segs[1:] + [(math.inf, math.inf)]):
        tc.assertLess(s, e)
        tc.assertGreaterEqual(s, 0.0)
        tc.assertLessEqual(e, total + 1e-9)
        tc.assertLessEqual(e - s, max_sec + 1e-6)
        tc.assertGreaterEqual(e - s, min(min_sec, total) - 1e-6)
        tc.assertLessEqual(e, nxt[0] + 1e-9)  # sorted, non-overlapping


class TestEnergyVad(unittest.TestCase):
    def test_silence_tone_silence(self) -> None:
        rng = np.random.default_rng(0)
        wave = np.concatenate([noise(2, rng), tone(3) + noise(3, rng), noise(2, rng)])
        segs = infer.energy_vad_segments(wave, SR, min_sec=1.0, max_sec=8.0, pad_sec=0.2)
        self.assertEqual(len(segs), 1, segs)
        (s, e), = segs
        self.assertAlmostEqual(s, 2.0 - 0.2, delta=0.1)          # onset - pad
        self.assertAlmostEqual(e, 5.0 + 0.3 + 0.2, delta=0.1)    # offset + hangover + pad
        check_segments(self, segs, 7.0)

    def test_int16_input_and_digital_silence(self) -> None:
        wave = np.concatenate([np.zeros(2 * SR), tone(3), np.zeros(2 * SR)])
        segs = infer.energy_vad_segments((wave * 32767).astype(np.int16), SR)
        self.assertEqual(len(segs), 1, segs)
        self.assertAlmostEqual(segs[0][0], 1.8, delta=0.1)
        self.assertAlmostEqual(segs[0][1], 5.5, delta=0.1)

    def test_long_speech_split_at_pause(self) -> None:
        rng = np.random.default_rng(1)
        wave = np.concatenate([noise(1, rng), tone(6) + noise(6, rng), noise(0.15, rng),
                               tone(2) + noise(2, rng), noise(0.15, rng), tone(4) + noise(4, rng), noise(1, rng)])
        total = len(wave) / SR
        segs = infer.energy_vad_segments(wave, SR, min_sec=1.0, max_sec=8.0, pad_sec=0.2)
        check_segments(self, segs, total)
        # 12.3 s of speech needs 2 pieces; the cut goes into one of the two 0.15 s pauses (7.0-7.15 / 9.15-9.3 s)
        self.assertEqual(len(segs), 2, segs)
        cut = segs[1][0]
        self.assertTrue(7.0 <= cut <= 7.2 or 9.15 <= cut <= 9.35, f"cut not in a pause: {segs}")

    def test_long_flat_speech_is_split_evenly(self) -> None:
        rng = np.random.default_rng(4)
        wave = np.concatenate([noise(1, rng), tone(20) + noise(20, rng), noise(1, rng)])
        segs = infer.energy_vad_segments(wave, SR, min_sec=1.0, max_sec=8.0, pad_sec=0.2)
        check_segments(self, segs, 22.0)
        self.assertEqual(len(segs), 3, segs)  # the minimum number of pieces, no 1-second slivers
        self.assertGreater(min(e - s for s, e in segs), 6.0, segs)

    def test_silence_only_gives_whole_timeline(self) -> None:
        self.assertEqual(infer.energy_vad_segments(np.zeros(3 * SR, np.float32), SR), [(0.0, 3.0)])
        segs = infer.energy_vad_segments(np.zeros(20 * SR, np.float32), SR, max_sec=8.0)
        check_segments(self, segs, 20.0)
        self.assertAlmostEqual(segs[0][0], 0.0)
        self.assertAlmostEqual(segs[-1][1], 20.0)
        self.assertGreater(min(e - s for s, e in segs), 4.0, segs)

    def test_short_input(self) -> None:
        segs = infer.energy_vad_segments(tone(0.3), SR, min_sec=1.0)
        self.assertEqual(len(segs), 1)
        self.assertAlmostEqual(segs[0][0], 0.0)
        self.assertAlmostEqual(segs[0][1], 0.3, places=3)

    def test_noisy_speech_is_still_covered(self) -> None:
        # Two 4 s utterances of 180 ms syllables (one loud, six at -8 dB) 4 dB above white noise, i.e. just above the
        # gating threshold: the adaptive margin keeps the quiet syllables, so each utterance is one segment (a fixed
        # 6 dB margin keeps only the loud syllables: 6 fragments covering 93 %), and the pause stays out.
        rng = np.random.default_rng(5)
        amps = [1.0] + [0.4] * 6
        wave = np.zeros(13 * SR, np.float32)
        speech = np.zeros(len(wave), bool)
        for a, b in ((1.0, 5.0), (8.0, 12.0)):
            for k, t0 in enumerate(np.arange(a, b - 0.2, 0.22)):
                syl = tone(0.18, amp=amps[k % len(amps)], freq=180.0 + 40 * (k % 3))
                wave[int(t0 * SR):int(t0 * SR) + len(syl)] += syl
            speech[int(a * SR):int(b * SR)] = True
        rms_speech = float(np.sqrt(np.mean(wave[speech] ** 2)))
        wave += (rms_speech / 10 ** (4.0 / 20)) * rng.standard_normal(len(wave)).astype(np.float32)
        segs = infer.energy_vad_segments(wave, SR, min_sec=1.0, max_sec=8.0, pad_sec=0.2)
        check_segments(self, segs, 13.0)
        self.assertEqual(len(segs), 2, segs)
        covered = np.zeros(len(wave), bool)
        for s, e in segs:
            covered[int(s * SR):int(e * SR)] = True
        self.assertGreater(covered[speech].mean(), 0.98, segs)
        self.assertFalse(covered[int(6.0 * SR):int(7.0 * SR)].any(), segs)

    def test_click_is_ignored(self) -> None:
        rng = np.random.default_rng(2)
        wave = np.concatenate([noise(2, rng), tone(3) + noise(3, rng), noise(3, rng)])
        wave[int(6.5 * SR):int(6.55 * SR)] += 0.5  # 50 ms click
        segs = infer.energy_vad_segments(wave, SR)
        self.assertEqual(len(segs), 1, segs)


class TestVisualVad(unittest.TestCase):
    @staticmethod
    def mouth_series(n: int, speech: list[tuple[float, float]], rng: np.random.Generator,
                     amp: float = 0.1) -> np.ndarray:
        t = np.arange(n) / 30.0
        ih = 0.02 + 0.002 * rng.standard_normal(n)
        for a, b in speech:
            m = (t >= a) & (t < b)
            ih[m] = 0.12 + amp * np.sin(2 * np.pi * 4 * t[m]) + 0.002 * rng.standard_normal(m.sum())
        return ih.astype(np.float32)

    def test_speech_burst(self) -> None:
        rng = np.random.default_rng(0)
        ih = self.mouth_series(300, [(3.0, 7.0)], rng)
        segs = infer.visual_vad_segments(ih, np.ones(300), 30.0)
        self.assertEqual(len(segs), 1, segs)
        s, e = segs[0]
        self.assertTrue(2.3 <= s <= 3.0, segs)
        self.assertTrue(7.2 <= e <= 8.0, segs)
        check_segments(self, segs, 10.0)

    def test_invalid_frames_are_interpolated(self) -> None:
        rng = np.random.default_rng(1)
        ih = self.mouth_series(300, [(3.0, 7.0)], rng)
        valid = np.ones(300)
        valid[140:160] = 0
        ih[140:160] = 0.0  # zeros where no face was found (as in the npz)
        segs = infer.visual_vad_segments(ih, valid, 30.0)
        self.assertEqual(len(segs), 1, segs)

    def test_mostly_speaking_video(self) -> None:
        rng = np.random.default_rng(2)
        ih = self.mouth_series(600, [(0.5, 9.8), (10.2, 19.5)], rng)
        segs = infer.visual_vad_segments(ih, np.ones(600), 30.0)
        check_segments(self, segs, 20.0)
        covered = sum(min(e, 19.5) - max(s, 0.5) for s, e in segs if e > 0.5 and s < 19.5)
        self.assertGreater(covered / 18.6, 0.95, segs)

    def test_flat_or_faceless(self) -> None:
        self.assertEqual(infer.visual_vad_segments(np.full(90, 0.05), np.ones(90), 30.0), [(0.0, 3.0)])
        self.assertEqual(infer.visual_vad_segments(np.zeros(90), np.zeros(90), 30.0), [(0.0, 3.0)])


# --------------------------------------------------------------------------------------------------------------
# resampling
# --------------------------------------------------------------------------------------------------------------
class TestResampling(unittest.TestCase):
    def test_dataset_30_to_25(self) -> None:
        idx = resample_indices(30, 30.0, 25.0)  # the index mapping build_segment_batch relies on
        self.assertEqual(len(idx), 25)
        np.testing.assert_array_equal(idx, np.floor(np.arange(25) * 1.2 + 0.5).astype(int))

    def test_time_resampler_constant_rate(self) -> None:
        src, _ = run_resampler([i / 30 for i in range(50)])
        np.testing.assert_array_equal(src, np.arange(50))
        src, _ = run_resampler([i / 25 for i in range(50)])  # 25 -> 30 fps: frames repeated
        self.assertEqual(len(src), 60)
        np.testing.assert_array_equal(src, np.minimum(np.floor(np.arange(60) * 25 / 30 + 0.5), 49))
        src, _ = run_resampler([i / 60 for i in range(60)])  # 60 -> 30 fps: every other frame
        np.testing.assert_array_equal(src, np.arange(30) * 2)
        src, _ = run_resampler([i / 29.97 for i in range(300)])  # NTSC: same span, nearest frames
        self.assertEqual(len(src), math.ceil(300 / 29.97 * 30 - 1e-6))
        assert_nearest(self, np.arange(300) / 29.97, src, 30.0)

    def test_time_resampler_variable_rate(self) -> None:
        ts = [i / 30 for i in range(30)] + [1.0 + 2 * j / 30 for j in range(15)] + [2.0 + 0.5 * j for j in range(3)]
        src, rs = run_resampler(ts)
        self.assertEqual(rs.n_fixed, 0)
        end = ts[-1] + (ts[-1] - ts[-2])
        self.assertEqual(len(src), math.ceil(end * 30 - 1e-6))
        assert_nearest(self, np.asarray(ts), src, 30.0)
        self.assertTrue(np.all(np.diff(src) >= 0))

    def test_time_resampler_bad_timestamps(self) -> None:
        src, rs = run_resampler([0.0] * 10, default=1 / 25)  # backend without timestamps: constant 25 fps
        self.assertEqual(rs.n_fixed, 9)
        np.testing.assert_array_equal(src, np.minimum(np.floor(np.arange(12) * 25 / 30 + 0.5), 9))
        src, rs = run_resampler([float("nan")] * 5 + [None] * 5, default=1 / 30)
        self.assertEqual(rs.n_fixed, 10)
        np.testing.assert_array_equal(src, np.arange(10))

    def test_extract_visual_stream_25fps(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v25.mp4"
            if not write_video(path, 30, 25.0):
                self.skipTest("OpenCV mp4v encoder unavailable")
            pts = fake_face_points()
            lmk = FakeLandmarker(pts, 1280, none_calls={0, 1, 2})
            vs = infer.extract_visual_stream(path, lmk, channels=1, progress=False)
            frames = decode_all(path)
        self.assertEqual(vs.n_frames, 36)
        self.assertEqual(vs.n_src, 30)
        self.assertAlmostEqual(vs.src_fps, 25.0, places=3)
        np.testing.assert_array_equal(vs.src_index, np.minimum(np.floor(np.arange(36) * 25 / 30 + 0.5), 29))
        self.assertEqual(len(lmk.calls), 30)  # each source frame analysed exactly once
        self.assertEqual(lmk.calls[0][0], (360, 640, 3))  # detection on the frame downscaled to max side 640
        self.assertTrue(all(b[1] > a[1] for a, b in zip(lmk.calls, lmk.calls[1:])))  # increasing timestamps
        np.testing.assert_array_equal(vs.valid, (vs.src_index >= 3).astype(np.uint8))
        self.assertEqual(vs.crops.shape, (36, 96, 96, 1))
        box = vf.mouth_box(pts)
        lm_ref, geom_ref = vf.skeleton_features(pts)
        for k, i in enumerate(vs.src_index):
            # points mapped back to full resolution -> same box on every frame, leading faceless frames backfilled
            ref = cv2.cvtColor(vf.crop_square(frames[i], *box, 96), cv2.COLOR_BGR2GRAY)
            np.testing.assert_array_equal(vs.crops[k, ..., 0], ref, f"crop of target frame {k}")
            if vs.valid[k]:
                np.testing.assert_allclose(vs.lm[k], lm_ref, atol=2e-3)
                np.testing.assert_allclose(vs.cue[k, :4], geom_ref, atol=2e-3)
                np.testing.assert_allclose(vs.cue[k, 4:], vf.color_cues(frames[i], pts), atol=2e-3)
            else:
                self.assertFalse(vs.lm[k].any() or vs.cue[k].any())

    def test_extract_visual_stream_rgb_channels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "v30.mp4"
            if not write_video(path, 10, 30.0, (320, 240)):
                self.skipTest("OpenCV mp4v encoder unavailable")
            pts = fake_face_points(320, 240) * 0.25 + np.float32([120, 90])
            vs = infer.extract_visual_stream(path, FakeLandmarker(pts, 320), channels=3, progress=False)
            frames = decode_all(path)
        self.assertEqual(vs.crops.shape, (10, 96, 96, 3))
        ref = vf.crop_square(frames[4], *vf.mouth_box(pts), 96)[..., ::-1]  # BGR -> RGB
        np.testing.assert_array_equal(vs.crops[4], ref)

    def test_extract_visual_stream_variable_fps(self) -> None:
        ffmpeg = ffmpeg_or_none()
        if ffmpeg is None:
            self.skipTest("ffmpeg not found")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "vfr.mp4"
            subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=30", "-t", "2",
                            "-vf", "setpts='if(lt(N,30),N,30+(N-30)*2)/(30*TB)'", "-fps_mode", "vfr",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
            cap = cv2.VideoCapture(str(path))
            ts = []
            while cap.read()[0]:
                ts.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
            cap.release()
            vs = infer.extract_visual_stream(path, FakeLandmarker(fake_face_points(), 1280, always_none=True),
                                             channels=1, progress=False)
        ts = np.asarray(ts)
        self.assertEqual(vs.n_src, len(ts))
        self.assertEqual(vs.n_frames, math.ceil((ts[-1] + ts[-1] - ts[-2]) * 30 - 1e-6))  # ~2 s at 30 fps
        assert_nearest(self, ts, vs.src_index, 30.0)
        self.assertEqual(int(vs.valid.sum()), 0)


# --------------------------------------------------------------------------------------------------------------
# gating
# --------------------------------------------------------------------------------------------------------------
class TestGating(unittest.TestCase):
    def test_choose_mode(self) -> None:
        self.assertEqual(infer.choose_mode(10.0, 5.0), "av")
        self.assertEqual(infer.choose_mode(5.0, 5.0), "av")  # at the threshold audio is still trusted
        self.assertEqual(infer.choose_mode(4.9, 5.0), "video")
        self.assertEqual(infer.choose_mode(None, 5.0), "video")  # no audio at all
        self.assertEqual(infer.choose_mode(-3.0, 5.0, "audio"), "audio")  # forced mode wins
        self.assertEqual(infer.choose_mode(30.0, 5.0, "video"), "video")
        self.assertEqual(infer.choose_mode(0.0, 5.0, visual_ratio=0.05), "audio")  # lips invisible
        self.assertEqual(infer.choose_mode(0.0, 5.0, visual_ratio=0.9), "video")
        self.assertEqual(infer.choose_mode(9.0, 5.0, visual_ratio=0.0), "av")
        with self.assertRaises(ValueError):
            infer.choose_mode(1.0, 5.0, "lips")

    def test_combine_snr(self) -> None:
        self.assertEqual(infer.combine_snr(10.0, 20.0, "dsp"), 10.0)
        self.assertEqual(infer.combine_snr(10.0, 20.0, "model"), 20.0)
        self.assertEqual(infer.combine_snr(10.0, 20.0, "hybrid"), 15.0)
        self.assertEqual(infer.combine_snr(None, 20.0, "dsp"), 20.0)
        self.assertEqual(infer.combine_snr(10.0, None, "hybrid"), 10.0)
        self.assertIsNone(infer.combine_snr(None, None, "hybrid"))
        with self.assertRaises(ValueError):
            infer.combine_snr(1.0, 2.0, "oracle")

    def test_snr_from_logits(self) -> None:
        for bucket, mid in enumerate((25.0, 15.0, 5.0, -5.0)):
            logits = torch.zeros(1, 4)
            logits[0, bucket] = 3.0
            self.assertEqual(infer.snr_from_logits(logits.to(torch.bfloat16)), mid)

    def test_choose_vad(self) -> None:
        rng = np.random.default_rng(0)
        clean = speech_like_wave(rng)
        vad, snr = infer.choose_vad(clean, 5.0)
        self.assertEqual(vad, "energy")
        self.assertGreater(snr, 20.0)
        noisy = clean + 0.2 * rng.standard_normal(len(clean)).astype(np.float32)
        vad, snr = infer.choose_vad(noisy, 5.0)
        self.assertEqual(vad, "visual")
        self.assertLess(snr, 5.0)
        self.assertEqual(infer.choose_vad(noisy, 5.0, mode="av")[0], "energy")  # user trusts the audio
        self.assertEqual(infer.choose_vad(noisy, 5.0, visual_ratio=0.9)[0], "visual")
        self.assertEqual(infer.choose_vad(noisy, 5.0, visual_ratio=0.05)[0], "energy")  # no lips to segment on
        self.assertEqual(infer.choose_vad(None, 5.0, visual_ratio=0.0), ("visual", None))
        self.assertEqual(infer.choose_vad(clean, 5.0, vad="visual")[0], "visual")
        self.assertEqual(infer.choose_vad(None, 5.0), ("visual", None))
        with self.assertRaises(ValueError):
            infer.choose_vad(None, 5.0, vad="energy")


# --------------------------------------------------------------------------------------------------------------
# model inputs / config
# --------------------------------------------------------------------------------------------------------------
class TestModelInputs(unittest.TestCase):
    @staticmethod
    def stream(n: int = 90, channels: int = 1) -> infer.VisualStream:
        crops = np.broadcast_to(np.arange(n, dtype=np.uint8)[:, None, None, None], (n, 96, 96, channels)).copy()
        if channels == 3:
            crops[..., 1] = 100
            crops[..., 2] = 200
        lm = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None, None], (n, 40, 2)).copy()
        cue = np.broadcast_to(np.arange(n, dtype=np.float32)[:, None], (n, 8)).copy()
        valid = (np.arange(n) % 2).astype(np.uint8)
        return infer.VisualStream(crops, lm, cue, valid, np.arange(n))

    def test_gray_batch_matches_dataset_layout(self) -> None:
        cfg = load_config(BASE_CFG, ["video.norm=global", "video.cue_norm=none"])
        wave = (np.random.default_rng(0).standard_normal(3 * SR) * 3000).astype(np.int16)
        batch = infer.build_segment_batch(self.stream(), wave, 1.0, 2.0, cfg)
        # 30 video frames -> 25 at 25 fps; 1 s of audio -> 98 fbank frames -> 24 stacked frames: trimmed to 24
        t = 24
        self.assertEqual(tuple(batch["video"].shape), (1, t, 1, 88, 88))
        self.assertEqual(tuple(batch["lm"].shape), (1, t, 80))
        self.assertEqual(tuple(batch["cue"].shape), (1, t, 8))
        self.assertEqual(tuple(batch["valid"].shape), (1, t))
        self.assertEqual(tuple(batch["audio"].shape), (1, t, 320))
        self.assertEqual(batch["lengths"].tolist(), [t])
        src = 30 + resample_indices(30, 30.0, 25.0)[:t]
        mean, std = float(cfg.video.mean), float(cfg.video.std)
        np.testing.assert_allclose(batch["video"][0, :, 0, 0, 0].numpy(), (src / 255.0 - mean) / std, rtol=1e-5)
        np.testing.assert_array_equal(batch["lm"][0, :, 0].numpy(), src)
        np.testing.assert_array_equal(batch["valid"][0].numpy(), src % 2)
        self.assertEqual(batch["video"].dtype, torch.float32)

    def test_utterance_norm_matches_dataset(self) -> None:
        """video.norm/cue_norm = utterance: the segment batch is normalised exactly like avsr.dataset does."""
        from avsr.dataset import normalize_cue, normalize_pixels
        cfg = load_config(BASE_CFG, ["video.norm=utterance", "video.cue_norm=utterance"])
        st = self.stream()
        batch = infer.build_segment_batch(st, None, 1.0, 2.0, cfg)
        t = int(batch["lengths"][0])
        src = 30 + resample_indices(30, 30.0, 25.0)[:t]
        pix = st.crops[src][:, 4:92, 4:92].transpose(0, 3, 1, 2)
        np.testing.assert_allclose(batch["video"][0].numpy(), normalize_pixels(pix, "utterance"), rtol=1e-5, atol=1e-5)
        self.assertAlmostEqual(float(batch["video"][0].mean()), 0.0, places=4)
        cue_ref = normalize_cue(st.cue[src].astype(np.float32), st.valid[src].astype(np.float32), "utterance")
        np.testing.assert_allclose(batch["cue"][0].numpy(), cue_ref, rtol=1e-5, atol=1e-5)

    def test_rgb_no_audio_and_past_the_end(self) -> None:
        cfg = load_config(BASE_CFG, ["video.channels=3", "video.norm=global", "video.cue_norm=none"])
        batch = infer.build_segment_batch(self.stream(channels=3), None, 2.5, 3.5, cfg)
        self.assertEqual(tuple(batch["video"].shape), (1, 25, 3, 88, 88))
        self.assertEqual(tuple(batch["audio"].shape), (1, 25, 320))
        self.assertFalse(batch["audio"].any())
        mean, std = float(cfg.video.mean), float(cfg.video.std)
        self.assertAlmostEqual(float(batch["video"][0, 0, 2, 0, 0]), (200 / 255 - mean) / std, places=4)
        # frames 75..89 exist; target frames mapped past the end of the video are blank and invalid
        idx = 75 + resample_indices(30, 30.0, 25.0)
        np.testing.assert_array_equal(batch["valid"][0].numpy(), np.where(idx < 90, idx % 2, 0))
        np.testing.assert_allclose(batch["video"][0, idx >= 90].numpy(), -mean / std, rtol=1e-5)
        with self.assertRaises(ValueError):  # gray crops do not fit an RGB model
            infer.build_segment_batch(self.stream(channels=1), None, 0.0, 1.0, cfg)

    def test_build_run_config(self) -> None:
        ckpt_cfg = {"work_dir": "w1", "model": {"d_model": 64}, "video": {"channels": 1, "mean": 0.5},
                    "audio": {"sr": 16000}, "infer": {"snr_threshold": 5.0, "segment_max_sec": 8.0}}
        cli = {"model": {"d_model": 512}, "video": {"channels": 3}, "infer": {"snr_threshold": 3.0}}
        cfg = infer.build_run_config(ckpt_cfg, cli, ["infer.segment_max_sec=6"])
        self.assertEqual(cfg.model.d_model, 64)          # architecture from the checkpoint
        self.assertEqual(cfg.video.channels, 1)          # input features from the checkpoint
        self.assertEqual(cfg.infer.snr_threshold, 3.0)   # run-time settings from the CLI config
        self.assertEqual(cfg.infer.segment_max_sec, 6)   # --set applied last
        self.assertEqual(cfg.work_dir, "w1")
        self.assertEqual(infer.build_run_config(None, cli).infer.snr_threshold, 3.0)


# --------------------------------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------------------------------
class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ffmpeg = ffmpeg_or_none()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.video = cls.dir / "talk.mp4"
        silent = cls.dir / "silent.mp4"
        cls.ok = write_video(silent, 100, 25.0) and cls.ffmpeg is not None
        if cls.ok:
            mux_audio(cls.ffmpeg, silent, speech_like_wave(np.random.default_rng(3)), cls.video)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def setUp(self) -> None:
        if not self.ok:
            self.skipTest("needs ffmpeg and the OpenCV mp4v encoder to build a synthetic talking video")

    def bundle(self, text: str = "안녕 하세요", bucket: int = 0) -> infer.ModelBundle:
        return infer.ModelBundle(StubModel(text, bucket), load_config(BASE_CFG), Tokenizer(), torch.device("cpu"))

    def test_auto_mode_clean_audio(self) -> None:
        bundle = self.bundle()
        res = infer.run_pipeline(bundle, self.video, landmarker=FakeLandmarker(fake_face_points(), 1280),
                                 progress=False)
        self.assertEqual(res.meta["vad"], "energy")
        self.assertGreater(res.meta["global_snr_db"], 20.0)
        self.assertEqual(res.meta["video_frames_30fps"], 120)
        segs = res.segments
        self.assertGreaterEqual(len(segs), 1)
        covered = np.zeros(400, bool)
        for s in segs:
            self.assertEqual(s.text, "안녕 하세요")
            self.assertEqual(s.mode, "av")
            self.assertEqual(s.snr_model, 25.0)
            self.assertAlmostEqual(s.snr_est, (s.snr_dsp + 25.0) / 2)
            self.assertEqual(s.visual_ratio, 1.0)
            covered[int(s.start * 100):int(s.end * 100)] = True
        self.assertTrue(covered[50:150].all() and covered[200:350].all(), [(s.start, s.end) for s in segs])
        decode_calls = [c for c in bundle.model.calls if c[0] == "decode"]
        self.assertEqual(len(decode_calls), len(segs))
        shapes = decode_calls[0][2]
        t = shapes["video"][1]
        self.assertEqual(shapes["video"], (1, t, 1, 88, 88))
        self.assertEqual(shapes["audio"], (1, t, 320))
        self.assertEqual(shapes["lm"], (1, t, 80))
        self.assertTrue(all(c[1] == "audio" for c in bundle.model.calls if c[0] == "forward"))
        with tempfile.TemporaryDirectory() as td:
            n = infer.write_srt(Path(td) / "o.srt", segs)
            self.assertEqual(n, len(segs))

    def test_forced_and_noisy_modes(self) -> None:
        lmk = FakeLandmarker(fake_face_points(), 1280)
        res = infer.run_pipeline(self.bundle(), self.video, mode="video", landmarker=lmk, progress=False)
        self.assertTrue(all(s.mode == "video" for s in res.segments))
        # threshold above the measured SNR: audio not trusted -> visual VAD + lip reading
        res = infer.run_pipeline(self.bundle(bucket=3), self.video, snr_threshold=100.0,
                                 landmarker=FakeLandmarker(fake_face_points(), 1280), progress=False)
        self.assertEqual(res.meta["vad"], "visual")
        self.assertEqual([(s.start, s.end) for s in res.segments], [(0.0, 4.0)])  # static mouth: whole timeline
        self.assertEqual(res.segments[0].mode, "video")
        # ... unless the lips are never visible: then audio is the only usable stream (segmentation included)
        res = infer.run_pipeline(self.bundle(bucket=3), self.video, snr_threshold=100.0,
                                 landmarker=FakeLandmarker(fake_face_points(), 1280, always_none=True),
                                 progress=False)
        self.assertEqual(res.meta["vad"], "energy")
        self.assertTrue(all(s.mode == "audio" for s in res.segments))
        self.assertEqual(res.meta["landmark_valid_ratio"], 0.0)

    def test_external_audio_and_video_without_audio(self) -> None:
        silent = self.dir / "silent.mp4"
        res = infer.run_pipeline(self.bundle(), silent, landmarker=FakeLandmarker(fake_face_points(), 1280),
                                 progress=False)
        self.assertIsNone(res.meta["audio"])
        self.assertEqual(res.meta["vad"], "visual")
        self.assertTrue(all(s.mode == "video" and s.snr_est is None for s in res.segments))
        res = infer.run_pipeline(self.bundle(), silent, audio_path=self.dir / "talk.wav",
                                 landmarker=FakeLandmarker(fake_face_points(), 1280), progress=False)
        self.assertEqual(res.meta["vad"], "energy")
        self.assertTrue(all(s.mode == "av" for s in res.segments))
        with self.assertRaises(RuntimeError):  # an explicit --audio file without audio is an error, not a fallback
            infer.run_pipeline(self.bundle(), silent, audio_path=silent,
                               landmarker=FakeLandmarker(fake_face_points(), 1280), progress=False)

    def test_cli_end_to_end_tiny_checkpoint(self) -> None:
        try:
            from avsr.models import build_model
        except Exception as exc:  # noqa: BLE001 - any import problem means the model package is not usable here
            self.skipTest(f"avsr.models not importable: {exc}")
        if not LANDMARKER.is_file():
            self.skipTest(f"MediaPipe model missing: {LANDMARKER}")
        cfg = load_config(BASE_CFG, TINY_MODEL)
        tok = Tokenizer()
        torch.manual_seed(0)
        model = build_model(cfg, tok.vocab_size)
        ckpt = self.dir / "tiny.pt"
        torch.save({"model": model.state_dict(), "cfg": to_dict(cfg), "vocab": list(tok.tokens), "epoch": 0,
                    "step": 0}, ckpt)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        out = self.dir / "subs" / "result.srt"
        rc = infer.main(["--ckpt", str(ckpt), "--video", str(self.video), "--out", str(out), "--device", device,
                         "--no-progress", "--set", "infer.snr_estimator=hybrid"])
        self.assertEqual(rc, 0)
        self.assertTrue(out.is_file())
        data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(data["vad"], "energy")
        self.assertEqual(data["device"], device)
        self.assertEqual(data["landmark_valid_ratio"], 0.0)  # synthetic frames contain no face
        self.assertGreaterEqual(len(data["segments"]), 1)
        for seg in data["segments"]:
            self.assertLess(seg["start"], seg["end"])
            self.assertIn(seg["mode"], ("av", "audio"))  # clean audio, lips invisible
            self.assertIsInstance(seg["text"], str)
        segs = infer.transcribe_video(self.video, ckpt, BASE_CFG, overrides=["infer.segment_max_sec=6"],
                                      device=device, mode="video")
        self.assertEqual(len(segs), len(data["segments"]))
        self.assertTrue(all(s.mode == "video" for s in segs))


if __name__ == "__main__":
    unittest.main(verbosity=2)

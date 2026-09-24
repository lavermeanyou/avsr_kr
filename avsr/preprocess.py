"""Dataset -> per-utterance features + manifest shards (SPEC section 4).

Usage (from the project root):
  $py -m avsr.preprocess --data-root "C:\\...\\009.립리딩(입모양) 음성인식 데이터" [--work-dir work] [--workers 14]
                         [--angles all|A,C] [--speakers C313,E014] [--limit-videos N] [--order a-first|natural]
                         [--decoder auto|nvdec|cpu] [--no-align]

Per video (one worker process each):
  1. extract the mp4 from its tar (direct seek) into <work>/tmp, probe fps / frame count with ffprobe
     (label FPS says 30 but part of the videos are 29.97 fps: the real fps is used for every time->frame mapping)
  2. decode the embedded audio track -> 16 kHz mono int16
  3. estimate a per-video label time offset from speech energy at sentence boundaries (some videos' labels are
     ~0.1-0.3 s early); applied only when it clearly improves the boundary contrast
  4. decode video frames restricted to the union of the label face boxes, downscaled so the face is ~360 px
     (NVDEC hardware crop+resize when available, else OpenCV + resize: identical output geometry)
     for frames inside a (padded) sentence span:
       label face box -> 320 px face crop -> MediaPipe FaceLandmarker (VIDEO mode) -> 478 points
       -> lip skeleton [40,2] + geometry cues [4] + colour cues [4], mouth box (face-size based) -> running median
       -> 96x96 mouth crop streamed into an ffmpeg libx264 encoder (one mp4 per utterance)
     frames without a detected face fall back to the label lip box for the crop and get valid=0
  5. write npz per utterance, manifest shard (atomic) and a .done marker; delete the temp mp4
Re-running skips videos that already have a .done marker, so the job is resumable.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from avsr.aihub import (LabelDoc, LabelRef, MediaRef, build_media_index, discover_sources, extract_media,
                        iter_label_refs, parse_stem)
from avsr.text import Tokenizer, normalize_text

SR = 16000
CROP = 96
FACE_TARGET = 320          # side of the face image given to MediaPipe
FACE_DECODE_SIDE = 360.0   # decode scale is chosen so the median label face box is about this many pixels
PAD_MAX = 0.20             # sentence padding (s), limited to half of the gap to the neighbour
SHIFT_MAX = 0.80           # label offset search range (s); speaker C085 has sessions offset by > 0.4 s
SHIFT_STEP = 0.02
SHIFT_MIN_GAIN_DB = 1.0    # apply the offset only if boundary contrast improves by at least this much


# --------------------------------------------------------------------------------------
# media helpers
# --------------------------------------------------------------------------------------
@dataclass
class VideoInfo:
    fps: float
    n_frames: int
    width: int
    height: int
    codec: str


def probe_video(path: Path) -> VideoInfo:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration", "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]

    def rate(v: str) -> float:
        a, b = v.split("/") if "/" in v else (v, "1")
        return float(a) / float(b) if float(b) else 0.0

    fps = rate(s.get("avg_frame_rate", "0/0")) or rate(s.get("r_frame_rate", "0/0"))
    n = int(s.get("nb_frames") or 0)
    if n <= 0 and s.get("duration"):
        n = int(round(float(s["duration"]) * fps))
    return VideoInfo(fps=fps, n_frames=n, width=int(s["width"]), height=int(s["height"]), codec=str(s.get("codec_name", "")))


def load_audio_track(path: Path, sr: int = SR) -> np.ndarray:
    """Decode the whole audio track of a media file to mono int16 at `sr` using ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg audio decode failed for {path}: {proc.stderr.decode('utf-8', 'replace')[:500]}")
    return np.frombuffer(proc.stdout, dtype=np.int16).copy()


def estimate_label_shift(doc: LabelDoc, audio: np.ndarray, sr: int = SR) -> Tuple[float, float]:
    """Best constant offset (s) to add to the label times, from the energy contrast at sentence boundaries.

    Returns (shift, gain_db); shift is 0 unless the gain over no shift is >= SHIFT_MIN_GAIN_DB."""
    hop = sr // 100
    n = len(audio) // hop
    if n < 100 or not doc.sentences:
        return 0.0, 0.0
    x = audio[: n * hop].astype(np.float32).reshape(n, hop)
    e = 10.0 * np.log10((x * x).mean(axis=1) + 1.0)
    cs = np.concatenate([[0.0], np.cumsum(e)])
    w = 15  # 150 ms windows

    def wmean(i0: np.ndarray) -> np.ndarray:
        return (cs[i0 + w] - cs[i0]) / w

    starts = np.array([s.start for s in doc.sentences])
    ends = np.array([s.end for s in doc.sentences])
    shifts = np.arange(-SHIFT_MAX, SHIFT_MAX + 1e-9, SHIFT_STEP)
    scores = []
    for sh in shifts:
        i0 = np.round((starts + sh) * 100).astype(np.int64)
        i1 = np.round((ends + sh) * 100).astype(np.int64)
        ok = (i0 - w >= 0) & (i1 + w <= n) & (i1 - w >= 0) & (i0 + w <= n)
        if ok.sum() < 3:
            scores.append(-1e9)
            continue
        c = (wmean(i0[ok]) - wmean(i0[ok] - w)) + (wmean(i1[ok] - w) - wmean(i1[ok]))
        scores.append(float(c.mean()))
    scores = np.asarray(scores)
    j0 = int(np.argmin(np.abs(shifts)))
    jb = int(np.argmax(scores))
    gain = float(scores[jb] - scores[j0])
    if gain < SHIFT_MIN_GAIN_DB:
        return 0.0, gain
    return float(round(shifts[jb], 3)), gain


def padded_spans(doc: LabelDoc, duration: float, shift: float = 0.0) -> List[Tuple[float, float]]:
    """Shift every sentence by `shift` and pad by min(PAD_MAX, gap/2) on each side, clamped to [0, duration]."""
    sents = doc.sentences
    spans = []
    for i, s in enumerate(sents):
        gap_prev = s.start - sents[i - 1].end if i > 0 else PAD_MAX * 2
        gap_next = sents[i + 1].start - s.end if i + 1 < len(sents) else PAD_MAX * 2
        pl = max(0.0, min(PAD_MAX, gap_prev / 2.0))
        pr = max(0.0, min(PAD_MAX, gap_next / 2.0))
        spans.append((max(0.0, s.start + shift - pl), min(duration, s.end + shift + pr)))
    return spans


@dataclass
class Region:
    """Decoded-frame geometry: full-frame (x, y) -> ((x - x0) * sx, (y - y0) * sy)."""
    x0: int
    y0: int
    x1: int
    y1: int
    out_w: int
    out_h: int

    @property
    def sx(self) -> float:
        return self.out_w / float(self.x1 - self.x0)

    @property
    def sy(self) -> float:
        return self.out_h / float(self.y1 - self.y0)

    def map_boxes(self, boxes: np.ndarray) -> np.ndarray:
        b = boxes.astype(np.float32).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - self.x0) * self.sx
        b[:, [1, 3]] = (b[:, [1, 3]] - self.y0) * self.sy
        return b


def decode_region(doc: LabelDoc, info: VideoInfo, frames: np.ndarray) -> Region:
    """Union of the face boxes over the needed frames (+12 %), scaled so the median face is ~FACE_DECODE_SIDE px."""
    fb = doc.face_boxes[frames] if frames.size else doc.face_boxes
    good = (fb[:, 2] > fb[:, 0]) & (fb[:, 3] > fb[:, 1])
    fb = fb[good] if good.any() else np.array([[0, 0, info.width, info.height]])
    x0, y0, x1, y1 = float(fb[:, 0].min()), float(fb[:, 1].min()), float(fb[:, 2].max()), float(fb[:, 3].max())
    mx, my = 0.12 * (x1 - x0), 0.12 * (y1 - y0)
    X0 = int(max(0, x0 - mx)) // 2 * 2
    Y0 = int(max(0, y0 - my)) // 2 * 2
    X1 = min(info.width, int(np.ceil(x1 + mx))) // 2 * 2
    Y1 = min(info.height, int(np.ceil(y1 + my))) // 2 * 2
    face_side = float(np.median(np.maximum(fb[:, 2] - fb[:, 0], fb[:, 3] - fb[:, 1])))
    s = float(np.clip(FACE_DECODE_SIDE / max(face_side, 1.0), 0.25, 1.0))
    W = max(64, int(round((X1 - X0) * s / 2)) * 2)
    H = max(64, int(round((Y1 - Y0) * s / 2)) * 2)
    return Region(X0, Y0, X1, Y1, W, H)


_CUVID = {"h264": "h264_cuvid", "hevc": "hevc_cuvid", "av1": "av1_cuvid", "mpeg4": "mpeg4_cuvid", "vp9": "vp9_cuvid"}


def iter_frames_nvdec(path: Path, info: VideoInfo, reg: Region, n_needed: int) -> Iterator[np.ndarray]:
    dec = _CUVID.get(info.codec)
    if dec is None:
        raise RuntimeError(f"no NVDEC decoder for codec {info.codec}")
    crop = f"{reg.y0}x{info.height - reg.y1}x{reg.x0}x{info.width - reg.x1}"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-c:v", dec, "-crop", crop, "-resize", f"{reg.out_w}x{reg.out_h}",
           "-i", str(path), "-an", "-frames:v", str(n_needed), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    fs = reg.out_w * reg.out_h * 3
    got = 0
    try:
        assert proc.stdout is not None
        while got < n_needed:
            buf = bytearray(fs)
            view = memoryview(buf)
            filled = 0
            while filled < fs:
                k = proc.stdout.readinto(view[filled:])
                if not k:
                    break
                filled += k
            if filled < fs:
                break
            got += 1
            yield np.frombuffer(buf, dtype=np.uint8).reshape(reg.out_h, reg.out_w, 3)
    finally:
        if proc.poll() is None:
            proc.kill()
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        proc.wait()
        if got == 0:
            raise RuntimeError(f"NVDEC produced no frames: {err[:300]}")


def iter_frames_cpu(path: Path, reg: Region, n_needed: int, needed: np.ndarray) -> Iterator[Optional[np.ndarray]]:
    """OpenCV decode with the same crop/resize as NVDEC; yields None for frames that are not needed (grab only)."""
    import cv2
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video {path}")
    try:
        for fi in range(n_needed):
            if not needed[fi]:
                if not cap.grab():
                    return
                yield None
                continue
            ok, frame = cap.read()
            if not ok:
                return
            sub = frame[reg.y0:reg.y1, reg.x0:reg.x1]
            yield cv2.resize(sub, (reg.out_w, reg.out_h), interpolation=cv2.INTER_AREA)
    finally:
        cap.release()


class MouthWriter:
    """Streams 96x96 BGR frames into an ffmpeg libx264 encoder."""

    def __init__(self, path: Path, fps: float, size: int = CROP, crf: int = 18) -> None:
        self.path = path
        self.tmp = path.with_name(path.stem + ".part.mp4")
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size}x{size}",
               "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-threads", "1", "-movflags", "+faststart", str(self.tmp)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.n = 0

    def write(self, frame: np.ndarray) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.n += 1

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        err = self.proc.stderr.read() if self.proc.stderr else b""
        rc = self.proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg encode failed ({rc}) for {self.path}: {err.decode('utf-8', 'replace')[:500]}")
        self.tmp.replace(self.path)

    def abort(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except OSError:
            pass
        self.proc.kill()
        self.proc.wait()
        try:
            self.tmp.unlink(missing_ok=True)
        except OSError:
            pass


# --------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------
_WORKER_STATE: Dict[str, object] = {}


def _worker_init(model_path: str, decoder: str, align: bool) -> None:
    import cv2
    cv2.setNumThreads(1)
    os.environ.setdefault("GLOG_minloglevel", "2")
    _WORKER_STATE.update({"model_path": model_path, "tok": Tokenizer(), "decoder": decoder, "align": align})


def process_video(label_ref: LabelRef, media_ref: MediaRef, work_dir: str, tmp_dir: str, keep_media: bool = False) -> dict:
    """Process one video end to end. Returns a summary dict (never raises)."""
    from avsr.video_feats import (BoxSmoother, FaceLandmarker, box_from_xyxy, color_cues, crop_square,
                                  face_input_from_box, mouth_box, skeleton_features)

    t0 = time.time()
    work = Path(work_dir)
    stem = label_ref.stem
    summary: dict = {"stem": stem, "ok": False}
    media_path: Optional[Path] = None
    writer: Optional[MouthWriter] = None
    try:
        tok: Tokenizer = _WORKER_STATE.get("tok") or Tokenizer()  # type: ignore[assignment]
        model_path = str(_WORKER_STATE.get("model_path"))
        decoder = str(_WORKER_STATE.get("decoder", "auto"))
        align = bool(_WORKER_STATE.get("align", True))
        doc = label_ref.load()
        media_path = extract_media(media_ref, Path(tmp_dir))
        t_extract = time.time()
        info = probe_video(media_path)
        if info.fps <= 0:
            raise RuntimeError("cannot determine video fps")
        if abs(info.n_frames - doc.n_frames) > max(2, 0.01 * doc.n_frames):
            raise RuntimeError(f"frame count mismatch: video {info.n_frames} vs label boxes {doc.n_frames}")
        fps = info.fps
        n_frames = min(info.n_frames, doc.n_frames)
        duration = n_frames / fps
        audio = load_audio_track(media_path)
        shift, gain = estimate_label_shift(doc, audio) if align else (0.0, 0.0)
        t_audio = time.time()

        spans = padded_spans(doc, duration, shift)
        frame_spans = [(int(round(a * fps)), min(n_frames, int(round(b * fps)))) for a, b in spans]
        span_of = np.full(n_frames, -1, dtype=np.int32)
        for si, (fa, fb) in enumerate(frame_spans):
            if fb > fa:
                span_of[fa:fb] = si
        last_needed = int(max((fb for _, fb in frame_spans), default=0))
        needed_idx = np.nonzero(span_of[:last_needed] >= 0)[0]
        reg = decode_region(doc, info, needed_idx)
        face_dec = reg.map_boxes(doc.face_boxes[:n_frames])
        lip_dec = reg.map_boxes(doc.lip_boxes[:n_frames])

        out_dir = work / "feats" / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        for leftover in out_dir.glob("*.part.mp4"):
            leftover.unlink(missing_ok=True)
        rows: List[dict] = []
        counters = {"valid": 0, "frames": 0}
        stemf = parse_stem(stem) or {}

        det = FaceLandmarker(model_path, running_mode="video")
        smoother = BoxSmoother()
        cur = -1
        lm_buf: List[np.ndarray] = []
        cue_buf: List[np.ndarray] = []
        val_buf: List[int] = []

        def finish_span(si: int) -> None:
            nonlocal writer
            assert writer is not None
            writer.close()
            writer = None
            sent = doc.sentences[si]
            a_s, a_e = spans[si]
            T = len(val_buf)
            ns = int(round(a_s * SR))
            ne = int(round(a_e * SR))
            clip = audio[ns:min(ne, len(audio))]
            np.savez_compressed(
                out_dir / f"{stem}__{sent.id:03d}.npz",
                lm=np.stack(lm_buf).astype(np.float16) if T else np.zeros((0, 40, 2), np.float16),
                cue=np.stack(cue_buf).astype(np.float16) if T else np.zeros((0, 8), np.float16),
                valid=np.asarray(val_buf, dtype=np.uint8),
                audio=clip.astype(np.int16),
                fps=np.float64(fps), sr=np.int64(SR),
            )
            text = normalize_text(sent.text)
            v = int(sum(val_buf))
            counters["valid"] += v
            counters["frames"] += T
            rows.append({
                "utt_id": f"{stem}__{sent.id:03d}", "video_stem": stem, "split_dir": doc.split_dir,
                "speaker": doc.speaker, "gender": doc.gender, "age": doc.age, "specificity": doc.specificity,
                "angle": doc.angle, "session": doc.session or stemf.get("session", ""), "noise_env": doc.noise_env,
                "topic": sent.topic, "sentence_id": sent.id, "start": round(a_s, 4), "end": round(a_e, 4),
                "duration": round(a_e - a_s, 4), "n_frames": T, "n_samples": int(len(clip)), "fps": round(fps, 5),
                "time_shift": shift,
                "text": text, "text_raw": sent.text, "has_unk": bool(("X" in sent.text) or tok.has_unk(text) or not text),
                "lm_valid_ratio": round(v / T, 4) if T else 0.0,
                "mouth_mp4": f"feats/{stem}/{stem}__{sent.id:03d}.mp4",
                "npz": f"feats/{stem}/{stem}__{sent.id:03d}.npz",
            })
            lm_buf.clear()
            cue_buf.clear()
            val_buf.clear()

        zeros_lm = np.zeros((40, 2), np.float32)
        zeros_cue = np.zeros(8, np.float32)
        used = "cpu"
        frame_iter: Iterator[Optional[np.ndarray]]
        if decoder in ("auto", "nvdec"):
            try:
                gen = iter_frames_nvdec(media_path, info, reg, last_needed)
                first = next(gen)
                used = "nvdec"

                def _chain(first_frame: np.ndarray, rest: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
                    yield first_frame
                    yield from rest
                frame_iter = _chain(first, gen)
            except (RuntimeError, StopIteration) as e:
                if decoder == "nvdec":
                    raise
                summary["nvdec_error"] = str(e)[:200]
                frame_iter = iter_frames_cpu(media_path, reg, last_needed, span_of[:last_needed] >= 0)
        else:
            frame_iter = iter_frames_cpu(media_path, reg, last_needed, span_of[:last_needed] >= 0)

        t_frames = time.time()
        n_read = 0
        for fi, frame in enumerate(frame_iter):
            n_read += 1
            si = int(span_of[fi])
            if si < 0 or frame is None:
                if cur >= 0:
                    finish_span(cur)
                    cur = -1
                continue
            if si != cur:
                if cur >= 0:
                    finish_span(cur)
                cur = si
                smoother.reset()
                writer = MouthWriter(out_dir / f"{stem}__{doc.sentences[si].id:03d}.mp4", fps=fps)
            fx0, fy0, fx1, fy1 = (float(v) for v in face_dec[fi])
            pts = None
            if fx1 > fx0 and fy1 > fy0:
                crop, scale, ox, oy = face_input_from_box(frame, (fx0, fy0, fx1, fy1), 1.3, FACE_TARGET)
                if crop is not None:
                    p = det.detect(crop, int(round(fi * 1000.0 / fps)))
                    if p is not None:
                        pts = p / scale + np.array([ox, oy], dtype=np.float32)
            if pts is not None:
                lm, geom = skeleton_features(pts)
                col = color_cues(frame, pts)
                box = smoother.update(mouth_box(pts))
                lm_buf.append(lm)
                cue_buf.append(np.concatenate([geom, col]))
                val_buf.append(1)
            else:
                lx0, ly0, lx1, ly1 = (float(v) for v in lip_dec[fi])
                box = smoother.update(box_from_xyxy(lx0, ly0, lx1, ly1) if lx1 > lx0 else None)
                lm_buf.append(zeros_lm)
                cue_buf.append(zeros_cue)
                val_buf.append(0)
            if box is None:
                box = (frame.shape[1] / 2.0, frame.shape[0] / 2.0, float(min(frame.shape[:2]) / 3))
            assert writer is not None
            writer.write(crop_square(frame, box[0], box[1], box[2], CROP))
        if cur >= 0:
            finish_span(cur)
        det.close()
        t_end = time.time()
        if n_read < last_needed - 2:
            raise RuntimeError(f"video ended early: read {n_read} of {last_needed} frames")

        man_dir = work / "manifests"
        man_dir.mkdir(parents=True, exist_ok=True)
        tmp_man = man_dir / f"{stem}.jsonl.part"
        with open(tmp_man, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp_man.replace(man_dir / f"{stem}.jsonl")
        (man_dir / f"{stem}.done").write_text(json.dumps({"n_utts": len(rows)}), encoding="utf-8")
        frames_total = counters["frames"]
        summary.update({
            "ok": True, "n_utts": len(rows), "n_sentences": len(doc.sentences), "frames": frames_total,
            "valid_ratio": round(counters["valid"] / max(frames_total, 1), 4), "speaker": doc.speaker, "angle": doc.angle,
            "fps": round(fps, 5), "label_fps": doc.fps, "time_shift": shift, "shift_gain_db": round(gain, 2),
            "decoder": used, "region": [reg.x0, reg.y0, reg.x1, reg.y1, reg.out_w, reg.out_h],
            "t_extract": round(t_extract - t0, 2), "t_audio": round(t_audio - t_extract, 2),
            "t_frames": round(t_end - t_frames, 2), "t_total": round(t_end - t0, 2),
            "fps_proc": round(n_read / max(t_end - t_frames, 1e-6), 1),
        })
    except Exception as e:  # noqa: BLE001 — summarised and reported by the parent
        if writer is not None:
            writer.abort()
        summary.update({"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
    finally:
        if media_path is not None and media_ref.source_kind == "tar" and not keep_media:
            try:
                media_path.unlink(missing_ok=True)
            except OSError:
                pass
    return summary


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def build_jobs(data_root: str, work: Path, angles: Optional[set], speakers: Optional[set], log=print) -> List[Tuple[LabelRef, MediaRef]]:
    sources = discover_sources(data_root)
    log(f"[preprocess] sources: {sum(s.role == 'label' for s in sources)} label, {sum(s.role == 'media' for s in sources)} media")
    index = build_media_index(sources, cache_path=work / "media_index.json", log=log)
    jobs: List[Tuple[LabelRef, MediaRef]] = []
    missing = 0
    for ref in iter_label_refs(sources):
        info = parse_stem(ref.stem) or {}
        if angles and info.get("angle") not in angles:
            continue
        if speakers and info.get("speaker") not in speakers:
            continue
        media = index.get(f"{ref.stem}.mp4")
        if media is None:
            missing += 1
            continue
        jobs.append((ref, media))
    if missing:
        log(f"[preprocess] WARNING: {missing} labels have no matching mp4 in the media sources (skipped)")
    return jobs


def parse_shard(value: str) -> Tuple[int, int]:
    """'K/N' -> (K, N) with 1 <= K <= N (argparse type for --shard)."""
    parts = str(value).strip().split("/")
    try:
        if len(parts) != 2:
            raise ValueError
        k, n = int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"--shard must look like K/N (e.g. 2/4), got {value!r}") from None
    if not 1 <= k <= n:
        raise argparse.ArgumentTypeError(f"--shard {value}: need 1 <= K <= N")
    return k, n


def select_shard(jobs: List[Tuple[LabelRef, MediaRef]], k: int, n: int) -> List[Tuple[LabelRef, MediaRef]]:
    """Shard K of N: the jobs sorted by stem, keeping every job with index % N == K - 1 (independent of --order)."""
    ordered = sorted(jobs, key=lambda j: j[0].stem)
    return [j for i, j in enumerate(ordered) if i % n == k - 1]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AI-Hub lip-reading dataset -> avsr features")
    ap.add_argument("--data-root", required=True, help="dataset root (folder containing 라벨링데이터/원천데이터, or any parent)")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--tmp-dir", default=None, help="where videos are extracted temporarily (default: <work>/tmp)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 6))
    ap.add_argument("--angles", default="all", help="'all' or comma list, e.g. A,C,E")
    ap.add_argument("--speakers", default="", help="comma list of speaker IDs to restrict to")
    ap.add_argument("--limit-videos", type=int, default=0)
    ap.add_argument("--order", default="a-first", choices=["a-first", "natural"],
                    help="a-first: frontal angle A videos of every speaker first (usable early), then the rest")
    ap.add_argument("--decoder", default="auto", choices=["auto", "nvdec", "cpu"],
                    help="video decoding: NVIDIA hardware (nvdec) with CPU fallback (auto), or CPU only")
    ap.add_argument("--no-align", action="store_true", help="do not estimate/apply the per-video label time offset")
    ap.add_argument("--keep-media", action="store_true", help="keep extracted mp4 files in tmp-dir")
    ap.add_argument("--model", default=str(Path(__file__).resolve().parents[1] / "assets" / "face_landmarker.task"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard", type=parse_shard, default=None, metavar="K/N",
                    help="process only part K of N (videos sorted by stem, every N-th starting at #K) to split the "
                         "work over N PCs holding the same dataset, e.g. 2/4")
    args = ap.parse_args(argv)

    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir).resolve() if args.tmp_dir else work / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    log_path = work / "preprocess_log.jsonl"

    def log(msg: str) -> None:
        print(msg, flush=True)

    angles = None if args.angles.lower() == "all" else {a.strip().upper() for a in args.angles.split(",") if a.strip()}
    speakers = {s.strip() for s in args.speakers.split(",") if s.strip()} or None
    jobs = build_jobs(args.data_root, work, angles, speakers, log)
    if args.shard is not None:
        n_all = len(jobs)
        jobs = select_shard(jobs, *args.shard)
        log(f"[preprocess] shard {args.shard[0]}/{args.shard[1]}: {len(jobs)} of {n_all} videos")
    done = [j for j in jobs if (work / "manifests" / f"{j[0].stem}.done").exists()]
    todo = [j for j in jobs if not (work / "manifests" / f"{j[0].stem}.done").exists()]
    if args.order == "a-first":
        todo.sort(key=lambda j: ((parse_stem(j[0].stem) or {}).get("angle", "Z") != "A", j[0].stem))
    if args.limit_videos:
        todo = todo[: args.limit_videos]
    log(f"[preprocess] videos: total {len(jobs)}, already done {len(done)}, to do {len(todo)}, workers {args.workers}, decoder {args.decoder}")
    if args.dry_run or not todo:
        for ref, media in todo[:20]:
            log(f"  {ref.stem}  <- {Path(media.source_path).name}:{media.member} ({media.size / 1e6:.0f} MB)")
        return 0

    t_start = time.time()
    n_ok = n_fail = 0
    frames_done = 0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_worker_init,
                             initargs=(args.model, args.decoder, not args.no_align)) as ex:
        futs = {ex.submit(process_video, ref, media, str(work), str(tmp_dir), args.keep_media): ref.stem for ref, media in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            stem = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # worker crashed hard
                res = {"stem": stem, "ok": False, "error": f"worker crash: {type(e).__name__}: {e}"}
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            el = time.time() - t_start
            eta = el / i * (len(todo) - i)
            if res.get("ok"):
                n_ok += 1
                frames_done += res.get("frames", 0)
                log(f"[{i}/{len(todo)}] {stem} ok utts={res['n_utts']} frames={res['frames']} valid={res['valid_ratio']:.3f} "
                    f"fps={res['fps']} shift={res['time_shift']:+.2f}s({res['shift_gain_db']:+.1f}dB) dec={res['decoder']} "
                    f"proc={res['fps_proc']}fps t={res['t_total']}s | agg {frames_done / el:.0f} fps, "
                    f"elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min")
            else:
                n_fail += 1
                log(f"[{i}/{len(todo)}] {stem} FAILED: {res.get('error')}")
    log(f"[preprocess] finished: ok {n_ok}, failed {n_fail}, {(time.time() - t_start) / 60:.1f} min. "
        f"Re-run the same command to retry failures.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

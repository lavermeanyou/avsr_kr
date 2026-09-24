"""Visual feature extraction shared by preprocessing and inference (SPEC section 6).

- MediaPipe FaceLandmarker (tasks API) wrapper -> 478 facial points in pixel coordinates
- lip "skeleton": 40 lip points normalised (mouth-centred, roll-aligned, scaled by inter-ocular distance)
- geometry cues (mouth width / heights / inner-lip area) and colour cues inside the mouth
  (dark cavity fraction, tongue/red fraction, mean V, mean S)
- square mouth crop centred on the lips with a running-median box smoother
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

# MediaPipe face-mesh indices. Outer lip contour (20) then inner lip contour (20), each ordered around the polygon.
OUTER_LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
INNER_LIP_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
LIP_IDX = OUTER_LIP_IDX + INNER_LIP_IDX
assert len(LIP_IDX) == 40 and len(set(LIP_IDX)) == 40

IDX_EYE_L, IDX_EYE_R = 33, 263      # outer eye corners (subject's right/left, image left/right)
IDX_MOUTH_L, IDX_MOUTH_R = 61, 291  # mouth corners
IDX_UP_INNER, IDX_LO_INNER = 13, 14
IDX_UP_OUTER, IDX_LO_OUTER = 0, 17

N_CUE = 8  # geom(4) + color(4)


class FaceLandmarker:
    """Thin wrapper over mediapipe.tasks FaceLandmarker returning pixel-space points."""

    def __init__(self, model_path: str, running_mode: str = "video", num_faces: int = 1,
                 min_detection_conf: float = 0.5, min_presence_conf: float = 0.5, min_tracking_conf: float = 0.5) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        self._vision = vision
        mode = vision.RunningMode.VIDEO if running_mode == "video" else vision.RunningMode.IMAGE
        self.running_mode = running_mode
        # pass the model as bytes: MediaPipe 1.0.1 cannot open model_asset_path when the path contains non-ASCII
        # characters (e.g. a project folder under a Korean-named directory)
        with open(model_path, "rb") as f:
            model_bytes = f.read()
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_tracking_conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int = 0) -> Optional[np.ndarray]:
        """Return [478, 2] float32 pixel coordinates (x, y) in `frame_bgr`, or None if no face is found."""
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.running_mode == "video":
            ts = int(timestamp_ms)
            if ts <= self._last_ts:  # MediaPipe requires strictly increasing timestamps
                ts = self._last_ts + 1
            self._last_ts = ts
            res = self._lm.detect_for_video(img, ts)
        else:
            res = self._lm.detect(img)
        if not res.face_landmarks:
            return None
        lms = res.face_landmarks[0]
        pts = np.fromiter((v for l in lms for v in (l.x * w, l.y * h)), dtype=np.float32, count=2 * len(lms)).reshape(-1, 2)
        return pts

    def close(self) -> None:
        self._lm.close()

    def __enter__(self) -> "FaceLandmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def skeleton_features(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalised lip skeleton and geometry cues from 478 face points (pixel coords).

    Returns (lm [40, 2] float32, geom [4] float32):
      lm   = lip points translated to the mouth centre, rotated so the eye line is horizontal, divided by IOD
      geom = [mouth_width/IOD, inner_height/IOD, outer_height/IOD, inner_polygon_area/IOD^2]
    """
    pts = np.asarray(pts, dtype=np.float32)
    eye_l, eye_r = pts[IDX_EYE_L], pts[IDX_EYE_R]
    d = eye_r - eye_l
    iod = float(max(np.hypot(d[0], d[1]), 1e-3))
    ang = float(np.arctan2(d[1], d[0]))
    c, s = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    lips = pts[LIP_IDX]
    centre = lips.mean(axis=0)
    lm = ((lips - centre) @ rot.T) / iod
    inner = pts[INNER_LIP_IDX]
    geom = np.array([
        float(np.linalg.norm(pts[IDX_MOUTH_R] - pts[IDX_MOUTH_L])) / iod,
        float(np.linalg.norm(pts[IDX_LO_INNER] - pts[IDX_UP_INNER])) / iod,
        float(np.linalg.norm(pts[IDX_LO_OUTER] - pts[IDX_UP_OUTER])) / iod,
        _polygon_area(inner) / (iod * iod),
    ], dtype=np.float32)
    return lm.astype(np.float32), geom


def color_cues(frame_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """[dark_frac, red_frac, mean_v, mean_s] over pixels inside the inner-lip polygon (zeros if area < 4 px)."""
    h, w = frame_bgr.shape[:2]
    poly = np.asarray(pts, dtype=np.float32)[INNER_LIP_IDX]
    x0 = int(max(0, np.floor(poly[:, 0].min())))
    y0 = int(max(0, np.floor(poly[:, 1].min())))
    x1 = int(min(w, np.ceil(poly[:, 0].max()) + 1))
    y1 = int(min(h, np.ceil(poly[:, 1].max()) + 1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.zeros(4, dtype=np.float32)
    sub = frame_bgr[y0:y1, x0:x1]
    mask = np.zeros(sub.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
    n = int(mask.sum())
    if n < 4:
        return np.zeros(4, dtype=np.float32)
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    m = mask.astype(bool)
    hh, ss, vv = hsv[..., 0][m], hsv[..., 1][m], hsv[..., 2][m]
    dark = vv < 60
    red = ((hh < 12) | (hh > 168)) & (ss > 90) & (vv > 60)
    return np.array([dark.mean(), red.mean(), vv.mean() / 255.0, ss.mean() / 255.0], dtype=np.float32)


def mouth_box(pts: np.ndarray, scale: float = 1.0) -> Tuple[float, float, float]:
    """Square box (cx, cy, side) centred on the mean lip point.

    side = scale x max(IOD, 1.25 x eye-to-mouth distance). The size depends on face size only (not on mouth width), so
    the crop does not zoom with articulation (spreading lips for 'ㅣ' stays visible). Using the max of a horizontal and a
    vertical face measure keeps the size stable under yaw (IOD shrinks) and pitch (eye-mouth distance shrinks).
    For a frontal face this is ~2x the mouth width, i.e. nose tip to chin, like the dataset's lip box."""
    pts = np.asarray(pts, dtype=np.float32)
    lips = pts[LIP_IDX]
    centre = lips.mean(axis=0)
    cx, cy = float(centre[0]), float(centre[1])
    iod = float(np.linalg.norm(pts[IDX_EYE_R] - pts[IDX_EYE_L]))
    emd = float(np.linalg.norm((pts[IDX_EYE_L] + pts[IDX_EYE_R]) / 2.0 - centre))
    return cx, cy, max(scale * max(iod, 1.25 * emd), 8.0)


def box_from_xyxy(x0: float, y0: float, x1: float, y1: float) -> Tuple[float, float, float]:
    """Convert a (x0, y0, x1, y1) box (e.g. the label lip box) into (cx, cy, side) using the longer edge."""
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0, float(max(x1 - x0, y1 - y0, 8.0))


def crop_square(frame_bgr: np.ndarray, cx: float, cy: float, side: float, out_size: int = 96) -> np.ndarray:
    """Crop a side x side square centred at (cx, cy) (black padding outside the frame) resized to out_size."""
    h, w = frame_bgr.shape[:2]
    s = max(int(round(side)), 2)
    x0 = int(round(cx - s / 2.0))
    y0 = int(round(cy - s / 2.0))
    x1, y1 = x0 + s, y0 + s
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 <= sx0 or sy1 <= sy0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    if sx0 == x0 and sy0 == y0 and sx1 == x1 and sy1 == y1:
        patch = frame_bgr[y0:y1, x0:x1]
    else:
        patch = np.zeros((s, s, 3), dtype=np.uint8)
        patch[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = frame_bgr[sy0:sy1, sx0:sx1]
    interp = cv2.INTER_AREA if s > out_size else cv2.INTER_LINEAR
    return cv2.resize(patch, (out_size, out_size), interpolation=interp)


class BoxSmoother:
    """Running median over the last `window` boxes; returns the previous smoothed box when given None."""

    def __init__(self, window: int = 9) -> None:
        self.window = max(1, int(window))
        self._hist: Deque[Tuple[float, float, float]] = deque(maxlen=self.window)
        self._last: Optional[Tuple[float, float, float]] = None

    def update(self, box: Optional[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
        if box is not None:
            self._hist.append((float(box[0]), float(box[1]), float(box[2])))
            arr = np.asarray(self._hist, dtype=np.float64)
            med = np.median(arr, axis=0)
            self._last = (float(med[0]), float(med[1]), float(med[2]))
        return self._last

    def reset(self) -> None:
        self._hist.clear()
        self._last = None


def face_input_from_box(frame_bgr: np.ndarray, box_xyxy, expand: float = 1.3, target: int = 320):
    """Square, expanded crop around a face box resized so its side == target. Returns (crop, scale, x0, y0).

    A point p in crop coordinates maps back to the frame as p / scale + (x0, y0)."""
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box_xyxy)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = max(x1 - x0, y1 - y0) * expand
    side = max(min(side, max(w, h)), 16.0)
    bx0 = int(round(cx - side / 2.0)); by0 = int(round(cy - side / 2.0))
    bx1 = int(round(cx + side / 2.0)); by1 = int(round(cy + side / 2.0))
    bx0, by0 = max(bx0, 0), max(by0, 0)
    bx1, by1 = min(bx1, w), min(by1, h)
    crop = frame_bgr[by0:by1, bx0:bx1]
    if crop.size == 0:
        return None, 1.0, 0, 0
    scale = float(target) / float(max(crop.shape[0], crop.shape[1]))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
    return crop, scale, bx0, by0


def downscale_for_detection(frame_bgr: np.ndarray, max_side: int = 640):
    """Downscale a full frame so max(H, W) <= max_side. Returns (small, scale) with small = frame * scale."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(frame_bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        return small, scale
    return frame_bgr, 1.0


__all__ = [
    "LIP_IDX", "OUTER_LIP_IDX", "INNER_LIP_IDX", "N_CUE", "FaceLandmarker", "skeleton_features", "color_cues",
    "mouth_box", "box_from_xyxy", "crop_square", "BoxSmoother", "face_input_from_box", "downscale_for_detection",
]

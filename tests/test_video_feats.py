"""Tests for avsr.video_feats. Run: $py tests\\test_video_feats.py (from project root).

Synthetic checks always run. If a real sample frame path is given as argv[1], landmarks are also exercised on it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avsr.video_feats import (  # noqa: E402
    LIP_IDX, INNER_LIP_IDX, BoxSmoother, FaceLandmarker, color_cues, crop_square, mouth_box, skeleton_features,
    box_from_xyxy, face_input_from_box, downscale_for_detection,
)


def _fake_face_points() -> np.ndarray:
    """478 points: eyes at (400,300)/(600,300), a lip ellipse around (500,500)."""
    pts = np.zeros((478, 2), dtype=np.float32)
    pts[:] = (500, 400)
    pts[33] = (400, 300)
    pts[263] = (600, 300)
    outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    inner = INNER_LIP_IDX
    for k, idx in enumerate(outer):
        a = np.pi - 2 * np.pi * k / len(outer)
        pts[idx] = (500 + 60 * np.cos(a), 500 - 25 * np.sin(a))
    for k, idx in enumerate(inner):
        a = np.pi - 2 * np.pi * k / len(inner)
        pts[idx] = (500 + 45 * np.cos(a), 500 - 12 * np.sin(a))
    return pts


def test_synthetic() -> None:
    pts = _fake_face_points()
    lm, geom = skeleton_features(pts)
    assert lm.shape == (40, 2) and lm.dtype == np.float32
    assert abs(float(lm.mean(axis=0)[0])) < 1e-4 and abs(float(lm.mean(axis=0)[1])) < 1e-4
    # mouth width 120 px / IOD 200 px = 0.6 ; inner height 24/200 = 0.12
    assert abs(geom[0] - 0.6) < 1e-3, geom
    assert abs(geom[1] - 0.12) < 2e-3, geom
    assert geom[3] > 0

    # rotation invariance: rotate all points by 20 degrees
    th = np.deg2rad(20)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    pts_r = (pts - 500) @ R.T + 500
    lm_r, geom_r = skeleton_features(pts_r)
    assert np.allclose(lm, lm_r, atol=1e-4), np.abs(lm - lm_r).max()
    assert np.allclose(geom, geom_r, atol=1e-4)

    # colour cues: black inside the inner lip -> dark_frac ~ 1
    frame = np.full((1000, 1000, 3), (120, 140, 200), dtype=np.uint8)
    poly = np.round(pts[INNER_LIP_IDX]).astype(np.int32)
    cv2.fillPoly(frame, [poly], (5, 5, 5))
    col = color_cues(frame, pts)
    assert col.shape == (4,) and col[0] > 0.9, col
    # red tongue-like colour
    frame2 = frame.copy()
    cv2.fillPoly(frame2, [poly], (40, 40, 200))  # BGR red
    col2 = color_cues(frame2, pts)
    assert col2[1] > 0.9 and col2[0] < 0.1, col2

    cx, cy, side = mouth_box(pts)
    # IOD = 200, eye-mid (500,300) -> mouth centre (500,500) = 200 -> side = max(200, 1.25*200) = 250
    assert abs(cx - 500) < 2 and abs(cy - 500) < 2 and abs(side - 250) < 1, (cx, cy, side)
    crop = crop_square(frame, cx, cy, side, 96)
    assert crop.shape == (96, 96, 3) and crop.dtype == np.uint8
    edge = crop_square(frame, 5, 5, 200, 96)  # mostly outside -> black padding
    assert edge.shape == (96, 96, 3) and edge[0, 0].sum() == 0
    none = crop_square(frame, -500, -500, 100, 96)
    assert none.sum() == 0

    sm = BoxSmoother(5)
    assert sm.update(None) is None
    for v in [100, 102, 500, 101, 99]:
        out = sm.update((v, v, 50))
    assert abs(out[0] - 101) < 1e-6  # median rejects the outlier
    assert sm.update(None) == out

    assert box_from_xyxy(0, 0, 10, 20) == (5.0, 10.0, 20.0)
    small, s = downscale_for_detection(np.zeros((1080, 1920, 3), np.uint8), 640)
    assert small.shape[1] == 640 and abs(s - 1 / 3) < 1e-6
    fc, sc, x0, y0 = face_input_from_box(np.zeros((1080, 1920, 3), np.uint8), (600, 200, 1400, 1000), 1.3, 320)
    assert max(fc.shape[:2]) == 320 and x0 == 480 and y0 == 80
    print("synthetic: OK")


def test_real(frame_path: str, model_path: str) -> None:
    frame = cv2.imread(frame_path)
    assert frame is not None, frame_path
    small, scale = downscale_for_detection(frame, 640)
    with FaceLandmarker(model_path, running_mode="image") as det:
        pts = det.detect(small)
    assert pts is not None, "no face found on real sample frame"
    pts = pts / scale
    lm, geom = skeleton_features(pts)
    col = color_cues(frame, pts)
    cx, cy, side = mouth_box(pts)
    crop = crop_square(frame, cx, cy, side, 96)
    out = Path(frame_path).with_name(Path(frame_path).stem + "_mouth96.png")
    vis = frame.copy()
    for i in LIP_IDX:
        cv2.circle(vis, (int(pts[i, 0]), int(pts[i, 1])), 2, (0, 255, 0), -1)
    cv2.imwrite(str(out), crop)
    cv2.imwrite(str(Path(frame_path).with_name(Path(frame_path).stem + "_lips.png")), vis[int(cy - side):int(cy + side), int(cx - side):int(cx + side)])
    print("real: geom", np.round(geom, 3), "color", np.round(col, 3), "box", (round(cx), round(cy), round(side)), "->", out)


if __name__ == "__main__":
    test_synthetic()
    if len(sys.argv) > 1:
        test_real(sys.argv[1], str(ROOT / "assets" / "face_landmarker.task"))
    print("test_video_feats: OK")

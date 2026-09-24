"""Independent lip-based validation of the per-syllable CTC alignment written by ``tools/align_chars.py``.

The aligner used audio only, so the mouth cues stored in the npz (MediaPipe lip geometry + mouth colour) are an
independent check of its timings. Each cue channel used here (mouth_width, inner_height, dark_frac, red_frac) is
z-scored within the utterance on landmark-valid frames only, then linearly interpolated (from valid frames only)
onto a 5-ms grid; grid points farther than 1.5 video frames from a valid frame are NaN. Time 0 = utterance start
(video frame k at k/fps, CTC time j/50, both from the same origin).

Articulatory predictions tested (all effect sizes are Cohen's d with the pooled SD, sign = expected direction, plus
AUC = P(expected-high value > expected-low value)):

a) bilabial onsets (ㅁ ㅂ ㅃ ㅍ) close the lips: min inner_height inside the onset token span +-1 video frame is lower
   than for alveolar/velar onsets, and much lower than at the syllable centre (paired);
b) open vowels (ㅏ ㅑ ㅓ ㅕ) have a larger inner_height than ㅣ/ㅡ at the vowel-span centre;
c) rounded vowels (ㅜ ㅗ ㅠ ㅛ ㅝ ㅘ) have a smaller mouth_width than spread ones (ㅣ ㅔ ㅐ);
d) final ㅁ/ㅂ close the lips inside the final's span (vs other finals);
e) time-shift control: the contrasts after shifting every aligned time by -300..+300 ms (20-ms grid, incl. the
   requested -240/-160/-80/0/+80/+160/+240); utterance-bootstrap CI of the best shift. Shift s means the lips are read
   at (aligned time + s), so a negative best shift = the lip event happens before the aligned time (visual lead);
f) everything split into TRAIN speakers vs the unseen val speaker C313 (all sampled utterances are angle A).

Also: an even-spacing baseline (same first/last syllable times, syllables spread uniformly in between) to show how
much of each contrast is due to the CTC timing, per-token lip event offsets (closure minimum / release) to bound the
jitter, a per-speaker best shift, a confidence split, and example trajectories of the worst cases.

Run (from the project root):  & $py -m tools.validate_alignment
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = 0.005                                     # s, resampling grid of the z-scored cues
CH_NAMES = ("mouth_width", "inner_height", "dark_frac", "red_frac")
CH_IDX = (0, 1, 4, 5)                            # columns of npz cue
MW, IH, DK, RD = 0, 1, 2, 3                      # rows of the resampled cue matrix
SHIFTS_MS = np.arange(-300, 301, 20)
SH_IDX = (SHIFTS_MS // 5).astype(int)            # shifts in grid steps
S0 = int(np.nonzero(SHIFTS_MS == 0)[0][0])
COARSE_MS = (-240, -160, -80, 0, 80, 160, 240)
REL_POS = np.round(np.arange(-0.5, 1.5001, 0.1), 2)
TRAJ_S = np.round(np.arange(-0.5, 0.5001, 0.01), 3)

BILABIAL_ON = {"ㅁ", "ㅂ", "ㅃ", "ㅍ"}
ALVVEL_ON = {"ㄱ", "ㄲ", "ㅋ", "ㄴ", "ㄷ", "ㄸ", "ㅌ", "ㄹ", "ㅅ", "ㅆ", "ㅈ", "ㅉ", "ㅊ"}
OPEN_V = {"ㅏ", "ㅑ", "ㅓ", "ㅕ"}
CLOSE_V = {"ㅣ", "ㅡ"}
ROUND_V = {"ㅜ", "ㅗ", "ㅠ", "ㅛ", "ㅝ", "ㅘ"}
SPREAD_V = {"ㅣ", "ㅔ", "ㅐ"}
BIL_CODA = {"ㅁ", "ㅂ"}
LABIAL_CODA_ANY = {"ㅁ", "ㅂ", "ㅍ", "ㅄ", "ㄻ", "ㄼ", "ㄿ"}   # finals that can end in a lip closure

#: Pass criteria, fixed before the first run (not tuned afterwards).
CRITERIA = {
    "a_onset_vs_alvvel": "d >= 0.5 and AUC >= 0.65 in train and in val (shift 0)",
    "a_onset_vs_own_centre": "paired dz >= 0.8 (syllable-centre inner_height minus onset min) for bilabial onsets, "
                             "train and val",
    "b_open_vs_close": "d >= 0.3 in train and val at the vowel-span centre [V.start, V.ext_end) (shift 0)",
    "c_round_vs_spread": "d >= 0.3 in train and val at the vowel-span centre (shift 0)",
    "d_coda_closure": "d >= 0.5 in train and val, min inner_height over the final's span [T.start, T.ext_end + 1 frame]",
    "e_shift_peak": "best shift within [-100, +20] ms (0..80 ms visual lead +- one 20-ms step) for the contrast",
}

PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "violet": "#4a3aa7",
       "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9", "surface": "#fcfcfb",
       "axis": "#c3c2b7"}


# ----------------------------------------------------------------------------------------------------------------------
# signal access
# ----------------------------------------------------------------------------------------------------------------------
def load_cues(npz_path: Path) -> tuple[np.ndarray | None, float, int, int, np.ndarray]:
    """z-scored cue channels on the 5-ms grid [4, n_grid] (NaN far from valid frames), fps, n_frames, n_invalid,
    audio (int16, 16 kHz)."""
    with np.load(npz_path) as z:
        cue = np.asarray(z["cue"], dtype=np.float64)
        valid = np.asarray(z["valid"]) > 0
        fps = float(z["fps"])
        audio = np.asarray(z["audio"])
    n = cue.shape[0]
    vi = np.nonzero(valid)[0]
    if vi.size < 10:
        return None, fps, n, int(n - vi.size), audio
    tf = np.arange(n) / fps
    ng = int(math.floor(tf[-1] / GRID + 1e-9)) + 1
    tg = np.arange(ng) * GRID
    tv = tf[vi]
    pos = np.searchsorted(tv, tg)
    dl = np.abs(tg - tv[np.clip(pos - 1, 0, vi.size - 1)])
    dr = np.abs(tv[np.clip(pos, 0, vi.size - 1)] - tg)
    far = np.minimum(dl, dr) > 1.5 / fps
    zg = np.empty((len(CH_IDX), ng), dtype=np.float64)
    for j, c in enumerate(CH_IDX):
        x = cue[vi, c]
        zx = (x - x.mean()) / max(float(x.std()), 1e-9)
        g = np.interp(tg, tv, zx)
        g[far] = np.nan
        zg[j] = g
    return zg, fps, n, int(n - vi.size), audio


def energy_grid(audio: np.ndarray, n_grid: int, sr: int = 16000) -> np.ndarray:
    """z-scored log energy of a 20-ms window centred on every 5-ms grid point."""
    x = audio.astype(np.float64)
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    c = np.arange(n_grid) * int(round(sr * GRID))
    a = np.clip(c - sr // 100, 0, x.size)
    b = np.clip(c + sr // 100, 0, x.size)
    le = np.log10((cs[b] - cs[a]) / np.maximum(b - a, 1) + 1.0)
    return (le - le.mean()) / max(float(le.std()), 1e-9)


AV_LAGS = np.arange(-40, 41)          # grid steps (+-200 ms)


def av_xcorr_sums(ih: np.ndarray, en: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alignment-free A/V lag: cross-correlation of the smoothed derivative of inner_height with that of the audio
    log energy (floored at its 20th percentile, as in tools/av_sync_check.py). Lag L pairs lips at audio time + L."""
    v = np.where(np.isfinite(ih), ih, 0.0)
    a = np.maximum(en, np.percentile(en, 20))
    ker = np.ones(7) / 7.0
    dv = np.gradient(np.convolve(v, ker, mode="same"))
    da = np.gradient(np.convolve(a, ker, mode="same"))
    dv = (dv - dv.mean()) / max(float(dv.std()), 1e-9)
    da = (da - da.mean()) / max(float(da.std()), 1e-9)
    n = dv.size
    acc = np.zeros(AV_LAGS.size)
    cnt = np.zeros(AV_LAGS.size)
    for j, L in enumerate(AV_LAGS):
        if n - abs(L) < 20:
            continue
        x, y = (dv[L:], da[:n - L]) if L >= 0 else (dv[:n + L], da[-L:])
        acc[j] += float((x * y).sum())
        cnt[j] += x.size
    return acc, cnt


def _gather(row: np.ndarray, idx: np.ndarray) -> np.ndarray:
    ok = (idx >= 0) & (idx < row.size)
    out = np.full(idx.shape, np.nan)
    out[ok] = row[idx[ok]]
    return out


def point_shifts(zg: np.ndarray, t: float) -> np.ndarray:
    """All channels at time t + every shift: [4, nS]."""
    i = int(round(t / GRID)) + SH_IDX
    return np.stack([_gather(zg[c], i) for c in range(zg.shape[0])])


def wmin_shifts(row: np.ndarray, t0: float, t1: float) -> np.ndarray:
    """min of one channel over [t0, t1] + every shift (NaN points ignored): [nS]."""
    i0 = int(math.ceil(t0 / GRID - 1e-9))
    i1 = max(int(math.floor(t1 / GRID + 1e-9)), i0)
    v = _gather(row, np.arange(i0, i1 + 1)[None, :] + SH_IDX[:, None])
    v = np.where(np.isnan(v), np.inf, v).min(axis=1)
    v[np.isinf(v)] = np.nan
    return v


def sample_at(zg: np.ndarray, ts: np.ndarray) -> np.ndarray:
    return np.stack([_gather(zg[c], np.rint(ts / GRID).astype(int)) for c in range(zg.shape[0])])


def argmin_idx(row: np.ndarray, t0: float, t1: float) -> tuple[int | None, bool]:
    """Grid index of the minimum inside [t0, t1] and whether it sits on the window edge."""
    i0 = max(0, int(math.ceil(t0 / GRID - 1e-9)))
    i1 = min(row.size - 1, int(math.floor(t1 / GRID + 1e-9)))
    if i1 < i0:
        return None, False
    seg = row[i0:i1 + 1]
    if np.all(np.isnan(seg)):
        return None, False
    j = int(np.nanargmin(seg))
    return i0 + j, (j == 0 or j == seg.size - 1)


def release_idx(row: np.ndarray, i_min: int, span_s: float = 0.2) -> float | None:
    """Grid position (may be fractional) of the steepest inner_height rise within span_s after the minimum."""
    seg = row[i_min:i_min + int(round(span_s / GRID)) + 1]
    if seg.size < 3 or np.all(np.isnan(np.diff(seg))):
        return None
    return i_min + int(np.nanargmax(np.diff(seg))) + 0.5


# ----------------------------------------------------------------------------------------------------------------------
# per-syllable features
# ----------------------------------------------------------------------------------------------------------------------
def syllable_features(zg: np.ndarray, fps: float, syls: list[dict], full: bool) -> dict[str, list]:
    fr = 1.0 / fps
    nan_s = np.full(SHIFTS_MS.size, np.nan)
    out: dict[str, list] = {k: [] for k in ("on_min", "vraw", "vctc", "sc", "tlit", "text", "endhalf",
                                            "relpos", "traj_on", "traj_end", "ev_on_min", "ev_on_edge",
                                            "ev_on_rel", "ev_coda_min", "ev_coda_edge")}
    for s in syls:
        toks = s["tokens"]
        st, ee = float(s["start_s"]), float(s["ext_end_s"])
        dur = ee - st
        L, V = toks[0], toks[1]
        T = toks[2] if len(toks) == 3 else None
        out["on_min"].append(wmin_shifts(zg[IH], L["start_s"] - fr, L["end_s"] + fr))
        out["vraw"].append(point_shifts(zg, 0.5 * (V["start_s"] + V["end_s"])))
        out["vctc"].append(point_shifts(zg, 0.5 * (V["start_s"] + V["ext_end_s"])))
        out["sc"].append(point_shifts(zg, 0.5 * (st + ee)))
        if T is not None:
            out["tlit"].append(wmin_shifts(zg[IH], T["start_s"] - fr, T["end_s"] + fr))
            out["text"].append(wmin_shifts(zg[IH], T["start_s"], T["ext_end_s"] + fr))
        else:
            out["tlit"].append(nan_s)
            out["text"].append(nan_s)
        out["endhalf"].append(wmin_shifts(zg[IH], st + 0.5 * dur, ee + fr))
        if not full:
            continue
        out["relpos"].append(sample_at(zg, st + REL_POS * dur).astype(np.float32))
        out["traj_on"].append(sample_at(zg, st + TRAJ_S).astype(np.float32))
        out["traj_end"].append(sample_at(zg, ee + TRAJ_S).astype(np.float32))
        i_min, edge = argmin_idx(zg[IH], st - 0.25, st + 0.15)
        out["ev_on_min"].append(np.nan if i_min is None else i_min * GRID - st)
        out["ev_on_edge"].append(bool(edge))
        rel = None if i_min is None else release_idx(zg[IH], i_min)
        out["ev_on_rel"].append(np.nan if rel is None else rel * GRID - st)
        if T is not None:
            i_c, edge_c = argmin_idx(zg[IH], T["start_s"], ee + 0.15)
            out["ev_coda_min"].append(np.nan if i_c is None else i_c * GRID - ee)
            out["ev_coda_edge"].append(bool(edge_c))
        else:
            out["ev_coda_min"].append(np.nan)
            out["ev_coda_edge"].append(False)
    return out


def uniform_syllables(syls: list[dict]) -> list[dict]:
    """Baseline timing: same first start and last extended end, syllables evenly spaced, jamo 20 ms each."""
    t0, t1 = float(syls[0]["start_s"]), float(syls[-1]["ext_end_s"])
    step = (t1 - t0) / len(syls)
    res = []
    for i, s in enumerate(syls):
        st = t0 + i * step
        en = st + step
        toks = []
        for j, tk in enumerate(s["tokens"]):
            a = st + 0.02 * j
            toks.append({"name": tk["name"], "start_s": a, "end_s": a + 0.02})
        for j, tk in enumerate(toks):
            tk["ext_end_s"] = toks[j + 1]["start_s"] if j + 1 < len(toks) else en
        res.append({"start_s": st, "ext_end_s": en, "tokens": toks})
    return res


# ----------------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------------
def _rank(x: np.ndarray) -> np.ndarray:
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    start = np.cumsum(cnt) - cnt
    return (start + (cnt + 1) / 2.0)[inv]


def two_group(hi: np.ndarray, lo: np.ndarray) -> dict[str, Any]:
    """Cohen's d (hi - lo, pooled SD) and AUC P(hi > lo), NaNs dropped."""
    hi = hi[~np.isnan(hi)]
    lo = lo[~np.isnan(lo)]
    n1, n2 = hi.size, lo.size
    if n1 < 3 or n2 < 3:
        return {"n_hi": int(n1), "n_lo": int(n2), "mean_hi": None, "mean_lo": None, "d": None, "auc": None}
    sp = math.sqrt(((n1 - 1) * hi.var(ddof=1) + (n2 - 1) * lo.var(ddof=1)) / (n1 + n2 - 2))
    r = _rank(np.concatenate([hi, lo]))
    auc = (r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n2)
    return {"n_hi": int(n1), "n_lo": int(n2), "mean_hi": float(hi.mean()), "mean_lo": float(lo.mean()),
            "d": float((hi.mean() - lo.mean()) / max(sp, 1e-9)), "auc": float(auc)}


def d_curve(X: np.ndarray, m_hi: np.ndarray, m_lo: np.ndarray) -> np.ndarray:
    """Cohen's d at every shift; X [n_syl, nS]."""
    res = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        r = two_group(X[m_hi, j], X[m_lo, j])
        res[j] = np.nan if r["d"] is None else r["d"]
    return res


def refine_peak(curve: np.ndarray, xs: np.ndarray = SHIFTS_MS) -> float | None:
    """x (ms) of the curve maximum, parabolic interpolation between the grid steps of xs."""
    if not np.isfinite(curve).any():
        return None
    step = float(xs[1] - xs[0])
    j = int(np.nanargmax(curve))
    if 0 < j < curve.size - 1 and np.isfinite(curve[j - 1:j + 2]).all():
        y0, y1, y2 = curve[j - 1], curve[j], curve[j + 1]
        den = y0 - 2 * y1 + y2
        delta = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        return float(xs[j] + delta * step)
    return float(xs[j])


def half_max(curve: np.ndarray) -> dict[str, Any]:
    """Where the contrast falls to half of its peak on either side of the peak (ms), linear interpolation."""
    if not np.isfinite(curve).any():
        return {"left_ms": None, "right_ms": None, "width_ms": None}
    j = int(np.nanargmax(curve))
    half = curve[j] / 2.0
    left = right = None
    for k in range(j, 0, -1):
        if curve[k - 1] <= half < curve[k]:
            left = SHIFTS_MS[k - 1] + (half - curve[k - 1]) / (curve[k] - curve[k - 1]) * 20.0
            break
    for k in range(j, curve.size - 1):
        if curve[k + 1] <= half < curve[k]:
            right = SHIFTS_MS[k] + (curve[k] - half) / (curve[k] - curve[k + 1]) * 20.0
            break
    return {"left_ms": None if left is None else round(float(left), 1),
            "right_ms": None if right is None else round(float(right), 1),
            "width_ms": None if left is None or right is None else round(float(right - left), 1)}


def boot_best_shift(X: np.ndarray, m_hi: np.ndarray, m_lo: np.ndarray, utt: np.ndarray, n_boot: int,
                    rng: np.random.Generator) -> dict[str, Any]:
    """Utterance-level bootstrap of the best shift (and of d at shift 0)."""
    sel_u = np.unique(utt[m_hi | m_lo])
    if sel_u.size < 5:
        return {"ci95_ms": None, "d0_ci95": None}
    remap = {u: i for i, u in enumerate(sel_u)}
    nu, ns = sel_u.size, X.shape[1]

    def sums(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Xm = X[mask]
        um = np.array([remap[u] for u in utt[mask]], dtype=np.int64)
        ok = ~np.isnan(Xm)
        Xz = np.where(ok, Xm, 0.0)
        s1, s2, cn = np.zeros((nu, ns)), np.zeros((nu, ns)), np.zeros((nu, ns))
        np.add.at(s1, um, Xz)
        np.add.at(s2, um, Xz ** 2)
        np.add.at(cn, um, ok.astype(np.float64))
        return s1, s2, cn

    h1, h2, hn = sums(m_hi)
    l1, l2, ln_ = sums(m_lo)
    best, d0 = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, nu, nu), minlength=nu).astype(np.float64)
        Nh, Nl = w @ hn, w @ ln_
        mh, ml = (w @ h1) / np.maximum(Nh, 1), (w @ l1) / np.maximum(Nl, 1)
        vh = ((w @ h2) - Nh * mh ** 2) / np.maximum(Nh - 1, 1)
        vl = ((w @ l2) - Nl * ml ** 2) / np.maximum(Nl - 1, 1)
        sp = np.sqrt(np.maximum(((Nh - 1) * vh + (Nl - 1) * vl) / np.maximum(Nh + Nl - 2, 1), 1e-12))
        d = (mh - ml) / sp
        b = refine_peak(d)
        if b is not None:
            best.append(b)
        d0.append(d[S0])
    return {"ci95_ms": [round(float(np.percentile(best, 2.5)), 1), round(float(np.percentile(best, 97.5)), 1)],
            "d0_ci95": [round(float(np.percentile(d0, 2.5)), 3), round(float(np.percentile(d0, 97.5)), 3)]}


def rnd(x: Any, k: int = 3) -> Any:
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray)):
        return [rnd(v, k) for v in x]
    if isinstance(x, dict):
        return {kk: rnd(v, k) for kk, v in x.items()}
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else round(float(x), k)
    return x


def robust(x: np.ndarray) -> dict[str, Any]:
    x = x[np.isfinite(x)]
    if x.size < 3:
        return {"n": int(x.size)}
    q = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(x.size), "median_ms": q[2] * 1000, "p25_ms": q[1] * 1000, "p75_ms": q[3] * 1000,
            "p5_ms": q[0] * 1000, "p95_ms": q[4] * 1000,
            "robust_sd_ms": 1.4826 * float(np.median(np.abs(x - q[2]))) * 1000, "mean_ms": float(x.mean()) * 1000}


# ----------------------------------------------------------------------------------------------------------------------
# main analysis
# ----------------------------------------------------------------------------------------------------------------------
def build_tables(recs: list[dict], work_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    F: dict[str, list] = {}
    U: dict[str, list] = {}
    meta: dict[str, list] = {k: [] for k in ("utt", "split", "spk", "on", "v", "t", "prev_t", "next_on", "lp", "dur",
                                             "capped", "idx", "char")}
    info: dict[str, Any] = {"n_utts": 0, "n_skipped": 0, "n_invalid_frames": 0, "n_frames": 0, "fps": {}}
    av: dict[str, list[np.ndarray]] = {}
    for ui, rec in enumerate(recs):
        if not rec.get("ok") or not rec.get("syllables"):
            info["n_skipped"] += 1
            continue
        zg, fps, n, n_bad, audio = load_cues(work_dir / rec["npz"])
        if zg is None:
            info["n_skipped"] += 1
            continue
        info["n_utts"] += 1
        info["n_frames"] += n
        info["n_invalid_frames"] += n_bad
        info["fps"][f"{fps:.2f}"] = info["fps"].get(f"{fps:.2f}", 0) + 1
        syls = rec["syllables"]
        en = energy_grid(audio, zg.shape[1])
        acc, cnt = av_xcorr_sums(zg[IH], en)
        slot = av.setdefault(rec["speaker"], [np.zeros(AV_LAGS.size), np.zeros(AV_LAGS.size)])
        slot[0] += acc
        slot[1] += cnt
        F.setdefault("traj_en", []).extend(
            _gather(en, np.rint((float(s["start_s"]) + TRAJ_S) / GRID).astype(int)).astype(np.float32) for s in syls)
        f = syllable_features(zg, fps, syls, full=True)
        u = syllable_features(zg, fps, uniform_syllables(syls), full=False)
        for k, v in f.items():
            F.setdefault(k, []).extend(v)
        for k in ("on_min", "vraw", "vctc", "sc", "tlit", "text", "endhalf"):
            U.setdefault(k, []).extend(u[k])
        for i, s in enumerate(syls):
            toks = s["tokens"]
            meta["utt"].append(ui)
            meta["split"].append(rec["split"])
            meta["spk"].append(rec["speaker"])
            meta["on"].append(toks[0]["name"][2:])
            meta["v"].append(toks[1]["name"][2:])
            meta["t"].append(toks[2]["name"][2:] if len(toks) == 3 else "")
            prev = syls[i - 1]["tokens"] if i > 0 else None
            meta["prev_t"].append(prev[-1]["name"][2:] if prev is not None and len(prev) == 3 else "")
            meta["next_on"].append(syls[i + 1]["tokens"][0]["name"][2:] if i + 1 < len(syls) else "")
            meta["lp"].append(float(s["logprob"]))
            meta["dur"].append(float(s["ext_end_s"]) - float(s["start_s"]))
            meta["capped"].append(bool(s.get("ext_capped", False)))
            meta["idx"].append(i)
            meta["char"].append(s["char"])
    Fa = {k: np.asarray(v) for k, v in F.items()}
    Ua = {k: np.asarray(v) for k, v in U.items()}
    Ma = {k: np.asarray(v) for k, v in meta.items()}
    return Fa, Ua, {"meta": Ma, "info": info, "av": av}


def speaker_sync(F: dict[str, np.ndarray], M: dict[str, np.ndarray], av: dict[str, list[np.ndarray]],
                 per_speaker: dict[str, Any]) -> dict[str, Any]:
    """Per speaker: alignment-free A/V lag (lip-opening vs loudness rise) and where the audio energy rises relative
    to the aligned syllable starts; compared with the per-speaker best shift of the lip contrasts."""
    lag_ms = AV_LAGS * GRID * 1000
    t_ms = TRAJ_S * 1000
    win = (t_ms >= -100) & (t_ms <= 150)
    rows = {}
    for spk in sorted(av):
        acc, cnt = av[spk]
        corr = acc / np.maximum(cnt, 1)
        sm = M["spk"] == spk
        e = np.nanmean(F["traj_en"][sm].astype(np.float64), axis=0)
        de = np.gradient(e, t_ms)
        j = int(np.nanargmax(np.where(win, de, -np.inf)))
        jm = int(np.nanargmin(np.where((t_ms >= -150) & (t_ms <= 50), e, np.inf)))
        ps = per_speaker.get(spk, {})
        rows[spk] = {"split": ps.get("split"), "av_lag_ms": refine_peak(corr, lag_ms), "av_corr_peak": float(corr.max()),
                     "energy_steepest_rise_ms": float(t_ms[j]), "energy_min_ms": float(t_ms[jm]),
                     "a_onset_best_ms": ps.get("a_onset", {}).get("best_shift_ms"),
                     "c_sc_best_ms": ps.get("c_sc", {}).get("best_shift_ms"),
                     "d_text_best_ms": ps.get("d_text", {}).get("best_shift_ms")}
    spks = [s for s in rows if rows[s]["a_onset_best_ms"] is not None and rows[s]["av_lag_ms"] is not None]
    x = np.array([rows[s]["av_lag_ms"] for s in spks])
    out: dict[str, Any] = {"per_speaker": rows,
                           "sign": "av_lag_ms < 0: lip opening leads the loudness rise (lips read at audio time + lag)"}
    for key in ("a_onset_best_ms", "c_sc_best_ms", "d_text_best_ms"):
        y = np.array([rows[s][key] for s in spks], dtype=float)
        ok = np.isfinite(y)
        r = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 4 else None
        res = y[ok] - x[ok]
        out[f"pearson_av_lag_vs_{key}"] = r
        out[f"{key}_minus_av_lag"] = {"mean": float(res.mean()), "sd": float(res.std(ddof=1)),
                                      "min": float(res.min()), "max": float(res.max())}
        out[f"{key}_across_speakers"] = {"mean": float(y[ok].mean()), "sd": float(y[ok].std(ddof=1)),
                                         "min": float(y[ok].min()), "max": float(y[ok].max())}
    er = np.array([rows[s]["energy_steepest_rise_ms"] for s in spks])
    out["energy_steepest_rise_across_speakers"] = {"mean": float(er.mean()), "sd": float(er.std(ddof=1)),
                                                   "min": float(er.min()), "max": float(er.max())}
    out["av_lag_across_speakers"] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "min": float(x.min()),
                                     "max": float(x.max())}
    # speaker-centred per-token jitter of the lip events
    mk = masks(M)
    jit = {}
    for name, key, m in (("bilabial_onset_closure_min", "ev_on_min", mk["bil_on"]),
                         ("bilabial_onset_release", "ev_on_rel", mk["bil_on"]),
                         ("coda_bil_closure_min", "ev_coda_min", mk["coda_bil"])):
        vals = []
        for spk in rows:
            v = F[key][m & (M["spk"] == spk)]
            v = v[np.isfinite(v)]
            if v.size >= 5:
                vals.append(v - np.median(v))
        allv = np.concatenate(vals)
        jit[name] = {"n": int(allv.size), "robust_sd_ms": 1.4826 * float(np.median(np.abs(allv))) * 1000,
                     "iqr_ms": float(np.subtract(*np.percentile(allv, [75, 25]))) * 1000}
    out["speaker_centred_event_jitter"] = jit
    return out


FL_LEADS = (-0.05, 0.0, 0.025, 0.05, 0.075, 0.1, 0.15)


def eval_frame_labels(recs: list[dict], work_dir: Path) -> dict[str, Any]:
    """Score the aligner's own per-video-frame labels (tools.align_chars.frame_labels) against the lips: frame-level
    Cohen's d for the jamo classes, for jamo_mode ctc / proportional and a sweep of lead_s."""
    from tools.align_chars import frame_labels  # heavy import (torch), only needed here
    id2name = {tk["id"]: tk["name"] for r in recs for s in r.get("syllables", []) for tk in s["tokens"]}
    ids = {k: np.array([i for i, nm in id2name.items() if nm[:2] == p and nm[2:] in st], dtype=np.int64)
           for k, p, st in (("L_bil", "L:", BILABIAL_ON), ("L_alvvel", "L:", ALVVEL_ON), ("V_open", "V:", OPEN_V),
                            ("V_a", "V:", {"ㅏ", "ㅑ"}), ("V_close", "V:", CLOSE_V), ("V_round", "V:", ROUND_V),
                            ("V_spread", "V:", SPREAD_V), ("T_bil", "T:", BIL_CODA),
                            ("T_other", "T:", {n[2:] for n in id2name.values() if n.startswith("T:")}
                             - LABIAL_CODA_ANY))}
    variants = [(m, ld) for m in ("ctc", "proportional") for ld in FL_LEADS]
    acc: dict[tuple[str, float], dict[str, list]] = {v: {"jamo": [], "z": [], "split": []} for v in variants}
    for rec in recs:
        if not rec.get("ok") or not rec.get("syllables"):
            continue
        with np.load(work_dir / rec["npz"]) as z:
            cue = np.asarray(z["cue"], dtype=np.float64)[:, list(CH_IDX)]
            valid = np.asarray(z["valid"]) > 0
            fps = float(z["fps"])
        if valid.sum() < 10:
            continue
        zc = (cue - cue[valid].mean(axis=0)) / np.maximum(cue[valid].std(axis=0), 1e-9)
        for v in variants:
            _, jamo = frame_labels(rec, n_frames=cue.shape[0], fps=fps, lead_s=v[1], jamo_mode=v[0])
            keep = valid & (jamo >= 0)
            acc[v]["jamo"].append(jamo[keep])
            acc[v]["z"].append(zc[keep])
            acc[v]["split"].append(np.full(int(keep.sum()), rec["split"]))
    out: dict[str, Any] = {"leads_s": list(FL_LEADS), "note": "lead_s > 0 labels frame k with the jamo aligned at "
                           "k/fps + lead_s (labels earlier than the audio); literal = ctc/0.0, proposed = "
                           "proportional/0.05. d over labelled, landmark-valid frames.", "variants": {}}
    for v in variants:
        jamo = np.concatenate(acc[v]["jamo"])
        zz = np.concatenate(acc[v]["z"])
        spl = np.concatenate(acc[v]["split"])
        row: dict[str, Any] = {}
        for sp in ("train", "val", "all"):
            sm = np.ones(spl.size, bool) if sp == "all" else spl == sp
            j = jamo[sm]
            z_ = zz[sm]

            def dd(hi: str, lo: str, ch: int) -> dict[str, Any]:
                r = two_group(z_[np.isin(j, ids[hi]), ch], z_[np.isin(j, ids[lo]), ch])
                return {"d": r["d"], "auc": r["auc"], "n_hi": r["n_hi"], "n_lo": r["n_lo"]}
            row[sp] = {"L_alvvel_minus_bil_ih": dd("L_alvvel", "L_bil", IH),
                       "V_open_minus_close_ih": dd("V_open", "V_close", IH),
                       "V_a_minus_close_ih": dd("V_a", "V_close", IH),
                       "V_spread_minus_round_mw": dd("V_spread", "V_round", MW),
                       "T_other_minus_bil_ih": dd("T_other", "T_bil", IH),
                       "frame_share": {"L": float((j < 25).mean()), "V": float(((j >= 25) & (j < 46)).mean()),
                                       "T": float((j >= 46).mean())}}
        out["variants"][f"{v[0]}@{v[1]:+.3f}"] = row
    return out


def masks(M: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    on, v, t = M["on"], M["v"], M["t"]
    prev_lab = np.isin(M["prev_t"], list(LABIAL_CODA_ANY))
    next_bil = np.isin(M["next_on"], list(BILABIAL_ON))
    bil_on = np.isin(on, list(BILABIAL_ON))
    t_other = (t != "") & ~np.isin(t, list(LABIAL_CODA_ANY))
    return {
        "bil_on": bil_on, "alvvel_on": np.isin(on, list(ALVVEL_ON)),
        "bil_on_clean": bil_on & ~prev_lab, "alvvel_on_clean": np.isin(on, list(ALVVEL_ON)) & ~prev_lab,
        "h_on": on == "ㅎ", "null_on": on == "ㅇ",
        "open_v": np.isin(v, list(OPEN_V)), "close_v": np.isin(v, list(CLOSE_V)),
        "a_v": np.isin(v, ["ㅏ", "ㅑ"]), "eo_v": np.isin(v, ["ㅓ", "ㅕ"]),
        "round_v": np.isin(v, list(ROUND_V)), "spread_v": np.isin(v, list(SPREAD_V)),
        "open_v_nolab": np.isin(v, list(OPEN_V)) & ~bil_on & ~np.isin(t, list(LABIAL_CODA_ANY)),
        "close_v_nolab": np.isin(v, list(CLOSE_V)) & ~bil_on & ~np.isin(t, list(LABIAL_CODA_ANY)),
        "round_v_nolab": np.isin(v, list(ROUND_V)) & ~bil_on & ~np.isin(t, list(LABIAL_CODA_ANY)),
        "spread_v_nolab": np.isin(v, list(SPREAD_V)) & ~bil_on & ~np.isin(t, list(LABIAL_CODA_ANY)),
        "coda_bil": np.isin(t, list(BIL_CODA)) & ~next_bil, "coda_other": t_other & ~next_bil,
        "coda_none": (t == "") & ~next_bil,
        "not_capped": ~M["capped"],
    }


#: contrast name -> (feature key, channel or None, hi mask, lo mask, description)
CONTRASTS = {
    "a_onset": ("on_min", None, "alvvel_on", "bil_on",
                "min inner_height in onset token span +-1 frame: alveolar/velar minus bilabial"),
    "a_onset_clean": ("on_min", None, "alvvel_on_clean", "bil_on_clean",
                      "same, excluding syllables after a labial final"),
    "b_vctc": ("vctc", IH, "open_v", "close_v", "inner_height at the vowel-span centre [V.start, V.ext_end): "
               "open minus close"),
    "b_raw": ("vraw", IH, "open_v", "close_v", "inner_height at the raw V token centre"),
    "b_sc": ("sc", IH, "open_v", "close_v", "inner_height at the extended-syllable centre"),
    "b_sc_nolab": ("sc", IH, "open_v_nolab", "close_v_nolab", "b_sc without labial onsets/finals"),
    "c_vctc": ("vctc", MW, "spread_v", "round_v", "mouth_width at the vowel-span centre: spread minus rounded"),
    "c_raw": ("vraw", MW, "spread_v", "round_v", "mouth_width at the raw V token centre"),
    "c_sc": ("sc", MW, "spread_v", "round_v", "mouth_width at the extended-syllable centre"),
    "c_sc_nolab": ("sc", MW, "spread_v_nolab", "round_v_nolab", "c_sc without labial onsets/finals"),
    "d_text": ("text", None, "coda_other", "coda_bil", "min inner_height over the final's span [T.start, "
               "T.ext_end + 1 frame]: other finals minus ㅁ/ㅂ (next onset not bilabial)"),
    "d_tlit": ("tlit", None, "coda_other", "coda_bil", "min inner_height over the raw T token +-1 frame"),
    "d_endhalf": ("endhalf", None, "coda_other", "coda_bil", "min inner_height over the 2nd half of the extended "
                  "syllable + 1 frame"),
    "d_endhalf_vs_open": ("endhalf", None, "coda_none", "coda_bil", "2nd-half min: open syllables minus ㅁ/ㅂ finals"),
    # post-hoc diagnostics for check b (added after b failed; they do not replace the pre-registered check)
    "bx_a_vs_close": ("vctc", IH, "a_v", "close_v", "POST-HOC: inner_height ㅏ/ㅑ minus ㅣ/ㅡ at the vowel-span centre"),
    "bx_a_vs_close_sc": ("sc", IH, "a_v", "close_v", "POST-HOC: inner_height ㅏ/ㅑ minus ㅣ/ㅡ at the syllable centre"),
    "bx_eo_vs_close": ("vctc", IH, "eo_v", "close_v", "POST-HOC: inner_height ㅓ/ㅕ minus ㅣ/ㅡ at the vowel-span centre"),
    "bx_open_vs_close_dark": ("vctc", DK, "open_v", "close_v", "POST-HOC: dark_frac (visible oral cavity) "
                              "ㅏㅑㅓㅕ minus ㅣㅡ at the vowel-span centre"),
    "bx_spread_vs_eo_mw": ("vctc", MW, "spread_v", "eo_v", "POST-HOC: mouth_width ㅣㅔㅐ minus ㅓ/ㅕ (is ㅓ "
                           "partly rounded?)"),
}
PRIMARY = ("a_onset", "b_vctc", "c_vctc", "d_text")


def feature_matrix(Fx: dict[str, np.ndarray], key: str, ch: int | None) -> np.ndarray:
    X = Fx[key]
    return X[:, ch, :] if ch is not None else X


def analyse(F: dict[str, np.ndarray], U: dict[str, np.ndarray], M: dict[str, np.ndarray], n_boot: int,
            seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    mk = masks(M)
    splits = {"train": M["split"] == "train", "val": M["split"] == "val", "all": np.ones(M["split"].size, bool)}
    res: dict[str, Any] = {"contrasts": {}, "shift_control": {}, "uniform_baseline": {}}
    coarse_idx = [int(np.nonzero(SHIFTS_MS == s)[0][0]) for s in COARSE_MS]
    for name, (key, ch, hi, lo, desc) in CONTRASTS.items():
        X = feature_matrix(F, key, ch)
        Xu = feature_matrix(U, key, ch)
        res["contrasts"][name] = {"description": desc}
        res["shift_control"][name] = {}
        res["uniform_baseline"][name] = {}
        for sp, sm in splits.items():
            m_hi, m_lo = mk[hi] & sm, mk[lo] & sm
            r0 = two_group(X[m_hi, S0], X[m_lo, S0])
            res["contrasts"][name][sp] = r0
            curve = d_curve(X, m_hi, m_lo)
            best = refine_peak(curve)
            sc: dict[str, Any] = {"d_by_shift": curve, "best_shift_ms": best,
                                  "d_best": float(np.nanmax(curve)) if np.isfinite(curve).any() else None,
                                  "d_at_coarse": dict(zip([str(s) for s in COARSE_MS], curve[coarse_idx])),
                                  "half_max": half_max(curve)}
            if best is not None and r0["d"]:
                jb = int(np.nanargmax(curve))
                sc["d_rel_to_peak"] = {str(o): (curve[jb + o // 20] / curve[jb] if 0 <= jb + o // 20 < curve.size
                                                else None) for o in (-160, -80, -40, 40, 80, 160)}
            if name in PRIMARY or name in ("b_sc", "c_sc", "d_endhalf"):
                sc.update(boot_best_shift(X, m_hi, m_lo, M["utt"], n_boot, rng))
            res["shift_control"][name][sp] = sc
            cu = d_curve(Xu, m_hi, m_lo)
            res["uniform_baseline"][name][sp] = {"d0": two_group(Xu[m_hi, S0], Xu[m_lo, S0])["d"],
                                                 "best_shift_ms": refine_peak(cu),
                                                 "d_best": float(np.nanmax(cu)) if np.isfinite(cu).any() else None}
    # a2: onset minimum vs the same syllable's centre (paired)
    res["a_paired"] = {}
    for sp, sm in splits.items():
        res["a_paired"][sp] = {}
        for g in ("bil_on", "alvvel_on"):
            m = mk[g] & sm
            dlt = F["sc"][m, IH, S0] - F["on_min"][m, S0]
            dlt = dlt[np.isfinite(dlt)]
            res["a_paired"][sp][g] = {"n": int(dlt.size), "mean_diff": float(dlt.mean()),
                                      "dz": float(dlt.mean() / max(dlt.std(ddof=1), 1e-9)),
                                      "frac_centre_higher": float((dlt > 0).mean()),
                                      "onset_min_mean": float(np.nanmean(F["on_min"][m, S0])),
                                      "centre_mean": float(np.nanmean(F["sc"][m, IH, S0]))}
    # per-token lip events
    ev: dict[str, Any] = {}
    for sp, sm in splits.items():
        ev[sp] = {"bilabial_onset_closure_min_rel_start": robust(F["ev_on_min"][mk["bil_on"] & sm]),
                  "bilabial_onset_release_rel_start": robust(F["ev_on_rel"][mk["bil_on"] & sm]),
                  "bilabial_onset_min_at_window_edge": float(F["ev_on_edge"][mk["bil_on"] & sm].mean()),
                  "coda_bil_closure_min_rel_ext_end": robust(F["ev_coda_min"][mk["coda_bil"] & sm]),
                  "coda_bil_min_at_window_edge": float(F["ev_coda_edge"][mk["coda_bil"] & sm].mean())}
        for c in ("ㅁ", "ㅂ", "ㅍ", "ㅃ"):
            m = (M["on"] == c) & sm
            ev[sp][f"onset_{c}"] = {"closure_min": robust(F["ev_on_min"][m]), "release": robust(F["ev_on_rel"][m])}
        for c in ("ㅁ", "ㅂ"):
            m = (M["t"] == c) & mk["coda_bil"] & sm
            ev[sp][f"coda_{c}"] = robust(F["ev_coda_min"][m])
    res["lip_events"] = ev
    # per speaker (shift curves of the syllable-level contrasts)
    res["per_speaker"] = {}
    for spk in sorted(set(M["spk"].tolist())):
        sm = M["spk"] == spk
        row = {"split": str(M["split"][sm][0]), "n_syl": int(sm.sum())}
        for name in ("a_onset", "b_sc", "c_sc", "d_text"):
            key, ch, hi, lo, _ = CONTRASTS[name]
            X = feature_matrix(F, key, ch)
            curve = d_curve(X, mk[hi] & sm, mk[lo] & sm)
            row[name] = {"d0": curve[S0], "best_shift_ms": refine_peak(curve),
                         "n_lo": int((mk[lo] & sm).sum())}
        res["per_speaker"][spk] = row
    # confidence split (bilabial onsets by their own syllable log-prob)
    res["confidence"] = {}
    for sp, sm in splits.items():
        hi = F["on_min"][mk["alvvel_on"] & sm, S0]
        out = {}
        for lab, cm in (("p_ge_0.5", M["lp"] >= math.log(0.5)), ("p_lt_0.5", M["lp"] < math.log(0.5))):
            r = two_group(hi, F["on_min"][mk["bil_on"] & sm & cm, S0])
            curve = d_curve(F["on_min"], mk["alvvel_on"] & sm, mk["bil_on"] & sm & cm)
            out[lab] = {"n_bil": r["n_lo"], "d0": r["d"], "auc0": r["auc"], "best_shift_ms": refine_peak(curve)}
        res["confidence"][sp] = out
    # relative-position profiles (not capped syllables)
    rp: dict[str, Any] = {"rel_pos": REL_POS}
    for sp, sm in splits.items():
        base = sm & mk["not_capped"]
        rp[sp] = {
            "d_open_minus_close_ih": [two_group(F["relpos"][mk["open_v"] & base, IH, j].astype(float),
                                                F["relpos"][mk["close_v"] & base, IH, j].astype(float))["d"]
                                      for j in range(REL_POS.size)],
            "d_spread_minus_round_mw": [two_group(F["relpos"][mk["spread_v"] & base, MW, j].astype(float),
                                                  F["relpos"][mk["round_v"] & base, MW, j].astype(float))["d"]
                                        for j in range(REL_POS.size)],
            "ih_bilabial_onset": np.nanmean(F["relpos"][mk["bil_on"] & base, IH, :], axis=0),
            "ih_coda_bil": np.nanmean(F["relpos"][mk["coda_bil"] & base, IH, :], axis=0),
            "ih_coda_other": np.nanmean(F["relpos"][mk["coda_other"] & base, IH, :], axis=0),
        }
    res["relpos_profile"] = rp
    # onset-locked / end-locked trajectories
    tr: dict[str, Any] = {"t_ms": TRAJ_S * 1000}
    groups_on = {"bilabial_onset": mk["bil_on"], "alvvel_onset": mk["alvvel_on"],
                 "open_v_nolab": mk["open_v_nolab"], "close_v_nolab": mk["close_v_nolab"],
                 "round_v_nolab": mk["round_v_nolab"], "spread_v_nolab": mk["spread_v_nolab"]}
    for sp, sm in splits.items():
        tr[sp] = {"onset_locked_ih": {g: np.nanmean(F["traj_on"][m & sm, IH, :], axis=0) for g, m in groups_on.items()},
                  "onset_locked_mw": {g: np.nanmean(F["traj_on"][m & sm, MW, :], axis=0) for g, m in groups_on.items()},
                  "end_locked_ih": {g: np.nanmean(F["traj_end"][mk[g] & sm, IH, :], axis=0)
                                    for g in ("coda_bil", "coda_other", "coda_none")}}
        d_bil = np.array([two_group(F["traj_on"][mk["alvvel_on"] & sm, IH, j].astype(float),
                                    F["traj_on"][mk["bil_on"] & sm, IH, j].astype(float))["d"] or np.nan
                          for j in range(TRAJ_S.size)])
        d_open = np.array([two_group(F["traj_on"][mk["open_v_nolab"] & sm, IH, j].astype(float),
                                     F["traj_on"][mk["close_v_nolab"] & sm, IH, j].astype(float))["d"] or np.nan
                           for j in range(TRAJ_S.size)])
        d_round = np.array([two_group(F["traj_on"][mk["spread_v_nolab"] & sm, MW, j].astype(float),
                                      F["traj_on"][mk["round_v_nolab"] & sm, MW, j].astype(float))["d"] or np.nan
                            for j in range(TRAJ_S.size)])
        d_coda = np.array([two_group(F["traj_end"][mk["coda_other"] & sm, IH, j].astype(float),
                                     F["traj_end"][mk["coda_bil"] & sm, IH, j].astype(float))["d"] or np.nan
                           for j in range(TRAJ_S.size)])
        tr[sp]["d_curves"] = {"bilabial_vs_alvvel_ih_onset_locked": d_bil, "open_vs_close_ih_onset_locked": d_open,
                              "spread_vs_round_mw_onset_locked": d_round, "coda_other_vs_bil_ih_end_locked": d_coda}
        tr[sp]["peaks_ms"] = {k: (float(TRAJ_S[int(np.nanargmax(v))] * 1000) if np.isfinite(v).any() else None)
                              for k, v in tr[sp]["d_curves"].items()}
        tr[sp]["peak_d"] = {k: (float(np.nanmax(v)) if np.isfinite(v).any() else None)
                            for k, v in tr[sp]["d_curves"].items()}
    res["trajectories"] = tr
    return res


def class_table(F: dict[str, np.ndarray], M: dict[str, np.ndarray], key: str) -> dict[str, dict[str, Any]]:
    """Mean z of the 4 channels at the given point (shift 0) per vowel / onset class, all splits pooled + by split."""
    classes: dict[str, np.ndarray] = {}
    for grp, st in (("OPEN ㅏㅑㅓㅕ", OPEN_V), ("CLOSE ㅣㅡ", CLOSE_V), ("ROUND ㅜㅗㅠㅛㅝㅘ", ROUND_V),
                    ("SPREAD ㅣㅔㅐ", SPREAD_V)):
        classes[grp] = np.isin(M["v"], list(st))
    for vw in sorted(set(M["v"].tolist())):
        if (M["v"] == vw).sum() >= 20:
            classes[f"V:{vw}"] = M["v"] == vw
    classes["ONSET bilabial ㅁㅂㅃㅍ"] = np.isin(M["on"], list(BILABIAL_ON))
    classes["ONSET alv/velar"] = np.isin(M["on"], list(ALVVEL_ON))
    classes["ONSET ㅎ"] = M["on"] == "ㅎ"
    classes["ONSET ㅇ (none)"] = M["on"] == "ㅇ"
    classes["FINAL ㅁ/ㅂ"] = np.isin(M["t"], list(BIL_CODA))
    classes["FINAL other"] = (M["t"] != "") & ~np.isin(M["t"], list(LABIAL_CODA_ANY))
    classes["NO FINAL"] = M["t"] == ""
    out = {}
    for name, m in classes.items():
        row: dict[str, Any] = {}
        for sp, sm in (("all", np.ones(m.size, bool)), ("train", M["split"] == "train"), ("val", M["split"] == "val")):
            X = F[key][m & sm, :, S0]
            row[sp] = {"n": int((m & sm).sum()), **{CH_NAMES[c]: float(np.nanmean(X[:, c])) if X.size else None
                                                     for c in range(4)}}
            if name.startswith("ONSET"):
                row[sp]["onset_min_inner_height"] = float(np.nanmean(F["on_min"][m & sm, S0])) if X.size else None
        out[name] = row
    return out


def table_text(tab: dict[str, dict[str, Any]], title: str) -> str:
    lines = [title, f"{'class':<24}{'n':>6}{'ih':>8}{'mw':>8}{'dark':>8}{'red':>8} | "
                    f"{'n_tr':>5}{'ih_tr':>7}{'mw_tr':>7} | {'n_val':>5}{'ih_val':>7}{'mw_val':>7}"]
    for name, row in tab.items():
        a, t, v = row["all"], row["train"], row["val"]

        def f(x: Any) -> str:
            return "   nan" if x is None or not math.isfinite(x) else f"{x:+.2f}"
        extra = f"  onset-min ih {row['all']['onset_min_inner_height']:+.2f}" if "onset_min_inner_height" in a else ""
        lines.append(f"{name:<24}{a['n']:>6}{f(a['inner_height']):>8}{f(a['mouth_width']):>8}{f(a['dark_frac']):>8}"
                     f"{f(a['red_frac']):>8} | {t['n']:>5}{f(t['inner_height']):>7}{f(t['mouth_width']):>7} | "
                     f"{v['n']:>5}{f(v['inner_height']):>7}{f(v['mouth_width']):>7}{extra}")
    return "\n".join(lines)


def evaluate_checks(res: dict[str, Any]) -> list[dict[str, Any]]:
    c, sh = res["contrasts"], res["shift_control"]
    checks = []

    def dd(name: str, sp: str) -> float:
        v = c[name][sp]["d"]
        return float("nan") if v is None else v

    for key, name, thr, extra in (("a_onset_vs_alvvel", "a_onset", 0.5, "auc"), ("b_open_vs_close", "b_vctc", 0.3, None),
                                  ("c_round_vs_spread", "c_vctc", 0.3, None), ("d_coda_closure", "d_text", 0.5, None)):
        ok = True
        parts = []
        for sp in ("train", "val"):
            d, auc = dd(name, sp), c[name][sp]["auc"]
            ok &= d >= thr and (extra is None or (auc is not None and auc >= 0.65))
            parts.append(f"{sp}: d={d:.2f}, AUC={auc:.2f}, n={c[name][sp]['n_lo']}/{c[name][sp]['n_hi']}")
        checks.append({"name": key, "expectation": CRITERIA[key], "result": "; ".join(parts), "passed": bool(ok)})
    ap = res["a_paired"]
    ok = all(ap[sp]["bil_on"]["dz"] >= 0.8 for sp in ("train", "val"))
    checks.append({"name": "a_onset_vs_own_centre", "expectation": CRITERIA["a_onset_vs_own_centre"],
                   "result": "; ".join(f"{sp}: dz={ap[sp]['bil_on']['dz']:.2f}, mean diff {ap[sp]['bil_on']['mean_diff']:.2f} z "
                                       f"(alv/velar dz {ap[sp]['alvvel_on']['dz']:.2f})" for sp in ("train", "val")),
                   "passed": bool(ok)})
    for name in PRIMARY:
        parts, ok = [], True
        for sp in ("train", "val"):
            b = sh[name][sp]["best_shift_ms"]
            ci = sh[name][sp].get("ci95_ms")
            ok &= b is not None and -100 <= b <= 20
            parts.append(f"{sp}: best {b:+.0f} ms (95% CI {ci}), d peak {sh[name][sp]['d_best']:.2f} vs d(0) "
                         f"{sh[name][sp]['d_by_shift'][S0]:.2f}")
        checks.append({"name": f"e_shift_peak_{name}", "expectation": CRITERIA["e_shift_peak"], "result": "; ".join(parts),
                       "passed": bool(ok)})
    return checks


# ----------------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------------
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font = Path(r"C:\Windows\Fonts\malgun.ttf")
    if font.is_file():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "axes.edgecolor": PAL["axis"], "axes.labelcolor": PAL["ink2"],
                         "xtick.color": PAL["muted"], "ytick.color": PAL["muted"], "axes.titlecolor": PAL["ink"],
                         "axes.grid": True, "grid.color": PAL["grid"], "grid.linewidth": 0.6,
                         "figure.facecolor": PAL["surface"], "axes.facecolor": PAL["surface"],
                         "legend.frameon": False, "font.size": 9, "axes.titlesize": 10, "lines.linewidth": 2.0,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def plot_main(res: dict[str, Any], path: Path) -> None:
    plt = _setup_mpl()
    tr = res["trajectories"]
    t = np.asarray(tr["t_ms"])
    fig, ax = plt.subplots(4, 2, figsize=(14, 19))
    ls = {"train": "-", "val": "--"}

    def lead(a):
        a.axvspan(-80, 0, color=PAL["grid"], alpha=0.6, lw=0)
        a.axvline(0, color=PAL["axis"], lw=1)

    a = ax[0, 0]
    lead(a)
    for g, col, lab in (("bilabial_onset", PAL["blue"], "bilabial onset ㅁㅂㅃㅍ"),
                        ("alvvel_onset", PAL["orange"], "alveolar/velar onset"),
                        ("open_v_nolab", PAL["aqua"], "open vowel ㅏㅑㅓㅕ (non-labial)")):
        for sp in ("train", "val"):
            a.plot(t, tr[sp]["onset_locked_ih"][g], ls[sp], color=col, label=f"{lab}, {sp}")
    a.set_title("inner_height (z) around the aligned syllable start")
    a.set_xlabel("time from aligned syllable start (ms)")
    a.set_ylabel("inner_height z (per utterance)")
    a.legend(fontsize=8, loc="lower left")

    a = ax[0, 1]
    lead(a)
    for g, col, lab in (("round_v_nolab", PAL["blue"], "rounded ㅜㅗㅠㅛㅝㅘ"), ("spread_v_nolab", PAL["orange"], "spread ㅣㅔㅐ"),
                        ("close_v_nolab", PAL["aqua"], "close ㅣㅡ (inner_height)")):
        key = "onset_locked_ih" if g == "close_v_nolab" else "onset_locked_mw"
        for sp in ("train", "val"):
            if g == "close_v_nolab":
                continue
            a.plot(t, tr[sp][key][g], ls[sp], color=col, label=f"{lab} mouth_width, {sp}")
    a.set_title("mouth_width (z) around the aligned syllable start (non-labial syllables)")
    a.set_xlabel("time from aligned syllable start (ms)")
    a.set_ylabel("mouth_width z")
    a.legend(fontsize=8, loc="lower left")

    a = ax[1, 0]
    lead(a)
    for g, col, lab in (("coda_bil", PAL["blue"], "final ㅁ/ㅂ"), ("coda_other", PAL["orange"], "other final"),
                        ("coda_none", PAL["aqua"], "no final")):
        for sp in ("train", "val"):
            a.plot(t, tr[sp]["end_locked_ih"][g], ls[sp], color=col, label=f"{lab}, {sp}")
    a.set_title("inner_height (z) around the extended syllable end (= next syllable start)")
    a.set_xlabel("time from ext_end (ms); next onset not bilabial")
    a.set_ylabel("inner_height z")
    a.legend(fontsize=8, loc="lower left")

    a = ax[1, 1]
    lead(a)
    sc = res["shift_control"]
    for name, col, lab in (("a_onset", PAL["blue"], "a) bilabial onset closure"),
                           ("b_vctc", PAL["orange"], "b) open vs close, vowel-span centre"),
                           ("c_vctc", PAL["aqua"], "c) spread vs round, vowel-span centre"),
                           ("d_text", PAL["yellow"], "d) final ㅁ/ㅂ closure")):
        for sp in ("train", "val"):
            a.plot(SHIFTS_MS, sc[name][sp]["d_by_shift"], ls[sp], color=col, label=f"{lab}, {sp}")
    a.set_title("shift control: contrast (Cohen's d) vs shift of all aligned times")
    a.set_xlabel("shift (ms); lips read at aligned time + shift (negative = lips earlier)")
    a.set_ylabel("Cohen's d (expected direction)")
    a.axhline(0, color=PAL["axis"], lw=1)
    a.set_ylim(top=2.3)
    a.legend(fontsize=7.5, loc="upper left", ncol=2)

    a = ax[2, 0]
    rp = res["relpos_profile"]
    x = np.asarray(rp["rel_pos"])
    a.axvspan(0, 1, color=PAL["grid"], alpha=0.5, lw=0)
    for key, col, lab in (("d_open_minus_close_ih", PAL["orange"], "open vs close (inner_height)"),
                          ("d_spread_minus_round_mw", PAL["aqua"], "spread vs round (mouth_width)")):
        for sp in ("train", "val"):
            a.plot(x, np.array(rp[sp][key], dtype=float), ls[sp], color=col, marker="o", ms=4, label=f"{lab}, {sp}")
    for key, col, lab in (("ih_bilabial_onset", PAL["blue"], "mean ih z, bilabial onset"),
                          ("ih_coda_bil", PAL["violet"], "mean ih z, final ㅁ/ㅂ")):
        a.plot(x, np.asarray(rp["all"][key], dtype=float), ":", color=col, lw=2, label=f"{lab} (all)")
    a.axhline(0, color=PAL["axis"], lw=1)
    a.set_title("vowel contrasts by position inside the extended syllable span (0 = start, 1 = ext_end)")
    a.set_xlabel("relative position in [start_s, ext_end_s)  (uncapped syllables)")
    a.set_ylabel("Cohen's d  /  mean z")
    a.legend(fontsize=7.5, loc="upper right")

    a = ax[2, 1]
    ev = res["lip_events_raw"]
    bins = np.arange(-250, 205, 10)
    for key, col, lab in (("on_min_train", PAL["blue"], "closure minimum, train"),
                          ("on_min_val", PAL["violet"], "closure minimum, val"),
                          ("on_rel_train", PAL["orange"], "release (fastest opening), train"),
                          ("on_rel_val", PAL["yellow"], "release, val")):
        v = np.asarray(ev[key]) * 1000
        v = v[np.isfinite(v)]
        a.hist(v, bins=bins, histtype="step", density=True, color=col, lw=2, label=f"{lab} (median {np.median(v):+.0f})")
    a.axvline(0, color=PAL["axis"], lw=1)
    a.set_title("bilabial onsets: per-token lip event time relative to the aligned start")
    a.set_xlabel("event time - aligned syllable start (ms)")
    a.set_ylabel("density")
    a.legend(fontsize=8, loc="upper left")

    a = ax[3, 0]
    ss = res["speaker_sync"]["per_speaker"]
    xs = np.array([ss[s]["av_lag_ms"] for s in ss], dtype=float)
    lim = (min(-130.0, float(np.nanmin(xs)) - 10), 90.0)
    a.plot(lim, lim, color=PAL["axis"], lw=1)
    for key, col, mkr, lab in (("a_onset_best_ms", PAL["blue"], "o", "a) bilabial onset closure"),
                               ("c_sc_best_ms", PAL["aqua"], "s", "c) rounding, syllable centre"),
                               ("d_text_best_ms", PAL["yellow"], "^", "d) final ㅁ/ㅂ closure")):
        ys = np.array([ss[s][key] if ss[s][key] is not None else np.nan for s in ss], dtype=float)
        a.scatter(xs, ys, s=40, color=col, marker=mkr, edgecolor=PAL["surface"], linewidth=1.5, label=lab, zorder=3)
    for s, x_, y_ in zip(ss, xs, [ss[s]["a_onset_best_ms"] for s in ss]):
        a.annotate(s + (" (val)" if ss[s]["split"] == "val" else ""), (x_, y_), textcoords="offset points",
                   xytext=(4, 3), fontsize=7.5, color=PAL["ink2"])
    r = res["speaker_sync"]["pearson_av_lag_vs_a_onset_best_ms"]
    a.set_title(f"per speaker: best shift vs alignment-free A/V lag (r = {r:.2f} for a)")
    a.set_xlabel("A/V lag from lip-opening vs loudness-rise cross-correlation (ms)")
    a.set_ylabel("best shift of the lip contrast (ms)")
    a.legend(fontsize=8, loc="upper left")

    a = ax[3, 1]
    fl = res["frame_labels"]
    leads = np.array(fl["leads_s"]) * 1000
    for key, col, lab in (("L_alvvel_minus_bil_ih", PAL["blue"], "L frames: alv/vel vs bilabial (ih)"),
                          ("V_a_minus_close_ih", PAL["orange"], "V frames: ㅏㅑ vs ㅣㅡ (ih)"),
                          ("V_spread_minus_round_mw", PAL["aqua"], "V frames: spread vs round (mw)"),
                          ("T_other_minus_bil_ih", PAL["yellow"], "T frames: other vs ㅁ/ㅂ (ih)")):
        for mode, lsty in (("proportional", "-"), ("ctc", "--")):
            ys = [fl["variants"][f"{mode}@{ld:+.3f}"]["all"][key]["d"] for ld in fl["leads_s"]]
            a.plot(leads, np.array(ys, dtype=float), lsty, color=col, marker="o", ms=4, label=f"{lab}, {mode}")
    a.axvline(50, color=PAL["axis"], lw=1)
    a.axhline(0, color=PAL["axis"], lw=1)
    a.set_title("aligner's frame labels (frame_labels) scored on the lips, all 500 utts")
    a.set_xlabel("lead_s of the frame labels (ms); proposed = proportional @ 50 ms, literal = ctc @ 0")
    a.set_ylabel("frame-level Cohen's d")
    a.set_ylim(bottom=-0.55)
    a.legend(fontsize=7.5, loc="lower center", ncol=2)
    fig.suptitle("Lip-based validation of the CTC syllable alignment (400 train utts / 13 speakers, 100 val utts C313;"
                 " angle A). Shaded: 0-80 ms visual lead", color=PAL["ink"], fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_examples(examples: list[dict], path: Path) -> None:
    plt = _setup_mpl()
    n = len(examples)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(15, 3.0 * rows), squeeze=False)
    for a, ex in zip(axs.ravel(), examples):
        t = np.asarray(ex["t_ms"])
        a.plot(t, ex["ih"], color=PAL["blue"], label="inner_height z")
        a.plot(t, ex["mw"], color=PAL["orange"], lw=1.2, label="mouth_width z")
        for s0, ch in ex["syl_starts"]:
            a.axvline(s0, color=PAL["axis"], lw=0.8)
            a.text(s0 + 3, a.get_ylim()[1] if False else 2.6, ch, fontsize=10, color=PAL["ink"])
        a.axvspan(ex["L_span_ms"][0], ex["L_span_ms"][1], color=PAL["aqua"], alpha=0.3, lw=0)
        a.set_ylim(-2.8, 3.2)
        a.set_xlim(t[0], t[-1])
        a.set_title(f"[{ex['kind']}] {ex['utt_id'][4:]}  '{ex['char']}' (#{ex['idx']}), p={math.exp(ex['logprob']):.2f},"
                    f" onset-min {ex['on_min']:+.2f}", fontsize=8.5)
        a.set_xlabel("ms from the aligned start of the target syllable", fontsize=8)
    for a in axs.ravel()[n:]:
        a.axis("off")
    axs[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Bilabial onsets: typical vs worst (lips most open at the aligned onset). Shaded = onset token span;"
                 " grey lines = aligned syllable starts", color=PAL["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def pick_examples(recs: list[dict], F: dict[str, np.ndarray], M: dict[str, np.ndarray], work_dir: Path,
                  n_typ: int = 3, n_bad_tr: int = 6, n_bad_val: int = 3) -> list[dict]:
    mk = masks(M)
    om = F["on_min"][:, S0]
    picks: list[tuple[str, int]] = []
    for sp, nb in (("train", n_bad_tr), ("val", n_bad_val)):
        idx = np.nonzero(mk["bil_on"] & (M["split"] == sp) & np.isfinite(om))[0]
        order = idx[np.argsort(-om[idx])]
        picks += [(f"worst {sp}", int(i)) for i in order[:nb]]
    idx = np.nonzero(mk["bil_on"] & np.isfinite(om))[0]
    med = idx[np.argsort(np.abs(om[idx] - np.median(om[idx])))]
    picks = [("typical", int(i)) for i in med[:n_typ]] + picks
    out = []
    for kind, i in picks:
        rec = recs[int(M["utt"][i])]
        s = rec["syllables"][int(M["idx"][i])]
        zg, fps, _, _, _ = load_cues(work_dir / rec["npz"])
        st = float(s["start_s"])
        ts = st + np.arange(-0.6, 0.6001, 0.01)
        v = sample_at(zg, ts)
        out.append({"kind": kind, "utt_id": rec["utt_id"], "speaker": rec["speaker"], "text": rec["text"],
                    "char": s["char"], "idx": int(M["idx"][i]), "logprob": float(s["logprob"]),
                    "on_min": float(om[i]), "closure_min_rel_ms": float(F["ev_on_min"][i] * 1000),
                    "t_ms": (ts - st) * 1000, "ih": v[IH], "mw": v[MW],
                    "L_span_ms": [(s["tokens"][0]["start_s"] - st) * 1000, (s["tokens"][0]["end_s"] - st) * 1000],
                    "syl_starts": [((x["start_s"] - st) * 1000, x["char"]) for x in rec["syllables"]
                                   if -600 <= (x["start_s"] - st) * 1000 <= 600],
                    "neighbours": "".join(x["char"] for x in rec["syllables"][max(0, int(M["idx"][i]) - 2):
                                                                             int(M["idx"][i]) + 3])})
    return out


# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--align", default="work/align/sample_alignments.json")
    ap.add_argument("--work-dir", default="work")
    ap.add_argument("--out-json", default="work/align/validation.json")
    ap.add_argument("--out-png", default="work/align/validation.png")
    ap.add_argument("--examples-png", default="work/align/validation_examples.png")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    t0 = time.time()
    work_dir = (ROOT / args.work_dir).resolve()
    with open(ROOT / args.align, "r", encoding="utf-8") as f:
        recs = json.load(f)
    F, U, extra = build_tables(recs, work_dir)
    M, info = extra["meta"], extra["info"]
    t_feat = time.time() - t0
    res = analyse(F, U, M, args.n_boot, args.seed)
    res["speaker_sync"] = speaker_sync(F, M, extra["av"], res["per_speaker"])
    t1 = time.time()
    res["frame_labels"] = eval_frame_labels(recs, work_dir)
    t_fl = time.time() - t1
    mk = masks(M)
    res["lip_events_raw"] = {f"on_min_{sp}": F["ev_on_min"][mk["bil_on"] & (M["split"] == sp)] for sp in ("train", "val")}
    res["lip_events_raw"].update({f"on_rel_{sp}": F["ev_on_rel"][mk["bil_on"] & (M["split"] == sp)]
                                  for sp in ("train", "val")})
    tab_sc = class_table(F, M, "sc")
    tab_vctc = class_table(F, M, "vctc")
    checks = evaluate_checks(res)
    examples = pick_examples(recs, F, M, work_dir)
    t_an = time.time() - t0 - t_feat
    plot_main(res, ROOT / args.out_png)
    plot_examples(examples, ROOT / args.examples_png)
    txt_sc = table_text(tab_sc, "mean z at the extended-syllable centre (shift 0):")
    txt_vctc = table_text(tab_vctc, "mean z at the vowel-span centre [V.start, V.ext_end) (shift 0):")
    info.update({"n_syllables": int(M["utt"].size), "n_syl_train": int((M["split"] == "train").sum()),
                 "n_syl_val": int((M["split"] == "val").sum()), "angles": sorted({r["angle"] for r in recs}),
                 "speakers_train": sorted({r["speaker"] for r in recs if r["split"] == "train"}),
                 "speakers_val": sorted({r["speaker"] for r in recs if r["split"] == "val"}),
                 "grid_s": GRID, "shifts_ms": SHIFTS_MS, "n_boot": args.n_boot, "seed": args.seed,
                 "sign_convention": "shift s: lips read at aligned time + s; negative best shift = lip event earlier "
                                    "than the aligned time (visual lead)",
                 "criteria": CRITERIA, "contrast_definitions": {k: v[4] for k, v in CONTRASTS.items()}})
    res.pop("lip_events_raw")
    out = {"info": info, "checks": checks, **res,
           "per_class_table_syllable_centre": tab_sc, "per_class_table_vowel_span_centre": tab_vctc,
           "per_class_text": txt_sc + "\n\n" + txt_vctc,
           "examples": [{k: v for k, v in e.items() if k not in ("t_ms", "ih", "mw")} for e in examples],
           "timing_s": {"features": t_feat, "analysis_incl_frame_labels": t_an, "frame_labels": t_fl,
                        "total": time.time() - t0}}
    with open(ROOT / args.out_json, "w", encoding="utf-8") as f:
        json.dump(rnd(out), f, ensure_ascii=False, indent=1)
    # console summary
    print(f"utterances {info['n_utts']} (skipped {info['n_skipped']}), syllables {info['n_syllables']} "
          f"(train {info['n_syl_train']}, val {info['n_syl_val']}), invalid frames {info['n_invalid_frames']}"
          f"/{info['n_frames']}")
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['result']}")
    print("\nshift control (d by shift, train | val):")
    for name in CONTRASTS:
        for sp in ("train", "val"):
            s = res["shift_control"][name][sp]
            print(f"  {name:<18}{sp:<6}" + " ".join(f"{s['d_at_coarse'][str(k)]:+.2f}" for k in COARSE_MS)
                  + f"  best {s['best_shift_ms']:+.0f} ms {s.get('ci95_ms', '')} d_best {s['d_best']:.2f}"
                  + f"  halfmax {s['half_max']}  | uniform d0 {res['uniform_baseline'][name][sp]['d0']:.2f}"
                  + f" best {res['uniform_baseline'][name][sp]['best_shift_ms']}")
    ssy = res["speaker_sync"]
    print("\nper speaker: av_lag | best shift a_onset c_sc d_text | energy steepest rise (ms from aligned start)")
    for spk, r in ssy["per_speaker"].items():
        print(f"  {spk} {r['split']:<5} {r['av_lag_ms']:+7.1f} | {r['a_onset_best_ms']:+7.1f} {r['c_sc_best_ms']:+7.1f} "
              f"{r['d_text_best_ms']:+7.1f} | {r['energy_steepest_rise_ms']:+5.0f} (min {r['energy_min_ms']:+5.0f})")
    for k, v in ssy.items():
        if k not in ("per_speaker", "sign"):
            print(f"  {k}: {rnd(v, 2)}")
    print("\nframe labels (all): d  L | V open | V a | V round | T   frame share")
    for name, row in res["frame_labels"]["variants"].items():
        a_ = row["all"]
        print(f"  {name:<22}" + " ".join(f"{a_[k]['d']:+.2f}" for k in ("L_alvvel_minus_bil_ih", "V_open_minus_close_ih",
                                                                     "V_a_minus_close_ih", "V_spread_minus_round_mw",
                                                                     "T_other_minus_bil_ih"))
              + f"   {rnd(a_['frame_share'], 3)}  | val L {row['val']['L_alvvel_minus_bil_ih']['d']:+.2f}"
              + f" Va {row['val']['V_a_minus_close_ih']['d']:+.2f} T {row['val']['T_other_minus_bil_ih']['d']:+.2f}")
    print("\n" + txt_sc + "\n\n" + txt_vctc)
    print(f"\nwrote {args.out_json}, {args.out_png}, {args.examples_png} in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()

"""Per-character (글자 = Hangul syllable) forced alignment with the trained CTC model.

For every utterance the audio-mode CTC posteriors of ``work/checkpoints/best.pt`` (50 Hz: BiLSTM at 25 Hz x
``model.ctc_upsample`` 2) are force-aligned to ``tokenizer.encode(text)`` (jamo ids). The best CTC path gives a span
(first..last frame) per target token; the tokens of a syllable (L V [T]) give the syllable span:

* raw span      ``[start_s, end_s)``  = first frame of its L token .. last frame of its last token (+1 frame);
* extended span ``[start_s, ext_end_s)`` tiles the time up to the next syllable / space / punctuation token, capped at
  ``end_s + --max-ext`` so that long pauses stay unlabelled. Tokens inside a syllable tile the same way.

Times are seconds from the utterance start (CTC frame j <-> j / 50 s; video frame k <-> k / fps; audio sample n <->
n / 16000 -- all three cover the same span, SPEC section 4); ``abs_*`` = ``row['start'] + t`` is the time in the source
video. :func:`frame_labels` turns an alignment into per-video-frame targets (syllable index, active jamo id).

Backend: ``torchaudio.functional.forced_align`` (present and warning-free in torchaudio 2.11) with an exact numpy CTC
Viterbi (:func:`ctc_viterbi`) as fallback; the Viterbi is always run as a cross-check (path score must agree).

Usage (from the project root)::

    & $py -m tools.align_chars --ckpt work/checkpoints/best.pt --n 400 --out work/align/sample_alignments.json --demo
    & $py -m tools.align_chars --ckpt work/checkpoints/best.pt --utt-ids lip_J_1_F_03_E220_A_001__007 --demo

Outputs (``work/align/``): the alignment list (``--out``), ``align_stats.json`` (sanity statistics + timings),
``frame_label_examples.json`` (per-video-frame labels of the demo utterances), ``sheet_<utt_id>.png`` contact sheets
(``--demo``) and ``label_times.json`` (cache of the raw label sentence times read from the AI-Hub label tars).
Read-only with respect to the package, configs, features, manifests and checkpoints.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tarfile
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch

from avsr.dataset import AVSRDataset, DurationBatchSampler, assign_split, collate_fn, load_manifests
from avsr.evaluate import eval_config
from avsr.models import build_model
from avsr.models.avsr_model import ctc_collapse
from avsr.text import Tokenizer, decompose_syllable, is_hangul_syllable, normalize_text
from avsr.utils import autocast_context, cer, load_checkpoint, safe_console

FONT_PATH = Path(r"C:\Windows\Fonts\malgun.ttf")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\malgunbd.ttf")
PUNCT = ".?!,"
NEG_INF = -np.inf


# ----------------------------------------------------------------------------------------------------------------------
# model / posteriors
# ----------------------------------------------------------------------------------------------------------------------
def load_model(ckpt_path: str, config_path: str | None, overrides: Sequence[str], device: torch.device
               ) -> tuple[torch.nn.Module, Any, Tokenizer, dict]:
    """Model + config exactly as ``avsr.evaluate`` builds them (architecture/input sections from the checkpoint)."""
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = eval_config(config_path, list(overrides), ckpt.get("cfg"))
    tok = Tokenizer()
    vocab = ckpt.get("vocab")
    if vocab is not None and list(vocab) != list(tok.tokens):
        raise ValueError("checkpoint vocabulary differs from avsr.text.Tokenizer")
    model = build_model(cfg, tok.vocab_size)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    info = {"epoch": ckpt.get("epoch"), "step": ckpt.get("step")}
    del ckpt
    return model, cfg, tok, info


@torch.inference_mode()
def ctc_posteriors(model: torch.nn.Module, rows: Sequence[dict], cfg: Any, tok: Tokenizer, device: torch.device,
                   mode: str, num_workers: int, max_frames: int) -> dict[str, np.ndarray]:
    """``{utt_id: float32 log-softmax [T_ctc, V]}`` over the real CTC frames (clean audio, no augmentation)."""
    ds = AVSRDataset(list(rows), cfg, False, tok)
    sampler = DurationBatchSampler(list(rows), max_frames=max_frames, shuffle=False, seed=0)
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs.update(multiprocessing_context="spawn", prefetch_factor=4)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn, num_workers=num_workers,
                                         pin_memory=device.type == "cuda", **kwargs)
    amp = cfg.train.get("amp", "bf16")
    out: dict[str, np.ndarray] = {}
    for batch in loader:
        gpu = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        gpu["tokens"] = gpu["token_lengths"] = None  # skip the attention decoder (only CTC is needed)
        with autocast_context(device, amp):
            res = model(gpu, mode=mode)
        lp = res["ctc_logits"].float().log_softmax(dim=-1).cpu().numpy()
        lens = res["ctc_lengths"].cpu().tolist()
        for b, uid in enumerate(batch["utt_ids"]):
            out[uid] = np.ascontiguousarray(lp[b, :int(lens[b])])
    if ds.n_bad:
        print(f"warning: {ds.n_bad} unreadable utterance(s) were skipped by the dataset", flush=True)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# CTC forced alignment
# ----------------------------------------------------------------------------------------------------------------------
def ctc_viterbi(lp: np.ndarray, targets: Sequence[int], blank: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact CTC Viterbi alignment over the 2L+1 blank-interleaved states.

    Returns (token index per frame [T] (-1 = blank), log-prob of the aligned label per frame [T], path log-prob);
    the path log-prob is -inf when no valid path exists (T too short)."""
    n_t = lp.shape[0]
    tg = np.asarray(targets, dtype=np.int64)
    n_s = 2 * tg.size + 1
    ext = np.full(n_s, blank, dtype=np.int64)
    ext[1::2] = tg
    skip_ok = np.zeros(n_s, dtype=bool)                      # s -> s-2 allowed for a label differing from s-2's
    if tg.size > 1:
        skip_ok[3::2] = tg[1:] != tg[:-1]
    emit = lp[:, ext].astype(np.float64)                     # [T, S]
    dp = np.full(n_s, NEG_INF)
    dp[0] = emit[0, 0]
    if n_s > 1:
        dp[1] = emit[0, 1]
    bp = np.zeros((n_t, n_s), dtype=np.int8)
    cand = np.full((3, n_s), NEG_INF)
    cols = np.arange(n_s)
    for t in range(1, n_t):
        cand[0] = dp
        cand[1, 0] = NEG_INF
        cand[1, 1:] = dp[:-1]
        cand[2, :2] = NEG_INF
        cand[2, 2:] = dp[:-2]
        cand[2, ~skip_ok] = NEG_INF
        arg = cand.argmax(axis=0)
        dp = cand[arg, cols] + emit[t]
        bp[t] = arg
    end = n_s - 1 if n_s == 1 or dp[n_s - 1] >= dp[n_s - 2] else n_s - 2
    score = float(dp[end])
    if not math.isfinite(score):
        return np.full(n_t, -1, np.int64), np.full(n_t, np.nan), NEG_INF
    states = np.empty(n_t, dtype=np.int64)
    states[-1] = end
    for t in range(n_t - 1, 0, -1):
        states[t - 1] = states[t] - bp[t, states[t]]
    tok_idx = np.where(states % 2 == 1, (states - 1) // 2, -1)
    return tok_idx, emit[np.arange(n_t), states], score


def labels_to_token_index(path: np.ndarray, blank: int = 0) -> np.ndarray:
    """Frame labels of a CTC path -> target index per frame (-1 = blank). A new token starts whenever a non-blank
    label follows a blank or a different label (repeats of one target token cannot touch in a valid CTC path)."""
    idx = np.full(path.size, -1, dtype=np.int64)
    k, prev = -1, blank
    for t, lab in enumerate(path.tolist()):
        if lab != blank:
            if lab != prev:
                k += 1
            idx[t] = k
        prev = lab
    return idx


def torchaudio_forced_align() -> Any | None:
    """``torchaudio.functional.forced_align`` if importable, warning-free and working on a tiny case; else None."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # a deprecation warning = do not rely on it
            import torchaudio.functional as taf
            fa = getattr(taf, "forced_align", None)
            if fa is None:
                return None
            lp = torch.log_softmax(torch.randn(1, 12, 5, generator=torch.Generator().manual_seed(0)), -1)
            ali, _ = fa(lp, torch.tensor([[1, 2, 2, 3]], dtype=torch.int32), blank=0)
            if ali.shape != (1, 12):
                return None
        return fa
    except Exception as e:  # noqa: BLE001 - any failure means "use our own Viterbi"
        print(f"torchaudio forced_align unusable ({type(e).__name__}: {e}); using the numpy Viterbi", flush=True)
        return None


def align_tokens(lp: np.ndarray, targets: Sequence[int], fa: Any | None
                 ) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Best CTC path -> (token index per frame, per-frame log-prob, path log-prob, cross-check info)."""
    v_idx, v_fr, v_score = ctc_viterbi(lp, targets)
    check: dict[str, Any] = {"viterbi_score": v_score}
    if fa is None:
        return v_idx, v_fr, v_score, {**check, "backend": "viterbi"}
    try:
        ali, sc = fa(torch.from_numpy(lp)[None], torch.tensor([list(targets)], dtype=torch.int32), blank=0)
    except RuntimeError as e:  # infeasible (T < needed frames) or other failure
        return v_idx, v_fr, v_score, {**check, "backend": "viterbi", "torchaudio_error": str(e)[:200]}
    path = ali[0].numpy().astype(np.int64)
    fr = sc[0].numpy().astype(np.float64)
    score = float(fr.sum())
    idx = labels_to_token_index(path)
    check.update(backend="torchaudio", score_diff=abs(score - v_score) if math.isfinite(v_score) else None,
                 frame_agree=float((idx == v_idx).mean()))
    return idx, fr, score, check


def token_spans(tok_idx: np.ndarray, frame_lp: np.ndarray, n_targets: int) -> list[tuple[int, int, float]] | None:
    """(first frame, last frame, mean log-prob) per target token; None if the path does not emit every token."""
    spans: list[tuple[int, int, float]] = []
    for k in range(n_targets):
        fr = np.nonzero(tok_idx == k)[0]
        if fr.size == 0:
            return None
        spans.append((int(fr[0]), int(fr[-1]), float(frame_lp[fr].mean())))
    return spans


# ----------------------------------------------------------------------------------------------------------------------
# syllable grouping and alignment records
# ----------------------------------------------------------------------------------------------------------------------
def text_groups(text: str, tok: Tokenizer) -> list[tuple[str, str, list[int]]]:
    """(char, kind 'syl'|'space'|'punct', target-token indices) per character of the normalised text, in the order of
    ``tok.encode(text)``; raises on characters the tokenizer maps to <unk>."""
    groups: list[tuple[str, str, list[int]]] = []
    ti = 0
    for ch in normalize_text(text):
        if ch == " ":
            kind, n = "space", 1
        elif is_hangul_syllable(ch):
            kind, n = "syl", 3 if decompose_syllable(ch)[2] else 2
        elif ch in PUNCT:
            kind, n = "punct", 1
        else:
            raise ValueError(f"character {ch!r} has no token (unk)")
        groups.append((ch, kind, list(range(ti, ti + n))))
        ti += n
    if ti != len(tok.encode(text)):
        raise AssertionError("syllable grouping disagrees with tokenizer.encode")
    return groups


def build_record(row: dict, split: str, lp: np.ndarray, tok: Tokenizer, fa: Any | None, ctc_hz: float,
                 max_ext: float, ext_stop: str = "syllable") -> dict:
    """Align one utterance and return its JSON record (``ok`` False + ``error`` when the alignment failed)."""
    text = str(row["text"])
    targets = tok.encode(text)
    n_ctc = int(lp.shape[0])
    rec: dict[str, Any] = {
        "utt_id": row["utt_id"], "speaker": row["speaker"], "angle": row["angle"], "split": split,
        "work_dir": row.get("work_dir"), "npz": row["npz"], "mouth_mp4": row["mouth_mp4"],
        "fps": float(row.get("fps") or 30.0), "n_frames": int(row["n_frames"]), "start": float(row["start"]),
        "end": float(row["end"]), "duration": float(row["duration"]), "text": text, "ctc_hz": ctc_hz,
        "ctc_frames": n_ctc, "n_targets": len(targets),
    }
    greedy = ctc_collapse(torch.from_numpy(lp.argmax(-1))[None], torch.ones(1, n_ctc, dtype=torch.bool),
                          (tok.blank_id, tok.pad_id, tok.sos_id, tok.eos_id))[0]
    rec["greedy_hyp"] = tok.decode(greedy)
    rec["greedy_cer"] = round(cer([tok.decode(targets)], [rec["greedy_hyp"]]), 4)
    groups = text_groups(text, tok)
    tok_idx, frame_lp, score, check = align_tokens(lp, targets, fa)
    rec["backend"] = check.pop("backend")
    rec["check"] = {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v) for k, v in check.items()
                    if k != "viterbi_score"}
    spans = token_spans(tok_idx, frame_lp, len(targets)) if math.isfinite(score) else None
    if spans is None:
        rec.update(ok=False, error=f"no valid CTC path ({n_ctc} frames for {len(targets)} tokens)", syllables=[])
        return rec
    rec["ok"] = True
    rec["path_logprob"] = round(score, 3)
    rec["path_logprob_per_frame"] = round(score / n_ctc, 4)
    rec["mean_token_logprob"] = round(float(np.mean([s[2] for s in spans])), 4)
    t_end = n_ctc / ctc_hz
    starts = [s[0] / ctc_hz for s in spans]
    syllables: list[dict] = []
    others: list[dict] = []
    word = 0
    for gi, (ch, kind, tis) in enumerate(groups):
        # the extended span stops at the next syllable (ext_stop 'syllable': space/punctuation emissions are CTC
        # word-boundary markers without duration, usually in the frame right after the word's last jamo) or at the
        # next token of any kind (ext_stop 'space'); always capped at end_s + max_ext
        later = [g for g in groups[gi + 1:] if ext_stop == "space" or g[1] == "syl"]
        nxt = starts[later[0][2][0]] if later else t_end
        toks = [{"id": targets[i], "name": tok.id_to_token(targets[i]), "start_s": round(spans[i][0] / ctc_hz, 3),
                 "end_s": round((spans[i][1] + 1) / ctc_hz, 3), "logprob": round(spans[i][2], 3)} for i in tis]
        if kind != "syl":
            if kind == "space":
                word += 1
            others.append({"char": ch, **toks[0]})
            continue
        s0, s1 = toks[0]["start_s"], toks[-1]["end_s"]
        ext = round(max(s1, min(nxt, s1 + max_ext)), 3)
        for j, t in enumerate(toks):  # tokens tile the syllable's extended span
            t["ext_end_s"] = toks[j + 1]["start_s"] if j + 1 < len(toks) else ext
        syllables.append({
            "char": ch, "word": word, "start_s": s0, "end_s": s1, "ext_end_s": ext,
            "abs_start_s": round(rec["start"] + s0, 3), "abs_end_s": round(rec["start"] + s1, 3),
            "abs_ext_end_s": round(rec["start"] + ext, 3), "ext_capped": bool(nxt > s1 + max_ext + 1e-9),
            "logprob": round(float(np.mean([t["logprob"] for t in toks])), 3), "tokens": toks,
        })
    rec["syllables"] = syllables
    rec["other_tokens"] = others
    return rec


JAMO_MODES = ("ctc", "proportional")
#: proportional jamo split of a syllable span: L until 25 %, V until 60 % (to the end without a final), T the rest.
#: Motivated by articulation_check: the CTC emits L, V, T in consecutive 20-ms frames at the syllable onset, so their
#: own timing says nothing about the vowel/final; the lip closure of a final ㅁ/ㅂ starts ~30-40 % into the
#: (unshifted) extended span, i.e. ~60 % once the labels lead by ~50 ms.
PROPORTIONAL_SPLIT = (0.25, 0.60)


def frame_labels(rec: dict, n_frames: int | None = None, fps: float | None = None, extended: bool = True,
                 lead_s: float = 0.0, jamo_mode: str = "ctc",
                 split: tuple[float, float] = PROPORTIONAL_SPLIT) -> tuple[np.ndarray, np.ndarray]:
    """Per-video-frame targets for one aligned utterance (the proposed training-label format).

    Video frame k is the instant ``k / fps`` from the utterance start. Returns int32 arrays [n_frames]:
    ``syl``  = index into ``rec['syllables']`` whose span contains the frame, -1 = none (pause / before / after);
    ``jamo`` = token id (L 6..24 / V 25..45 / T 46..72) of the jamo active at that frame, -1 = none.
    ``extended`` uses the tiling spans ``[start_s, ext_end_s)``; otherwise the raw CTC spans ``[start_s, end_s)``.
    ``lead_s`` > 0 moves every label earlier by that much (the lips lead the sound: the closure of a bilabial onset
    lies mostly before the aligned syllable start, see :func:`articulation_check`).
    ``jamo_mode``: 'ctc' = each token from its own CTC start to the next token's start (the last token to the
    syllable end); 'proportional' = fixed fractions ``split`` of the syllable span (see PROPORTIONAL_SPLIT)."""
    if jamo_mode not in JAMO_MODES:
        raise ValueError(f"jamo_mode must be one of {JAMO_MODES}")
    n = int(rec["n_frames"] if n_frames is None else n_frames)
    fps = float(rec["fps"] if fps is None else fps)
    t = np.arange(n, dtype=np.float64) / fps + float(lead_s)
    eps = 1e-9
    syl = np.full(n, -1, dtype=np.int32)
    jamo = np.full(n, -1, dtype=np.int32)
    for si, s in enumerate(rec.get("syllables", [])):
        s0, s1 = s["start_s"], (s["ext_end_s"] if extended else s["end_s"])
        inside = (t >= s0 - eps) & (t < s1 - eps)
        syl[inside] = si
        toks = s["tokens"]
        if jamo_mode == "ctc":
            bounds = [tk["start_s"] for tk in toks[1:]] + [s1]
        else:
            fr = [split[0], split[1]] if len(toks) == 3 else [split[0]]
            bounds = [s0 + f * (s1 - s0) for f in fr] + [s1]
        lo = s0
        for tk, hi in zip(toks, bounds):
            jamo[inside & (t >= lo - eps) & (t < hi - eps)] = tk["id"]
            lo = hi
    return syl, jamo


def label_coverage(recs: Sequence[dict], lead_s: float, jamo_mode: str) -> dict[str, Any]:
    """Share of video frames per label class over ``recs`` (none / onset L / vowel V / final T), overall and inside
    the speech region (first syllable start .. last syllable extended end)."""
    counts = np.zeros(4, dtype=np.int64)
    inner = np.zeros(4, dtype=np.int64)
    for rec in recs:
        syl, jamo = frame_labels(rec, lead_s=lead_s, jamo_mode=jamo_mode)
        cls = np.select([jamo < 0, jamo < 25, jamo < 46], [0, 1, 2], 3)
        counts += np.bincount(cls, minlength=4)
        hit = np.nonzero(syl >= 0)[0]
        if hit.size:
            inner += np.bincount(cls[hit[0]:hit[-1] + 1], minlength=4)
    names = ("none", "L", "V", "T")
    return {"lead_s": lead_s, "jamo_mode": jamo_mode, "n_frames": int(counts.sum()),
            "frac_all": {n: round(float(c / max(counts.sum(), 1)), 4) for n, c in zip(names, counts)},
            "frac_inside_speech": {n: round(float(c / max(inner.sum(), 1)), 4) for n, c in zip(names, inner)}}


# ----------------------------------------------------------------------------------------------------------------------
# raw label times (for the onset sanity check)
# ----------------------------------------------------------------------------------------------------------------------
def labels_root_from_index(work_dir: Path) -> Path | None:
    """Split folder of the source dataset (``...\\2.Validation``), found via ``work/media_index.json``."""
    p = work_dir / "media_index.json"
    if not p.is_file():
        return None
    with open(p, "r", encoding="utf-8") as f:
        index = json.load(f).get("index", {})
    for ref in index.values():
        src = Path(ref.get("source_path", ""))
        root = src.parent.parent if ref.get("source_kind") == "tar" else None
        if root is not None and (root / "라벨링데이터").is_dir():
            return root
    return None


def load_label_times(stems: Iterable[str], labels_root: Path | None, cache_path: Path) -> dict[str, dict[int, list]]:
    """``{stem: {sentence_id: [start_time, end_time]}}`` from the AI-Hub label JSONs (tars or plain dirs), cached in
    ``cache_path``. Stems not found are simply absent."""
    cache: dict[str, dict[str, list]] = {}
    if cache_path.is_file():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    want = {s for s in stems if s not in cache}
    if want and labels_root is not None and (labels_root / "라벨링데이터").is_dir():
        def take(stem: str, raw: bytes) -> None:
            doc = json.loads(raw.decode("utf-8-sig"))
            doc = doc[0] if isinstance(doc, list) else doc
            cache[stem] = {str(int(s["ID"])): [float(s["start_time"]), float(s["end_time"])]
                           for s in doc.get("Sentence_info", [])}
            want.discard(stem)

        lab = labels_root / "라벨링데이터"
        for tar_path in sorted(lab.glob("*.tar")):
            if not want:
                break
            with tarfile.open(tar_path, "r") as tf:
                for m in tf:
                    stem = Path(m.name).stem
                    if m.isfile() and m.name.lower().endswith(".json") and stem in want:
                        fh = tf.extractfile(m)
                        if fh is not None:
                            take(stem, fh.read())
        for js in sorted(lab.rglob("*.json")) if want else []:
            if js.stem in want:
                take(js.stem, js.read_bytes())
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    return {s: {int(k): v for k, v in d.items()} for s, d in cache.items()}


def add_label_offsets(rec: dict, row: dict, label_times: dict[str, dict[int, list]]) -> None:
    """Label onset/offset inside the utterance (= the padding actually applied) and the aligned first/last syllable
    relative to them (``onset_delta_s`` > 0: first syllable starts after the labelled onset)."""
    sent = label_times.get(str(row["video_stem"]), {}).get(int(row["sentence_id"]))
    if sent is None:
        return
    shift = float(row.get("time_shift") or 0.0)
    pad_l = sent[0] + shift - float(row["start"])
    pad_r = float(row["end"]) - (sent[1] + shift)
    rec["label_pad_left_s"] = round(pad_l, 3)
    rec["label_pad_right_s"] = round(pad_r, 3)
    if rec.get("ok") and rec["syllables"]:
        rec["onset_delta_s"] = round(rec["syllables"][0]["start_s"] - pad_l, 3)
        rec["offset_delta_s"] = round((float(row["duration"]) - pad_r) - rec["syllables"][-1]["end_s"], 3)


# ----------------------------------------------------------------------------------------------------------------------
# timing sanity against the signals (lip closure of bilabials, audio energy)
# ----------------------------------------------------------------------------------------------------------------------
BILABIAL = {"ㅁ", "ㅂ", "ㅃ", "ㅍ"}


def _zscore(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    if int(ok.sum()) >= 2:
        out[ok] = (x[ok] - x[ok].mean()) / max(float(x[ok].std()), 1e-6)
    return out


def articulation_check(recs: Sequence[dict], max_off_ms: float = 240.0) -> dict[str, Any]:
    """Where do the CTC syllable onsets sit relative to the articulation?

    * lips: mean per-utterance z-scored ``inner_height`` (npz cue[:, 1]) at video frames around each syllable start,
      bilabial onsets (ㅁ ㅂ ㅃ ㅍ: lips closed just before the release) minus all other onsets. The offset of the
      minimum = where the bilabial closure sits relative to the aligned start (negative = before it).
    * audio: mean z-scored 20-ms log energy around each syllable start (a rise after 0 = start at the acoustic onset).
    """
    fps_ref = 30.0
    offs = np.arange(-int(round(max_off_ms / 1000 * fps_ref)), int(round(max_off_ms / 1000 * fps_ref)) + 1)
    e_offs = np.arange(-int(max_off_ms // 20), int(max_off_ms // 20) + 1)
    acc = {"bilabial": [[] for _ in offs], "other": [[] for _ in offs], "coda_bilabial": [[] for _ in offs]}
    energy = [[] for _ in e_offs]
    rel_edges = np.round(np.arange(-0.5, 1.5001, 0.1), 2)       # position inside the extended span (0 = start, 1 = end)
    rel = {"onset_bilabial": [[] for _ in rel_edges[:-1]], "coda_bilabial": [[] for _ in rel_edges[:-1]],
           "other": [[] for _ in rel_edges[:-1]]}
    for rec in recs:
        if not rec.get("ok") or not rec.get("work_dir"):
            continue
        with np.load(Path(rec["work_dir"]) / rec["npz"]) as z:
            inner = np.asarray(z["cue"][:, 1], dtype=np.float64)
            valid = np.asarray(z["valid"]) > 0
            audio = np.asarray(z["audio"], dtype=np.float64)
            fps = float(z["fps"])
        zi = _zscore(inner, valid)
        t_v = np.arange(zi.size) / fps
        n_e = audio.size // 320
        le = np.log10(np.square(audio[:n_e * 320].reshape(n_e, 320)).mean(axis=1) + 1.0)
        ze = _zscore(le, np.ones(n_e, bool))
        for s in rec["syllables"]:
            onset = s["tokens"][0]["name"][2:]
            coda = s["tokens"][-1]
            coda_bil = coda["name"].startswith("T:") and coda["name"][2:] in {"ㅁ", "ㅂ"}
            k0 = int(round(s["start_s"] * fps))
            key = "bilabial" if onset in BILABIAL else "other"
            for j, d in enumerate(offs):
                if 0 <= k0 + d < zi.size and not np.isnan(zi[k0 + d]):
                    acc[key][j].append(zi[k0 + d])
            if coda_bil:
                kc = int(round(coda["start_s"] * fps))
                for j, d in enumerate(offs):
                    if 0 <= kc + d < zi.size and not np.isnan(zi[kc + d]):
                        acc["coda_bilabial"][j].append(zi[kc + d])
            e0 = int(round(s["start_s"] * 50))
            for j, d in enumerate(e_offs):
                if 0 <= e0 + d < n_e:
                    energy[j].append(ze[e0 + d])
            dur = s["ext_end_s"] - s["start_s"]
            if dur >= 0.12 and not s["ext_capped"]:  # >= ~4 video frames and not before a pause
                rkey = "onset_bilabial" if onset in BILABIAL else ("coda_bilabial" if coda_bil else "other")
                pos = (t_v - s["start_s"]) / dur
                b = np.searchsorted(rel_edges, pos, side="right") - 1
                for bi, v in zip(b.tolist(), zi.tolist()):
                    if 0 <= bi < len(rel_edges) - 1 and not math.isnan(v):
                        rel[rkey][bi].append(v)
    mean = {k: np.array([np.mean(v) if v else np.nan for v in lst]) for k, lst in acc.items()}
    diff = mean["bilabial"] - mean["other"]
    j_min = int(np.nanargmin(diff)) if np.isfinite(diff).any() else None
    e_mean = np.array([np.mean(v) if v else np.nan for v in energy])
    grad = np.diff(e_mean)
    rel_mean = {k: np.array([np.mean(v) if v else np.nan for v in lst]) for k, lst in rel.items()}
    centres = 0.5 * (rel_edges[:-1] + rel_edges[1:])
    closed_on = centres[rel_mean["onset_bilabial"] < -0.5]
    closed_coda = centres[(rel_mean["coda_bilabial"] < -0.5) & (centres >= 0)]
    return {
        "relative_position_bins": [round(float(c), 2) for c in centres],
        "inner_height_z_by_relpos": {k: [round(float(x), 3) for x in v] for k, v in rel_mean.items()},
        "n_by_relpos": {k: int(len(v[len(v) // 2])) for k, v in rel.items()},
        "onset_bilabial_closed_relpos": ([round(float(closed_on.min()), 2), round(float(closed_on.max()), 2)]
                                         if closed_on.size else None),
        "coda_bilabial_closure_starts_relpos": round(float(closed_coda.min()), 2) if closed_coda.size else None,
        "offsets_ms_video": [round(float(d) * 1000 / fps_ref, 1) for d in offs],
        "inner_height_z_bilabial_onset": [round(float(x), 3) for x in mean["bilabial"]],
        "inner_height_z_other_onset": [round(float(x), 3) for x in mean["other"]],
        "inner_height_z_bilabial_coda_vs_T_start": [round(float(x), 3) for x in mean["coda_bilabial"]],
        "n_bilabial_onsets": len(acc["bilabial"][len(offs) // 2]), "n_other_onsets": len(acc["other"][len(offs) // 2]),
        "n_bilabial_codas": len(acc["coda_bilabial"][len(offs) // 2]),
        "bilabial_minus_other": [round(float(x), 3) for x in diff],
        "bilabial_closure_offset_ms": None if j_min is None else round(float(offs[j_min]) * 1000 / fps_ref, 1),
        "coda_closure_offset_ms": (round(float(offs[int(np.nanargmin(mean["coda_bilabial"]))]) * 1000 / fps_ref, 1)
                                   if np.isfinite(mean["coda_bilabial"]).any() else None),
        "offsets_ms_audio": [int(d * 20) for d in e_offs],
        "log_energy_z_at_syllable_start": [round(float(x), 3) for x in e_mean],
        "energy_steepest_rise_offset_ms": (int(e_offs[int(np.nanargmax(grad))] * 20 + 10)
                                           if np.isfinite(grad).any() else None),
        "energy_peak_offset_ms": int(e_offs[int(np.nanargmax(e_mean))] * 20) if np.isfinite(e_mean).any() else None,
    }


# ----------------------------------------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------------------------------------
def _dist_ms(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "median_ms": None, "p5_ms": None, "p95_ms": None, "mean_ms": None}
    v = np.asarray(values) * 1000.0
    return {"n": int(v.size), "median_ms": round(float(np.median(v)), 1), "p5_ms": round(float(np.percentile(v, 5)), 1),
            "p95_ms": round(float(np.percentile(v, 95)), 1), "mean_ms": round(float(v.mean()), 1),
            "frac_le_40ms": round(float((v <= 40.0 + 1e-6).mean()), 4),
            "frac_lt_40ms": round(float((v < 40.0 - 1e-6).mean()), 4),
            "frac_gt_600ms": round(float((v > 600.0 + 1e-6).mean()), 4)}


def summarize(recs: Sequence[dict], ctc_hz: float) -> dict[str, Any]:
    """Sanity statistics over one group of alignment records."""
    ok = [r for r in recs if r.get("ok")]
    raw = [s["end_s"] - s["start_s"] for r in ok for s in r["syllables"]]
    ext = [s["ext_end_s"] - s["start_s"] for r in ok for s in r["syllables"]]
    tok_lp = [t["logprob"] for r in ok for s in r["syllables"] for t in s["tokens"]]
    all_tok_lp = tok_lp + [t["logprob"] for r in ok for t in r.get("other_tokens", [])]
    n_syl = sum(len(r["syllables"]) for r in ok)
    gaps = [r["syllables"][i + 1]["start_s"] - r["syllables"][i]["end_s"] for r in ok
            for i in range(len(r["syllables"]) - 1)]
    on = [r["onset_delta_s"] for r in ok if "onset_delta_s" in r]
    off = [r["offset_delta_s"] for r in ok if "offset_delta_s" in r]
    tol = 1.0 / ctc_hz
    out: dict[str, Any] = {
        "n_utts": len(recs), "n_ok": len(ok), "success_rate": round(len(ok) / len(recs), 4) if recs else None,
        "failures": [{"utt_id": r["utt_id"], "error": r.get("error")} for r in recs if not r.get("ok")],
        "n_syllables": n_syl, "speakers": sorted({r["speaker"] for r in recs}),
        "syllable_raw_span": _dist_ms(raw), "syllable_extended_span": _dist_ms(ext),
        "frac_ext_capped": round(float(np.mean([s["ext_capped"] for r in ok for s in r["syllables"]])), 4) if n_syl else None,
        "gap_between_syllables": _dist_ms(gaps),
        "mean_token_logprob": round(float(np.mean(all_tok_lp)), 4) if all_tok_lp else None,
        "mean_syllable_token_logprob": round(float(np.mean(tok_lp)), 4) if tok_lp else None,
        "frac_tokens_logprob_lt_log0.5": round(float((np.asarray(all_tok_lp) < math.log(0.5)).mean()), 4) if all_tok_lp else None,
        "frac_utts_with_syllable_prob_lt_0.1": round(float(np.mean([min(s["logprob"] for s in r["syllables"])
                                                                      < math.log(0.1) for r in ok if r["syllables"]])), 4)
        if ok else None,
        "mean_path_logprob_per_frame":round(float(np.mean([r["path_logprob_per_frame"] for r in ok])), 4) if ok else None,
        "greedy_cer_mean": round(float(np.mean([r["greedy_cer"] for r in recs])), 4) if recs else None,
        "syllables_per_sec_median": round(float(np.median([len(r["syllables"]) / r["duration"] for r in ok])), 2) if ok else None,
        "torchaudio_vs_viterbi": {
            "n_compared": sum(1 for r in ok if r["check"].get("score_diff") is not None),
            "max_score_diff": max((r["check"]["score_diff"] for r in ok if r["check"].get("score_diff") is not None),
                                  default=None),
            "mean_frame_agree": round(float(np.mean([r["check"]["frame_agree"] for r in ok if "frame_agree" in r["check"]])), 4)
            if any("frame_agree" in r["check"] for r in ok) else None},
        "first_syllable_at_frame0": round(float(np.mean([r["syllables"][0]["start_s"] <= 1e-9 for r in ok])), 4) if ok else None,
    }
    if on:
        a = np.asarray(on)
        out["onset"] = {
            "n": int(a.size), "definition": "first syllable start - labelled sentence onset inside the utterance "
                                            "(= left pad actually applied); >0 = starts after the pad",
            "frac_starts_after_pad": round(float((a >= -tol).mean()), 4),
            "frac_more_than_100ms_before_pad": round(float((a < -0.1).mean()), 4),
            "frac_more_than_300ms_after_pad": round(float((a > 0.3).mean()), 4),
            "median_ms": round(float(np.median(a)) * 1000, 1), "p5_ms": round(float(np.percentile(a, 5)) * 1000, 1),
            "p95_ms": round(float(np.percentile(a, 95)) * 1000, 1),
            "pad_left_median_ms": round(float(np.median([r["label_pad_left_s"] for r in ok if "onset_delta_s" in r])) * 1000, 1),
            "median_ms_by_speaker": {spk: round(float(np.median([r["onset_delta_s"] for r in ok
                                                                  if r["speaker"] == spk and "onset_delta_s" in r])) * 1000, 1)
                                     for spk in sorted({r["speaker"] for r in ok if "onset_delta_s" in r})}}
    if off:
        a = np.asarray(off)
        out["offset"] = {
            "n": int(a.size), "definition": "labelled sentence offset - last syllable raw end; >0 = ends before the "
                                            "labelled offset",
            "frac_ends_before_label_end": round(float((a >= -tol).mean()), 4),
            "frac_more_than_100ms_after_label_end": round(float((a < -0.1).mean()), 4),
            "median_ms": round(float(np.median(a)) * 1000, 1), "p5_ms": round(float(np.percentile(a, 5)) * 1000, 1),
            "p95_ms": round(float(np.percentile(a, 95)) * 1000, 1)}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# demo output
# ----------------------------------------------------------------------------------------------------------------------
def table_rows(rec: dict) -> list[str]:
    rows = []
    for s in rec["syllables"]:
        jamo = " ".join(f"{t['name'][2:]}{int(round(t['start_s'] * 1000))}" for t in s["tokens"])
        rows.append(f"{s['char']} | {int(round(s['start_s'] * 1000))}-{int(round(s['end_s'] * 1000))} ms"
                    f" | ext {int(round(s['ext_end_s'] * 1000))} | {jamo} | p={math.exp(s['logprob']):.2f}")
    return rows


def print_table(rec: dict) -> None:
    print(f"\n{rec['utt_id']}  speaker {rec['speaker']} ({rec['split']})  {rec['duration']:.2f} s  fps {rec['fps']}  "
          f"video start {rec['start']:.2f} s", flush=True)
    print(f"  REF    {rec['text']}\n  GREEDY {rec['greedy_hyp']}  (CER {100 * rec['greedy_cer']:.1f}%)")
    if "onset_delta_s" in rec:
        print(f"  label pad left {1000 * rec['label_pad_left_s']:.0f} ms, first syllable at "
              f"{1000 * rec['syllables'][0]['start_s']:.0f} ms (delta {1000 * rec['onset_delta_s']:+.0f} ms)")
    print("  글자 | start-end (ms) | extended end (ms) | jamo@start ms | mean token prob")
    for line in table_rows(rec):
        print("  " + line)


def read_mouth_frames(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok or img is None:
            break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise OSError(f"cannot decode {path}")
    return np.stack(frames)


def _wrap(draw: Any, text: str, font: Any, width: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > width and cur:
            lines.append(cur)
            cur = ch.lstrip() if ch == " " else ch
        else:
            cur += ch
    return lines + ([cur] if cur else [])


def make_sheet(rec: dict, out_png: Path, cols: int = 10, cell: int = 128) -> Path:
    """Contact sheet: per syllable the mouth crop at the centre of its raw span, the syllable and its time."""
    from PIL import Image, ImageDraw, ImageFont

    frames = read_mouth_frames(Path(rec["work_dir"]) / rec["mouth_mp4"])
    f_char = ImageFont.truetype(str(FONT_BOLD_PATH if FONT_BOLD_PATH.is_file() else FONT_PATH), 24)
    f_small = ImageFont.truetype(str(FONT_PATH), 13)
    f_head = ImageFont.truetype(str(FONT_PATH), 16)
    pad, text_h = 8, 50
    width = cols * (cell + pad) + pad
    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    head = [f"{rec['utt_id']}   speaker {rec['speaker']} ({rec['split']}), angle {rec['angle']}, "
            f"{rec['duration']:.2f} s, video {rec['fps']:g} fps, starts at {rec['start']:.2f} s of the source video"]
    head += _wrap(probe, "문장: " + rec["text"], f_head, width - 2 * pad)
    head += _wrap(probe, "각 칸: 음절(글자) CTC 구간의 중심 시각 영상 프레임 / 글자 / 구간 start-end (ms, 발화 시작 기준) "
                         "/ 프레임 번호. 글자 색은 어절마다 바뀜.", f_small, width - 2 * pad)
    head_h = pad + 22 * len(head) + pad
    syl = rec["syllables"]
    n_rows = max(1, math.ceil(len(syl) / cols))
    img = Image.new("RGB", (width, head_h + n_rows * (cell + text_h + pad) + pad), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = pad
    for i, line in enumerate(head):
        draw.text((pad, y), line, fill=(0, 0, 0) if i == 0 else (40, 40, 40), font=f_head if i < len(head) - 1 else f_small)
        y += 22
    colors = [(20, 20, 20), (20, 70, 200)]
    fps = float(rec["fps"])
    for i, s in enumerate(syl):
        r, c = divmod(i, cols)
        x0 = pad + c * (cell + pad)
        y0 = head_h + r * (cell + text_h + pad)
        centre = 0.5 * (s["start_s"] + s["end_s"])
        k = int(np.clip(round(centre * fps), 0, len(frames) - 1))
        crop = cv2.resize(frames[k], (cell, cell), interpolation=cv2.INTER_CUBIC)
        img.paste(Image.fromarray(crop), (x0, y0))
        label = s["char"]
        tw = draw.textlength(label, font=f_char)
        draw.text((x0 + (cell - tw) / 2, y0 + cell - 2), label, fill=colors[s["word"] % 2], font=f_char)
        t_txt = f"{int(round(s['start_s'] * 1000))}-{int(round(s['end_s'] * 1000))}ms  f{k}"
        tw = draw.textlength(t_txt, font=f_small)
        draw.text((x0 + (cell - tw) / 2, y0 + cell + 30), t_txt, fill=(90, 90, 90), font=f_small)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png


def choose_demo(recs: Sequence[dict], rng: np.random.Generator, n_train: int = 3) -> list[dict]:
    """``n_train`` TRAIN utterances of different speakers + 1 validation utterance, angle A, 3-6 s long."""
    ok = [r for r in recs if r.get("ok") and r["angle"] == "A" and 3.0 <= r["duration"] <= 6.0]
    train = [r for r in ok if r["split"] == "train"]
    val = [r for r in ok if r["split"] == "val"]
    picks: list[dict] = []
    for i in rng.permutation(len(train)).tolist():
        if all(train[i]["speaker"] != p["speaker"] for p in picks):
            picks.append(train[i])
        if len(picks) == n_train:
            break
    if val:
        picks.append(val[int(rng.integers(0, len(val)))])
    return picks


# ----------------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------------
def sample_rows(splits: dict[str, list[dict]], n_train: int, n_val: int, seed: int, angle: str
                ) -> list[tuple[dict, str]]:
    """Stratified random TRAIN sample (``n_train`` split evenly over the train speakers) + ``n_val`` validation rows."""
    rng = np.random.default_rng(seed)
    out: list[tuple[dict, str]] = []
    train = [r for r in splits["train"] if str(r["angle"]) == angle]
    speakers = sorted({str(r["speaker"]) for r in train})
    for i, spk in enumerate(speakers):
        pool = sorted((r for r in train if r["speaker"] == spk), key=lambda r: r["utt_id"])
        k = n_train // len(speakers) + (1 if i < n_train % len(speakers) else 0)
        out += [(pool[j], "train") for j in sorted(rng.choice(len(pool), size=min(k, len(pool)), replace=False))]
    val = sorted((r for r in splits["val"] if str(r["angle"]) == angle), key=lambda r: r["utt_id"])
    if val and n_val > 0:
        out += [(val[j], "val") for j in sorted(rng.choice(len(val), size=min(n_val, len(val)), replace=False))]
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-syllable (글자) CTC forced alignment with the trained model.")
    p.add_argument("--ckpt", default="work/checkpoints/best.pt")
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    p.add_argument("--n", type=int, default=400, help="random TRAIN utterances (stratified over the train speakers)")
    p.add_argument("--n-val", type=int, default=100, help="random utterances of the validation speaker(s)")
    p.add_argument("--angle", default="A")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--utt-ids", nargs="+", default=None, help="align these utterances instead of a random sample")
    p.add_argument("--out", default=None, help="alignment JSON (default work/align/sample_alignments.json, or "
                                               "work/align/alignments_selected.json with --utt-ids)")
    p.add_argument("--demo", action="store_true", help="print tables + contact sheets (3 train + 1 val, or --utt-ids)")
    p.add_argument("--mode", default="audio", choices=("audio", "av"), help="model modality for the posteriors")
    p.add_argument("--backend", default="auto", choices=("auto", "torchaudio", "viterbi"))
    p.add_argument("--max-ext", type=float, default=0.3,
                   help="extended spans reach at most this far (s) beyond the raw span end")
    p.add_argument("--ext-stop", default="syllable", choices=("syllable", "space"),
                   help="extended span ends at the next syllable (default) or at the next token incl. space/punct")
    p.add_argument("--label-lead", type=float, default=0.05,
                   help="lead (s) of the 'proposed' per-frame label variant (labels move earlier: lips lead sound)")
    p.add_argument("--jamo-mode", default="proportional", choices=JAMO_MODES,
                   help="jamo tiling of the 'proposed' per-frame label variant")
    p.add_argument("--labels-root", default=None, help="dataset split folder holding 라벨링데이터 (default: from "
                                                       "work/media_index.json; 'none' disables the onset check)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=3200, help="25-fps frames per forward batch")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    safe_console()
    args = parse_args(argv)
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, tok, info = load_model(args.ckpt, args.config, args.set, device)
    work = Path(str(cfg.work_dir)).resolve()
    align_dir = work / "align"
    align_dir.mkdir(parents=True, exist_ok=True)
    ctc_hz = 1000.0 / 10.0 / int(cfg.audio.stack) * int(model.ctc_upsample)
    timings["load_model_s"] = time.perf_counter() - t0
    print(f"model {args.ckpt} (epoch {info['epoch']}, step {info['step']}) on {device}; CTC at {ctc_hz:g} Hz, "
          f"mode={args.mode}", flush=True)

    t = time.perf_counter()
    rows_all = load_manifests(work)
    splits = assign_split(rows_all, cfg)
    split_of = {r["utt_id"]: name for name, rs in splits.items() for r in rs}
    if args.utt_ids:
        by_id = {r["utt_id"]: r for r in rows_all}
        missing = [u for u in args.utt_ids if u not in by_id]
        if missing:
            raise SystemExit(f"unknown utt_ids: {missing}")
        bad = [u for u in args.utt_ids if by_id[u]["has_unk"]]
        if bad:
            raise SystemExit(f"utterances with <unk> text cannot be aligned: {bad}")
        chosen = [(by_id[u], split_of.get(u, "none")) for u in args.utt_ids]
    else:
        chosen = sample_rows(splits, args.n, args.n_val, args.seed, args.angle)
    out_path = Path(args.out) if args.out else align_dir / ("alignments_selected.json" if args.utt_ids
                                                              else "sample_alignments.json")
    timings["manifests_s"] = time.perf_counter() - t
    print(f"{len(chosen)} utterances ({sum(s == 'train' for _, s in chosen)} train, "
          f"{sum(s == 'val' for _, s in chosen)} val, {len({r['speaker'] for r, _ in chosen})} speakers)", flush=True)

    t = time.perf_counter()
    posts = ctc_posteriors(model, [r for r, _ in chosen], cfg, tok, device, args.mode, args.workers, args.max_frames)
    timings["forward_s"] = time.perf_counter() - t
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    t = time.perf_counter()
    fa = None if args.backend == "viterbi" else torchaudio_forced_align()
    if args.backend == "torchaudio" and fa is None:
        raise SystemExit("torchaudio forced_align requested but unusable")
    backend = "torchaudio.functional.forced_align" if fa is not None else "numpy CTC Viterbi"
    recs = []
    for row, split in chosen:
        lp = posts.get(row["utt_id"])
        if lp is None:
            recs.append({"utt_id": row["utt_id"], "speaker": row["speaker"], "angle": row["angle"], "split": split,
                         "ok": False, "error": "no posteriors (unreadable utterance)", "syllables": [],
                         "greedy_cer": 1.0, "duration": row["duration"]})
            continue
        recs.append(build_record(row, split, lp, tok, fa, ctc_hz, args.max_ext, args.ext_stop))
    timings["align_s"] = time.perf_counter() - t
    print(f"aligned {sum(r['ok'] for r in recs)}/{len(recs)} with {backend} "
          f"({timings['align_s']:.1f} s incl. the numpy Viterbi cross-check)", flush=True)

    t = time.perf_counter()
    if args.labels_root is None:
        root = labels_root_from_index(work)
    else:
        root = None if args.labels_root.lower() == "none" else Path(args.labels_root)
    label_times = load_label_times({str(r["video_stem"]) for r, _ in chosen}, root, align_dir / "label_times.json")
    for rec, (row, _) in zip(recs, chosen):
        add_label_offsets(rec, row, label_times)
    timings["label_times_s"] = time.perf_counter() - t

    demo_recs: list[dict] = []
    if args.demo:
        t = time.perf_counter()
        demo_recs = [r for r in recs if r.get("ok")] if args.utt_ids else choose_demo(recs, np.random.default_rng(args.seed))
        examples = []
        for rec in demo_recs:
            print_table(rec)
            png = make_sheet(rec, align_dir / f"sheet_{rec['utt_id']}.png")
            rec["sheet_png"] = str(png)
            variants = {}
            for name, lead, mode in (("literal", 0.0, "ctc"), ("proposed", args.label_lead, args.jamo_mode)):
                syl, jamo = frame_labels(rec, lead_s=lead, jamo_mode=mode)
                variants[name] = {"lead_s": lead, "jamo_mode": mode, "syl": syl.tolist(), "jamo": jamo.tolist(),
                                  "jamo_names": [tok.id_to_token(j)[2:] if j >= 0 else "-" for j in jamo.tolist()]}
            examples.append({"utt_id": rec["utt_id"], "fps": rec["fps"], "n_frames": rec["n_frames"],
                             "syllables": [s["char"] for s in rec["syllables"]], "variants": variants})
            print(f"  sheet: {png}")
            for name, v in variants.items():
                print(f"  per-frame labels, {name} (lead {1000 * v['lead_s']:.0f} ms, jamo {v['jamo_mode']}), "
                      f"frames 0-59:\n    " + " ".join(
                          (f"{rec['syllables'][i]['char']}{jn}" if i >= 0 else "·")
                          for i, jn in zip(v["syl"][:60], v["jamo_names"][:60])))
        with open(align_dir / "frame_label_examples.json", "w", encoding="utf-8") as f:
            json.dump({"format": FRAME_LABEL_FORMAT, "examples": examples}, f, ensure_ascii=False)
        timings["demo_s"] = time.perf_counter() - t

    t = time.perf_counter()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False)
    stats = {"ckpt": args.ckpt, "epoch": info["epoch"], "mode": args.mode, "backend": backend, "ctc_hz": ctc_hz,
             "max_ext_s": args.max_ext, "ext_stop": args.ext_stop, "seed": args.seed, "n_requested": {"train": args.n, "val": args.n_val},
             "groups": {g: summarize([r for r in recs if r["split"] == g], ctc_hz)
                        for g in sorted({r["split"] for r in recs})},
             "all": summarize(recs, ctc_hz), "demo_utts": [r["utt_id"] for r in demo_recs]}
    t_art = time.perf_counter()
    stats["articulation_check"] = articulation_check(recs)
    timings["articulation_check_s"] = time.perf_counter() - t_art
    stats["frame_label_coverage"] = {
        name: label_coverage([r for r in recs if r.get("ok")], lead, mode)
        for name, lead, mode in (("literal", 0.0, "ctc"), ("proposed", args.label_lead, args.jamo_mode))}
    timings["write_s"] = time.perf_counter() - t
    timings["total_s"] = time.perf_counter() - t0
    stats["timings_s"] = {k: round(v, 2) for k, v in timings.items()}
    stats_path = out_path.with_name("align_stats.json" if out_path.name == "sample_alignments.json"
                                    else out_path.stem + "_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    for g, s in stats["groups"].items():
        raw, ext = s["syllable_raw_span"], s["syllable_extended_span"]
        print(f"\n[{g}] {s['n_ok']}/{s['n_utts']} aligned, {s['n_syllables']} syllables, greedy CER "
              f"{100 * (s['greedy_cer_mean'] or 0):.1f}%, mean token logprob {s['mean_token_logprob']}")
        print(f"   raw span  median {raw['median_ms']} ms (p5 {raw['p5_ms']}, p95 {raw['p95_ms']}), "
              f"<=40 ms {raw.get('frac_le_40ms')}, >600 ms {raw.get('frac_gt_600ms')}")
        print(f"   extended  median {ext['median_ms']} ms (p5 {ext['p5_ms']}, p95 {ext['p95_ms']}), "
              f"<=40 ms {ext.get('frac_le_40ms')}, >600 ms {ext.get('frac_gt_600ms')}, capped {s['frac_ext_capped']}")
        if "onset" in s:
            o = s["onset"]
            print(f"   onset: first syllable starts after the label pad in {100 * o['frac_starts_after_pad']:.1f}% "
                  f"(median {o['median_ms']:+.0f} ms, p5 {o['p5_ms']:+.0f}, p95 {o['p95_ms']:+.0f}; pad median "
                  f"{o['pad_left_median_ms']:.0f} ms)")
    art = stats["articulation_check"]
    print(f"\nlip check: bilabial-onset closure minimum at {art['bilabial_closure_offset_ms']} ms from the aligned "
          f"syllable start (closed over relative positions {art['onset_bilabial_closed_relpos']} of the extended "
          f"span); final ㅁ/ㅂ closure minimum {art['coda_closure_offset_ms']} ms after the T token start, closure "
          f"begins at relative position {art['coda_bilabial_closure_starts_relpos']}; audio energy rises fastest "
          f"{art['energy_steepest_rise_offset_ms']} ms after the syllable start (peak {art['energy_peak_offset_ms']} ms)")
    for name, cov in stats["frame_label_coverage"].items():
        print(f"frame labels {name}: inside speech {cov['frac_inside_speech']}, all frames {cov['frac_all']}")
    print(f"\nwrote {out_path} and {stats_path}; timings {stats['timings_s']}", flush=True)
    return 0


FRAME_LABEL_FORMAT = (
    "per utterance, int arrays of length n_frames at the mouth-video fps (npz fps, 30 or 29.97); frame k = instant "
    "k/fps + lead_s from the utterance start (same origin as the npz audio). syl[k] = index into the utterance's "
    "syllables (글자, text order without spaces/punctuation) whose extended span [start_s, ext_end_s) contains that "
    "instant, -1 = none (pause / before the first / after the last syllable). jamo[k] = tokenizer id of the active jamo "
    "(L 6..24 onset, V 25..45 vowel, T 46..72 final), -1 = none. Variant 'literal': lead 0, jamo_mode 'ctc' (each jamo "
    "from its own CTC start). Variant 'proposed': lead_s 0.05, jamo_mode 'proportional' (L first 25 % of the span, "
    "V to 60 % (to 100 % without a final), T the rest).")


if __name__ == "__main__":
    sys.exit(main())

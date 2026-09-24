"""Hangul jamo tokenizer (SPEC section 5).

Vocabulary (77 ids):
  0 <blank>  1 <pad>  2 <sos>  3 <eos>  4 <unk>  5 <space>
  6..24   초성 (19, Unicode L index order)
  25..45  중성 (21, Unicode V index order)
  46..72  종성 (27, Unicode T index 1..27)
  73..76  '.' '?' '!' ','
초성 and 종성 get distinct id ranges so a token sequence can be recomposed into syllables unambiguously.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

_HANGUL_BASE = 0xAC00
_N_L, _N_V, _N_T = 19, 21, 28  # T index 0 means "no final consonant"

# Compatibility jamo used for human-readable token names / orphan output.
_L_CHARS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_V_CHARS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_T_CHARS = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"  # T index 1..27
_PUNCT = ".?!,"

_SPECIAL = ["<blank>", "<pad>", "<sos>", "<eos>", "<unk>", "<space>"]

_FULLWIDTH_MAP = {
    "．": ".", "？": "?", "！": "!", "，": ",",
    "…": ".", "‘": "", "’": "", "“": "", "”": "",
}
_DROP_CHARS = "()/\"'[]{}<>~*_-"
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """NFC-normalise, map NBSP/newlines/tabs to spaces, drop brackets/slashes, collapse whitespace."""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\xa0", " ").replace("\n", " ").replace("\t", " ").replace("\r", " ")
    for k, v in _FULLWIDTH_MAP.items():
        s = s.replace(k, v)
    for ch in _DROP_CHARS:
        s = s.replace(ch, "")
    s = _WS_RE.sub(" ", s).strip()
    return s


def is_hangul_syllable(ch: str) -> bool:
    return "가" <= ch <= "힣"


def decompose_syllable(ch: str) -> tuple[int, int, int]:
    """Return (L, V, T) indices of a precomposed Hangul syllable (T == 0 when no final)."""
    code = ord(ch) - _HANGUL_BASE
    l_idx = code // (_N_V * _N_T)
    v_idx = (code % (_N_V * _N_T)) // _N_T
    t_idx = code % _N_T
    return l_idx, v_idx, t_idx


def compose_syllable(l_idx: int, v_idx: int, t_idx: int = 0) -> str:
    return chr(_HANGUL_BASE + (l_idx * _N_V + v_idx) * _N_T + t_idx)


class Tokenizer:
    """Fixed-vocabulary jamo tokenizer. No files needed."""

    def __init__(self) -> None:
        self.tokens: List[str] = list(_SPECIAL)
        self._l_base = len(self.tokens)
        self.tokens += [f"L:{c}" for c in _L_CHARS]
        self._v_base = len(self.tokens)
        self.tokens += [f"V:{c}" for c in _V_CHARS]
        self._t_base = len(self.tokens)
        self.tokens += [f"T:{c}" for c in _T_CHARS]
        self._p_base = len(self.tokens)
        self.tokens += list(_PUNCT)
        self._punct_ids = {c: self._p_base + i for i, c in enumerate(_PUNCT)}
        assert len(self.tokens) == 77, len(self.tokens)

    # --- ids -------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    blank_id = 0
    pad_id = 1
    sos_id = 2
    eos_id = 3
    unk_id = 4
    space_id = 5

    def id_to_token(self, i: int) -> str:
        return self.tokens[i]

    # --- encode ------------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for ch in normalize_text(text):
            if ch == " ":
                ids.append(self.space_id)
            elif is_hangul_syllable(ch):
                l_idx, v_idx, t_idx = decompose_syllable(ch)
                ids.append(self._l_base + l_idx)
                ids.append(self._v_base + v_idx)
                if t_idx:
                    ids.append(self._t_base + t_idx - 1)
            elif ch in self._punct_ids:
                ids.append(self._punct_ids[ch])
            else:
                ids.append(self.unk_id)
        return ids

    def has_unk(self, text: str) -> bool:
        return self.unk_id in self.encode(text)

    # --- decode ------------------------------------------------------------
    def _kind(self, i: int) -> tuple[str, int]:
        if self._l_base <= i < self._v_base:
            return "L", i - self._l_base
        if self._v_base <= i < self._t_base:
            return "V", i - self._v_base
        if self._t_base <= i < self._p_base:
            return "T", i - self._t_base + 1
        if self._p_base <= i < len(self.tokens):
            return "P", i - self._p_base
        return "S", i  # special

    def decode(self, ids: Iterable[int], strip_special: bool = True) -> str:
        out: List[str] = []
        pend_l: int | None = None
        pend_v: int | None = None

        def flush() -> None:
            nonlocal pend_l, pend_v
            if pend_l is not None and pend_v is not None:
                out.append(compose_syllable(pend_l, pend_v, 0))
            elif pend_l is not None:
                out.append(_L_CHARS[pend_l])
            elif pend_v is not None:
                out.append(_V_CHARS[pend_v])
            pend_l = pend_v = None

        for i in ids:
            i = int(i)
            kind, idx = self._kind(i)
            if kind == "L":
                flush()
                pend_l = idx
            elif kind == "V":
                if pend_l is not None and pend_v is None:
                    pend_v = idx
                else:
                    flush()
                    out.append(_V_CHARS[idx])
            elif kind == "T":
                if pend_l is not None and pend_v is not None:
                    out.append(compose_syllable(pend_l, pend_v, idx))
                    pend_l = pend_v = None
                else:
                    flush()
                    out.append(_T_CHARS[idx - 1])
            elif kind == "P":
                flush()
                out.append(_PUNCT[idx])
            else:  # special
                if i == self.space_id:
                    flush()
                    out.append(" ")
                elif i == self.unk_id:
                    flush()
                    out.append("X")
                elif i == self.eos_id and strip_special:
                    break
                else:
                    # blank / pad / sos: ignored
                    continue
        flush()
        return "".join(out)


__all__ = ["Tokenizer", "normalize_text", "is_hangul_syllable", "decompose_syllable", "compose_syllable"]

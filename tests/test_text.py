"""Tests for the jamo tokenizer. Run: $py -m tests.test_text (from project root)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avsr.text import Tokenizer, normalize_text  # noqa: E402


def main() -> None:
    tok = Tokenizer()
    assert tok.vocab_size == 77
    assert tok.blank_id == 0 and tok.pad_id == 1 and tok.sos_id == 2 and tok.eos_id == 3
    assert tok.unk_id == 4 and tok.space_id == 5

    # round trip on real-looking sentences
    sents = [
        # invented sentences (no dataset transcript is included in the repository)
        "어제 저녁에 친구와 함께 강변을 따라 걸으면서 앞으로 배우고 싶은 것들에 대해 오래 이야기했어요.",
        "저는 망설이지 않고 새로운 언어를 공부하겠다고 했고 특히 발음 연습은 매일 하겠다고 약속했었는데요.",
        "요즘은 아이들도 코딩을 일찍 배우기 시작한다고 해요?",
        "닭갈비, 밟다! 값",
    ]
    for s in sents:
        ids = tok.encode(s)
        assert all(0 <= i < tok.vocab_size for i in ids)
        assert tok.unk_id not in ids, s
        back = tok.decode(ids)
        assert back == normalize_text(s), (back, s)

    # normalisation
    assert normalize_text("안녕\xa0하세요\n(테스트)/ 1") == "안녕 하세요 테스트 1"
    assert tok.has_unk("이 부분은 X 처리") is True
    assert tok.has_unk("숫자 1 포함") is True
    assert tok.has_unk("정상 문장.") is False

    # decode robustness: orphan jamo, specials, eos stop
    l_k = tok.encode("가")[0]
    v_a = tok.encode("가")[1]
    t_n = tok.encode("간")[2]
    assert tok.decode([l_k]) == "ㄱ"
    assert tok.decode([v_a]) == "ㅏ"
    assert tok.decode([t_n]) == "ㄴ"
    assert tok.decode([l_k, v_a, t_n, tok.space_id, l_k, v_a]) == "간 가"
    assert tok.decode([tok.sos_id, l_k, v_a, tok.eos_id, l_k, v_a]) == "가"
    assert tok.decode([tok.blank_id, tok.pad_id, l_k, tok.blank_id, v_a, tok.blank_id]) == "가"
    assert tok.decode([l_k, v_a, v_a]) == "가ㅏ"  # second vowel is orphan
    assert tok.decode([tok.unk_id]) == "X"
    # ids are distinct for 초성 vs 종성 of the same consonant
    assert tok.encode("각")[0] != tok.encode("각")[2]
    print("test_text: OK (vocab", tok.vocab_size, ")")


if __name__ == "__main__":
    main()

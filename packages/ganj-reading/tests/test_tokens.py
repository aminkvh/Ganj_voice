# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import pytest

from ganj_reading import tokens as T


def test_special_ids_match_piper_convention():
    assert T.SYMBOL_TO_ID[T.PAD] == 0 and T.SYMBOL_TO_ID[T.BOS] == 1 and T.SYMBOL_TO_ID[T.EOS] == 2 and T.SYMBOL_TO_ID[T.SPACE] == 3
    assert len(T.SYMBOLS) <= 256 and len(set(T.SYMBOLS)) == len(T.SYMBOLS)


def test_ge2pe_sh_is_remapped_away_from_eos():
    assert T.remap_ge2pe("$ab") == "šab"
    assert "$" not in T.remap_ge2pe("ba$/d")


def test_encode_interleaves_pad_and_wraps_bos_eos():
    ids = T.encode("ab")
    a, b = T.SYMBOL_TO_ID["a"], T.SYMBOL_TO_ID["b"]
    assert ids == [1, 0, a, 0, b, 0, 2]


def test_style_tokens_are_single_codepoints_in_the_table():
    for tok in (T.VERSE, T.PROSE, T.HEMI, T.BEYT):
        assert len(tok) == 1 and tok in T.SYMBOL_TO_ID


def test_encode_rejects_unknown_symbol():
    with pytest.raises(ValueError):
        T.encode("a$b")  # raw GE2PE output must be remapped first
    with pytest.raises(ValueError):
        T.encode("")


def test_decode_roundtrip():
    s = T.VERSE + "h/m/late1 borunm/rzi" + T.HEMI + "b/d"
    assert T.decode(T.encode(s)) == s


def test_tokens_txt_format(tmp_path):
    p = tmp_path / "tokens.txt"
    T.write_tokens_txt(p)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "_ 0" and lines[1] == "^ 1" and lines[2] == "$ 2" and lines[3] == "  3"
    assert len(lines) == len(T.SYMBOLS)


def test_existing_ids_unchanged_by_metre_tokens():
    from ganj_reading import tokens as T
    old = [T.PAD, T.BOS, T.EOS, T.SPACE] + [chr(c) for c in range(0x21, 0x7F) if chr(c) not in (T.PAD, T.BOS, T.EOS)] + [T.SH] + T.STYLE
    assert T.SYMBOLS[:len(old)] == old
    assert all(0xE010 <= ord(s) < 0xE100 for s in T.SYMBOLS[len(old):])
    assert len(T.SYMBOLS) <= 256


def test_metre_token_is_stable_and_unknown_metres_get_none():
    from ganj_reading import tokens as T
    first = T.METRES[0]
    assert T.metre_token(first) == chr(0xE010)
    assert T.metre_token(first + " (نام بحر)") == T.metre_token(first)  # the family name in brackets does not matter
    assert T.metre_token("فعلن فعلن") == ""

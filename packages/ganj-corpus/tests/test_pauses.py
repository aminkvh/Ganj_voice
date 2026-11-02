# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import pytest

from ganj_corpus.pauses import (PAUSE_LONG, PAUSE_SHORT, mark_pauses, pause_token, punctuation_after_words,
                                     rule_tokens, strip_pauses)
from ganj_reading import tokens as T

S, L = PAUSE_SHORT, PAUSE_LONG
PREFIX = T.VERSE + ""


def test_a_pause_becomes_short_from_150_ms_and_long_from_450_ms():
    assert [pause_token(s) for s in (0.0, 0.149, 0.15, 0.449, 0.45, 2.0)] == ["", "", S, S, L, L]


def test_tokens_go_between_words_and_before_the_hemistich_mark():
    ph = PREFIX + "a b c" + T.HEMI + "d e"
    out = mark_pauses(ph, ["", S, L, ""])
    assert out == PREFIX + f"a b {S} c{L}" + T.HEMI + "d e"
    assert strip_pauses(out) == ph


def test_prose_without_a_prefix_or_hemistich_works_too():
    assert mark_pauses("goft sahebe1 š/r@", [S, ""]) == f"goft {S} sahebe1 š/r@"


def test_a_label_whose_words_do_not_match_the_boundaries_is_refused():
    with pytest.raises(ValueError):
        mark_pauses(PREFIX + "a b" + T.HEMI + "c", ["", ""," "])
    with pytest.raises(ValueError):
        mark_pauses("a b c", [S])


def test_the_pause_tokens_are_not_taken_for_style_or_metre_tokens():
    from ganj_corpus.relabel import PREFIX_CHARS
    assert S not in PREFIX_CHARS and L not in PREFIX_CHARS and S != L


def test_punctuation_is_found_after_the_word_it_follows_even_when_typed_apart():
    assert punctuation_after_words("گفت: صاحب شرع ، مرا خبر داده است. مگر نه؟") == [":", "", "،", "", "", "", ".", "", "؟"]


def test_reading_rules_pause_long_at_the_hemistich_break_and_at_sentence_ends_short_at_commas():
    lines = ["سال‌ها، نرد خدایی باختی", "این گره را زان گره نشناختی"]
    assert rule_tokens(lines) == [S, "", "", L, "", "", "", "", ""]
    assert rule_tokens(["گفت: صاحب شرع. مگر نه"]) == [S, "", L, ""]


def test_a_reading_label_falls_back_to_the_hemistich_pause_when_the_words_do_not_line_up():
    from ganj_corpus.pauses import rule_label
    ph = PREFIX + "s/lha n/rd" + T.HEMI + "@in gereh"
    assert rule_label(ph, ["سال‌ها، نرد", "این گره"]) == PREFIX + f"s/lha {S} n/rd{L}" + T.HEMI + "@in gereh"
    split = PREFIX + "yek to b/ravor/m" + T.HEMI + "@/j/b"     # یک‌تو written as two words by the G2P
    assert rule_label(split, ["یک‌تو، برآورم", "عجب"]) == PREFIX + "yek to b/ravor/m" + L + T.HEMI + "@/j/b"


def test_a_reading_label_takes_the_vav_rule_in_prose_and_the_rule_pauses_when_the_voice_has_them():
    from ganj_corpus.pauses import reading_label
    ph = T.PROSE + "bord/nd o s/@y k/rd/nd yeki"
    lines = ["بردند و سعی کردند: یکی"]
    assert reading_label(ph, lines, prose=True, pauses=False, va=True) == T.PROSE + "bord/nd v/ s/@y k/rd/nd yeki"
    assert reading_label(ph, lines, prose=True, pauses=True, va=True) == T.PROSE + f"bord/nd v/ s/@y k/rd/nd {S} yeki"
    assert reading_label(ph, lines, prose=True, pauses=False, va=False) == ph


def test_a_verse_reading_label_keeps_the_metres_o_and_pauses_long_between_hemistichs():
    from ganj_corpus.pauses import reading_label
    ph = PREFIX + "jan o del" + T.HEMI + "o g/r"
    lines = ["جان و دل", "و گر"]
    assert reading_label(ph, lines, prose=False, pauses=True, va=True) == PREFIX + f"jan o del{L}" + T.HEMI + "v/ g/r"

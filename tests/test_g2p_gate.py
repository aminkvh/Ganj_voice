# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from ganj_reading import tokens as T
from tools.g2p_gate import label_words, score


def test_label_words_drop_style_tokens_and_split_at_hemistich_breaks():
    assert label_words(T.VERSE + "dele1 m/n" + T.HEMI + "gol") == ["dele1", "m/n", "gol"]


def test_score_counts_fixed_flags_changed_unflagged_words_and_unaligned_rows():
    rows = [{"id": "a", "lines": ["خدمت سلطان کردی"], "phonemes": "xedm/te1 soltane1 k/rdi"},
            {"id": "b", "lines": ["دل من"], "phonemes": "dele1 m/n"}]
    errors = [{"id": "a", "word": "سلطان", "got": "soltane1", "expected": "soltan", "type": "ezafe_spurious"},
              {"id": "a", "word": "کردی", "got": "k/rdi", "expected": "k/rdi", "type": "non_persian_line"}]
    new = {"a": "xedm/t soltan k/rdi", "b": "dele1"}
    res = score(rows, new, errors, lambda line: line.split())
    assert (res["flagged_checkable"], res["flagged_fixed"]) == (1, 1)
    assert (res["unflagged_words"], res["unflagged_changed"]) == (2, 1)     # خدمت changed, کردی (skipped type) not
    assert res["changed"] == [["a", "خدمت", "xedm/te1", "xedm/t"]]
    assert res["unaligned_rows"] == 1


def test_flagged_words_are_found_by_text_even_when_the_old_label_does_not_align():
    rows = [{"id": "a", "lines": ["دیوانه ای احمق"], "phonemes": "divanei @/hm/q"}]   # old label merged two words
    errors = [{"id": "a", "word": "احمق", "got": "@ahm/q", "expected": "@/hm/q", "type": "short_vowel"}]
    res = score(rows, {"a": "divane @i @/hm/q"}, errors, lambda line: line.split())
    assert (res["flagged_checkable"], res["flagged_fixed"]) == (1, 1)


def test_diff_lists_every_word_two_label_sets_disagree_on():
    from tools.g2p_gate import diff
    rows = [{"id": "a", "lines": ["روز پسین"]}, {"id": "b", "lines": ["دل"]}]
    assert diff(rows, {"a": "ruzeye1 p/sin", "b": "del"}, {"a": "ruze1 p/sin", "b": "del"}, lambda l: l.split()) == \
        [["a", "روز", "ruzeye1", "ruze1"]]

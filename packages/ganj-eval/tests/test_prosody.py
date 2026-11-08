# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np
import pytest

from ganj_eval.prosody import compare_pauses, line_words, silence_in_gap, silent_run_in_gap, stretch_corr

VOCAB = {ch: i for i, ch in enumerate("|سالهنردخدایبتیگزش")}


def test_words_drop_half_spaces_and_punctuation_and_remember_their_line():
    ws = line_words(["سال‌ها نرد،", "خدایی!"], VOCAB)
    assert [(w.line, w.text, w.chars) for w in ws] == [(0, "سال‌ها", "سالها"), (0, "نرد،", "نرد"), (1, "خدایی!", "خدایی")]


def test_only_the_longest_silent_run_inside_a_gap_counts_as_its_pause():
    silent = np.zeros(200, dtype=bool)
    silent[50:70] = True     # 0.20 s pause in the gap
    silent[75:78] = True     # a stop closure, not a pause
    assert silence_in_gap(silent, 0.45, 0.85, frame_s=0.01) == pytest.approx(0.20)
    assert silence_in_gap(silent, 1.0, 1.2, frame_s=0.01) == 0.0          # words glued: no silence between them


def test_the_longest_silent_run_in_a_gap_is_found_where_it_lies():
    silent = np.zeros(200, dtype=bool)
    silent[50:70] = True
    silent[75:78] = True
    assert silent_run_in_gap(silent, 0.45, 0.85, frame_s=0.01) == pytest.approx((0.50, 0.70))
    assert silent_run_in_gap(silent, 1.0, 1.2, frame_s=0.01) is None


def test_pauses_are_compared_at_each_word_boundary():
    ref = [0.30, 0.0, 0.50, 0.0, 0.05]
    hyp = [0.20, 0.40, 0.0, 0.0, 0.10]
    r = compare_pauses(ref, hyp, min_pause=0.15)
    assert r["hits"] == [0] and r["extra"] == [1] and r["missing"] == [2]
    assert r["precision"] == pytest.approx(0.5) and r["recall"] == pytest.approx(0.5)


def test_no_pauses_on_either_side_is_full_agreement_not_a_division_by_zero():
    r = compare_pauses([0.0, 0.05], [0.0, 0.0], min_pause=0.15)
    assert r["precision"] == 1.0 and r["recall"] == 1.0 and r["f1"] == 1.0


def test_stretch_correlation_compares_relative_word_lengths_not_the_overall_pace():
    ref = [0.2, 0.4, 0.3, 0.8]
    assert stretch_corr(ref, [2 * d for d in ref]) == pytest.approx(1.0)
    assert stretch_corr(ref, ref[::-1]) < 0

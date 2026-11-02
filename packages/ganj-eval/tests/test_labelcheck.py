# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import pytest

from ganj_eval.labelcheck import boundary_window, ezafe_errors, ezafe_targets, pick_draw, words_present

E2, S = "", ""


def test_ezafe_targets_mark_consonant_final_words_by_their_label():
    # شرحِ داغِ دلِ پروانه چو گفتم: three ezafes after a consonant; پروانه ends in a vowel letter (left out); چو too
    lines = ["شرحِ داغِ دلِ پروانه چو گفتم", "آتشی در دلش"]
    label = "š/rhe1 daqe1 dele1 p/rvane co goft/m" + E2 + "@at/ši d/r del/š"
    assert ezafe_targets(lines, label) == [(0, True), (1, True), (2, True), (7, False)]


def test_ezafe_targets_skip_the_last_word_of_each_hemistich_and_ignore_pause_tokens():
    lines = ["کار تو", "فکر تو بر"]
    label = "kare1 " + S + " to" + E2 + "fekre1 to b/r"
    assert ezafe_targets(lines, label) == [(0, True), (2, True)]


def test_ezafe_targets_are_empty_when_the_label_does_not_line_up():
    assert ezafe_targets(["شرح داغ"], "š/rhe1") == []


def test_a_word_before_vav_is_no_ezafe_target_the_vowel_there_is_the_conjunctions():
    # «بردند و سعی»: nothing takes an ezafe before «و»; the o said there (glued: bord/ndo) is not an ezafe vowel
    assert ezafe_targets(["احمق و از خلق"], "@/hm/q o @/z xolqe1") == [(2, False)]


def test_boundary_window_spans_the_end_of_one_word_and_the_start_of_the_next():
    spans = [(0.0, 0.50), (0.60, 1.0), (1.0, 1.4)]
    assert boundary_window(spans, 0) == pytest.approx((0.42, 0.63))
    assert boundary_window([(0.0, 0.05), (0.05, 0.4)], 0) == pytest.approx((0.0, 0.08))   # never before the word


def test_words_present_compares_the_text_with_and_without_each_word():
    said = {"b/r", "p/hlu", "šod/n"}        # the audio says everything but "foru"

    def loglik(words):                      # worse for each said word left out and each unsaid word put in
        return -10.0 * len(said - set(words)) - 10.0 * len(set(words) - said)

    out = words_present(loglik, ["b/r", "p/hlu", "foru", "šod/n"])
    assert out == [True, True, False, True]


def test_pick_draw_prefers_every_word_then_every_ezafe_then_the_best_fit():
    draws = {"d1": {"missing_words": 1, "wrong_ezafe": 0, "fit": -50.0},
             "d2": {"missing_words": 0, "wrong_ezafe": 2, "fit": -40.0},
             "d3": {"missing_words": 0, "wrong_ezafe": 1, "fit": -90.0},
             "d4": {"missing_words": 0, "wrong_ezafe": 1, "fit": -60.0}}
    assert pick_draw(draws) == "d4"


def test_ezafe_errors_count_unheard_and_unwanted_ezafes():
    targets = [(0, True), (1, True), (3, False), (5, False)]
    assert ezafe_errors(targets, [0.9, 0.2, 0.7, 0.1]) == (1, 1)


def test_pick_draw_counts_a_wrong_short_vowel_like_a_wrong_ezafe():
    # d8 had the best CTC fit but said «آبش» abesh; d1 said ab/sh
    draws = {"d1": {"missing_words": 0, "wrong_ezafe": 0, "wrong_vowel": 0, "fit": -97.8},
             "d8": {"missing_words": 0, "wrong_ezafe": 0, "wrong_vowel": 1, "fit": -78.4},
             "d4": {"missing_words": 1, "wrong_ezafe": 0, "wrong_vowel": 0, "fit": -60.0}}
    assert pick_draw(draws) == "d1"
    assert pick_draw({"a": {"missing_words": 0, "wrong_ezafe": 1, "fit": -90.0},      # no vowel model: as before
                      "b": {"missing_words": 0, "wrong_ezafe": 0, "fit": -95.0}}) == "b"


def test_vowel_errors_are_the_measured_vowels_the_model_calls_wrong():
    from ganj_eval.labelcheck import vowel_errors
    from ganj_eval.vowelcheck import VowelModel
    m = VowelModel({"/": [6.55, 7.32], "e": [6.12, 7.61], "o": [6.18, 6.87]}, [0.08, 0.08])
    measured = [{"word": 4, "vowel": "/", "f1": 405.0, "f2": 2040.0}, {"word": 2, "vowel": "e", "f1": 455.0, "f2": 2010.0}]
    assert vowel_errors(m, measured) == [measured[0]]
    assert vowel_errors(None, measured) == []

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import pytest

from ganj_corpus.pauses import PAUSE_SHORT
from ganj_eval.vowelcheck import VowelModel, line_vowel_targets, vowel_targets, vowel_window
from ganj_reading.tokens import HEMI


def test_a_short_vowel_between_two_written_consonants_is_a_target():
    # آبش: ab/sh (a voice said abesh); آ is a vowel letter, so only ب-ش is checked
    assert vowel_targets("آبش", "@ab/š") == [(1, 2, "/")]
    assert vowel_targets("رجا", "r/ja") == [(0, 1, "/")]


def test_a_cluster_or_a_vowel_letter_between_the_consonants_is_no_target():
    # نخورد: ن-خ fatha; خ-و-ر has a vowel letter between; ر-د is a cluster (no vowel)
    assert vowel_targets("نخورد", "n/xord") == [(0, 1, "/")]
    # شرحِ: ش-ر fatha; the ezafe after ح belongs to the ezafe check
    assert vowel_targets("شرحِ", "š/rhe1") == [(0, 1, "/")]


def test_kasre_and_zamma_are_targets_and_tashdid_counts_as_one_letter():
    assert vowel_targets("دلش", "del/š") == [(0, 1, "e"), (1, 2, "/")]
    assert vowel_targets("گفتم", "goft/m") == [(0, 1, "o"), (2, 3, "/")]
    assert vowel_targets("حقِّ", "h/qqe1") == [(0, 1, "/")]


def test_a_fatha_right_before_kaf_or_gaf_is_not_judged():
    # ک/گ pull the fatha before them toward e in every source (F2 +200 Hz in the human recordings too); مگر: 90 % m/g/r
    # by ear, both draws flagged. The fatha after گ is judged as usual.
    assert vowel_targets("مگر", "m/g/r") == [(1, 2, "/")]
    assert vowel_targets("نکرد", "n/k/rd") == [(1, 2, "/")]
    assert vowel_targets("دلک", "del/k") == [(0, 1, "e")]


def test_letters_that_map_to_one_sound_all_match_it():
    assert vowel_targets("صبر", "s/br") == [(0, 1, "/")]       # ص s
    assert vowel_targets("طلب", "t/l/b") == [(0, 1, "/"), (1, 2, "/")]
    assert vowel_targets("مژده", "mo;de") == [(0, 1, "o")]      # the labels write ژ as ;


def test_a_word_whose_label_does_not_follow_its_letters_has_no_targets():
    assert vowel_targets("دلش", "kar") == []
    assert vowel_targets("", "") == []


def test_line_targets_follow_the_hemistichs_and_skip_pause_tokens():
    lines = ["شرحِ داغِ دلِ", "آتشی در دلش"]
    label = "š/rhe1 " + PAUSE_SHORT + " daqe1 dele1" + HEMI + "@at/ši d/r del/š"
    got = line_vowel_targets(lines, label)
    assert got == [(0, 0, 1, "/"), (2, 0, 1, "e"), (3, 1, 2, "/"), (4, 0, 1, "/"), (5, 0, 1, "e"), (5, 1, 2, "/")]
    assert line_vowel_targets(["شرح داغ"], "š/rhe1") == []      # label and text do not line up


def test_vowel_window_runs_from_just_after_one_consonant_to_the_next():
    chars = [(1.00, 1.02), (1.10, 1.12), (1.30, 1.34)]
    assert vowel_window(chars, 1, 2) == pytest.approx((1.12, 1.30))
    assert vowel_window([(1.0, 1.02), (1.01, 1.03)], 0, 1) is None   # too short to measure


def _model():
    #            F1   F2 (Hz): a fatha, a kasre, a zamma, each a small cloud
    pts = {"/": [(700, 1500), (720, 1550), (680, 1450), (710, 1520)],
           "e": [(450, 2000), (470, 2050), (430, 1950), (460, 2020)],
           "o": [(480, 950), (500, 1000), (460, 900), (490, 980)]}
    return VowelModel.fit([(v, f1, f2) for v, xs in pts.items() for f1, f2 in xs])


def test_a_vowel_is_wrong_only_when_another_vowel_fits_it_clearly_better():
    m = _model()
    assert m.wrong("/", 405, 2040)            # آبش said abesh: F1 405, F2 2040
    assert not m.wrong("/", 630, 1780)        # said ab/sh
    assert not m.wrong("e", 580, 1750)        # between a and e: not clear, not flagged
    assert m.wrong("e", 700, 1500)


def test_a_vowel_model_round_trips_through_json(tmp_path):
    m = _model()
    p = tmp_path / "vowels.json"
    m.save(p)
    assert VowelModel.load(p).wrong("/", 405, 2040) and not VowelModel.load(p).wrong("/", 700, 1500)

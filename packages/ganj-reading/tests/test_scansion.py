# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from ganj_reading.scansion import metre_patterns, scans, syllables

MUTAQARIB = "فعولن فعولن فعولن فعل (متقارب مثمن محذوف یا وزن شاهنامه)"
RAMAL = "فاعلاتن فاعلاتن فاعلن (رمل مسدس محذوف یا وزن مثنوی)"


def test_metre_name_to_weight_patterns():
    assert "U--U--U--U-" in metre_patterns(MUTAQARIB)
    assert "-U---U---U-" in metre_patterns(RAMAL)


def test_syllable_weights_closed_open_long_overlong():
    # t/ va na: short open, long open, long open; "dast" = short + two codas = overlong
    assert [s.weight for s in syllables("t/vana")] == ["U", "-", "-"]
    assert [s.weight for s in syllables("d/st")] == ["-U"]


def test_ferdowsi_line_scans():
    assert scans("t/vana bov/d h/r ke dana bov/d", MUTAQARIB)


def test_masnavi_line_scans_with_hamza_elision_and_silent_nun():
    assert scans("bešnov @in ney con šekay/t mikon/d", RAMAL)


def test_a_wrong_short_vowel_breaks_the_metre():
    assert not scans("t/vana bov/d h/r ke d/na bov/d", MUTAQARIB)  # «دانا» read with a short vowel


def test_ezafe_marker_is_not_a_syllable():
    # «دلِ» is de-le: GE2PE writes the ezafe vowel e followed by the marker 1
    assert [s.weight for s in syllables("dele1")] == ["U", "U"]


def test_ramal_first_foot_licence_works_both_ways():
    assert "-U--UU--UU--UU-" in metre_patterns("فعلاتن فعلاتن فعلاتن فعلن")
    assert "UU---U---U-" in metre_patterns(RAMAL)


def test_long_vowel_before_a_vowel_may_be_short():
    # «میان» mi-yan: the i before the glide-y + vowel may count short
    assert "U" in syllables("miyan")[0].options
    assert syllables("mina")[0].options == frozenset({"-"})  # no hiatus: stays long

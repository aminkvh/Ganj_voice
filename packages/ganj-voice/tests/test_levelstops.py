# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np
import pytest

from ganj_voice.levelstops import level_stops, trim_by_levels

SR = 24000


def line(*parts):
    """Audio from (kind, seconds) parts: 'tone' is speech, 'gap' a near-silence."""
    rng = np.random.default_rng(0)
    out = []
    for kind, s in parts:
        n = int(round(s * SR))
        out.append(0.3 * np.sin(2 * np.pi * 220 * np.arange(n) / SR) if kind == "tone" else 1e-4 * rng.normal(size=n))
    return np.concatenate(out).astype(np.float32)


VERSE = [("gap", 0.4), ("tone", 0.5), ("gap", 0.6), ("tone", 0.5), ("gap", 0.9), ("tone", 0.5), ("gap", 0.2),
         ("tone", 0.4), ("gap", 0.3)]


def test_the_stops_are_the_silences_inside_the_line_not_before_or_after_it():
    stops = level_stops(line(*VERSE), SR)
    assert [round(e - s, 1) for s, e in stops] == [0.6, 0.9, 0.2]
    assert stops[0][0] == pytest.approx(0.9, abs=0.02)


def test_a_verse_keeps_its_longest_stop_as_the_hemistich_break_and_the_others_are_cut_to_the_cap():
    y = line(*VERSE)
    out, cut = trim_by_levels(y, SR, n_breaks=1, cap=0.30)
    assert [round(e - s, 1) for s, e in cut] == [0.6]
    assert len(out) / SR == pytest.approx(len(y) / SR - 0.3, abs=0.02)


def test_prose_has_no_break_so_every_long_stop_is_cut():
    y = line(*VERSE)
    out, cut = trim_by_levels(y, SR, n_breaks=0, cap=0.30)
    assert [round(e - s, 1) for s, e in cut] == [0.6, 0.9]
    assert len(out) / SR == pytest.approx(len(y) / SR - 0.3 - 0.6, abs=0.03)


def test_a_line_without_long_stops_is_left_as_it_is():
    y = line(("gap", 0.2), ("tone", 0.5), ("gap", 0.25), ("tone", 0.5))
    out, cut = trim_by_levels(y, SR, n_breaks=0)
    assert cut == [] and np.array_equal(out, y)


def test_lengths_are_compared_in_whole_frames_so_a_stop_of_exactly_the_cap_is_kept():
    # frames 3-33: in float seconds 0.33 - 0.03 = 0.30000000000000004, which is over 0.30
    y = line(("tone", 0.03), ("gap", 0.30), ("tone", 0.5), ("gap", 0.31), ("tone", 0.5), ("gap", 0.2))
    out, cut = trim_by_levels(y, SR, n_breaks=0, cap=0.30)
    assert [round(e - s, 2) for s, e in cut] == [0.31]


def test_of_two_equally_long_stops_the_earlier_is_the_break():
    y = line(("gap", 0.2), ("tone", 0.5), ("gap", 0.6), ("tone", 0.5), ("gap", 0.6), ("tone", 0.5), ("gap", 0.2))
    _, cut = trim_by_levels(y, SR, n_breaks=1, cap=0.30)
    assert len(cut) == 1 and cut[0][0] > 1.5

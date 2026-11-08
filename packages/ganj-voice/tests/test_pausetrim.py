# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np
import pytest

from ganj_voice.pausetrim import shorten_runs, silent_frames, stops_to_trim

SR = 1000


def test_a_frame_is_silent_well_below_the_speech_level_and_near_the_floor():
    db = np.array([-60.0] * 10 + [-10.0] * 80 + [-58.0] * 10)
    s = silent_frames(db)
    assert s[:10].all() and not s[10:90].any() and s[90:].all()


def _tone(s, f=50.0):
    return 0.5 * np.sin(2 * np.pi * f * np.arange(int(s * SR)) / SR)


def test_shorten_runs_cuts_each_run_to_the_cap_from_its_middle():
    y = np.concatenate([_tone(0.4), np.zeros(int(0.6 * SR)), _tone(0.4)])
    out = shorten_runs(y, SR, [(0.4, 1.0)], cap=0.3, fade=0.01)
    assert len(out) == pytest.approx(len(y) - 0.3 * SR, abs=2)
    assert np.allclose(out[:390], y[:390]) and np.allclose(out[-390:], y[-390:])    # the words are untouched
    assert np.abs(out[420:680]).max() == 0.0                                         # still silent in between


def test_shorten_runs_leaves_a_run_within_the_cap_alone():
    y = np.concatenate([_tone(0.4), np.zeros(int(0.2 * SR)), _tone(0.4)])
    assert np.array_equal(shorten_runs(y, SR, [(0.4, 0.6)], cap=0.3, fade=0.01), y)


def test_shorten_runs_handles_several_runs_in_order():
    y = np.concatenate([_tone(0.2), np.zeros(500), _tone(0.2), np.zeros(700), _tone(0.2)])
    out = shorten_runs(y, SR, [(0.2, 0.7), (0.9, 1.6)], cap=0.25, fade=0.01)
    assert len(out) == pytest.approx(len(y) - 250 - 450, abs=3)


def test_stops_to_trim_keeps_every_hemistich_break():
    runs = [(1.0, 1.5), (2.0, 2.8), (3.0, 3.1)]          # longest silence at each of three word boundaries
    assert stops_to_trim(runs, breaks={1}, cap=0.3) == [(1.0, 1.5)]

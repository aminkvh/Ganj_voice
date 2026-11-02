# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np
import pytest

from collections import namedtuple

from ganj_eval.align import SR, boundary_pauses, frame_db, group_chars


def _tone(s):
    t = np.arange(int(s * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_frame_levels_are_10_ms_frames_in_db():
    db = frame_db(np.concatenate([_tone(0.1), np.zeros(int(0.1 * SR), np.float32)]))
    assert len(db) == 20 and db[:10].min() > -15 and db[10:].max() < -100


def test_the_silence_between_two_words_is_their_pause_and_glued_words_have_none():
    rng = np.random.default_rng(0)
    y = np.concatenate([_tone(0.3), np.zeros(int(0.3 * SR), np.float32), _tone(0.3), _tone(0.3)])
    y = y + rng.normal(0, 1e-4, len(y)).astype(np.float32)          # a recording floor, not digital silence
    p = boundary_pauses([(0.0, 0.28), (0.62, 0.9), (0.9, 1.2)], y)  # CTC ends a word a little early
    assert p[0] == pytest.approx(0.3, abs=0.02) and p[1] == 0.0


def test_ctc_tokens_group_into_per_letter_spans_of_each_word():
    # the vowel check (ganj_eval.vowelcheck) needs each letter's span, not only the word's
    Tok = namedtuple("Tok", "token start end")
    toks = [Tok(5, 10, 11), Tok(6, 14, 15), Tok(0, 20, 21), Tok(7, 30, 32)]   # a b | c, 20 ms frames
    assert group_chars(toks, sep=0, frame_s=0.02) == [[(0.2, 0.22), (0.28, 0.3)], [(0.6, 0.64)]]

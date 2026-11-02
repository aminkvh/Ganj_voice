# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np
from scipy.signal import butter, sosfilt

from ganj_voice.deess import deess

SR = 24000


def _band_db(y, lo, hi):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return 10 * np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-20)


def _vowel_hiss_vowel():
    # a vowel (harmonics of 150 Hz up to 1.5 kHz), a hiss (white noise above 3 kHz), the vowel again: 0.4 s each
    rng = np.random.default_rng(0)
    t = np.arange(int(0.4 * SR)) / SR
    vowel = sum(0.05 * np.sin(2 * np.pi * 150 * k * t) for k in range(1, 11))
    hiss = 0.08 * sosfilt(butter(8, 3000, "high", fs=SR, output="sos"), rng.normal(0, 1, len(t)))
    return np.concatenate([vowel, hiss, vowel]).astype(np.float32), len(t)


def test_no_depth_leaves_the_audio_as_it_is():
    y, _ = _vowel_hiss_vowel()
    assert np.array_equal(deess(y, SR, 0.0), y)


def test_the_hiss_loses_its_top_band_by_the_depth_and_keeps_its_low_band():
    # measured: the voices' س/ش sit 5-11 dB over the human recordings in 5-9 kHz, level with them in 2-5 kHz
    y, n = _vowel_hiss_vowel()
    z = deess(y, SR, 8.0)
    assert len(z) == len(y)
    mid = slice(n + int(0.05 * SR), 2 * n - int(0.05 * SR))
    assert abs((_band_db(z[mid], 5500, 9000) - _band_db(y[mid], 5500, 9000)) + 8.0) < 1.0
    assert abs(_band_db(z[mid], 3000, 4400) - _band_db(y[mid], 3000, 4400)) < 1.0


def test_a_vowel_is_untouched_and_nothing_gets_louder():
    y, n = _vowel_hiss_vowel()
    z = deess(y, SR, 8.0)
    first = slice(int(0.02 * SR), n - int(0.05 * SR))
    assert abs(_band_db(z[first], 100, 1600) - _band_db(y[first], 100, 1600)) < 0.1
    for lo, hi in ((100, 2000), (2000, 5000), (5000, 9000), (9000, 12000)):
        assert _band_db(z, lo, hi) <= _band_db(y, lo, hi) + 0.05


def _noise(lo, hi, n, seed):
    rng = np.random.default_rng(seed)
    return sosfilt(butter(8, (lo, hi), "bandpass", fs=SR, output="sos"), rng.normal(0, 1, n))


def test_a_shin_like_hiss_is_cut_less_and_only_where_the_voices_are_too_loud_and_a_sin_like_hiss_all_the_way_up():
    # measured: ش is too loud in 5-7 kHz and human-level above; س is too loud in 5-9 kHz. A cut of 5 dB
    # put ش at the human level and left س 3-7 dB over it, 9 dB did the reverse; so ش gets SHIN_DEPTH of the depth
    from ganj_voice.deess import SHIN_DEPTH
    n = int(0.5 * SR)
    shin = (0.08 * (_noise(2500, 6500, n, 2) + 0.3 * _noise(6500, 10000, n, 3))).astype(np.float32)
    sin = (0.08 * _noise(4000, 11000, n, 4)).astype(np.float32)
    mid = slice(int(0.05 * SR), n - int(0.05 * SR))
    zs, zn = deess(shin, SR, 9.0), deess(sin, SR, 9.0)
    assert abs((_band_db(zs[mid], 5500, 6500) - _band_db(shin[mid], 5500, 6500)) + 9.0 * SHIN_DEPTH) < 1.0
    assert abs(_band_db(zs[mid], 8000, 9500) - _band_db(shin[mid], 8000, 9500)) < 0.5
    assert abs((_band_db(zn[mid], 7500, 9000) - _band_db(sin[mid], 7500, 9000)) + 9.0) < 1.0

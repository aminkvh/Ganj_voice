# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np

from ganj_voice.melwarp import mel_centres, warp_logmel

C = mel_centres(24000, 100)


def _peak(hz, frames=4):
    """A log-mel with one peak (at the bin nearest hz) in every frame."""
    lm = np.full((100, frames), -8.0)
    lm[np.argmin(abs(C - hz))] = 0.0
    return lm


def _peak_hz(lm):
    return C[np.argmax(lm[:, 0])]


def test_mel_centres_rise_to_below_nyquist():
    assert len(C) == 100 and np.all(np.diff(C) > 0) and 0 < C[0] < 150 and 11000 < C[-1] < 12000


def test_alpha_one_returns_the_input():
    lm = np.random.default_rng(0).normal(size=(100, 7))
    assert np.array_equal(warp_logmel(lm, C, 1.0), lm)


def test_the_warp_moves_a_formant_up_by_alpha():
    # the disguise in the mel domain (WORLD's resynthesis added ≈ 5 cents of held-vowel wobble)
    out = warp_logmel(_peak(1000), C, 1.10)
    assert abs(_peak_hz(out) / 1000 - 1.10) < 0.04


def test_the_warp_can_fade_out_above_a_band_and_give_fricatives_their_own_factor():
    # as world_shift: the hiss band does not ride the shift; a frame whose energy lives above 2.5 kHz
    # (ش, س) is moved by its own factor (د۴: 0.80)
    high = warp_logmel(_peak(6000), C, 1.10, fade_hz=(3000, 4000))
    assert abs(_peak_hz(high) - 6000) / 6000 < 0.04
    low = warp_logmel(_peak(1000), C, 1.10, fade_hz=(3000, 4000))
    assert abs(_peak_hz(low) / 1000 - 1.10) < 0.04
    fric = warp_logmel(_peak(5000), C, 1.10, fade_hz=(3000, 4000), fricative_alpha=0.80)
    assert abs(_peak_hz(fric) / 5000 - 0.80) < 0.04
    vowel = warp_logmel(_peak(1000), C, 1.10, fade_hz=(3000, 4000), fricative_alpha=0.80)
    assert abs(_peak_hz(vowel) / 1000 - 1.10) < 0.04

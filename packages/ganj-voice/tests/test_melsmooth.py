# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import torch

from ganj_voice.melsmooth import despike_mel


def flat(c=100, t=20, v=-5.0):
    return torch.full((1, c, t), v)


def test_single_bin_spike_in_the_high_band_is_removed():
    mel = flat()
    mel[0, 87, :] += 4.0          # a stuck-high bin: what Vocos renders as a steady 8 kHz tone
    out = despike_mel(mel, start_bin=60)
    assert torch.allclose(out[0, 87], torch.full((20,), -5.0))


def test_bins_below_the_start_are_untouched():
    mel = flat()
    mel[0, 30, :] += 4.0          # a harmonic in the low band must survive
    out = despike_mel(mel, start_bin=60)
    assert torch.equal(out[0, :60], mel[0, :60])


def test_a_smooth_spectral_slope_passes_unchanged():
    slope = torch.linspace(0, -10, 100)[None, :, None].repeat(1, 1, 20)
    assert torch.allclose(despike_mel(slope, start_bin=60), slope)


def test_time_option_removes_a_single_frame_speck():
    mel = flat()
    mel[0, 75, 10] += 4.0
    mel[0, 76, 10] += 4.0         # two adjacent bins: a frequency median alone keeps it
    out = despike_mel(mel, start_bin=60, time=True)
    assert torch.allclose(out[0, 75:77, 10], torch.full((2,), -5.0))


def test_input_is_not_modified():
    mel = flat()
    mel[0, 87, :] += 4.0
    before = mel.clone()
    despike_mel(mel, start_bin=60, time=True)
    assert torch.equal(mel, before)


def test_time_smoothing_spreads_a_one_frame_wobble_in_the_low_band_and_leaves_the_high_band():
    # A slight shake in the long vowels: the model's frame-to-frame jitter in the harmonic band adds
    # ~5 cents of pitch wobble over human readings; a short binomial average over frames takes it out.
    from ganj_voice.melsmooth import smooth_time
    mel = flat()
    mel[0, 10, 10] += 1.6
    mel[0, 80, 10] += 1.6
    before = mel.clone()
    out = smooth_time(mel, frames=5, hi_bin=60)
    assert torch.allclose(out[0, 10, 8:13], torch.tensor([0.1, 0.4, 0.6, 0.4, 0.1]) - 5.0)
    assert torch.equal(out[0, 60:], mel[0, 60:]) and torch.equal(mel, before)


def test_time_smoothing_keeps_a_steady_vowel_and_the_length():
    from ganj_voice.melsmooth import smooth_time
    slope = torch.linspace(0, -10, 100)[None, :, None].repeat(1, 1, 7)
    out = smooth_time(slope, frames=3, hi_bin=60)
    assert out.shape == slope.shape and torch.allclose(out, slope)

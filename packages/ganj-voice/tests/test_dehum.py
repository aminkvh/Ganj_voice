# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np

from ganj_voice.dehum import notch_mains


def _level_db(y, sr, f):
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    k = int(round(f * len(y) / sr))
    return 20 * np.log10(spec[k - 1:k + 2].max() + 1e-12)


def test_notch_removes_the_mains_hum_and_its_harmonics_but_keeps_a_voice_partial():
    sr, secs = 24000, 4.0
    t = np.arange(int(sr * secs)) / sr
    hum = 0.05 * np.sin(2 * np.pi * 50 * t) + 0.02 * np.sin(2 * np.pi * 100 * t) + 0.01 * np.sin(2 * np.pi * 250 * t)
    voice = 0.1 * np.sin(2 * np.pi * 137 * t)  # a voiced partial between two hum harmonics
    y = (hum + voice).astype(np.float32)
    z = notch_mains(y, sr, base_hz=50.0)
    assert z.shape == y.shape and z.dtype == np.float32
    for f in (50, 100, 250):
        assert _level_db(y, sr, f) - _level_db(z, sr, f) > 30
    assert abs(_level_db(y, sr, 137) - _level_db(z, sr, 137)) < 1.0


def test_mains_frequency_is_estimated_from_the_clip_and_an_off_nominal_hum_is_removed():
    from ganj_voice.dehum import estimate_mains
    sr, secs = 22050, 6.0
    t = np.arange(int(sr * secs)) / sr
    hum = 0.05 * np.sin(2 * np.pi * 49.75 * t) + 0.01 * np.sin(2 * np.pi * 99.5 * t)
    y = (hum + 0.1 * np.sin(2 * np.pi * 137 * t)).astype(np.float32)
    base = estimate_mains(y, sr, nominal_hz=50.0)
    assert abs(base - 49.75) < 0.1
    z = notch_mains(y, sr, base_hz=base)
    assert _level_db(y, sr, 49.75) - _level_db(z, sr, 49.75) > 30


def _speechlike_with_noise(sr=22050, secs=6.0, seed=0):
    """Voiced bursts (harmonics of 150 Hz up to 5 kHz) with gaps, plus steady hiss and a steady 117 Hz tone."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * secs)) / sr
    env = ((t % 1.5) < 1.0).astype(np.float64)  # 1 s voiced, 0.5 s gap
    voice = env * sum(0.1 / k * np.sin(2 * np.pi * 150 * k * t) for k in range(1, 34))
    noise = 0.003 * rng.standard_normal(len(t)) + 0.004 * np.sin(2 * np.pi * 117 * t)
    return t, env, voice.astype(np.float32), noise.astype(np.float32)


def test_denoise_lowers_steady_hiss_and_tones_in_the_gaps_and_keeps_the_voice():
    from ganj_voice.dehum import denoise_stationary
    sr = 22050
    t, env, voice, noise = _speechlike_with_noise(sr)
    z = denoise_stationary(voice + noise, sr, floor_db=-20.0)
    assert z.shape == voice.shape and z.dtype == np.float32
    gap = (env == 0) & (t > 0.2) & (t < 5.8)
    gap = gap & np.roll(gap, 2000) & np.roll(gap, -2000)  # away from burst edges
    before = 10 * np.log10(np.mean(noise[gap] ** 2))
    after = 10 * np.log10(np.mean(z[gap] ** 2))
    assert 12.0 < before - after <= 21.0      # down by most of the floor, never past it
    on = env == 1
    kept = 10 * np.log10(np.mean(z[on] ** 2) / np.mean(voice[on] ** 2))
    assert abs(kept) < 1.0                    # the voice keeps its level
    assert _level_db(z[gap], sr, 117) < _level_db((voice + noise)[gap], sr, 117) - 12


def test_highpass_removes_the_vocoders_low_hum_lines_and_keeps_a_low_male_pitch():
    """ZipVoice outputs of a humming prompt carry the hum as two lines near 48 and 56 Hz (Vocos's lowest mel bins);
    the voices' pitch rarely goes below 80 Hz."""
    from ganj_voice.dehum import highpass
    sr, secs = 24000, 4.0
    t = np.arange(int(sr * secs)) / sr
    y = sum(a * np.sin(2 * np.pi * f * t) for a, f in ((0.03, 48.3), (0.02, 55.7), (0.1, 85.0), (0.1, 110.0)))
    y = y.astype(np.float32)
    z = highpass(y, sr)
    assert z.shape == y.shape and z.dtype == np.float32
    for f in (48.3, 55.7):
        assert _level_db(y, sr, f) - _level_db(z, sr, f) > 25
    assert abs(_level_db(y, sr, 85.0) - _level_db(z, sr, 85.0)) < 1.5
    assert abs(_level_db(y, sr, 110.0) - _level_db(z, sr, 110.0)) < 0.5

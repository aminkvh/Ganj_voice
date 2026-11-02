# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np

SR = 24000


def _bursts(noise_db, secs=4.0, seed=0):
    """Voiced bursts (harmonics of 140 Hz) 0.7 s on / 0.3 s off, plus white noise noise_db below the speech power."""
    t = np.arange(int(SR * secs)) / SR
    on = (t % 1.0) < 0.7
    voice = on * sum(0.2 / k * np.sin(2 * np.pi * 140 * k * t) for k in range(1, 40))
    p = np.mean(voice[on] ** 2)
    noise = np.random.default_rng(seed).standard_normal(len(t)) * np.sqrt(p * 10 ** (noise_db / 10))
    return (voice + noise).astype(np.float32)


def test_pause_bands_track_the_background_level_below_speech():
    from ganj_eval.noise import BANDS, pause_bands
    quiet, loud = pause_bands(_bursts(-60), SR), pause_bands(_bursts(-30), SR)
    assert len(quiet) == len(loud) == len(BANDS)
    assert all(25 < l - q < 35 for q, l in zip(quiet, loud))   # 30 dB more noise reads ≈ 30 dB higher in every band


def test_voiced_aperiodicity_is_higher_for_a_noisier_voice():
    from ganj_eval.noise import voiced_aperiodicity
    clean, noisy = voiced_aperiodicity(_bursts(-60), SR), voiced_aperiodicity(_bursts(-15), SR)
    assert all(n > c + 3 for c, n in zip(clean, noisy))

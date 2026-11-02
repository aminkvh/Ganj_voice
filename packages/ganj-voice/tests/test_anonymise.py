# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np

from ganj_voice.anonymise import mcadams, warp_poles


def _voice(sr=16000, f0=140.0, secs=1.0):
    t = np.arange(int(sr * secs)) / sr
    src = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 30))
    from scipy.signal import lfilter
    # two formants at 700 Hz and 1200 Hz
    a = np.poly([0.97 * np.exp(1j * 2 * np.pi * 700 / sr), 0.97 * np.exp(-1j * 2 * np.pi * 700 / sr),
                 0.96 * np.exp(1j * 2 * np.pi * 1200 / sr), 0.96 * np.exp(-1j * 2 * np.pi * 1200 / sr)]).real
    return lfilter([1.0], a, src).astype(np.float32) * 0.01


def test_warp_poles_raises_complex_pole_angles_to_alpha_and_keeps_radius():
    p = np.array([0.9 * np.exp(1j * 0.5), 0.9 * np.exp(-1j * 0.5), 0.5 + 0j])
    w = warp_poles(p, 0.8)
    assert np.allclose(np.abs(w), np.abs(p))
    assert np.isclose(np.angle(w[0]), 0.5 ** 0.8) and np.isclose(np.angle(w[1]), -(0.5 ** 0.8))
    assert w[2] == p[2]


def test_alpha_one_returns_the_input():
    y = _voice()
    assert np.allclose(mcadams(y, 16000, alpha=1.0), y, atol=1e-3)


def test_mcadams_keeps_length_and_moves_the_spectral_envelope():
    y = _voice()
    z = mcadams(y, 16000, alpha=0.8)
    assert z.shape == y.shape and np.isfinite(z).all()
    spec = lambda s: np.abs(np.fft.rfft(s * np.hanning(len(s))))
    f = np.fft.rfftfreq(len(y), 1 / 16000)
    band = (f > 1500) & (f < 3000)
    # alpha < 1 lifts formant angles (φ**0.8 > φ for φ < 1 rad), so energy moves up into 1.5–3 kHz
    assert spec(z)[band].sum() / spec(z).sum() > spec(y)[band].sum() / spec(y).sum()


def _f0_median(y, sr):
    import pyworld as pw
    f0, _ = pw.harvest(y.astype(np.float64), sr, f0_floor=40.0, frame_period=5.0)
    return np.median(f0[f0 > 0])


def test_world_shift_raises_pitch_by_the_given_semitones_and_keeps_length_and_loudness():
    from ganj_voice.anonymise import world_shift
    y = _voice()
    z = world_shift(y, 16000, formant=1.1, semitones=2.0)
    assert len(z) == len(y)
    assert np.isclose(_f0_median(z, 16000) / _f0_median(y, 16000), 2 ** (2 / 12), rtol=0.03)
    assert np.isclose(np.sqrt(np.mean(z ** 2)), np.sqrt(np.mean(y ** 2)), rtol=0.02)


def test_world_shift_moves_the_first_formant_up_by_the_formant_factor():
    from ganj_voice.anonymise import world_shift
    import pyworld as pw
    y = _voice()
    z = world_shift(y, 16000, formant=1.15, semitones=0.0)

    def f1(x):
        x = x.astype(np.float64); f0, t = pw.harvest(x, 16000, frame_period=5.0)
        sp = np.median(pw.cheaptrick(x, f0, t, 16000)[f0 > 0], axis=0)
        hz = np.arange(len(sp)) * 16000 / (2 * (len(sp) - 1)); band = (hz > 400) & (hz < 1000)
        return hz[band][np.argmax(sp[band])]
    assert np.isclose(f1(z) / f1(y), 1.15, rtol=0.05)


def test_world_shift_tracks_a_bass_voice_below_wolds_default_floor():
    from ganj_voice.anonymise import world_shift
    y = _voice(f0=50.0, secs=1.5)
    z = world_shift(y, 16000, formant=1.0, semitones=2.0)
    import pyworld as pw
    f0, _ = pw.harvest(z.astype(np.float64), 16000, f0_floor=40.0, frame_period=5.0)
    assert (f0 > 0).mean() > 0.8  # with WORLD's default 71 Hz floor the bass was resynthesised as noise ("husky")
    assert np.isclose(np.median(f0[f0 > 0]), 50.0 * 2 ** (2 / 12), rtol=0.05)


def test_the_formant_shift_can_fade_out_above_a_band():
    # the ش of «شرحِ … شمع» sounds like a laser: د۱'s ش sits at 5.5 kHz (human 3.8), the prompt's ش
    # (source 4.7 kHz) lifted by the ×1.10 shift; the vowel formants keep their shift, the hiss band need not follow
    from ganj_voice.anonymise import envelope_sources
    hz = np.linspace(0, 11025, 1025)
    src = envelope_sources(1025, 22050, 1.10)
    assert np.allclose(src, np.arange(1025) / 1.10)
    src = envelope_sources(1025, 22050, 1.10, fade_hz=(3000, 4000)) * 11025 / 1024
    low, high = hz < 3000, hz >= 4000
    assert np.allclose(src[low], hz[low] / 1.10) and np.allclose(src[high], hz[high]) and np.all(np.diff(src) >= 0)


def test_fricative_frames_are_those_whose_envelope_lives_above_2500_hz():
    from ganj_voice.anonymise import fricative_frames
    sp = np.full((3, 1025), 1e-6)
    sp[0, :100] = 1.0           # a vowel: energy below 1.1 kHz
    sp[1, 300:700] = 1.0        # a ش: 3.2–7.5 kHz
    sp[2, :] = 1.0              # flat: 77 % above 2.5 kHz, still a fricative by share
    assert fricative_frames(sp, 22050).tolist() == [False, True, True]


def test_fricative_frames_can_be_limited_to_a_hiss_centred_below_a_frequency():
    # ش (hiss centred ≈ 5 kHz in the prompt) moves, س (≈ 8 kHz) stays: the whistle is the ش closing in on the س
    # (centroid ratio س/ش human 1.74, د۱ 1.19; one factor for both moved the ratio only to 1.35)
    from ganj_voice.anonymise import fricative_frames
    sp = np.full((2, 1025), 1e-6)
    sp[0, 370:560] = 1.0        # 4.0–6.0 kHz: ش-like
    sp[1, 650:880] = 1.0        # 7.0–9.5 kHz: س-like
    assert fricative_frames(sp, 22050).tolist() == [True, True]
    assert fricative_frames(sp, 22050, below_hz=6500).tolist() == [True, False]


def _hiss(sr, lo, hi, secs=0.3, seed=0):
    from scipy.signal import butter, sosfiltfilt
    n = np.random.default_rng(seed).standard_normal(int(sr * secs))
    return (sosfiltfilt(butter(6, [lo, hi], btype="band", fs=sr, output="sos"), n) * 0.02).astype(np.float32)


def _centroid(x, sr, lo=2000, hi=10000):
    P = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr); m = (f > lo) & (f < hi)
    return np.sum(f[m] * P[m]) / np.sum(P[m])


def test_world_shift_can_move_a_fricative_with_its_own_factor_and_keep_the_vowel_shift():
    from ganj_voice.anonymise import world_shift
    sr = 22050
    vowel, hiss = _voice(sr=sr, secs=0.6), _hiss(sr, 4000, 6000)
    vowel = vowel / np.abs(vowel).max() * 0.3   # at 22.05 kHz _voice peaks at 11, and WORLD's D4C returns NaN on it
    y = np.concatenate([vowel, hiss, vowel])
    s, e = len(vowel), len(vowel) + len(hiss)
    mid = slice(s + 1000, e - 1000)
    plain = world_shift(y, sr, formant=1.10, semitones=0.0)
    fixed = world_shift(y, sr, formant=1.10, semitones=0.0, fade_hz=(3000, 4000), fricative_formant=0.85)
    c0 = _centroid(y[mid], sr)
    assert _centroid(plain[mid], sr) / c0 > 1.05                          # today: the hiss rides the ×1.10 shift
    assert np.isclose(_centroid(fixed[mid], sr) / c0, 0.85, rtol=0.06)    # its own factor
    v = slice(2000, s - 2000)
    assert np.isclose(_centroid(fixed[v], sr, 300, 2500), _centroid(plain[v], sr, 300, 2500), rtol=0.02)  # vowel as before
    assert np.isclose(len(fixed), len(y)) and np.isfinite(fixed).all()


def test_square_voiced_aperiodicity_squares_only_voiced_frames():
    from ganj_voice.anonymise import square_voiced_ap
    ap = np.full((3, 4), 0.5)
    f0 = np.array([0.0, 120.0, 0.0])
    out = square_voiced_ap(ap, f0)
    assert np.allclose(out[1], 0.25) and np.allclose(out[[0, 2]], 0.5)
    assert np.allclose(ap, 0.5)  # input untouched


def test_ltas_match_moves_a_tilted_spectrum_to_the_reference_and_caps_the_gain():
    from ganj_voice.anonymise import ltas, ltas_match
    from scipy.signal import lfilter
    sr = 16000
    rng = np.random.default_rng(0)
    flat = rng.standard_normal(sr * 4).astype(np.float32) * 0.05
    tilted = lfilter([1.0, -0.9], [1.0], flat).astype(np.float32)  # high-pass tilt, about +25 dB from 100 Hz to 8 kHz
    f, ref = ltas(flat, sr)
    z = ltas_match(tilted, sr, f, ref, max_db=6.0)
    _, before = ltas(tilted, sr)
    _, after = ltas(z, sr)
    band = (f > 300) & (f < 6000)
    gain = (after - before)[band]
    assert np.ptp(gain) <= 12.5  # the EQ shape is capped at +-6 dB (the loudness restore adds one overall offset)
    assert np.mean(np.abs((after - ref)[band])) < np.mean(np.abs((before - ref)[band]))
    assert np.isclose(np.sqrt(np.mean(z ** 2)), np.sqrt(np.mean(tilted ** 2)), rtol=0.02)  # loudness kept


def test_praat_shift_moves_the_median_pitch_by_the_requested_semitones_and_keeps_length_and_loudness():
    import numpy as np
    from ganj_voice.anonymise import praat_shift
    sr = 22050
    t = np.arange(int(sr * 2.0)) / sr
    f0 = 120 * (1 + 0.05 * np.sin(2 * np.pi * 2 * t))  # a gently moving pitch around 120 Hz
    ph = 2 * np.pi * np.cumsum(f0) / sr
    y = sum(0.2 / k * np.sin(k * ph) for k in range(1, 30)).astype(np.float32)
    z = praat_shift(y, sr, formant=1.1, semitones=2.0)
    assert z.shape == y.shape and z.dtype == np.float32
    import parselmouth
    from parselmouth.praat import call

    def median(a):
        return call(parselmouth.Sound(a.astype(np.float64), sr).to_pitch(pitch_floor=60, pitch_ceiling=400), "Get quantile", 0, 0, 0.5, "Hertz")
    assert abs(12 * np.log2(median(z) / median(y)) - 2.0) < 0.3
    assert abs(10 * np.log10(np.mean(z ** 2) / np.mean(y ** 2))) < 0.5

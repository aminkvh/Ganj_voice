# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np

from ganj_voice.dehum import denoise_stationary, estimate_mains, notch_mains
from tools.make_prompt import build


def _clip(sr=22050, secs=3.0):
    t = np.arange(int(sr * secs)) / sr
    rng = np.random.default_rng(1)
    return (0.1 * np.sin(2 * np.pi * 140 * t) * (t % 1 < 0.7) + 0.01 * np.sin(2 * np.pi * 49.8 * t)
            + 0.003 * rng.standard_normal(len(t))).astype(np.float32)


def test_build_without_steps_returns_the_clip_unchanged():
    y = _clip()
    assert np.array_equal(build(y, 22050), y)


def test_build_notches_before_it_denoises():
    y, sr = _clip(), 22050
    want = denoise_stationary(notch_mains(y, sr, base_hz=estimate_mains(y, sr)), sr, floor_db=-20.0)
    assert np.allclose(build(y, sr, notch=True, denoise_floor=-20.0), want)


def test_build_passes_the_fricative_options_to_the_world_shift():
    from ganj_voice.anonymise import world_shift
    y, sr = _clip(), 22050
    want = world_shift(y, sr, formant=1.1, semitones=1.0, ap_squared=True, fade_hz=(3000, 4000), fricative_formant=0.85)
    got = build(y, sr, formant=1.1, semitones=1.0, ap_squared=True, formant_fade=(3000, 4000), fricative_formant=0.85)
    assert np.array_equal(got, want)


def test_the_cli_reads_the_fricative_options():
    from tools.make_prompt import parse_args
    a = parse_args(["s.wav", "o.wav", "--formant", "1.1", "--formant-fade", "3000", "4000", "--fricative-formant", "0.85",
                    "--fricative-below", "6500"])
    assert a.formant_fade == [3000.0, 4000.0] and a.fricative_formant == 0.85 and a.fricative_below == 6500.0
    a = parse_args(["s.wav", "o.wav"])
    assert a.formant_fade is None and a.fricative_formant is None and a.fricative_below is None


def test_build_passes_the_fricative_ceiling_to_the_world_shift():
    from ganj_voice.anonymise import world_shift
    y, sr = _clip(), 22050
    want = world_shift(y, sr, formant=1.1, semitones=1.0, fricative_formant=0.7, fricative_below_hz=6500)
    assert np.array_equal(build(y, sr, formant=1.1, semitones=1.0, fricative_formant=0.7, fricative_below=6500), want)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np

from ganj_voice.vocos_data import aligned_crop, pick_input


def test_aligned_crop_maps_frames_to_hop_samples():
    rng = np.random.default_rng(0)
    for _ in range(50):
        f0, s0 = aligned_crop(200, 199 * 256, 64, 256, rng)
        assert 1 <= f0 <= 200 - 64 and s0 == f0 * 256
        assert s0 + 64 * 256 <= 199 * 256


def test_aligned_crop_returns_none_for_a_clip_shorter_than_the_crop():
    assert aligned_crop(60, 59 * 256, 64, 256, np.random.default_rng(0)) is None


def test_pick_input_takes_the_real_mel_at_the_given_rate():
    rng = np.random.default_rng(0)
    picks = [pick_input("gen", "real", 0.25, rng) for _ in range(4000)]
    assert abs(picks.count("real") / 4000 - 0.25) < 0.03
    assert pick_input("gen", "real", 0.0, rng) == "gen"


def _pair(tmp_path, n_frames=200, hop=4):
    audio = (np.arange((n_frames - 1) * hop) % 1000).astype(np.int16)
    gen = np.tile(np.arange(n_frames, dtype=np.float16), (3, 1))
    p = tmp_path / "c.npz"
    np.savez(p, gen=gen, real=gen + 1, audio=audio)
    return p, audio


def test_load_crop_returns_the_audio_under_the_mel_frames(tmp_path):
    from ganj_voice.vocos_data import load_crop
    p, audio = _pair(tmp_path)
    mel, y = load_crop(p, 9, 4, 0.0, np.random.default_rng(1))
    f0 = int(mel[0, 0])
    assert mel.shape == (3, 9) and mel.dtype == np.float32
    assert y.shape == (8 * 4,) and y.dtype == np.float32
    assert np.allclose(y * 32768, audio[f0 * 4: f0 * 4 + 32])


def test_load_crop_uses_the_real_mel_at_rate_one(tmp_path):
    from ganj_voice.vocos_data import load_crop
    p, _ = _pair(tmp_path)
    real, _ = load_crop(p, 9, 4, 1.0, np.random.default_rng(1))
    gen, _ = load_crop(p, 9, 4, 0.0, np.random.default_rng(1))
    assert np.array_equal(real - gen, np.ones_like(gen))


def test_load_crop_skips_a_clip_shorter_than_the_crop(tmp_path):
    from ganj_voice.vocos_data import load_crop
    p, _ = _pair(tmp_path, n_frames=8)
    assert load_crop(p, 9, 4, 0.0, np.random.default_rng(1)) is None

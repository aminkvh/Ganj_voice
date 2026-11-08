# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np
import pytest

from ganj_voice.phonepost import AFTER, phone_post

SR = 24000


def test_the_gain_comes_before_the_clamp_so_a_loud_peak_is_scaled_not_cut():
    y = np.zeros(SR, dtype=np.float32)
    y[1000] = 1.2
    out, cut = phone_post(y, SR, {**AFTER, "gain": 0.5}, n_breaks=0, skip={"highpass", "deess", "trim"})
    assert out[1000] == pytest.approx(0.6) and cut == []


def test_without_the_gain_the_clamp_cuts_the_peak():
    y = np.zeros(SR, dtype=np.float32)
    y[1000] = 1.2
    out, _ = phone_post(y, SR, {**AFTER, "gain": 0.5}, n_breaks=0, skip={"gain", "highpass", "deess", "trim"})
    assert out[1000] == pytest.approx(1.0)


def test_the_high_pass_removes_a_hum_below_70_hz():
    t = np.arange(2 * SR) / SR
    y = (0.1 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out, _ = phone_post(y, SR, AFTER, n_breaks=0, skip={"deess", "trim"})
    spec = np.abs(np.fft.rfft(out[SR // 2:-SR // 2]))
    freqs = np.fft.rfftfreq(len(out) - SR, 1 / SR)
    hum, tone = spec[np.argmin(abs(freqs - 50))], spec[np.argmin(abs(freqs - 440))]
    assert hum < tone * 0.05


def test_an_unknown_step_name_is_refused():
    with pytest.raises(ValueError):
        phone_post(np.zeros(100, dtype=np.float32), SR, AFTER, 0, skip={"deesser"})

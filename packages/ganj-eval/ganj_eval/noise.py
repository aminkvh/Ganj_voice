# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Background and voice noise of a clip (the "fluorescent" background noise of early outputs).

pause_bands: the quietest 10 % of frames' power per band, in dB below the speech (the loudest half of the frames). It
reads the background only when a clip pauses for at least 10 % of its length; for a clip read without pauses it reads
speech. voiced_aperiodicity: WORLD D4C aperiodicity (noise power over total) in voiced frames, median dB per band;
pitch-adaptive, so a moving (chanting) pitch does not read as noise the way fixed-window harmonic valleys do.
"""
import numpy as np
import pyworld as pw

BANDS = ((30, 200), (200, 1000), (1000, 4000), (4000, 8000), (8000, 12000))
AP_BANDS = ((500, 1000), (1000, 2000), (2000, 4000), (4000, 7000))
N_FFT, HOP = 2048, 512


def pause_bands(y: np.ndarray, sr: int, bands=BANDS, quiet_frac: float = 0.1) -> list:
    x = np.asarray(y, dtype=np.float64)
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP] * np.hanning(N_FFT)
    S = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    order = np.argsort(S.sum(axis=1))
    quiet = S[order[: max(1, int(len(order) * quiet_frac))]].mean(axis=0)
    speech = S[order[len(order) // 2:]].sum(axis=1).mean()
    f = np.fft.rfftfreq(N_FFT, 1 / sr)
    return [float(10 * np.log10(quiet[(f >= lo) & (f < hi)].sum() / speech + 1e-20)) for lo, hi in bands]


def voiced_aperiodicity(y: np.ndarray, sr: int, bands=AP_BANDS) -> list:
    x = np.asarray(y, dtype=np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=70, f0_ceil=400, frame_period=10.0)
    n = pw.get_cheaptrick_fft_size(sr, 70)
    ap = pw.d4c(x, f0, t, sr, fft_size=n)[f0 > 0]
    return [float(np.median(10 * np.log10(ap[:, int(lo / sr * n):int(hi / sr * n)] + 1e-6))) for lo, hi in bands]

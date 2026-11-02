# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Sibilant de-essing for model outputs (the voices' س and ش whistle; turning the prompt's hiss
down made no audible difference). Measured on the 50 eval lines, ش/س frames, level against each line's
median voiced frame: Vocos copy-synthesis of the human recordings is within 0.3 dB of them in every band, but every
model voice tried is 5-11 dB louder in 5-9 kHz and level in 2-5 kHz. So the acoustic model, not the vocoder or
the prompt, makes the hiss too loud up there, and the prompt factor moved only its centre.

The excess differs by sound: ش is too loud in 5-7 kHz and human-level above, س in 5-9 kHz (a cut of the whole band left
ش 7-8 dB under the human recordings at 7-9 kHz). So the cut's top follows each frame's spectral centre: a ش-like
frame (centre <= 5 kHz) is cut up to 7 kHz, a س-like one (>= 6.2 kHz) up to 9.5 kHz, in between in proportion.
The depth follows the centre the same way (measured on the 50 lines: a 5 dB cut put ش at the human level and
left س 3-7 dB over it, 9 dB put س there and cut ش 2-5 dB under it): a ش-like frame gets SHIN_DEPTH of depth_db.

deess: in frames whose energy is mostly above HISS_FROM_HZ (a fricative), 4.5 kHz up to that top (raised-cosine edges)
is turned down by depth_db (times SHIN_DEPTH for ش); vowels, voiced consonants and everything below 4.5 kHz are left
as they are. Attenuation only.
"""
import numpy as np
from scipy.signal import istft, stft

NFFT, HOP = 512, 128                         # 21 ms windows every 5.3 ms at 24 kHz
RISE_HZ = (4500.0, 5500.0)                   # the cut fades in between these
CENTRE_HZ, TOP_HZ = (5000.0, 6200.0), (7000.0, 9500.0)   # a frame's 2-11 kHz centre -> where the full cut ends
FADE_HZ = 1000.0                             # and fades out over this
SHIN_DEPTH = 0.6                             # share of depth_db a ش-like frame gets (س-like: all of it)
HISS_FROM_HZ = 2500.0
HISS_SHARE = (0.4, 0.6)                      # share of a frame's energy above HISS_FROM_HZ: no cut below, the full cut above
SMOOTH_FRAMES = 3


def _ramp(x: np.ndarray) -> np.ndarray:
    """0 -> 1 raised cosine over x in [0, 1]."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))


def centre_hz(f: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Per frame, the power-weighted mean frequency over 2-11 kHz."""
    band = (f >= 2000) & (f <= 11000)
    return (f[band, None] * power[band]).sum(axis=0) / (power[band].sum(axis=0) + 1e-20)


def band_weight(f: np.ndarray, power: np.ndarray) -> np.ndarray:
    """(bins, frames) share of the cut: rises over RISE_HZ, full up to each frame's top, gone FADE_HZ above it,
    scaled from SHIN_DEPTH (ش-like centre) to 1 (س-like)."""
    centre = centre_hz(f, power)
    top = np.interp(centre, CENTRE_HZ, TOP_HZ)
    depth = np.interp(centre, CENTRE_HZ, (SHIN_DEPTH, 1.0))
    rise = _ramp((f - RISE_HZ[0]) / (RISE_HZ[1] - RISE_HZ[0]))
    return rise[:, None] * (1.0 - _ramp((f[:, None] - top[None, :]) / FADE_HZ)) * depth[None, :]


def hiss_weight(power: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Per frame, how much of the cut it gets (0 … 1), from the share of its energy above HISS_FROM_HZ."""
    share = power[f > HISS_FROM_HZ].sum(axis=0) / (power.sum(axis=0) + 1e-20)
    lo, hi = HISS_SHARE
    m = np.clip((share - lo) / (hi - lo), 0.0, 1.0)
    return np.convolve(m, np.ones(SMOOTH_FRAMES) / SMOOTH_FRAMES, mode="same")


def deess(y: np.ndarray, sr: int, depth_db: float) -> np.ndarray:
    y = np.asarray(y)
    if depth_db <= 0:
        return y.copy()
    x = y.astype(np.float64)
    f, _, Z = stft(x, fs=sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)
    power = np.abs(Z) ** 2
    cut = band_weight(f, power) * hiss_weight(power, f)[None, :]
    _, z = istft(Z * 10 ** (-depth_db * cut / 20), fs=sr, window="hann", nperseg=NFFT, noverlap=NFFT - HOP)
    return z[:len(x)].astype(y.dtype)

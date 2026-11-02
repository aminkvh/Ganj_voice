# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Recording-noise removal for a voice prompt (2026-10-28: the F2 prompt's source clip carries a 50 Hz hum 33 dB over its
floor and a broadband hiss ≈ 19 dB above the human recordings at 4-8 kHz; ZipVoice copies the prompt's background into
every output, heard as a "fluorescent" noise).

notch_mains: zero-phase IIR notches at the mains frequency and its harmonics, each about `width_hz` wide, so a voiced
partial a few Hz away from a harmonic passes unchanged.
denoise_stationary: gentle suppression of a steady background (hiss, hum, tones) learned from the clip's quietest frames.
highpass: for model outputs (2026-10-29): a humming prompt's hum comes back from Vocos as lines near 48 and 56 Hz,
not at the mains frequency, so a notch misses them; the voices' pitch stays above ≈ 80 Hz.
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, istft, sosfiltfilt, stft


def estimate_mains(y: np.ndarray, sr: int, nominal_hz: float = 50.0, search_hz: float = 2.0) -> float:
    """The hum's actual frequency (mains drifts: 49.75 Hz in clip 18347/15): the spectral peak within +-search_hz of
    nominal, in the quietest third of long frames, refined by parabolic interpolation on a zero-padded FFT."""
    x = np.asarray(y, dtype=np.float64)
    seg = min(len(x), 4 * sr)
    f, _, Z = stft(x, fs=sr, nperseg=seg, noverlap=seg * 3 // 4, nfft=8 * seg)
    p = np.abs(Z) ** 2
    quiet = np.argsort(p.sum(axis=0))[: max(1, p.shape[1] // 3)]
    spec = np.log(p[:, quiet].mean(axis=1) + 1e-30)
    m = np.where((f >= nominal_hz - search_hz) & (f <= nominal_hz + search_hz))[0]
    k = m[np.argmax(spec[m])]
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    return float(f[k] + shift * (f[1] - f[0]))


def notch_mains(y: np.ndarray, sr: int, base_hz: float = 50.0, max_hz: float = 1000.0, width_hz: float = 2.0) -> np.ndarray:
    z = np.asarray(y, dtype=np.float64)
    for k in range(1, int(max_hz // base_hz) + 1):
        f = k * base_hz
        if f >= sr / 2:
            break
        b, a = iirnotch(f, f / width_hz, fs=sr)
        z = filtfilt(b, a, z)
    return z.astype(np.float32)


def highpass(y: np.ndarray, sr: int, cutoff_hz: float = 70.0, order: int = 8) -> np.ndarray:
    """Zero-phase Butterworth high-pass (run forward and back, so the slope is doubled and no phase shift)."""
    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, np.asarray(y, dtype=np.float64)).astype(np.float32)


def denoise_stationary(y: np.ndarray, sr: int, floor_db: float = -20.0, quiet_frac: float = 0.15, n_fft: int = 1024,
                       smooth: float = 0.98) -> np.ndarray:
    """Decision-directed Wiener suppression (Ephraim-Malah a-priori SNR) of a steady background whose spectrum is the mean
    of the clip's quietest `quiet_frac` of frames. The gain never drops below `floor_db`: no musical noise (which the
    model would copy as a robotic texture), and soft consonants and breaths stay audible."""
    x = np.asarray(y, dtype=np.float64)
    hop = n_fft // 4
    _, _, X = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    post = np.abs(X) ** 2
    quiet = np.argsort(post.sum(axis=0))[: max(1, int(post.shape[1] * quiet_frac))]
    post = post / (post[:, quiet].mean(axis=1, keepdims=True) + 1e-20)  # a-posteriori SNR per bin
    gmin, prev = 10 ** (floor_db / 20), np.ones(post.shape[0])
    G = np.empty_like(post)
    for k in range(post.shape[1]):
        xi = smooth * prev + (1 - smooth) * np.maximum(post[:, k] - 1, 0)
        G[:, k] = np.maximum(xi / (1 + xi), gmin)
        prev = G[:, k] ** 2 * post[:, k]
    _, z = istft(X * G, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    return np.pad(z, (0, max(0, len(x) - len(z))))[: len(x)].astype(np.float32)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""The voice-prompt disguise in the mel domain: warp the frequency axis of the prompt's Vocos log-mel and
decode it with Vocos (tools/melwarp_prompt.py). WORLD's analysis–resynthesis, the disguise so far, adds ≈ 5 cents of
held-vowel wobble and ≈ 0.5 dB of level flutter on its own (undisguised prompt 15.7 cents, WORLD د۱ 20.7, human 17.2),
which the model copies as a shake in stretched vowels. The warp moves pitch with the formants (×1.10 ≈ +1.65 st).
numpy only: the decode runs in the ZipVoice environment, which has Vocos."""
import numpy as np


def mel_centres(sr: int = 24000, n_mels: int = 100) -> np.ndarray:
    """Centre frequencies of the HTK mel filters from 0 Hz to sr / 2 (torchaudio's MelSpectrogram, as Vocos uses)."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    return hz(np.linspace(0, mel(sr / 2), n_mels + 2))[1:-1]


def _fricative(lm: np.ndarray, centres: np.ndarray, above_hz: float = 2500.0, share: float = 0.6) -> np.ndarray:
    """Frames whose mel energy has more than `share` above `above_hz` (ش, س), as anonymise.fricative_frames."""
    p = np.exp(2 * lm)
    return p[centres > above_hz].sum(0) / np.maximum(p.sum(0), 1e-30) > share


def warp_logmel(lm: np.ndarray, centres: np.ndarray, alpha: float, fade_hz=None, fricative_alpha=None) -> np.ndarray:
    """lm: (n_mels, frames) log-mel. Output frequency f reads input f / alpha (formants and harmonics move up by alpha);
    with fade_hz=(lo, hi) the warp fades out from lo to hi Hz, and with fricative_alpha the fricative frames are
    warped by that factor instead (د۴'s ×0.80 lowered the whistling ش)."""
    if alpha == 1.0 and not fade_hz and not fricative_alpha:
        return lm.copy()
    src = centres / alpha
    if fade_hz:
        w = np.clip((centres - fade_hz[0]) / (fade_hz[1] - fade_hz[0]), 0, 1)
        src = (1 - w) * src + w * centres
    fric = _fricative(lm, centres) if fricative_alpha else np.zeros(lm.shape[1], dtype=bool)
    fsrc = centres / fricative_alpha if fricative_alpha else src
    return np.stack([np.interp(fsrc if f else src, centres, col) for col, f in zip(lm.T, fric)], axis=1)

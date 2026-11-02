# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""McAdams-coefficient voice anonymisation (Patino et al. 2021, VoicePrivacy baseline B2).

Per frame: LPC fit, raise each complex pole's angle φ to φ**alpha (radius kept), re-filter the LPC residual
through the warped all-pole filter, overlap-add. alpha < 1 lifts formants, giving a different, non-existent
timbre while keeping pitch, timing and words. Used to make a "nobody's voice" ZipVoice prompt from a clean clip.

world_shift: WORLD analysis/resynthesis with the spectral envelope stretched by `formant` and f0 raised by `semitones`.
McAdams also moves the low-angle poles that carry spectral tilt, leaving a thin, buzzy prompt that ZipVoice copies;
the WORLD shift keeps the tilt and most of the prompt's UTMOS.
"""
import librosa
import numpy as np
import pyworld as pw
from scipy.signal import lfilter


def warp_poles(poles: np.ndarray, alpha: float) -> np.ndarray:
    out = poles.copy()
    cx = np.abs(poles.imag) > 1e-12
    ang = np.angle(poles[cx])
    out[cx] = np.abs(poles[cx]) * np.exp(1j * np.sign(ang) * np.abs(ang) ** alpha)
    return out


def mcadams(y: np.ndarray, sr: int, alpha: float = 0.8, order: int = 20, win_ms: float = 20.0) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if alpha == 1.0:
        return y.astype(np.float32)
    n = int(sr * win_ms / 1000) // 2 * 2
    hop = n // 2
    win = np.hanning(n + 1)[:-1]  # periodic Hann: overlap-adds to 1 at 50 % hop
    pad = np.concatenate([np.zeros(n), y, np.zeros(n)])
    out = np.zeros_like(pad)
    for s in range(0, len(pad) - n, hop):
        fr = pad[s:s + n] * win
        if not np.any(fr):
            continue
        a = librosa.lpc(fr + 1e-9 * np.random.default_rng(s).standard_normal(n), order=order)
        res = lfilter(a, [1.0], fr)
        a2 = np.real(np.poly(warp_poles(np.roots(a), alpha)))
        out[s:s + n] += lfilter([1.0], a2, res)
    z = out[n:n + len(y)]
    z *= np.sqrt(np.mean(y ** 2) / max(np.mean(z ** 2), 1e-12))  # keep the input loudness
    return z.astype(np.float32)


F0_FLOOR = 40.0  # WORLD defaults to 71 Hz; a bass reciter (≈ 50–100 Hz) lost 40 % of his voicing to noise


def square_voiced_ap(ap: np.ndarray, f0: np.ndarray) -> np.ndarray:
    """Voiced frames' aperiodicity squared: less noise in the excitation (sweep 10, against the "husky" prompt)."""
    out = ap.copy()
    out[f0 > 0] = out[f0 > 0] ** 2
    return out


def envelope_sources(n_bins: int, sr: int, formant: float, fade_hz=None) -> np.ndarray:
    """The envelope bin each output bin reads: output f reads input f / formant, so formants move up by `formant`. With
    fade_hz=(lo, hi) the shift fades out from lo to hi Hz and output f reads f above hi: the vowel formants move, the
    hiss band of a sibilant stays (the whistling ش)."""
    bins = np.arange(n_bins)
    src = bins / formant
    if fade_hz:
        per_bin = sr / 2 / (n_bins - 1)
        w = np.clip((bins - fade_hz[0] / per_bin) / ((fade_hz[1] - fade_hz[0]) / per_bin), 0, 1)
        src = (1 - w) * src + w * bins
    return np.clip(src, 0, bins[-1])


def fricative_frames(sp: np.ndarray, sr: int, above_hz: float = 2500.0, share: float = 0.6, below_hz=None) -> np.ndarray:
    """Frames whose envelope has more than `share` of its energy above `above_hz` (ش, س, ف…); with below_hz, only those
    whose energy above `above_hz` is centred below it (ش, not س). WORLD's harvest calls the prompt's sibilants voiced
    (at 50–116 Hz), so f0 cannot pick them out."""
    hz = np.linspace(0, sr / 2, sp.shape[1])
    hi = sp[:, hz > above_hz]
    out = hi.sum(1) / np.maximum(sp.sum(1), 1e-30) > share
    if below_hz:
        out &= hi @ hz[hz > above_hz] / np.maximum(hi.sum(1), 1e-30) < below_hz
    return out


def world_shift(y: np.ndarray, sr: int, formant: float = 1.1, semitones: float = 1.0,
                ap_squared: bool = False, fade_hz=None, fricative_formant=None, fricative_below_hz=None) -> np.ndarray:
    """fade_hz and fricative_formant (all off by default: the stored prompts are reproduced bit for bit): fade the
    shift out above a band (envelope_sources), and give fricative frames their own factor (0.85 moves a hiss down);
    fricative_below_hz limits that to a hiss centred below it (ش, not س; fricative_frames)."""
    x = np.asarray(y, dtype=np.float64)
    f0, t = pw.harvest(x, sr, f0_floor=F0_FLOOR, frame_period=5.0)
    n = pw.get_cheaptrick_fft_size(sr, F0_FLOOR)
    sp, ap = pw.cheaptrick(x, f0, t, sr, fft_size=n), pw.d4c(x, f0, t, sr, fft_size=n)
    if ap_squared:
        ap = square_voiced_ap(ap, f0)
    bins = np.arange(sp.shape[1])
    src = envelope_sources(sp.shape[1], sr, formant, fade_hz)
    fric = fricative_frames(sp, sr, below_hz=fricative_below_hz) if fricative_formant else np.zeros(len(sp), dtype=bool)
    fsrc = envelope_sources(sp.shape[1], sr, fricative_formant) if fricative_formant else src
    sp2 = np.stack([np.interp(fsrc if f else src, bins, row) for row, f in zip(sp, fric)])
    z = pw.synthesize(f0 * 2 ** (semitones / 12), sp2, ap, sr, frame_period=5.0)
    z = np.pad(z, (0, max(0, len(x) - len(z))))[:len(x)]
    z *= np.sqrt(np.mean(x ** 2) / max(np.mean(z ** 2), 1e-12))  # keep the input loudness
    return z.astype(np.float32)


def praat_shift(y: np.ndarray, sr: int, formant: float = 1.1, semitones: float = 1.0, pitch_range: float = 1.0) -> np.ndarray:
    """Praat's "Change gender" (PSOLA pitch, resampled formants; praat-parselmouth, GPL-3, an offline prompt tool):
    the disguise without WORLD's analysis-resynthesis (F6). The median pitch moves by `semitones`.
    Not deterministic: Praat places the overlap-add pulses of unvoiced parts at random (two runs correlate ≈ 0.98, no
    seed in parselmouth), so a stored prompt wav, not its recipe, is the reference."""
    import parselmouth
    from parselmouth.praat import call
    x = np.asarray(y, dtype=np.float64)
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    med = call(snd.to_pitch(time_step=0.01, pitch_floor=60, pitch_ceiling=400), "Get quantile", 0, 0, 0.5, "Hertz")
    z = call(snd, "Change gender", 60, 400, formant, med * 2 ** (semitones / 12), pitch_range, 1.0).values[0]
    z = np.pad(z, (0, max(0, len(x) - len(z))))[:len(x)]
    z *= np.sqrt(np.mean(x ** 2) / max(np.mean(z ** 2), 1e-12))  # keep the input loudness
    return z.astype(np.float32)


LTAS_NFFT = 2048


def ltas(y: np.ndarray, sr: int, nfft: int = LTAS_NFFT):
    """Long-term average spectrum in dB (Welch, Hann), with its frequency axis."""
    from scipy.signal import welch
    f, p = welch(np.asarray(y, dtype=np.float64), fs=sr, nperseg=nfft)
    return f, 10 * np.log10(p + 1e-20)


def _smooth_octave(f: np.ndarray, db: np.ndarray, frac: float) -> np.ndarray:
    """Average each bin over +-frac/2 octave around it."""
    out = db.copy()
    for i, fc in enumerate(f):
        if fc <= 0:
            continue
        m = (f >= fc * 2 ** (-frac / 2)) & (f <= fc * 2 ** (frac / 2))
        out[i] = db[m].mean()
    return out


def ltas_match(y: np.ndarray, sr: int, ref_f: np.ndarray, ref_db: np.ndarray, max_db: float = 6.0,
               smooth_oct: float = 1 / 3) -> np.ndarray:
    """EQ y towards a reference LTAS (Sweep 9: the prompt tone-matched to the mean human LTAS): a smoothed gain per
    frequency, capped at +-max_db, applied in the STFT domain; the input loudness is kept."""
    from scipy.signal import istft, stft
    x = np.asarray(y, dtype=np.float64)
    f, own = ltas(x, sr)
    ref = np.interp(f, ref_f, ref_db)
    diff = (ref - own) - np.mean((ref - own)[(f > 100) & (f < 8000)])  # shape only: loudness is restored below
    gain_db = np.clip(_smooth_octave(f, diff, smooth_oct), -max_db, max_db)
    fs, _, Z = stft(x, fs=sr, nperseg=LTAS_NFFT)
    Z *= (10 ** (np.interp(fs, f, gain_db) / 20))[:, None]
    _, z = istft(Z, fs=sr, nperseg=LTAS_NFFT)
    z = np.pad(z, (0, max(0, len(x) - len(z))))[:len(x)]
    z *= np.sqrt(np.mean(x ** 2) / max(np.mean(z ** 2), 1e-12))
    return z.astype(np.float32)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Remove isolated spikes from the high band of a generated log-mel before the vocoder.

ZipVoice's flow model leaves single high mel bins stuck above their neighbours (and scattered single-frame specks);
Vocos renders a stuck bin as a steady sine at the bin's centre (8.04 / 8.30 / 9.12 / 11.30 kHz) — a "laser" tone.
A median over neighbouring bins (and optionally frames) removes such outliers and leaves smooth spectral shape
untouched; the low band, where the voice's harmonics are resolved, is never touched."""
import torch
import torch.nn.functional as F

START_BIN = 60  # ≈ 3.9 kHz for Vocos's 100-band 24 kHz mel


def despike_mel(mel: torch.Tensor, start_bin: int = START_BIN, time: bool = False) -> torch.Tensor:
    """mel: (B, C, T) log-mel. Returns a new tensor with bins >= start_bin median-filtered over 3 bins
    (3 bins x 3 frames when time=True)."""
    kt = 3 if time else 1
    x = F.pad(mel[:, None], (kt // 2, kt // 2, 1, 1), mode="replicate")[:, 0]   # (B, C+2, T+kt-1)
    win = x.unfold(1, 3, 1).unfold(2, kt, 1)                                     # (B, C, T, 3, kt)
    med = win.reshape(*win.shape[:3], -1).median(dim=-1).values
    out = mel.clone()
    out[:, start_bin:] = med[:, start_bin:]
    return out


BINOMIAL = {3: (1, 2, 1), 5: (1, 4, 6, 4, 1)}


def smooth_time(mel: torch.Tensor, frames: int = 5, hi_bin: int = START_BIN) -> torch.Tensor:
    """mel: (B, C, T) log-mel. Returns a new tensor with bins < hi_bin averaged over `frames` frames (binomial weights).
    The flow model leaves frame-to-frame jitter in the harmonic band that Vocos renders as a pitch wobble in long vowels
    (≈ 21-23 cents vs 16.5 in human readings); the high band is despike_mel's."""
    w = torch.tensor(BINOMIAL[frames], dtype=mel.dtype, device=mel.device)
    x = F.pad(mel[:, :hi_bin], (frames // 2, frames // 2), mode="replicate")
    low = F.conv1d(x.reshape(-1, 1, x.shape[-1]), (w / w.sum()).view(1, 1, -1)).reshape(mel.shape[0], hi_bin, -1)
    return torch.cat([low, mel[:, hi_bin:]], dim=1)

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Training pairs for a Vocos fine-tune on the flow model's own output (regen_pairs.py writes them): a mel crop and
the real audio under it. Frame i of a centred STFT (hop 256) sits at sample i * hop, so a crop of n frames from frame
f0 is vocoded (centred inverse STFT) to the (n - 1) * hop samples from f0 * hop. Frame 0 is never a crop start: its centre padding has no audio
to its left."""
from typing import Optional, Tuple

import numpy as np


def aligned_crop(n_frames: int, n_samples: int, crop_frames: int, hop: int,
                 rng: np.random.Generator) -> Optional[Tuple[int, int]]:
    """(f0, s0) of a random crop that fits both the mel and the audio, or None if the clip is too short."""
    last = min(n_frames - crop_frames, n_samples // hop - crop_frames)
    if last < 1:
        return None
    f0 = int(rng.integers(1, last + 1))
    return f0, f0 * hop


def pick_input(gen, real, real_rate: float, rng: np.random.Generator):
    """The real mel at real_rate (keeps copy-synthesis anchored), else the model's."""
    return real if rng.random() < real_rate else gen


def load_crop(path, crop_frames: int, hop: int, real_rate: float,
              rng: np.random.Generator) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """One training crop of a pair file: (mel float32 (C, crop_frames), audio float32 ((crop_frames - 1) * hop,)) —
    a centred inverse STFT of n frames gives (n - 1) * hop samples — or None when the clip is too short."""
    with np.load(path) as z:
        gen, real, audio = z["gen"], z["real"], z["audio"]
    crop = aligned_crop(gen.shape[1], len(audio), crop_frames, hop, rng)
    if crop is None:
        return None
    f0, s0 = crop
    mel = pick_input(gen, real, real_rate, rng)[:, f0:f0 + crop_frames].astype(np.float32)
    return mel, audio[s0:s0 + (crop_frames - 1) * hop].astype(np.float32) / 32768

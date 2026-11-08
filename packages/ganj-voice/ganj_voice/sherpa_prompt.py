# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""The voice prompt for sherpa-onnx at a length whose frames are ZipVoice's. ZipVoice's VocosFbank keeps
(n + hop // 2) // hop frames of the centred STFT (lhotse's compute_num_frames) and drops the rest; sherpa keeps every
centred frame, 1 + n // hop. The two agree when n % hop >= hop // 2; otherwise sherpa reads one prompt frame more (ours:
176401 samples, 690 frames against 689) and makes the line one frame longer. Cutting the last n % hop + 1 samples gives
both ZipVoice's count; the frames are the same except the last two windows, which see a tail shorter by that much."""
import numpy as np

HOP = 256


def zipvoice_frames(n: int, hop: int = HOP) -> int:
    return (n + hop // 2) // hop


def sherpa_frames(n: int, hop: int = HOP) -> int:
    return 1 + n // hop


def frame_matched(samples: np.ndarray, hop: int = HOP) -> np.ndarray:
    """samples cut at the end so that sherpa_frames == zipvoice_frames of the uncut prompt (at most hop // 2 samples)."""
    if len(samples) < hop:
        raise ValueError(f"a prompt of {len(samples)} samples is shorter than one hop ({hop})")
    r = len(samples) % hop
    return samples if r >= hop // 2 else samples[:len(samples) - r - 1]

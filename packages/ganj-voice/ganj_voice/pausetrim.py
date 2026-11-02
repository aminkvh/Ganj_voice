# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Shorter stops inside a hemistich, after synthesis (the model stops too long; a native reader's inner stops are
≈ 0.27 s, the model's 0.47–0.52 s). Only silence is cut, from the middle of each run, with a short crossfade; the
hemistich break keeps the length the model gave it. Whether a stop belongs there at all stays the model's (and the
pause tokens') business: this sets only how long it lasts."""
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

CAP_S, FADE_S = 0.30, 0.01


def stops_to_trim(runs: Sequence[Optional[Tuple[float, float]]], breaks: Iterable[int], cap: float = CAP_S) -> list:
    """The silent runs (one per word boundary, None where there is none) longer than cap, hemistich breaks left out."""
    breaks = set(breaks)
    return [r for i, r in enumerate(runs) if r is not None and i not in breaks and r[1] - r[0] > cap]


def shorten_runs(y: np.ndarray, sr: int, runs: Sequence[Tuple[float, float]], cap: float = CAP_S,
                 fade: float = FADE_S) -> np.ndarray:
    """y with each (start s, end s) silent run longer than cap cut to cap from its middle (crossfaded)."""
    n_fade = max(int(fade * sr), 1)
    pieces: List[np.ndarray] = []
    pos = 0
    for start, end in sorted(runs):
        a, b = int(round(start * sr)), int(round(end * sr))
        keep = int(round(cap * sr))
        if b - a <= keep + n_fade:
            continue
        cut0 = a + keep // 2                       # keep the first half of the cap ...
        cut1 = b - (keep - keep // 2) - n_fade     # ... and the last half, plus the fade the two halves overlap by
        head, tail = y[pos:cut0], y[cut1:cut1 + n_fade]
        if len(head) >= n_fade and len(tail) == n_fade:
            w = np.linspace(1.0, 0.0, n_fade)
            head = np.concatenate([head[:-n_fade], head[-n_fade:] * w + tail * (1 - w)])
            pos = cut1 + n_fade
        else:
            pos = cut1
        pieces.append(head)
    pieces.append(y[pos:])
    return np.concatenate(pieces)

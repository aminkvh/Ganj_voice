# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Re-cut clip boundaries at the aligned silence between lines.

Ganjoor's hemistich marks are set for highlighting and are often late: cutting at them leaves the next line's onset
(or music) at a clip's end and starts the next clip mid-sound. Each unit is aligned inside a window widened around
its marks; the cut between two neighbouring lines goes to the middle of the pause between their aligned words, with
the padding capped so long pauses do not become long silences.
"""
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ganj_corpus.align import WORD_DELIM, accepts, targets_for
from ganj_reading.tokens import MAX_SECONDS

Span = Optional[Tuple[int, int]]

PRE = 150        # ms of lead-in before the first aligned word when the neighbour is unknown
POST = 250       # ms after the last aligned character (CTC ends are early) when the neighbour is unknown
MAX_PRE = 300    # caps when the neighbour is known and the pause is long
MAX_POST = 500
MIN_GAP = 40     # aligned words of two lines closer than this: the alignments disagree, drop both
WIDEN_MS = 1500  # how far the alignment window reaches past the Ganjoor marks on each side
DEFAULT_LEN_MS = int(MAX_SECONDS * 1000)


def recut_bounds(spans: Sequence[Span], contiguous: Sequence[bool]) -> List[Span]:
    """spans[k]: aligned (first word start, last word end) in ms, or None. contiguous[k]: cut k is followed directly by k+1."""
    n = len(spans)
    out: List[Span] = []
    for k, sp in enumerate(spans):
        if sp is None:
            out.append(None)
            continue
        s, e = sp
        prev = spans[k - 1] if k > 0 and contiguous[k - 1] else None
        nxt = spans[k + 1] if k + 1 < n and contiguous[k] else None
        if (prev and prev[1] >= s - MIN_GAP) or (nxt and nxt[0] <= e + MIN_GAP):
            out.append(None)
            continue
        start = max((prev[1] + s) // 2, s - MAX_PRE) if prev else s - PRE
        end = min((e + nxt[0]) // 2, e + MAX_POST) if nxt else e + POST
        out.append((max(0, start), end))
    return out


def _n_words(aligner, text: str) -> int:
    vocab = getattr(aligner, "vocab", None)
    if vocab is None:
        return len(text.split())
    ids, _ = targets_for(text, vocab)
    return sum(1 for k, t in enumerate(ids) if t != vocab[WORD_DELIM] and (k == 0 or ids[k - 1] == vocab[WORD_DELIM]))


def _window(c, widen_ms):
    end = c.end_ms if c.end_ms is not None else c.start_ms + DEFAULT_LEN_MS
    return max(0, c.start_ms - widen_ms), end + widen_ms


def _align(aligner, y, sr, ws, we, text):
    al = aligner.align(y[int(ws * sr / 1000):int(we * sr / 1000)], text)
    return al if al.words and accepts(al) else None


def align_spans(aligner, y: np.ndarray, sr: int, cuts, widen_ms: int = WIDEN_MS):
    """Speech spans (ms) per cut, and which neighbour pairs were aligned jointly.

    Each pair of directly adjacent lines is aligned as one text over one window, so the pause between them comes
    from a single monotonic alignment (separate windows let both lines claim the same audio). A cut whose pair
    alignment fails falls back to aligning its own line alone; that edge then gets fixed pads.
    """
    n = len(cuts)
    starts, ends = [None] * n, [None] * n
    joint = []
    for k in range(n - 1):
        a, b = cuts[k], cuts[k + 1]
        ok = False
        if a.end_ms is not None and a.end_ms == b.start_ms:
            ws, we = _window(a, widen_ms)[0], _window(b, widen_ms)[1]
            al = _align(aligner, y, sr, ws, we, " ".join(a.lines + b.lines))
            na = _n_words(aligner, " ".join(a.lines))
            if al and 0 < na < len(al.words):
                ends[k], starts[k + 1] = ws + int(al.words[na - 1][2] * 1000), ws + int(al.words[na][1] * 1000)
                ok = True
        joint.append(ok)
    spans: List[Span] = []
    for k, c in enumerate(cuts):
        s, e = starts[k], ends[k]
        if s is None or e is None:
            ws, we = _window(c, widen_ms)
            al = _align(aligner, y, sr, ws, we, " ".join(c.lines))
            if al is None:
                spans.append(None)
                continue
            s = s if s is not None else ws + int(al.words[0][1] * 1000)
            e = e if e is not None else ws + int(al.words[-1][2] * 1000)
        spans.append((s, e) if s < e else None)
    return spans, joint


def recut(aligner, y: np.ndarray, sr: int, cuts) -> list:
    """Return [(cut with re-cut start/end) or None] in the order of `cuts` (sorted by start)."""
    cuts = sorted(cuts, key=lambda c: c.start_ms)
    spans, contiguous = align_spans(aligner, y, sr, cuts)
    bounds = recut_bounds(spans, contiguous)
    return [(c, replace(c, start_ms=b[0], end_ms=b[1]) if b else None) for c, b in zip(cuts, bounds)]

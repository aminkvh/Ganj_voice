# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Pace and phrasing of a reading against a reference reading of the same line: where it stops, which
words it runs together, and how long each word lasts.

A line's known text is force-aligned per word (CTC; tools/prosody_compare.py); a pause at a word boundary is the longest
run of silent frames between the two words' spans (CTC leaves blank frames while a sound is still ringing, so a gap
alone is not a pause). Readings are compared boundary by boundary (which pauses agree) and by relative word length.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Word:
    line: int     # which hemistich (line of the item) the word is in
    text: str     # as written
    chars: str    # the letters the aligner knows (half-spaces and punctuation dropped)


def line_words(lines, vocab) -> list:
    out = []
    for i, line in enumerate(lines):
        for w in line.split():
            chars = "".join(ch for ch in w.replace("‌", "") if ch in vocab and ch not in "|-")
            if chars:
                out.append(Word(i, w, chars))
    return out


def silent_run_in_gap(silent: np.ndarray, start: float, end: float, frame_s: float):
    """(start s, end s) of the longest silent run between start and end (s), or None if no frame there is silent."""
    f0 = max(int(round(start / frame_s)), 0)
    run = best = 0
    best_end = None
    for k, s in enumerate(silent[f0:int(round(end / frame_s))], start=f0):
        run = run + 1 if s else 0
        if run > best:
            best, best_end = run, k + 1
    return None if best_end is None else ((best_end - best) * frame_s, best_end * frame_s)


def silence_in_gap(silent: np.ndarray, start: float, end: float, frame_s: float) -> float:
    """Seconds of the longest silent run between start and end (s)."""
    run = silent_run_in_gap(silent, start, end, frame_s)
    return 0.0 if run is None else run[1] - run[0]


def compare_pauses(ref, hyp, min_pause: float) -> dict:
    """Boundary by boundary: pauses (>= min_pause s) in both (hits), only in hyp (extra), only in ref (missing)."""
    r = [p >= min_pause for p in ref]
    h = [p >= min_pause for p in hyp]
    hits = [i for i, (a, b) in enumerate(zip(r, h)) if a and b]
    extra = [i for i, (a, b) in enumerate(zip(r, h)) if b and not a]
    missing = [i for i, (a, b) in enumerate(zip(r, h)) if a and not b]
    precision = len(hits) / (len(hits) + len(extra)) if hits or extra else 1.0
    recall = len(hits) / (len(hits) + len(missing)) if hits or missing else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"hits": hits, "extra": extra, "missing": missing, "precision": precision, "recall": recall, "f1": f1}


def stretch_corr(ref, hyp) -> float:
    """Pearson correlation of log word durations: which words are drawn out, whatever the overall pace."""
    a, b = np.log(np.asarray(ref, float)), np.log(np.asarray(hyp, float))
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])

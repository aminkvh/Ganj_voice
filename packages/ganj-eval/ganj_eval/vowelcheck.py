# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Does a synthesised line say the short vowels its label says? (an example: «آبش», label @ab/š: 6 of 8 draws said
a, and the draw picker kept one that said e: the CTC model writes no fatha, kasre or zamma, so
neither the word check nor the CTC fit sees them.)

Targets: a short vowel (/ e o) of the label between two consonant letters written next to each other (no vowel letter
between, so the vowel is the only sound there), except a fatha right before ک or گ. Window: from the end of the first letter's CTC span to the start of
the second's (ganj_eval.align). Measure: Praat (Burg) F1 and F2, median over the voiced frames of the window.
A VowelModel holds one voice's vowel centres (log Hz, fitted on its own renders: a disguised prompt moves every
formant) and calls a vowel wrong only when another vowel is clearly nearer, so a draw is never traded for a doubt."""
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ganj_corpus.pauses import split_label, strip_pauses
from ganj_corpus.rules_check import text_words
from ganj_reading.g2p import EZAFE, strip_harakat

SHORT = "/eo"
CONSONANT = {"ب": "b", "پ": "p", "ت": "t", "ث": "s", "ج": "j", "چ": "c", "ح": "h", "خ": "x", "د": "d", "ذ": "z",
             "ر": "r", "ز": "z", "ژ": ";", "س": "s", "ش": "š", "ص": "s", "ض": "z", "ط": "t", "ظ": "z", "ع": "@",
             "غ": "q", "ف": "f", "ق": "q", "ک": "k", "ك": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n"}
SKIPPABLE = set(SHORT + "aiu@yvh")     # what vowel letters (ا آ و ی ه ء …) and word starts can stand for
FRONTING = set("کكگ")                  # a fatha right before these is not judged: they pull its F2 up toward e in
                                       # every source (+200 Hz in the human recordings; مگر m/g/r flagged, fine by ear)
MIN_S = 0.03                           # a window shorter than this is not measured
PAD_S = 0.05                           # context Praat gets on each side of a window
MAX_FORMANT = 5500.0
MIN_VOICED = 3                         # voiced 5 ms frames a measurement needs
MARGIN = 2.0                           # wrong: the label's vowel is this many times farther than another vowel
MIN_D = 2.0                            # and farther than this (in spreads) from its own centre
SPREAD_FLOOR = 0.06                    # log-Hz spread used when a voice's clouds are tighter than this


def letters(word: str) -> str:
    return strip_harakat(word).replace("‌", "").replace("ٔ", "")


def vowel_targets(word: str, label: str) -> List[Tuple[int, int, str]]:
    """(letter i, letter i + 1, short vowel) for each pair of adjacent consonant letters whose label has one short
    vowel between their sounds; [] when the label does not follow the letters."""
    ph = label.replace(EZAFE, "")
    out, k, prev = [], 0, None           # prev: (letter index, label index after its sound) of a consonant letter
    for i, ch in enumerate(letters(word)):
        c = CONSONANT.get(ch)
        if c is None:                    # a vowel letter (or one the table does not know): no pair across it
            prev = None
            continue
        start = k
        while k < len(ph) and ph[k] != c and ph[k] in SKIPPABLE:
            k += 1
        if k >= len(ph) or ph[k] != c:
            return []
        between = ph[start:k]
        k += 2 if ph[k + 1:k + 2] == c else 1     # a tashdid doubles the sound of one letter
        if prev is not None and prev[0] == i - 1 and len(between) == 1 and between in SHORT \
                and not (between == "/" and ch in FRONTING):
            out.append((i - 1, i, between))
        prev = (i, k)
    if any(ch not in SKIPPABLE for ch in ph[k:]):
        return []
    return out


def line_vowel_targets(lines: Sequence[str], label: str) -> List[Tuple[int, int, int, str]]:
    """(word index across the line's hemistichs, letter i, letter j, vowel); [] when the label's words do not line
    up with the text's."""
    lines = [l for l in lines if l.strip()]
    _, parts = split_label(strip_pauses(label))
    if len(parts) != len(lines):
        return []
    out, base = [], 0
    for line, part in zip(lines, parts):
        words = text_words(line)
        part = [p for p in part if p]
        if len(words) != len(part):
            return []
        for w, (word, ph) in enumerate(zip(words, part)):
            out += [(base + w, i, j, v) for i, j, v in vowel_targets(word, ph)]
        base += len(words)
    return out


def vowel_window(chars: Sequence[Tuple[float, float]], i: int, j: int) -> Optional[Tuple[float, float]]:
    a, b = chars[i][1], chars[j][0]
    return (a, b) if b - a >= MIN_S else None


def formants(snd, window: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    """Median (F1, F2) in Hz over the voiced frames of `window` of a parselmouth Sound; None if too few."""
    t0, t1 = window
    lo = max(0.0, t0 - PAD_S)
    part = snd.extract_part(lo, t1 + PAD_S)
    fm = part.to_formant_burg(max_number_of_formants=5, maximum_formant=MAX_FORMANT)
    pitch = part.to_pitch()
    f1, f2 = [], []
    for t in np.arange(t0 - lo, t1 - lo, 0.005):
        if np.isnan(pitch.get_value_at_time(t)):
            continue
        a, b = fm.get_value_at_time(1, t), fm.get_value_at_time(2, t)
        if not (np.isnan(a) or np.isnan(b)):
            f1.append(a)
            f2.append(b)
    if len(f1) < MIN_VOICED:
        return None
    return float(np.median(f1)), float(np.median(f2))


def measure_vowels(y: np.ndarray, sr: int, words: Sequence[str], chars: Sequence[Sequence[Tuple[float, float]]],
                   targets: Sequence[Tuple[int, int, int, str]], aligner_letters) -> List[dict]:
    """{"word", "i", "j", "vowel", "f1", "f2"} for each target that can be measured: its word's aligner letters
    (`aligner_letters(word)`) must be its letters, so the spans in `chars` (per word, per letter) line up."""
    import parselmouth
    snd = parselmouth.Sound(np.asarray(y, dtype=np.float64), sr)
    out = []
    for w, i, j, v in targets:
        if aligner_letters(words[w]) != letters(words[w]):
            continue
        win = vowel_window(chars[w], i, j)
        f = formants(snd, win) if win else None
        if f:
            out.append({"word": w, "i": i, "j": j, "vowel": v, "f1": round(f[0], 1), "f2": round(f[1], 1)})
    return out


class VowelModel:
    """Centres of / e o in (log F1, log F2) and one spread per axis, pooled over the three."""

    def __init__(self, centres: dict, spread: Sequence[float]):
        self.centres = {v: np.asarray(c, dtype=float) for v, c in centres.items()}
        self.spread = np.asarray(spread, dtype=float)

    @classmethod
    def fit(cls, points: Sequence[Tuple[str, float, float]]) -> "VowelModel":
        by = {}
        for v, f1, f2 in points:
            by.setdefault(v, []).append((np.log(f1), np.log(f2)))
        centres = {v: np.median(np.array(xs), axis=0) for v, xs in by.items()}
        dev = np.concatenate([np.abs(np.array(xs) - centres[v]) for v, xs in by.items()])
        spread = np.maximum(1.4826 * np.median(dev, axis=0), SPREAD_FLOOR)
        return cls(centres, spread)

    def distances(self, f1: float, f2: float) -> dict:
        x = np.log([f1, f2])
        return {v: float(np.linalg.norm((x - c) / self.spread)) for v, c in self.centres.items()}

    def wrong(self, vowel: str, f1: float, f2: float) -> bool:
        d = self.distances(f1, f2)
        if vowel not in d:
            return False
        other = min(dv for v, dv in d.items() if v != vowel)
        return d[vowel] > MIN_D and d[vowel] > MARGIN * other

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"centres": {v: c.tolist() for v, c in self.centres.items()},
                                          "spread": self.spread.tolist(), "margin": MARGIN, "min_d": MIN_D}))

    @classmethod
    def load(cls, path: Path) -> "VowelModel":
        d = json.loads(Path(path).read_text())
        return cls(d["centres"], d["spread"])

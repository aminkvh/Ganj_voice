# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Aruz scansion of GE2PE phonemes against a poem's metre.

A hemistich whose phonemes scan in the poem's metre is very likely pronounced right (short vowels, ezafe, word
joins); one that does not points at a G2P error. Used as an accuracy metric that needs no listening, and to find
the words that most often break the metre.

Weights: U light, - heavy. Short vowels are / e o (the ezafe is e followed by the marker 1); long vowels are a i u. A syllable takes the one
consonant before its vowel as onset. CV = U, CVC and CV̄ = -, CV̄C and CVCC = -U (overlong).
Licences honoured (the common ekhtiyarat): the last syllable of a hemistich is free; a word-initial glottal stop
(@) may drop and link the previous consonant (bešnav-in); an n after a long vowel may be silent (an, in, un); an
open syllable at a word end may be read long or short; a long vowel before another vowel may be short; the first foot of ramal may be fa'ilatun for fa'ilatun.
"""
from dataclasses import dataclass, field
from itertools import product
from typing import FrozenSet, List, Sequence, Set

SHORT = frozenset("/eo")
MARKERS = frozenset("1")  # GE2PE ezafe marker: follows the ezafe vowel, is not itself a sound
LONG = frozenset("aiu")
VOWELS = SHORT | LONG
CONSONANTS = frozenset("bcdfghjklmnpqrstvxyzš@;")

FEET = {
    "فاعلاتن": ["-U--"], "فعولن": ["U--"], "فعلاتن": ["UU--"], "مفاعلن": ["U-U-"], "مفاعیلن": ["U---"],
    "فاعلن": ["-U-"], "مفعول": ["--U"], "فعلن": ["UU-", "--"], "مفاعیل": ["U--U"], "مستفعلن": ["--U-"],
    "مفتعلن": ["-UU-"], "فعل": ["U-"], "فاعلات": ["-U-U"], "فعلات": ["UU-U"], "فع": ["-"],
    "متفاعلن": ["UU-U-"], "متفاعلاتن": ["UU-U--"], "مستفعلتن": ["--UU-"],
}
MAX_ELISION_WORDS = 6


@dataclass(frozen=True)
class Syllable:
    weight: str
    options: FrozenSet[str] = field(default_factory=frozenset)


def metre_patterns(metre: str) -> Set[str]:
    feet = metre.split("(")[0].split()
    if not feet or any(f not in FEET for f in feet):
        return set()
    alts = [FEET[f] for f in feet]
    if feet[0] in ("فاعلاتن", "فعلاتن"):  # the first foot of ramal/khafif may be either
        alts[0] = FEET["فاعلاتن"] + FEET["فعلاتن"]
    return {"".join(p) for p in product(*alts)}


def _clean(word: str) -> str:
    return "".join(c for c in word if c in VOWELS or c in CONSONANTS)


def syllables(seq: str, word_ends: Sequence[int] = ()) -> List[Syllable]:
    """Syllables of a phoneme string with word boundaries removed; word_ends are indices of word-final phonemes."""
    keep = [i for i, c in enumerate(seq) if c not in MARKERS]
    ends = {k for k, i in enumerate(keep) if i in set(word_ends) or (i + 1 < len(seq) and seq[i + 1] in MARKERS and i + 1 in set(word_ends))}
    seq = "".join(seq[i] for i in keep)
    vs = [i for i, c in enumerate(seq) if c in VOWELS]
    out = []
    for k, v in enumerate(vs):
        nxt_onset = vs[k + 1] - 1 if k + 1 < len(vs) and seq[vs[k + 1] - 1] not in VOWELS else (vs[k + 1] if k + 1 < len(vs) else len(seq))
        coda = seq[v + 1:nxt_onset]
        long = seq[v] in LONG
        if long:
            w = "-" if not coda else "-U"
        else:
            w = "U" if not coda else ("-" if len(coda) == 1 else "-U")
        opts = {w}
        if long and not coda and k + 1 < len(vs) and (nxt_onset == vs[k + 1] or seq[nxt_onset] in "y@"):
            opts.add("U")  # a long vowel before another vowel (hiatus, often written with a glide y) may be short
        if long and coda == "n":
            opts.add("-")  # silent nun after a long vowel
        last = v + len(coda)
        if not coda and last in ends:
            opts |= {"U", "-"}  # open syllable at a word end: either length
        out.append(Syllable(w, frozenset(opts)))
    return out


def _variants(words: List[str]):
    """Phoneme strings for the line with each droppable word-initial @ kept or dropped."""
    drop_at = [i for i in range(1, len(words)) if words[i].startswith("@") and words[i - 1] and words[i - 1][-1] in CONSONANTS]
    drop_at = drop_at[:MAX_ELISION_WORDS]
    for mask in product((False, True), repeat=len(drop_at)):
        ws = list(words)
        for i, d in zip(drop_at, mask):
            if d:
                ws[i] = ws[i][1:]
        seq, ends, pos = "", [], 0
        for w in ws:
            seq += w
            pos += len(w)
            if w:
                ends.append(pos - 1)
        yield seq, ends


def _matches(syls: List[Syllable], pattern: str) -> bool:
    if not syls:
        return False
    cur = {0}
    for s in syls[:-1]:
        cur = {p + len(o) for p in cur for o in s.options if pattern.startswith(o, p)}
        if not cur:
            return False
    return len(pattern) - 1 in cur  # the hemistich's last syllable is free


def scans(phonemes: str, metre: str) -> bool:
    patterns = metre_patterns(metre)
    words = [_clean(w) for w in phonemes.replace("$", "š").split(" ")]
    words = [w for w in words if w]
    if not patterns or not words:
        return False
    for seq, ends in _variants(words):
        syls = syllables(seq, ends)
        if any(_matches(syls, p) for p in patterns):
            return True
    return False

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Pause tokens in the labels: a model that reads the words right can still stop in the wrong places and run words
together that should be kept apart.

The model's input had no pause mark (the G2P drops punctuation), so it learned to stop wherever reciters tend to,
about one pause per line more than a human reader. Training labels get a token at every word boundary where the
reciter is silent (short from 150 ms, long from 450 ms, measured by forced alignment); no token means the words run
on. At reading time the tokens are placed by rule (rule_tokens), so where the voice stops is decided, not guessed.
Tokens sit between words (" ⟨S⟩ ") or straight before the hemistich mark ("w⟨L⟩<HEMI>"); they are private-use code
points outside the style/metre range, appended to the token list so no existing id moves.
"""
from ganj_corpus.relabel import PREFIX_CHARS
from ganj_reading import tokens as T
from ganj_reading.g2p import G2P, LONG_MARKS, SHORT_MARKS, punctuation_after_words, read_vav  # noqa: F401 (re-exported)

PAUSE_SHORT, PAUSE_LONG = "", ""
SHORT_S, LONG_S = 0.15, 0.45


def pause_token(seconds: float) -> str:
    return PAUSE_LONG if seconds >= LONG_S else PAUSE_SHORT if seconds >= SHORT_S else ""


def split_label(ph: str):
    """(prefix of style/metre tokens, [[words of each hemistich]])."""
    k = len(ph) - len(ph.lstrip(PREFIX_CHARS))
    return ph[:k], [part.split(" ") for part in ph[k:].split(T.HEMI)]


def mark_pauses(ph: str, toks: list) -> str:
    """Put toks[i] ("" or a pause token) at the i-th word boundary of the label, counted across hemistichs."""
    prefix, lines = split_label(ph)
    if len(toks) != sum(len(words) for words in lines) - 1:
        raise ValueError(f"{len(toks)} boundaries given for {sum(len(w) for w in lines)} words")
    it, out = iter(toks), []
    for li, words in enumerate(lines):
        s = words[0]
        for w in words[1:]:
            t = next(it)
            s += (f" {t} " if t else " ") + w
        if li < len(lines) - 1:
            s += next(it)
        out.append(s)
    return prefix + T.HEMI.join(out)


def strip_pauses(ph: str) -> str:
    for t in (PAUSE_SHORT, PAUSE_LONG):
        ph = ph.replace(f" {t} ", " ").replace(t, "")
    return ph


def rule_tokens(lines: list) -> list:
    """Reading-time pauses: long at every hemistich break and after . ! ؟, short after ، ؛ : and nowhere else."""
    lines = [l for l in lines if l.strip()]
    toks = []
    for li, line in enumerate(lines):
        n = len(G2P.words(line))
        marks = punctuation_after_words(line)
        if len(marks) != n:
            marks = [""] * n
        for m in marks[:-1]:
            toks.append(PAUSE_LONG if any(c in LONG_MARKS for c in m) else PAUSE_SHORT if m else "")
        if li < len(lines) - 1:
            toks.append(PAUSE_LONG)
    return toks


def rule_label(ph: str, lines: list) -> str:
    """A reading label with rule_tokens; when the label's words do not line up with the text's, only the hemistich
    breaks get their long pause (a line must never run on into the next)."""
    try:
        return mark_pauses(ph, rule_tokens(lines))
    except ValueError:
        prefix, parts = split_label(ph)
        return prefix + (PAUSE_LONG + T.HEMI).join(" ".join(words) for words in parts)


def reading_label(ph: str, lines: list, prose: bool, pauses: bool = True, va: bool = False) -> str:
    """The label a line is read with: with va, the «و» rule (g2p.read_vav) on each hemistich whose words line up;
    with pauses (a voice trained with the pause tokens), the reading-time pauses (rule_label)."""
    lines = [l for l in lines if l.strip()]
    if va:
        prefix, parts = split_label(ph)
        if len(parts) == len(lines):
            ph = prefix + T.HEMI.join(" ".join(read_vav(line, words, prose)) for line, words in zip(lines, parts))
    return rule_label(ph, lines) if pauses else ph

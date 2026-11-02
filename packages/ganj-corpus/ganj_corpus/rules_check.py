# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""The rules every generation must carry, from one label set to the next: the text's own marks (a written ezafe is
read, tashdid doubles, sukun takes no vowel and no ezafe), the per-line readings (audit_g2p/review/overrides_*.json),
and reading lists built from the current labels (a list copied from older labels is stale). Pause tokens are ignored
by every check."""
from dataclasses import dataclass

from ganj_corpus.pauses import split_label, strip_pauses
from ganj_corpus.relabel import _same_word
from ganj_reading.g2p import (EZAFE, G2P, HAMZA_ABOVE, HEH_HAMZA, NO_EZAFE, SUKUN, harakat_agrees, strip_harakat,
                                 written_ezafe)

UNALIGNED = "label words do not line up"


@dataclass(frozen=True)
class Violation:
    line: int
    word: str
    label: str
    rule: str


def text_words(line: str) -> list:
    return G2P.words(line.replace(HEH_HAMZA, "ه" + HAMZA_ABOVE))   # as G2P._prepare reads ۀ


def ezafe_read(word: str, ph: str) -> bool:
    """A written ezafe is read: the label carries the ezafe, or the word never takes one and the kasra is its own
    vowel («زِ» = ze)."""
    return ph.endswith(EZAFE) or (strip_harakat(word) in NO_EZAFE and ph.endswith("e"))


def check_label(lines, label: str) -> list:
    lines = [l for l in lines if l.strip()]
    _, parts = split_label(strip_pauses(label))
    if len(parts) != len(lines):
        return [Violation(-1, "", "", UNALIGNED)]
    out = []
    for li, (part, line) in enumerate(zip(parts, lines)):
        words = text_words(line)
        if len(words) != len(part):
            out.append(Violation(li, "", "", UNALIGNED))
            continue
        for i, (w, ph) in enumerate(zip(words, part)):
            if w.endswith(SUKUN) and ph.endswith(EZAFE):
                out.append(Violation(li, w, ph, "sukun: no ezafe"))
            elif not harakat_agrees(w, ph):
                out.append(Violation(li, w, ph, "written vowels, tashdid or sukun"))
            elif written_ezafe(w) and i < len(words) - 1 and not ezafe_read(w, ph):
                out.append(Violation(li, w, ph, "written ezafe"))
    return out


def missing_overrides(labels: dict, rows: dict, overrides: list) -> list:
    """(id, word, expected label, label found) for every per-line reading not in the labels (ids absent: skipped)."""
    out = []
    for o in overrides:
        if o["id"] not in labels:
            continue
        lines = [l for l in rows.get(o["id"], []) if l.strip()]
        _, parts = split_label(strip_pauses(labels[o["id"]]))
        if o["line"] >= min(len(lines), len(parts)):
            out.append((o["id"], o["text"], o["label"], "line missing"))
            continue
        words, labs = text_words(lines[o["line"]]), parts[o["line"]]
        i = o["word"] if o["word"] < len(words) and _same_word(words[o["word"]], o["text"]) else None
        if i is None:
            hits = [j for j, w in enumerate(words) if _same_word(w, o["text"])]
            i = hits[0] if len(hits) == 1 else None
        got = labs[i] if i is not None and len(labs) == len(words) else "word not found"
        if got != o["label"]:
            out.append((o["id"], o["text"], o["label"], got))
    return out


def stale_keys(list_labels: dict, current: dict) -> list:
    """Keys whose label (pause tokens aside) is not the current one."""
    return [k for k, v in list_labels.items() if k in current and strip_pauses(v) != current[k]]


REPORT_ONLY = ("arabic", "turkish")   # read as they are: the Persian G2P cannot follow their marks


def triage(key: str, cls: str, violations: list, waivers: list) -> tuple:
    """(failing, reported): non-Persian lines and waived words (each waiver names a line id, a word or "*" for the whole
    line, and why) are reported without failing a generation."""
    waived = {(w["id"], w["word"]) for w in waivers}
    if (key, "*") in waived:
        return [], list(violations)
    fail, report = [], []
    for v in violations:
        (report if cls in REPORT_ONLY or (key, v.word) in waived or v.rule == UNALIGNED else fail).append(v)
    return fail, report

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Metre-validated lexicon entries.

For each candidate word -> phonemes, substitute it (keeping the model's ezafe) into every corpus hemistich where the
word sits at an aligned position, rescan, and compare the scansion pass rate before and after. An entry is kept
only if it helps clearly: +0.15 or more on at least 10 hemistichs.
"""
import json
import sys

from ganj_reading.g2p import G2P, _with_ezafe, strip_harakat
from ganj_reading.scansion import metre_patterns, scans
from tools.scansion_report import hemistichs

MIN_N, MIN_GAIN = 10, 0.15


def main(clips_path, candidates_path, out_path):
    cands = json.load(open(candidates_path, encoding="utf-8"))
    items = []
    for l in open(clips_path, encoding="utf-8"):
        r = json.loads(l)
        if r["kind"] != "verse" or not metre_patterns(r.get("metre") or ""):
            continue
        for ph, text in hemistichs(r):
            words = [strip_harakat(w) for w in G2P.words(text)]
            pws = ph.split()
            if len(words) == len(pws):
                items.append((words, pws, r["metre"]))
    result = {}
    for word, new in cands.items():
        before = after = n = 0
        for words, pws, metre in items:
            if word not in words:
                continue
            n += 1
            before += scans(" ".join(pws), metre)
            alt = [_with_ezafe(new, p.endswith("1")) if w == word else p for w, p in zip(words, pws)]
            after += scans(" ".join(alt), metre)
        b, a = before / max(1, n), after / max(1, n)
        result[word] = {"phonemes": new, "n": n, "before": round(b, 3), "after": round(a, 3), "keep": n >= MIN_N and a - b >= MIN_GAIN}
        print(f"{'KEEP' if result[word]['keep'] else 'drop'}  {word}={new}  n={n}  {b:.2f} -> {a:.2f}")
    json.dump(result, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:4])

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Freeze the metre-token list: existing metres keep their place, new ones are appended by frequency.

Usage: python -m tools.freeze_metres POEMS_DIR packages/ganj-reading/ganj_reading/metres.json
"""
import glob
import json
import sys
from collections import Counter

from ganj_reading.tokens import METRE_BASE

LIMIT = 150  # 100 older symbols + 150 metre tokens stay under Piper's 256-symbol table


def merge_metres(existing, counts: Counter, limit: int = LIMIT):
    new = [m for m, _ in counts.most_common() if m not in existing]
    return (list(existing) + new)[:max(limit, len(existing))]


def main(poems_dir, path):
    from ganj_corpus.build import metre_of
    counts = Counter()
    for p in glob.glob(f"{poems_dir}/*.json"):
        try:
            m = metre_of(json.load(open(p, encoding="utf-8")))
        except (OSError, ValueError):
            continue
        if m:
            counts[m.split("(")[0].strip()] += 1
    existing = json.load(open(path, encoding="utf-8"))
    merged = merge_metres(existing, counts)
    json.dump(merged, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)
    print(f"metres: {len(existing)} -> {len(merged)} (tokens U+{METRE_BASE:04X}..U+{METRE_BASE + len(merged) - 1:04X})")


if __name__ == "__main__":
    main(*sys.argv[1:3])

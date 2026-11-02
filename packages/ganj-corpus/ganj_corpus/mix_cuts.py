# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Shuffle several lhotse cut manifests into one, so a fine-tune sees the sources interleaved (lhotse's own shuffle
buffer is too small to mix two concatenated files). python -m ganj_corpus.mix_cuts OUT IN [IN ...]"""
import gzip
import random
import sys
from pathlib import Path


def mix(inputs, out, seed: int = 0) -> int:
    lines = [l for p in inputs for l in gzip.open(p, "rt", encoding="utf-8") if l.strip()]
    random.Random(seed).shuffle(lines)
    with gzip.open(out, "wt", encoding="utf-8") as f:
        f.writelines(lines)
    return len(lines)


if __name__ == "__main__":
    print("mixed", mix([Path(p) for p in sys.argv[2:]], Path(sys.argv[1])))

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""The letter n-gram weights of packages/ganj-reading/ganj_reading/language.py (naive Bayes, add-one: log P(gram | Arabic) - log P(gram |
untagged)) from Ganjoor's hemistich tags in $GANJ_DATA/poems (languageId 2 against none or 1), the held-out score
(trained on even poem ids, scored on odd), and the app's fixture (packages/ganj-reading/tests/fixtures/hemistich_language.json: Hafez's
ghazal 1, a sample of tagged and untagged hemistichs and the untagged ones the detector skips, each with its answer).
  python tools/train_arabic_detector.py [--data $GANJ_DATA]"""
import argparse
import collections
import glob
import json
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ganj_reading import language as L  # noqa: E402


def hemistichs(data: Path) -> list:
    """(poem id, poem title, text, Ganjoor languageId) for every hemistich (versePosition 0 / 1)."""
    out = []
    for p in sorted(glob.glob(str(data / "poems" / "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        for v in d.get("verses") or []:
            if v.get("versePosition") in (0, 1):
                out.append((d["id"], d["fullTitle"], v["text"], v.get("languageId")))
    return out


def grams(text: str) -> list:
    return [g for w in L.WORD.findall(L.normalise(text)) for g in L.word_grams(w)]


def train(rows: list) -> dict:
    count = {True: collections.Counter(), False: collections.Counter()}
    for _, _, text, lid in rows:
        if lid in (None, 1, 2):
            count[lid == 2].update(grams(text))
    vocab = set(count[True]) | set(count[False])
    total = {y: sum(count[y].values()) + len(vocab) for y in count}
    return {g: round(math.log((count[True][g] + 1) / total[True]) - math.log((count[False][g] + 1) / total[False]), 4)
            for g in sorted(vocab)}


def held_out(rows: list) -> str:
    w = train([r for r in rows if r[0] % 2 == 0])
    test = [r for r in rows if r[0] % 2 == 1 and r[3] in (None, 1, 2)]
    tagged = [r for r in test if r[3] == 2]
    caught = sum(L.looks_arabic(r[2], w) for r in tagged)
    flagged = sum(L.looks_arabic(r[2], w) for r in test if r[3] != 2)
    return (f"held out (odd poem ids): {caught} of {len(tagged)} tagged Arabic hemistichs found ({caught / len(tagged):.1%}); "
            f"{flagged} of {len(test) - len(tagged)} untagged ones flagged")


def fixture(rows: list, rng: random.Random) -> list:
    ghazal = [r for r in rows if r[0] == 2130]
    tagged = [r for r in rows if r[3] not in (None, 1)]
    untagged = [r for r in rows if r[3] in (None, 1)]
    flagged = [r for r in untagged if L.looks_arabic(r[2])]
    pick = ghazal + rng.sample(tagged, 40) + rng.sample(flagged, min(60, len(flagged))) + rng.sample(untagged, 200)
    cases = []
    for r in pick:
        for lid in ([r[3], None] if r[3] is not None else [None]):
            cases.append({"poem": r[0], "text": r[2], "language_id": lid, "skip": L.skip_hemistich(r[2], lid)})
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.environ.get("GANJ_DATA"))
    a = ap.parse_args(argv)
    rows = hemistichs(Path(a.data))
    print(f"{len(rows)} hemistichs, by languageId: {dict(collections.Counter(r[3] for r in rows))}")
    print(held_out(rows))
    w = train(rows)
    (ROOT / "packages" / "ganj-reading" / "ganj_reading" / "arabic_ngrams.json").write_text(json.dumps(w, ensure_ascii=False, separators=(",", ":")),
                                                                     encoding="utf-8")
    L.weights.cache_clear()
    untagged = [r for r in rows if r[3] in (None, 1)]
    print(f"weights: {len(w)} n-grams; untagged hemistichs the detector skips: {sum(L.looks_arabic(r[2]) for r in untagged)}")
    out = ROOT / "packages" / "ganj-reading" / "tests" / "fixtures" / "hemistich_language.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(fixture(rows, random.Random(1)), ensure_ascii=False, indent=0), encoding="utf-8")
    print("fixture:", out)


if __name__ == "__main__":
    main()

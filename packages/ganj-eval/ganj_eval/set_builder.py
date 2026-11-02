# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Fixed evaluation set: 5 text classes × n units from held-out poems."""
import json
import random
import re
from pathlib import Path

from ganj_corpus.select import split_by_poem

HARAKAT = re.compile("[ً-ٰٟ]")
CLASSES = ("prose", "arabic", "vowelled", "masnavi", "plain")


def classify(row: dict) -> str:
    if row["kind"] == "prose":
        return "prose"
    langs = [str(l).lower() for l in row.get("langs", []) if l]
    if any(l == "3" or l.startswith(("az", "tr")) for l in langs):
        return "turkish"  # Ganjoor languageId 3: Azeri Turkish — not Persian, kept out of the eval set
    if any(not l.startswith("fa") for l in langs):
        return "arabic"  # languageId 2
    if any(HARAKAT.search(l) for l in row["lines"]):
        return "vowelled"
    if "مثنوی" in (row.get("title") or ""):
        return "masnavi"
    return "plain"


def build_set(rows, val_poems, n_per_class: int = 10, seed: int = 1) -> list:
    rng = random.Random(seed)
    pool = {c: [] for c in CLASSES}
    for r in rows:
        c = classify(r)
        if r["poem_id"] in val_poems and c in pool:
            pool[c].append(r)
    out = []
    for c in CLASSES:
        rng.shuffle(pool[c])
        for r in pool[c][:n_per_class]:
            unit = r["unit"] if "unit" in r else Path(r["clip"]).stem  # clips.jsonl carries the unit in the clip name
            out.append({"key": f"{r['rec_id']}-{unit}", "cls": c, "lines": r["lines"], "phonemes": r["phonemes"],
                        "human": r["clip"], "rec_id": r["rec_id"], "unit": unit})
    return out


def main(data: Path):
    rows = [json.loads(l) for l in open(data / "clips.jsonl", encoding="utf-8")]
    sp = json.loads((data / "speakers.json").read_text(encoding="utf-8"))
    trainable = [r for r in rows if r["artist"] in sp]
    # clips of reciters below the speaker minimum are never trained on either; they widen the
    # held-out pool so the thin prose/Arabic classes are represented.
    unseen = [r for r in rows if r["artist"] not in sp]
    for r in trainable:
        r["poemId"] = r["poem_id"]
    _, val = split_by_poem(trainable, val_fraction=0.02)
    held_out = {r["poem_id"] for r in val} | {r["poem_id"] for r in unseen}
    train_poems = {r["poem_id"] for r in trainable} - {r["poem_id"] for r in val}
    pool = [r for r in rows if r["poem_id"] in held_out and r["poem_id"] not in train_poems]
    s = build_set(pool, {r["poem_id"] for r in pool})
    (data / "eval").mkdir(exist_ok=True)
    with open(data / "eval" / "set.jsonl", "w", encoding="utf-8") as f:
        for r in s:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(r["cls"] for r in s))


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]))

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Rare-pattern oversampling for a ZipVoice cut manifest.

A clip's rarity is the corpus count of its rarest within-word phoneme n-gram (the model gets about that many chances
to learn it — خمیازه's "amy" occurs in 13 clips). Clips whose rarest n-gram is seen fewer than `threshold` times are
repeated, up to `max_repeat` copies in total, rarer clips more; copies get new ids (<id>_rep<k>).

python -m ganj_corpus.oversample IN.jsonl.gz OUT.jsonl.gz [threshold] [max_repeat]
"""
import collections
import copy
import gzip
import json
import math
import sys

N = 3


def _words(text: str):
    return ["".join(c for c in w if not "" <= c <= "") for w in text.split()]


def ngrams(text: str, n: int = N):
    return [w[i:i + n] for w in _words(text) for i in range(len(w) - n + 1)]


def rare_score(text: str, counts: dict, n: int = N) -> int:
    grams = ngrams(text, n)
    return min((counts.get(g, 0) for g in grams), default=10 ** 9)


def repeats(score: int, threshold: int = 50, max_repeat: int = 3) -> int:
    if score >= threshold or max_repeat <= 1:
        return 1
    step = threshold / (max_repeat - 1)
    return 1 + min(max_repeat - 1, math.ceil((threshold - score) / step))


def _text(cut: dict) -> str:
    return " ".join(s.get("text", "") for s in cut.get("supervisions", []))


def oversample(src, out, threshold: int = 50, max_repeat: int = 3) -> int:
    cuts = [json.loads(l) for l in gzip.open(src, "rt", encoding="utf-8") if l.strip()]
    counts = collections.Counter(g for c in cuts for g in set(ngrams(_text(c))))   # clips containing the n-gram
    added = 0
    with gzip.open(out, "wt", encoding="utf-8") as f:
        for c in cuts:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            for k in range(1, repeats(rare_score(_text(c), counts), threshold, max_repeat)):
                d = copy.deepcopy(c)
                d["id"] = f"{c['id']}_rep{k}"
                for s in d.get("supervisions", []):
                    s["id"] = f"{s.get('id', c['id'])}_rep{k}"
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
                added += 1
    return added


if __name__ == "__main__":
    a = sys.argv[1:]
    print("added", oversample(a[0], a[1], *(int(x) for x in a[2:4])))

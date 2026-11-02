# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Metre-scansion pass rate of a corpus's phonemes and the words most tied to failures."""
import collections
import json
import sys

from ganj_reading.scansion import metre_patterns, scans
from ganj_reading.tokens import HEMI, STYLE


def hemistichs(row):
    ph = row["phonemes"]
    for s in STYLE:
        if s != HEMI:
            ph = ph.replace(s, "")
    return list(zip(ph.split(HEMI), row["lines"]))


def main(clips_path, out_path):
    tot = collections.Counter()
    by_metre = collections.defaultdict(collections.Counter)
    word_fail, word_all = collections.Counter(), collections.Counter()
    examples = []
    for l in open(clips_path, encoding="utf-8"):
        r = json.loads(l)
        if r["kind"] != "verse" or not metre_patterns(r.get("metre") or ""):
            continue
        hs = hemistichs(r)
        if len(hs) != len(r["lines"]):
            continue
        for ph, text in hs:
            ok = scans(ph, r["metre"])
            tot[ok] += 1
            by_metre[r["metre"].split("(")[0].strip()][ok] += 1
            pairs = list(zip(text.split(), ph.split()))
            for w, p in pairs:
                word_all[(w, p)] += 1
                if not ok:
                    word_fail[(w, p)] += 1
            if not ok and len(examples) < 40:
                examples.append({"text": text, "phonemes": ph, "metre": r["metre"]})
    rate = tot[True] / max(1, sum(tot.values()))
    worst = sorted(((word_fail[k] / word_all[k], word_all[k], k) for k in word_all if word_all[k] >= 15), reverse=True)[:60]
    rep = {"hemistichs": sum(tot.values()), "pass_rate": round(rate, 4),
           "by_metre": {m: round(c[True] / sum(c.values()), 3) for m, c in sorted(by_metre.items(), key=lambda kv: -sum(kv[1].values()))},
           "worst_words": [{"word": k[0], "phonemes": k[1], "fail_share": round(f, 2), "n": n} for f, n, k in worst],
           "examples": examples}
    json.dump(rep, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"hemistichs {rep['hemistichs']}  pass rate {rate:.3f}")
    for m, v in list(rep["by_metre"].items())[:12]:
        print(f"  {v:.2f}  {m}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

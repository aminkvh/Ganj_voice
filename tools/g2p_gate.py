# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""G2P gate on the 140-row hand audit: how many flagged label errors a G2P version fixes, and how many
unflagged (audited-correct) words it changes. Run before every relabel; the unflagged-changed count is the precision
guard. Usage:
  python tools/g2p_gate.py [--module ganj_reading.g2p] [--model DIR] [--out gate.json]
"""
import argparse
import importlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from ganj_reading import tokens as T  # noqa: E402
from ganj_reading.g2p import strip_harakat  # noqa: E402

from tools._env import env_path  # noqa: E402
DATA = env_path("GANJ_DATA")
SKIP_TYPES = {"non_persian_line", "extra_word"}
SEP = re.compile("[\\s" + re.escape(T.HEMI + T.BEYT) + "]+")


def label_words(ph: str) -> list:
    """Words of a label: style and metre tokens dropped, hemistich and beyt tokens are word breaks."""
    ph = "".join(c for c in ph if not (0xE000 <= ord(c) <= 0xE0FF) or c in T.HEMI + T.BEYT)
    return [w for w in SEP.split(ph) if w]


def score(rows: list, new_labels: dict, errors: list, words_of) -> dict:
    """rows: audit rows (id, lines, phonemes = the audited label); new_labels: id -> new label; errors: the audit's
    flagged words (id, word, got, expected, type); words_of: line -> its G2P words."""
    flagged = {}
    for e in errors:
        if e["type"] not in SKIP_TYPES:
            flagged.setdefault(e["id"], []).append(e)
    out = {"rows": len(rows), "unaligned_rows": 0, "old_unaligned_rows": 0, "flagged_checkable": 0, "flagged_fixed": 0,
           "unflagged_words": 0, "unflagged_changed": 0, "changed": [], "still_wrong": []}
    for r in rows:
        words = [w for line in r["lines"] for w in words_of(line)]
        new = label_words(new_labels.get(r["id"], ""))
        if len(new) != len(words):
            out["unaligned_rows"] += 1
            continue
        hits = set()
        for e in flagged.get(r["id"], []):  # found by the word itself: the audited label may split words differently
            i = next((i for i, w in enumerate(words) if i not in hits and e["word"] in (w, strip_harakat(w))), None)
            if i is None:
                continue
            hits.add(i)
            out["flagged_checkable"] += 1
            if new[i] == e["expected"]:
                out["flagged_fixed"] += 1
            else:
                out["still_wrong"].append([r["id"], words[i], new[i], e["expected"], e["type"]])
        old = label_words(r["phonemes"])
        if len(old) != len(words):
            out["old_unaligned_rows"] += 1
            continue
        for i, (w, o, n) in enumerate(zip(words, old, new)):
            if i in hits:
                continue
            out["unflagged_words"] += 1
            if n != o:
                out["unflagged_changed"] += 1
                out["changed"].append([r["id"], w, o, n])
    return out


def diff(rows: list, a: dict, b: dict, words_of) -> list:
    """Every word where label sets a and b disagree, on rows where both align with the text: (id, word, a, b)."""
    out = []
    for r in rows:
        words = [w for line in r["lines"] for w in words_of(line)]
        wa, wb = label_words(a.get(r["id"], "")), label_words(b.get(r["id"], ""))
        if len(wa) == len(wb) == len(words):
            out += [[r["id"], w, x, y] for w, x, y in zip(words, wa, wb) if x != y]
    return out


def _meta():
    """id -> (kind, metre) for the audit rows: eval rows from set_q, training rows from clips.jsonl."""
    meta = {}
    for line in open(os.path.join(DATA, "eval/set_q.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        meta[r["key"]] = ("prose" if r["cls"] == "prose" else "verse", r.get("metre") or None)
    for line in open(os.path.join(DATA, "corpus_q/clips.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        key = f"{r['rec_id']}-{os.path.splitext(os.path.basename(r['clip']))[0]}"
        meta.setdefault(key, (r.get("kind", "verse"), r.get("metre") or None))
    return meta


def _label(module: str, rows: list, meta: dict, model: str = None, device: str = "cpu"):
    mod = importlib.import_module(module)
    kw = {"model_id": model} if model else {}
    g = mod.G2P(lexicon=mod.load_lexicon(os.path.join(ROOT, "packages/ganj-reading/ganj_reading/lexicon.json")), device=device, **kw)
    units = [(r["lines"],) + meta.get(r["id"], ("verse", None)) for r in rows]
    return {r["id"]: ph or "" for r, ph in zip(rows, g.phonemize_units(units))}, mod.G2P.words


def _random_training_rows(n: int, seed: int = 0) -> list:
    import random
    rows = []
    for line in open(os.path.join(DATA, "corpus_q/clips.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        if r.get("lines"):
            rows.append({"id": f"{r['rec_id']}-{os.path.splitext(os.path.basename(r['clip']))[0]}", "lines": r["lines"]})
    return random.Random(seed).sample(rows, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="ganj_reading.g2p", help="G2P module to gate (e.g. a copy of an old version)")
    ap.add_argument("--ref-module", default=None, help="also label with this module and list every word that differs")
    ap.add_argument("--extra", type=int, default=0, help="random training units added to the diff (not to the score)")
    ap.add_argument("--model", default=None, help="the G2P's model (a fine-tuned copy's directory); the reference keeps the default")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = json.load(open(os.path.join(DATA, "audit_g2p/sample.json"), encoding="utf-8"))
    errors = json.load(open(os.path.join(DATA, "audit_g2p/errors.json"), encoding="utf-8"))
    meta = _meta()
    extra = _random_training_rows(a.extra) if a.extra else []
    new, words_of = _label(a.module, rows + extra, meta, a.model, a.device)
    res = score(rows, new, errors, words_of)
    print(f"{a.module}{' + ' + a.model if a.model else ''}: flagged fixed {res['flagged_fixed']}/{res['flagged_checkable']}, unflagged changed "
          f"{res['unflagged_changed']}/{res['unflagged_words']}, unaligned rows {res['unaligned_rows']}/{res['rows']}")
    if a.ref_module:
        ref, _ = _label(a.ref_module, rows + extra, meta, device=a.device)
        ref_res = score(rows, ref, errors, words_of)
        print(f"{a.ref_module}: flagged fixed {ref_res['flagged_fixed']}/{ref_res['flagged_checkable']}, unflagged changed "
              f"{ref_res['unflagged_changed']}/{ref_res['unflagged_words']}")
        res["diff_vs_ref"] = diff(rows + extra, ref, new, words_of)
        n_words = sum(len(label_words(new[r["id"]])) for r in rows + extra)
        print(f"words that differ from {a.ref_module}: {len(res['diff_vs_ref'])} of about {n_words}")
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

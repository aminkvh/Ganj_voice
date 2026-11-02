# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Check that a generation carries the rules (ganj_corpus.rules_check) before it is synthesised or trained on.

  python tools/check_rules.py list LIST.tsv [--current eval/set_q_q9p.jsonl] [--waivers …/waivers.json]
      a reading list: every target label against the text's written marks, and against the current labels
  python tools/check_rules.py cuts CUTS.jsonl.gz ... [--overrides audit_g2p/review/overrides_q9e.json]
      training cuts: every per-line reading present, the written marks counted
Exit 1 when a list is stale, a written mark is not read on a Persian line, or a per-line reading is missing. Arabic and
Turkish lines, labels whose words do not line up, and words in audit_g2p/review/waivers.json are reported only.
"""
import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ganj_corpus.rules_check import REPORT_ONLY, check_label, missing_overrides, stale_keys, triage  # noqa: E402

from tools._env import env_path  # noqa: E402
DATA = env_path("GANJ_DATA")


def by_key(path) -> dict:
    """{key: row} of a JSONL file; {} when there is none (without the eval set every line of a list is unchecked)."""
    if not Path(path).exists():
        return {}
    return {r["key"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def check_list(path, current_path, waivers_path) -> int:
    rows = by_key(DATA / "eval" / "set_q.jsonl")
    current = {k: r["phonemes"] for k, r in by_key(current_path).items()}
    waivers = json.load(open(waivers_path, encoding="utf-8")) if Path(waivers_path).exists() else []
    labels = {l.split("\t")[0]: l.rstrip("\n").split("\t")[3] for l in open(path, encoding="utf-8") if l.strip()}
    stale = stale_keys(labels, current)
    fail, report, unchecked = [], collections.Counter(), 0
    for k, lab in labels.items():
        if k not in rows:
            unchecked += 1
            continue
        f, r = triage(k, rows[k]["cls"], check_label(rows[k]["lines"], lab), waivers)
        fail += [(k, v) for v in f]
        report.update(f"{rows[k]['cls'] if rows[k]['cls'] in REPORT_ONLY else 'waived or unaligned'}" for _ in r)
    for k in stale:
        print(f"STALE {k}: list has {labels[k]!r}, current is {current[k]!r}")
    for k, v in fail:
        print(f"RULE {k} line {v.line}: {v.word} → {v.label} ({v.rule})")
    print(f"{path}: {len(labels)} lines, {len(stale)} stale, {len(fail)} rule violations; reported, not failing: "
          f"{dict(report)}; {unchecked} lines not in the eval set (unchecked)")
    return 1 if stale or fail else 0


def check_cuts(paths, overrides_path) -> int:
    from ganj_corpus.manatts import transcripts
    rows = {f"{r['rec_id']}-{r['unit']}": r["lines"]
            for r in map(json.loads, open(DATA / "corpus_q" / "clips.jsonl", encoding="utf-8"))}
    rows.update({k: [v] for k, v in transcripts(DATA / "manatts" / "raw").items()})
    labels = {}
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for l in f:
                c = json.loads(l)
                labels[c["id"]] = c["supervisions"][0]["text"]
    missing = missing_overrides(labels, rows, json.load(open(overrides_path, encoding="utf-8")))
    counts = collections.Counter(v.rule for k, lab in labels.items() if k in rows for v in check_label(rows[k], lab))
    for m in missing:
        print("MISSING per-line reading", m)
    print(f"{len(labels)} cuts; per-line readings missing: {len(missing)}; written-mark findings: {dict(counts)}")
    return 1 if missing else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("list")
    a1.add_argument("path")
    a1.add_argument("--current", default=str(DATA / "eval" / "set_q_q9p.jsonl"), help="the current eval labels")
    a1.add_argument("--waivers", default=str(DATA / "audit_g2p" / "review" / "waivers.json"))
    a2 = sub.add_parser("cuts")
    a2.add_argument("paths", nargs="+")
    a2.add_argument("--overrides", default=str(DATA / "audit_g2p" / "review" / "overrides_q9e.json"))
    a = ap.parse_args(argv)
    sys.exit(check_list(a.path, a.current, a.waivers) if a.cmd == "list" else check_cuts(a.paths, a.overrides))


if __name__ == "__main__":
    main()

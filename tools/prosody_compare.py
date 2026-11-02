# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Compare phrasing and pace of voices with a reference reader's recordings, line by line (ganj_eval.prosody).

  python tools/prosody_compare.py OUT.json reader:readings/wav24 human run:eval/runs/<tag> ...

Each source is name:dir (wavs named <key>.wav) or "human" (the corpus recitation). Only the keys the reader recorded are used.
Per source and line: word spans from CTC forced alignment of the known text (ganj_eval.align), the pause at every word boundary, word durations. Each source is then compared
with every other: pause agreement at boundaries and correlation of relative word lengths; the summary is printed.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ganj_eval.align import SR, Aligner, boundary_pauses  # noqa: E402
from ganj_eval.prosody import compare_pauses, line_words, stretch_corr  # noqa: E402

from tools._env import env_path  # noqa: E402
DATA = env_path("GANJ_DATA")
MIN_PAUSE = 0.15


def measure(aligner, path, lines) -> dict:
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    ws = line_words(lines, aligner.vocab)
    spans, _ = aligner.spans(y, [w.text for w in ws])
    pauses = boundary_pauses(spans, y)
    speech = (spans[-1][1] - spans[0][0]) - sum(pauses)
    return {"words": [w.text for w in ws], "lines": [w.line for w in ws], "durations": [e - s for s, e in spans],
            "pauses": pauses, "speech_s": speech}


def summary(per: dict, names: list) -> dict:
    out = {}
    for a in names:
        for b in names:
            if a == b:
                continue
            keys = [k for k in per[a] if k in per[b]]
            cmp = [compare_pauses(per[a][k]["pauses"], per[b][k]["pauses"], MIN_PAUSE) for k in keys]
            corr = [stretch_corr(per[a][k]["durations"], per[b][k]["durations"]) for k in keys]
            out[f"{b} vs {a}"] = {
                "n": len(keys),
                "pause_f1": float(np.mean([c["f1"] for c in cmp])),
                "extra_per_line": float(np.mean([len(c["extra"]) for c in cmp])),
                "missing_per_line": float(np.mean([len(c["missing"]) for c in cmp])),
                "stretch_corr": float(np.nanmean(corr)),
                "pace_ratio": float(np.median([per[b][k]["speech_s"] / per[a][k]["speech_s"] for k in keys])),
            }
    for a in names:
        p = [x for k in per[a] for x in per[a][k]["pauses"]]
        out[a] = {"pauses_per_line": float(np.mean([sum(x >= MIN_PAUSE for x in per[a][k]["pauses"]) for k in per[a]])),
                  "median_pause_s": float(np.median([x for x in p if x >= MIN_PAUSE] or [0]))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("sources", nargs="+")
    a = ap.parse_args(argv)
    rows = {json.loads(l)["key"]: json.loads(l) for l in open(DATA / "eval" / "set_q.jsonl", encoding="utf-8")}
    srcs = [(s, None) if s == "human" else tuple(s.split(":", 1)) for s in a.sources]
    reader_dir = DATA / dict(srcs).get("reader", "readings/wav24")
    keys = sorted(p.stem for p in Path(reader_dir).glob("*.wav") if p.stem in rows)
    aligner, per = Aligner(), {}
    for name, d in srcs:
        per[name] = {}
        for k in keys:
            path = DATA / rows[k]["human"] if name == "human" else DATA / d / f"{k}.wav"
            if path.exists():
                per[name][k] = measure(aligner, path, rows[k]["lines"])
        print(f"{name}: {len(per[name])} lines", flush=True)
    res = {"min_pause_s": MIN_PAUSE, "summary": summary(per, [n for n, _ in srcs]), "per": per}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res["summary"].items():
        print(k, {x: round(y, 3) if isinstance(y, float) else y for x, y in v.items()})


if __name__ == "__main__":
    main()

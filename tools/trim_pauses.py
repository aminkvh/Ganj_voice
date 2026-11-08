# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Cut every stop inside a hemistich to at most --cap seconds (ganj_voice.pausetrim; the hemistich break keeps its
length). Word boundaries come from the CTC alignment of the line's text (ganj_eval.align), the silence from the
same frame levels as the pause measure (ganj_eval.prosody).
  python tools/trim_pauses.py SRC_DIR DST_DIR [--cap 0.30] [--set eval/set_q_q9p.jsonl]   (KEY.wav or KEY_d<k>.wav)"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ganj_corpus.rules_check import text_words  # noqa: E402
from ganj_eval.align import FRAME_S, SLACK_S, SR, Aligner, frame_db  # noqa: E402
from ganj_eval.prosody import silent_run_in_gap  # noqa: E402
from ganj_voice.pausetrim import CAP_S, shorten_runs, silent_frames, stops_to_trim  # noqa: E402

from tools._env import env_path  # noqa: E402
DATA = env_path("GANJ_DATA")


def trim(al: Aligner, path: str, lines: list, cap: float):
    lines = [l for l in lines if l.strip()]
    words = [w for l in lines for w in text_words(l)]
    if any(not al.chars(w) for w in words):
        return None, []
    y, sr = sf.read(path, dtype="float32")
    y16 = librosa.resample(y, orig_sr=sr, target_sr=SR)
    spans, _ = al.spans(y16, words)
    silent = silent_frames(frame_db(y16))
    runs = [silent_run_in_gap(silent, e - SLACK_S, spans[i + 1][0] + SLACK_S, FRAME_S) for i, (_, e) in enumerate(spans[:-1])]
    breaks, n = set(), 0
    for l in lines[:-1]:
        n += len(text_words(l))
        breaks.add(n - 1)
    cut = stops_to_trim(runs, breaks, cap)
    return (shorten_runs(y, sr, cut, cap), sr), cut


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--cap", type=float, default=CAP_S)
    ap.add_argument("--set", type=Path, default=DATA / "eval/set_q_q9p.jsonl")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    rows = {r["key"]: r for r in map(json.loads, open(a.set, encoding="utf-8"))}
    al, done = Aligner(a.device), {}
    a.dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(glob.glob(str(a.src / "*.wav"))):
        name = os.path.basename(p)
        key = re.sub(r"_d\d+$", "", name[:-4])
        out, cut = trim(al, p, rows[key]["lines"], a.cap) if key in rows else (None, [])
        if out is None:
            y, sr = sf.read(p, dtype="float32")
            out = (y, sr)
        sf.write(a.dst / name, out[0], out[1], subtype="PCM_16")
        done[name] = [round(e - s, 2) for s, e in cut]
    json.dump(done, open(a.dst / "trimmed.json", "w"), indent=1)
    n = sum(len(v) for v in done.values())
    print(f"{len(done)} files, {n} stops cut to {a.cap} s (median before {np.median([x for v in done.values() for x in v] or [0]):.2f} s) -> {a.dst}")


if __name__ == "__main__":
    main()

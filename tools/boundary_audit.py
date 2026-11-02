# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Audit clip boundaries: is there speech energy at the very end/start of a clip (cut mid-word)? Where do aligned words sit?"""
import json, os, random, sys
import numpy as np, soundfile as sf

D = os.environ["GANJ_DATA"]
EDGE_MS = 60


def edge_ratios(y, sr):
    n = int(sr * EDGE_MS / 1000)
    frame = int(sr * 0.02)
    rms = np.sqrt(np.convolve(y ** 2, np.ones(frame) / frame, mode="valid") + 1e-12)
    speech = np.percentile(rms, 75)  # a loud-speech reference level
    return float(np.sqrt(np.mean(y[-n:] ** 2)) / speech), float(np.sqrt(np.mean(y[:n] ** 2)) / speech)


def main(out_path, n_align):
    rows = [json.loads(l) for l in open(f"{D}/clips.jsonl", encoding="utf-8")]
    out = open(out_path, "w")
    random.seed(1)
    sample = set(random.sample(range(len(rows)), n_align))
    aligner = None
    if n_align:
        from ganj_corpus.align import Aligner
        aligner = Aligner(device="cuda")
    for i, r in enumerate(rows):
        p = r["clip"] if os.path.isabs(r["clip"]) else f"{D}/{r['clip']}"
        y, sr = sf.read(p, dtype="float32")
        end_r, start_r = edge_ratios(y, sr)
        rec = {"clip": r["clip"], "artist": r["artist"], "seconds": len(y) / sr, "end_ratio": end_r, "start_ratio": start_r,
               "last_unit": r.get("last_unit", None)}
        if i in sample:
            al = aligner.align(y, " ".join(r["lines"]))
            if al.words:
                rec.update(first_word_s=al.words[0][1], last_word_end_s=al.words[-1][2], tail_s=len(y) / sr - al.words[-1][2], score=al.score)
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 4000 == 0:
            print(i, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))

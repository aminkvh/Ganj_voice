# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Background (pause) noise and voiced aperiodicity of eval runs and single files, against the human references.
  python tools/noise_probe.py [run-tag | path.wav ...] [--n 50]
A run tag reads $GANJ_DATA/eval/runs/<tag>/<key>.wav for the first n set_q items; "human" is always shown first.
"""
import argparse
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ganj_eval.noise import AP_BANDS, BANDS, pause_bands, voiced_aperiodicity  # noqa: E402

from tools._env import env_path  # noqa: E402
DATA = env_path("GANJ_DATA")
SR = 24000


def load(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    return librosa.resample(y, orig_sr=sr, target_sr=SR) if sr != SR else y


def measure(paths: list) -> tuple:
    ys = [load(p) for p in paths if os.path.exists(p)]
    return (np.median([pause_bands(y, SR) for y in ys], axis=0), np.mean([voiced_aperiodicity(y, SR) for y in ys], axis=0),
            len(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="*")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    evals = [json.loads(l) for l in open(os.path.join(DATA, "eval/set_q.jsonl"), encoding="utf-8")][: a.n]
    rows = [("human", [os.path.join(DATA, e["human"]) for e in evals])]
    for it in a.items:
        rows.append((it, [it] if it.endswith(".wav") else [os.path.join(DATA, "eval/runs", it, f"{e['key']}.wav") for e in evals]))
    print("pause dB below speech (median over clips):", " ".join(f"{lo}-{hi}" for lo, hi in BANDS),
          "| voiced aperiodicity dB (mean):", " ".join(f"{lo}-{hi}" for lo, hi in AP_BANDS))
    for name, paths in rows:
        pb, va, n = measure(paths)
        print(f"{os.path.basename(name)[-38:]:38s} n={n:2d} | " + " ".join(f"{v:6.1f}" for v in pb) + " | "
              + " ".join(f"{v:6.1f}" for v in va))


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""The pause trim (ganj_voice.pausetrim) from the audio's levels alone, for the phone: tools/trim_pauses.py finds the
word boundaries with the CTC aligner, which the phone does not have. Here a stop is any silent run (pausetrim's
silent_frames, 10 ms frames) between the line's first and last sound; the line's n_breaks longest stops are taken as its
hemistich breaks and keep their length, and every other stop longer than the cap is cut to it.
  python -m ganj_voice.levelstops SRC_DIR DST_DIR --set SET.jsonl [--cap 0.30]   (KEY.wav or KEY_d<k>.wav)"""
import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ganj_voice.pausetrim import silent_frames
from ganj_voice.pausetrim import CAP_S, shorten_runs

FRAME_S = 0.01


def frame_db(y: np.ndarray, sr: int, frame_s: float = FRAME_S) -> np.ndarray:
    n = int(round(frame_s * sr))
    k = len(y) // n
    return 10 * np.log10(np.mean(np.asarray(y[:k * n], dtype=np.float64).reshape(k, n) ** 2, axis=1) + 1e-12)


def stop_frames(y: np.ndarray, sr: int, frame_s: float = FRAME_S) -> List[Tuple[int, int]]:
    """(first frame, frame after the last) of each silent run inside the line, in order."""
    silent = silent_frames(frame_db(y, sr, frame_s))
    sound = np.flatnonzero(~silent)
    if len(sound) == 0:
        return []
    stops, start = [], None
    for k in range(sound[0], sound[-1] + 1):
        if silent[k] and start is None:
            start = k
        elif not silent[k] and start is not None:
            stops.append((start, k))
            start = None
    return stops


def level_stops(y: np.ndarray, sr: int, frame_s: float = FRAME_S) -> List[Tuple[float, float]]:
    """(start s, end s) of each silent run inside the line, in order."""
    return [(a * frame_s, b * frame_s) for a, b in stop_frames(y, sr, frame_s)]


def trim_by_levels(y: np.ndarray, sr: int, n_breaks: int, cap: float = CAP_S, frame_s: float = FRAME_S):
    """(y with every stop but the n_breaks longest cut to cap, the stops cut). Lengths are compared in whole frames (a
    stop of exactly the cap is kept; in float seconds 33 x 0.01 - 3 x 0.01 is over 0.30), and of equally long stops the
    earlier is the break."""
    stops = stop_frames(y, sr, frame_s)
    cap_frames = int(round(cap / frame_s))
    by_length = sorted(range(len(stops)), key=lambda i: stops[i][1] - stops[i][0], reverse=True)   # stable: ties keep order
    breaks = set(by_length[:max(n_breaks, 0)])
    cut = [(a * frame_s, b * frame_s) for i, (a, b) in enumerate(stops) if i not in breaks and b - a > cap_frames]
    return (shorten_runs(y, sr, cut, cap) if cut else y), cut


def main(argv=None):
    import soundfile as sf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--set", type=Path, required=True, help="the lines' text (JSONL with key and lines)")
    ap.add_argument("--cap", type=float, default=CAP_S)
    a = ap.parse_args(argv)
    rows = {r["key"]: r for r in map(json.loads, open(a.set, encoding="utf-8"))}
    a.dst.mkdir(parents=True, exist_ok=True)
    done = {}
    for p in sorted(glob.glob(str(a.src / "*.wav"))):
        name = os.path.basename(p)
        key = re.sub(r"_d\d+$", "", name[:-4])
        y, sr = sf.read(p, dtype="float32")
        lines = [l for l in rows[key]["lines"] if l.strip()] if key in rows else []
        out, cut = trim_by_levels(y, sr, len(lines) - 1, a.cap) if lines else (y, [])
        sf.write(a.dst / name, out, sr, subtype="PCM_16")
        done[name] = [[round(s, 2), round(e, 2)] for s, e in cut]
    json.dump(done, open(a.dst / "trimmed.json", "w"), indent=1)
    print(f"{len(done)} files, {sum(len(v) for v in done.values())} stops cut to {a.cap} s -> {a.dst}")


if __name__ == "__main__":
    main()

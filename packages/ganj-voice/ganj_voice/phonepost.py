# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""What the app does to a line after sherpa-onnx (the phone chain; the settings are phone.json's "after", written by
recipes/zipvoice/export_distill.sh): the gain ZipVoice applies and sherpa does not (prompt RMS / target RMS when the prompt
is the quieter), the clamp to [-1, 1], the 70 Hz high-pass (dehum.highpass), the de-ess (deess.deess) and the pause trim
from the levels (levelstops; the hemistich breaks keep their length). skip leaves steps out, for listening to what each
one does.
  python -m ganj_voice.phonepost SRC_DIR DST_DIR --phone PHONE.json --set SET.jsonl [--skip deess,trim]"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

from ganj_voice.deess import deess
from ganj_voice.dehum import highpass
from ganj_voice.levelstops import trim_by_levels

STEPS = ("gain", "highpass", "deess", "trim")
AFTER = {"gain": 1.0, "clamp": 1.0, "highpass_hz": 70, "highpass_order": 8, "deess_db": 6, "pause_cap_s": 0.30}


def phone_post(y: np.ndarray, sr: int, after: dict, n_breaks: int, skip=frozenset()):
    """(the cleaned line, the stops cut)."""
    unknown = set(skip) - set(STEPS)
    if unknown:
        raise ValueError(f"unknown steps {sorted(unknown)}; the steps are {STEPS}")
    y = np.asarray(y, dtype=np.float32)
    if "gain" not in skip:
        y = y * np.float32(after["gain"])
    y = np.clip(y, -after["clamp"], after["clamp"])
    if "highpass" not in skip and after["highpass_hz"]:
        y = highpass(y, sr, after["highpass_hz"], after["highpass_order"])
    if "deess" not in skip and after["deess_db"]:
        y = deess(y, sr, after["deess_db"])
    cut = []
    if "trim" not in skip:
        y, cut = trim_by_levels(y, sr, n_breaks, after["pause_cap_s"])
    return np.clip(y, -1.0, 1.0).astype(np.float32), cut


def main(argv=None):
    import soundfile as sf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--phone", type=Path, required=True)
    ap.add_argument("--set", type=Path, required=True, help="the lines' text (JSONL with key and lines)")
    ap.add_argument("--skip", default="")
    a = ap.parse_args(argv)
    after = json.load(open(a.phone, encoding="utf-8"))["after"]
    skip = set(filter(None, a.skip.split(",")))
    rows = {r["key"]: r for r in map(json.loads, open(a.set, encoding="utf-8"))}
    a.dst.mkdir(parents=True, exist_ok=True)
    done = {}
    for p in sorted(glob.glob(str(a.src / "*.wav"))):
        name = os.path.basename(p)
        key = re.sub(r"_d\d+$", "", name[:-4])
        y, sr = sf.read(p, dtype="float32")
        lines = [l for l in rows[key]["lines"] if l.strip()] if key in rows else []
        out, cut = phone_post(y, sr, after, max(len(lines) - 1, 0), skip | (set() if lines else {"trim"}))
        sf.write(a.dst / name, out, sr, subtype="PCM_16")
        done[name] = [[round(s, 2), round(e, 2)] for s, e in cut]
    json.dump({"after": after, "skip": sorted(skip), "trimmed": done}, open(a.dst / "phonepost.json", "w"), indent=1)
    print(f"{len(done)} files -> {a.dst} (skipped: {','.join(sorted(skip)) or 'none'})")


if __name__ == "__main__":
    main()

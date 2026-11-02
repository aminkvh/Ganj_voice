# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Clean model outputs after the vocoder (a humming prompt's hum comes back from the vocoder).
  python tools/postclean.py SRC_TAG DST_TAG [--highpass 70] [--denoise-floor -20] [--deess 6]
  python tools/postclean.py --paths SRC_DIR DST_DIR ...     (synth_voice.sh: the same dir, in place)
Reads $GANJ_DATA/eval/runs/SRC_TAG/*.wav, writes the cleaned clips (same rate, PCM16) and postclean.json to
$GANJ_DATA/eval/runs/DST_TAG/, ready for `ganj_eval.run --tag DST_TAG` (it scores wavs already in the run dir).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ganj_voice.deess import deess  # noqa: E402
from ganj_voice.dehum import denoise_stationary, highpass  # noqa: E402

from tools._env import env_path  # noqa: E402
DATA = env_path("GANJ_DATA")


def postclean_dir(src: Path, dst: Path, highpass_hz=70.0, denoise_floor_db=None, deess_db=None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(src).glob("*.wav")):
        y, sr = sf.read(str(p), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        if highpass_hz:
            y = highpass(y, sr, highpass_hz)
        if denoise_floor_db is not None:
            y = denoise_stationary(y, sr, floor_db=denoise_floor_db)
        if deess_db:
            y = deess(y, sr, deess_db)
        sf.write(str(dst / p.name), y.clip(-1, 1), sr, subtype="PCM_16")
        n += 1
    (dst / "postclean.json").write_text(json.dumps({"source": str(src), "highpass_hz": highpass_hz,
                                                    "denoise_floor_db": denoise_floor_db, "deess_db": deess_db}))
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src_tag")
    ap.add_argument("dst_tag")
    ap.add_argument("--paths", action="store_true", help="SRC and DST are directories, not run tags (the same dir = in place)")
    ap.add_argument("--highpass", type=float, default=70.0, help="cutoff in Hz; 0 = off")
    ap.add_argument("--denoise-floor", type=float, default=None, help="dB, e.g. -20; off when not given")
    ap.add_argument("--deess", type=float, default=None, help="dB cut of 5-9 kHz in hiss frames (ganj_voice.deess); off when not given")
    a = ap.parse_args(argv)
    src, dst = (Path(a.src_tag), Path(a.dst_tag)) if a.paths else (DATA / "eval/runs" / a.src_tag, DATA / "eval/runs" / a.dst_tag)
    n = postclean_dir(src, dst, a.highpass or None, a.denoise_floor, a.deess)
    print(f"{n} clips -> {dst}")


if __name__ == "__main__":
    main()

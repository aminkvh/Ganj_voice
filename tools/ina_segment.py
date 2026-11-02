# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Speech / music / gender segmentation with inaSpeechSegmenter (MIT). Writes one CSV per WAV."""
import argparse
import csv
from pathlib import Path

from inaSpeechSegmenter import Segmenter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg = Segmenter(vad_engine="smn", detect_gender=True)
    wavs = sorted(Path(a.wav_dir).glob("*.wav"))
    for i, wav in enumerate(wavs, 1):
        out = out_dir / (wav.stem + ".csv")
        if out.exists():
            continue
        rows = seg(str(wav))
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "start", "stop"])
            w.writerows(rows)
        if i % 50 == 0:
            print(f"{i}/{len(wavs)}")


if __name__ == "__main__":
    main()

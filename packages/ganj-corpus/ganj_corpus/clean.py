# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Cleaning helpers: music detection summary, gender per reciter, Demucs vocals,
loudness normalisation, clipping and bandwidth checks. All components MIT/BSD/Apache."""
import csv
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

PEAK = 10 ** (-1 / 20)  # -1 dBFS


def summarize_ina(csv_path) -> dict:
    total = music = male = female = 0.0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            d = float(row["stop"]) - float(row["start"])
            total += d
            if row["label"] == "music":
                music += d
            elif row["label"] == "male":
                male += d
            elif row["label"] == "female":
                female += d
    return {"music_fraction": music / total if total else 0.0, "male_s": male, "female_s": female}


def reciter_genders(rows, ina_dir, overrides_csv: Optional[Path] = None) -> dict:
    secs = defaultdict(lambda: {"male": 0.0, "female": 0.0})
    for r in rows:
        p = Path(ina_dir) / f"{r['id']}.csv"
        if not p.exists():
            continue
        s = summarize_ina(p)
        secs[r["audioArtist"]]["male"] += s["male_s"]
        secs[r["audioArtist"]]["female"] += s["female_s"]
    out = {}
    for artist, d in secs.items():
        out[artist] = "unknown" if d["male"] + d["female"] == 0 else ("male" if d["male"] >= d["female"] else "female")
    if overrides_csv and Path(overrides_csv).exists():
        with open(overrides_csv, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("gender") in ("male", "female"):
                    out[row["audioArtist"]] = row["gender"]
    return out


def write_reciters_csv(path, genders: dict, hours: dict) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audioArtist", "gender", "hours_measured"])
        for a in sorted(genders):
            w.writerow([a, genders[a], f"{hours.get(a, 0.0):.2f}"])


def separate_vocals(wav_path, out_dir, sr: int = 22050) -> Path:
    out_dir = Path(out_dir)
    final = out_dir / (Path(wav_path).stem + ".wav")
    if final.exists():
        return final
    tmp = out_dir / "tmp"
    subprocess.run(["python", "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", str(tmp), str(wav_path)], check=True)
    vocals = tmp / "htdemucs" / Path(wav_path).stem / "vocals.wav"
    y, s = librosa.load(str(vocals), sr=sr, mono=True)
    sf.write(str(final), y, sr, subtype="PCM_16")
    return final


def loudness_normalize(y: np.ndarray, sr: int, target: float = -23.0) -> np.ndarray:
    meter = pyln.Meter(sr)
    loud = meter.integrated_loudness(y.astype(np.float64))
    if not np.isfinite(loud):
        return y
    z = pyln.normalize.loudness(y.astype(np.float64), loud, target)
    peak = np.abs(z).max()
    if peak > PEAK:
        z = z * (PEAK / peak)
    return z.astype(np.float32)


def clipping_fraction(y: np.ndarray) -> float:
    return float(np.mean(np.abs(y) >= 0.999))


def bandwidth_hz(mp3_path) -> float:
    y, sr = librosa.load(str(mp3_path), sr=None, mono=True, duration=60.0)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
    return float(np.median(roll))

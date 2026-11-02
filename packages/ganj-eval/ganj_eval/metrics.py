# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Scoring primitives: text normalisation, WER, defect probes, pitch stability, ASR and MOS wrappers."""
import json
from pathlib import Path

import jiwer
import librosa
import numpy as np

from ganj_reading.g2p import normalize_fa, strip_harakat

TINY = {"ز", "و", "که", "چو", "گر", "کز", "وز", "به", "ای"}


def norm(text: str) -> str:
    return strip_harakat(normalize_fa(text)).replace("‌", " ").replace("  ", " ").strip()


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    return jiwer.wer(r, h) if r and h else 1.0


def tiny_word_recall(ref: str, hyp: str):
    want = [w for w in norm(ref).split() if w in TINY]
    if not want:
        return None
    have = norm(hyp).split()
    hits = 0
    for w in want:
        if w in have:
            have.remove(w)
            hits += 1
    return hits / len(want)


def line_end_recall(lines, hyp: str) -> float:
    have = set(norm(hyp).split())
    ends = [norm(l).split()[-1] for l in lines if norm(l).split()]
    return sum(e in have for e in ends) / len(ends) if ends else 0.0


def f0_logstd(y: np.ndarray, sr: int) -> float:
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
    f0 = f0[voiced & np.isfinite(f0)]
    return float(np.std(np.log(f0))) if len(f0) > 10 else 0.0


class PersianASR:
    MODEL = "vhdm/whisper-v3-turbo-persian-v1.1"  # evaluation only; check its model-card licence before any other use

    def __init__(self, device: str = "cpu", cache_path=None):
        import torch
        from transformers import pipeline
        self.pipe = pipeline("automatic-speech-recognition", model=self.MODEL, device=0 if device == "cuda" else -1,
                             torch_dtype=torch.float16 if device == "cuda" else torch.float32, chunk_length_s=30)
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = json.loads(self.cache_path.read_text(encoding="utf-8")) if self.cache_path and self.cache_path.exists() else {}

    def forget(self, path) -> None:
        """Drop a cached transcript (the cache is keyed by path; a rewritten wav must not reuse a stale one)."""
        self.cache.pop(str(path), None)

    def transcribe(self, path) -> str:
        key = str(path)
        if key in self.cache:
            return self.cache[key]
        out = self.pipe(key, generate_kwargs={"language": "fa", "task": "transcribe"})["text"]
        self.cache[key] = out
        if self.cache_path:
            self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        return out


class Mos:
    """UTMOSv2 on the clip as shipped, i.e. high-passed at 70 Hz exactly once (synth_voice.sh does it; a run made by
    postclean.py already is, so prefiltered=True). UTMOSv2 scores a clip +0.26..0.46 higher with a steady 48/56 Hz hum
    added, human recordings too, and also moves with what is left below 100 Hz (F2 filtered once 2.74, twice 2.91, at
    100 Hz 3.05), so scores compare only between clips with the same low-band treatment."""
    SR = 16000
    HIGHPASS_HZ = 70.0

    def __init__(self, device: str = "cpu"):
        import utmosv2
        self.device = device
        self.model = utmosv2.create_model(pretrained=True, device=device)

    def score(self, path, prefiltered: bool = False) -> float:
        import soundfile as sf
        from ganj_voice.dehum import highpass
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.mean(axis=1) if prefiltered else highpass(y.mean(axis=1), sr, self.HIGHPASS_HZ)
        y = librosa.resample(y, orig_sr=sr, target_sr=self.SR)  # as UTMOSv2 loads a file
        r = self.model.predict(data=y, sr=self.SR, device=self.device)  # predict() defaults to cuda otherwise
        return float(np.asarray(r).reshape(-1)[0])

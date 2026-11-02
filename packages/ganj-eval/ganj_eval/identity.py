# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""ECAPA speaker embeddings (speechbrain, Apache-2.0) for the "nobody's voice" check."""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


class Ecapa:
    def __init__(self, device: str = "cpu"):
        from speechbrain.inference.speaker import EncoderClassifier
        self.enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": device})

    def embed(self, wav: np.ndarray, sr: int = 22050) -> np.ndarray:
        """sr is the clip's own rate: corpus clips are 22.05 kHz, ZipVoice outputs 24 kHz (before 2026-10-29 every
        output was read as 22.05 kHz, i.e. heard 8 % slower and lower)."""
        x = torchaudio.functional.resample(torch.from_numpy(wav).float(), sr, 16000)
        with torch.no_grad():
            e = self.enc.encode_batch(x[None])
        return e.squeeze().cpu().numpy()


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def reciter_embeddings(rows, data: Path, ecapa: Ecapa, per_artist: int = 30, seed: int = 1) -> dict:
    cache = data / "eval" / "reciter_embeddings.json"
    if cache.exists():
        return {k: np.array(v) for k, v in json.loads(cache.read_text()).items()}
    by = defaultdict(list)
    for r in rows:
        by[r["artist"]].append(r["clip"])
    rng = random.Random(seed)
    out = {}
    for artist, clips in by.items():
        embs = []
        for c in rng.sample(clips, min(per_artist, len(clips))):
            y, sr = sf.read(str(data / c), dtype="float32")
            embs.append(ecapa.embed(y, sr))
        out[artist] = np.mean(embs, axis=0)
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps({k: v.tolist() for k, v in out.items()}))
    return out


def novelty(synth_embs, reciters: dict):
    mean = np.mean(synth_embs, axis=0)
    table = sorted(((cos(mean, e), a) for a, e in reciters.items()), reverse=True)
    return table[0][0], table[0][1], table

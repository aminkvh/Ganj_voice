# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""sherpa-onnx synthesis from phoneme strings (the same path the app will use)."""
from pathlib import Path

import numpy as np
import sherpa_onnx

from ganj_reading import tokens as T


class Synth:
    def __init__(self, model_dir, sid: int = 0, speed: float = 1.0, debug: bool = False, num_threads: int = 4):
        d = Path(model_dir)
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(model=str(d / "model.onnx"), lexicon="", tokens=str(d / "tokens.txt"), data_dir="")
        cfg = sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=num_threads, debug=debug, provider="cpu"))
        if not cfg.validate():
            raise RuntimeError("invalid sherpa-onnx config")
        self.tts = sherpa_onnx.OfflineTts(cfg)
        self.sid, self.speed = sid, speed

    @staticmethod
    def ids_for(phonemes: str):
        return T.encode(phonemes)

    def say(self, phonemes: str) -> np.ndarray:
        audio = self.tts.generate(phonemes, sid=self.sid, speed=self.speed)
        if audio.sample_rate != 22050:
            raise RuntimeError(f"unexpected sample rate {audio.sample_rate}")
        return np.asarray(audio.samples, dtype=np.float32)

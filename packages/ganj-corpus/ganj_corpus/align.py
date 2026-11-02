# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""CTC forced alignment with torchaudio (BSD) and an Apache-2.0 Persian wav2vec2 model.

Used for three things: transcript verification (score), the last unit's end time, and word spans for
Pocket TTS manifests. Never use ctc-forced-aligner / MMS (CC-BY-NC).
"""
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torchaudio.functional import forced_align, merge_tokens

from ganj_reading.g2p import normalize_fa, strip_harakat

MIN_SCORE = 0.35
LOW_TOKEN = 0.3  # a token the audio barely supports
MAX_LOW_FRAC = 0.4  # calibrated 2026-10-15 on 894 accepted clips: 0 of 400 clean (score >= 0.6) exceed it; 409 of 494 weak (0.35-0.6) do
WORD_DELIM = "|"


@dataclass
class Alignment:
    score: float
    words: List[Tuple[str, float, float]]
    end_s: float
    dropped_chars: int
    low_frac: float = 0.0


def targets_for(text: str, vocab: dict):
    # ZWNJ is dropped, not turned into a space: the word count must stay equal to G2P's (the word spans rely on it).
    t = strip_harakat(normalize_fa(text)).replace("‌", "")
    ids, dropped = [], 0
    for ch in t:
        if ch == " ":
            if ids and ids[-1] != vocab[WORD_DELIM]:
                ids.append(vocab[WORD_DELIM])
        elif ch in vocab:
            ids.append(vocab[ch])
        else:
            dropped += 1
    while ids and ids[-1] == vocab[WORD_DELIM]:
        ids.pop()
    return ids, dropped


class Aligner:
    def __init__(self, model_id: str = "jonatasgrosman/wav2vec2-large-xlsr-53-persian", device: str = "cuda"):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.proc = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()
        self.vocab = self.proc.tokenizer.get_vocab()
        self.blank = self.model.config.pad_token_id
        self.device = device
        self.sr = 16000

    def log_probs(self, wav16k: np.ndarray) -> torch.Tensor:
        inp = self.proc(wav16k, sampling_rate=self.sr, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            return torch.log_softmax(self.model(inp).logits.float().cpu(), dim=-1)

    def frame_rate(self) -> float:
        return 50.0  # wav2vec2: 20 ms per frame

    def align(self, wav22k: np.ndarray, text: str) -> Alignment:
        wav16 = torchaudio.functional.resample(torch.from_numpy(wav22k).float(), 22050, self.sr).numpy()
        lp = self.log_probs(wav16)
        targets, dropped = targets_for(text, self.vocab)
        if not targets or lp.shape[1] < len(targets):
            return Alignment(0.0, [], 0.0, dropped, 1.0)
        aligned, scores = forced_align(lp, torch.tensor([targets], dtype=torch.int32), blank=self.blank)
        spans = merge_tokens(aligned[0], scores[0].exp())
        fr = self.frame_rate()
        inv = {v: k for k, v in self.vocab.items()}
        words, cur, start = [], "", None
        for sp in spans:
            tok = inv[sp.token]
            if tok == WORD_DELIM:
                if cur:
                    words.append((cur, start, last_end))
                cur, start = "", None
                continue
            if start is None:
                start = sp.start / fr
            cur += tok
            last_end = sp.end / fr
        if cur:
            words.append((cur, start, last_end))
        score = float(np.mean([sp.score for sp in spans])) if spans else 0.0
        low = float(np.mean([sp.score < LOW_TOKEN for sp in spans])) if spans else 1.0
        end_s = words[-1][2] if words else 0.0
        return Alignment(score, words, end_s, dropped, low)


def accepts(al: Alignment) -> bool:
    """A clip is kept only if the audio supports the text overall AND almost everywhere (a skipped hemistich fails the second)."""
    return al.score >= MIN_SCORE and al.low_frac <= MAX_LOW_FRAC


def last_unit_end_ms(aligner: Aligner, wav22k: np.ndarray, start_ms: int, text: str, pad_ms: int = 250, window_s: float = 45.0):
    a = int(start_ms * 22050 / 1000)
    seg = wav22k[a:a + int(window_s * 22050)]
    if len(seg) < 22050 // 2:
        return None
    al = aligner.align(seg, text)
    if not accepts(al) or not al.words:
        return None
    return start_ms + int(al.end_s * 1000) + pad_ms

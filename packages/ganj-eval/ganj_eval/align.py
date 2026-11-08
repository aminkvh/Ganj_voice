# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""CTC forced alignment of a line's known words (jonatasgrosman/wav2vec2-large-xlsr-53-persian, Apache-2.0; analysis
and labelling only, never shipped) and the pause at each word boundary (ganj_eval.prosody)."""
import numpy as np

from ganj_eval.prosody import silence_in_gap
from ganj_voice.pausetrim import silent_frames

MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-persian"
SR = 16000
FRAME_S = 0.01
SLACK_S = 0.04      # a word's sound can outlast its last CTC frame by a little: look this far into each word


def frame_db(y: np.ndarray) -> np.ndarray:
    n = int(FRAME_S * SR)
    k = len(y) // n
    return 10 * np.log10(np.mean(y[:k * n].reshape(k, n) ** 2, axis=1) + 1e-12)


def boundary_pauses(spans, y: np.ndarray) -> list:
    """Seconds of silence at each boundary between consecutive (start, end) word spans."""
    silent = silent_frames(frame_db(y))
    return [silence_in_gap(silent, e - SLACK_S, spans[i + 1][0] + SLACK_S, FRAME_S) for i, (_, e) in enumerate(spans[:-1])]


def group_chars(tokens, sep: int, frame_s: float) -> list:
    """[[(start s, end s) per letter] per word] from merged CTC token spans, words split at the `sep` token."""
    words, cur = [], []
    for t in tokens:
        if t.token == sep:
            words.append(cur)
            cur = []
        else:
            cur.append((t.start * frame_s, t.end * frame_s))
    words.append(cur)
    return words


class Aligner:
    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.torch, self.device = torch, device
        self.proc = Wav2Vec2Processor.from_pretrained(MODEL)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval().to(device)
        if device != "cpu":
            self.model = self.model.half()
        self.vocab = self.proc.tokenizer.get_vocab()

    def chars(self, word: str) -> str:
        return "".join(ch for ch in word.replace("‌", "") if ch in self.vocab and ch not in "|-")

    def spans(self, y: np.ndarray, words: list) -> tuple:
        """([(start s, end s)] per word, mean token confidence). Every word must have aligner letters (chars())."""
        groups, conf = self.char_spans(y, words)
        return [(g[0][0], g[-1][1]) for g in groups], conf

    def char_spans(self, y: np.ndarray, words: list) -> tuple:
        """([[(start s, end s)] per letter of chars(word)] per word, mean token confidence)."""
        import torchaudio
        torch = self.torch
        ids = []
        for i, w in enumerate(words):
            c = self.chars(w)
            if not c:
                raise ValueError(f"no alignable letters in {w!r}")
            ids += [self.vocab[ch] for ch in c] + ([self.vocab["|"]] if i < len(words) - 1 else [])
        x = self.proc(y, sampling_rate=SR, return_tensors="pt").input_values.to(self.device)
        with torch.inference_mode():
            lp = self.model(x.half() if self.device != "cpu" else x).logits.float().log_softmax(-1)
        ali, scores = torchaudio.functional.forced_align(lp, torch.tensor([ids], device=lp.device), blank=self.vocab["<pad>"])
        toks = torchaudio.functional.merge_tokens(ali[0].cpu(), scores[0].exp().cpu())
        return (group_chars(toks, self.vocab["|"], len(y) / SR / lp.shape[1]),
                float(np.mean([t.score for t in toks])))

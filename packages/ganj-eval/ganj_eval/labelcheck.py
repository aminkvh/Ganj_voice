# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Does a synthesised line say what its label says? Three checks, all on the Persian CTC model of ganj_eval.align
(analysis only, never shipped), and the choice among several noise draws of one line (a dropped word or ezafe
belongs to one draw, not to the line).

Words: the CTC log-likelihood of the line's text minus that of the text without one word; below 0 the audio fits
the line better without it (a dropped «فرو» scores −60, a said one +50 … +90).

Ezafe: the recognizer never writes a kasra (the text with ِ scores −14 … −20 on human recordings too), so a small
classifier on its hidden states around the word boundary decides. Targets come from labels: after word i (never the
last of a hemistich) whose text ends in a consonant letter, the reading ends in the ezafe after that consonant
(`…Ce1`: said) or in the consonant (none). Words ending in a vowel letter (ا و ی ه ة: the ezafe is a ye or rides on
the word's own vowel) and the NO_EZAFE words are left out. Trained on a native reader's recordings of the corrected labels.

Short vowels (2026-11-02): ganj_eval.vowelcheck measures each fatha, kasre and zamma of the label written between
two consonant letters; a voice's VowelModel says which are wrong. A wrong one counts like a wrong ezafe."""
from typing import Callable, Dict, List, Sequence, Tuple

from ganj_corpus.pauses import split_label, strip_pauses
from ganj_corpus.rules_check import text_words
from ganj_eval.vowelcheck import line_vowel_targets, measure_vowels
from ganj_reading.g2p import EZAFE, NO_EZAFE, VAV, strip_harakat

LABEL_VOWELS = set("/eoaiu")
VOWEL_LETTERS = set("اآأإوؤیيىئهةۀ")
BEFORE_S, AFTER_S = 0.08, 0.03      # the window reaches this far back into word i and forward into word i + 1


def _consonant_final(word: str) -> bool:
    plain = strip_harakat(word).rstrip("‌ٔ")
    return bool(plain) and plain[-1] not in VOWEL_LETTERS and plain not in NO_EZAFE


def ezafe_targets(lines: Sequence[str], label: str) -> List[Tuple[int, bool]]:
    """(word index across the line's hemistichs, ezafe said?) for each checkable boundary; [] when the label's words
    do not line up with the text's."""
    lines = [l for l in lines if l.strip()]
    _, parts = split_label(strip_pauses(label))
    if len(parts) != len(lines):
        return []
    out, base = [], 0
    for line, part in zip(lines, parts):
        words = text_words(line)
        if len(words) != len(part):
            return []
        for i, (w, ph) in enumerate(zip(words[:-1], part[:-1])):
            if not _consonant_final(w) or words[i + 1] == VAV:   # before «و» the vowel is the conjunction's
                continue
            if ph.endswith("e" + EZAFE) and len(ph) > 2 and ph[-3] not in LABEL_VOWELS:
                out.append((base + i, True))
            elif ph and ph[-1] not in LABEL_VOWELS and ph[-1] != EZAFE:
                out.append((base + i, False))
        base += len(words)
    return out


def boundary_window(spans: Sequence[Tuple[float, float]], i: int) -> Tuple[float, float]:
    """Seconds (a, b) around the boundary after word i: the end of word i's sound and the start of word i + 1's."""
    return max(spans[i][1] - BEFORE_S, spans[i][0]), spans[i + 1][0] + AFTER_S


def words_present(loglik: Callable[[List[str]], float], words: List[str]) -> List[bool]:
    """For each word, whether the audio fits the text with it better than without it."""
    full = loglik(words)
    return [full > loglik(words[:i] + words[i + 1:]) for i in range(len(words))]


def pick_draw(draws: Dict[str, dict]) -> str:
    """The draw with the fewest missing words, then the fewest vowels wrong (ezafes unheard or unwanted, short vowels
    the vowel model calls wrong), then the best CTC fit of the text."""
    return min(draws, key=lambda d: (draws[d]["missing_words"], draws[d]["wrong_ezafe"] + draws[d].get("wrong_vowel", 0),
                                     -draws[d]["fit"]))


def vowel_errors(model, measured: Sequence[dict]) -> List[dict]:
    """The measured short vowels (ganj_eval.vowelcheck.measure_vowels) `model` calls wrong; none without a model."""
    return [m for m in measured if model is not None and model.wrong(m["vowel"], m["f1"], m["f2"])]


class LineChecker:
    """The two checks on one recording (needs torch, transformers, librosa). Hidden states of LAYERS
    (mean and max over the boundary window) are the ezafe classifier's features."""
    LAYERS = (12, 16, 20, 24)

    def __init__(self, device: str = "cuda"):
        from ganj_eval.align import Aligner
        self.al = Aligner(device)

    def _ids(self, words):
        ids = []
        for i, w in enumerate(words):
            ids += [self.al.vocab[ch] for ch in self.al.chars(w)] + ([self.al.vocab["|"]] if i < len(words) - 1 else [])
        return ids

    def analyse(self, path: str, lines: Sequence[str], label: str) -> dict:
        """{"words", "present" (per word), "fit" (CTC log-likelihood of the text), "targets" [(i, said)],
        "features" (one row per target), "vowels" (measured short vowels)}; words without aligner letters are left
        out of the word checks."""
        import librosa
        import numpy as np
        import torch
        from ganj_eval.align import SR
        lines = [l for l in lines if l.strip()]
        words = [w for l in lines for w in text_words(l)]
        y = librosa.load(path, sr=SR)[0]
        x = self.al.proc(y, sampling_rate=SR, return_tensors="pt").input_values.to(self.al.device)
        with torch.inference_mode():
            out = self.al.model(x.half() if self.al.device != "cpu" else x, output_hidden_states=True)
        lp = out.logits.float().log_softmax(-1)[0].cpu()

        def loglik(ws):
            t = torch.tensor(self._ids(ws))
            return -float(torch.nn.functional.ctc_loss(lp[:, None], t[None], torch.tensor([lp.shape[0]]),
                                                       torch.tensor([len(t)]), blank=self.al.vocab["<pad>"],
                                                       reduction="sum", zero_infinity=True))

        checkable = [w for w in words if self.al.chars(w)]
        res = {"words": checkable, "present": words_present(loglik, checkable), "fit": loglik(checkable),
               "targets": [], "features": [], "vowels": []}
        targets, vowels = ezafe_targets(lines, label), line_vowel_targets(lines, label)
        if (targets or vowels) and len(checkable) == len(words):
            chars, _ = self.al.char_spans(y, words)
            spans = [(g[0][0], g[-1][1]) for g in chars]
            res["vowels"] = measure_vowels(y, SR, words, chars, vowels, self.al.chars)
            fr = len(y) / SR / out.hidden_states[0].shape[1]
            for i, said in targets:
                a, b = boundary_window(spans, i)
                f0 = int(a / fr)
                f1 = max(int(np.ceil(b / fr)), f0 + 1)
                hs = [out.hidden_states[L][0, f0:f1].float() for L in self.LAYERS]
                res["features"].append(np.concatenate([np.concatenate([h.mean(0).cpu().numpy(), h.max(0).values.cpu().numpy()])
                                                       for h in hs]))
                res["targets"].append((i, said))
        return res


def ezafe_errors(targets: Sequence[Tuple[int, bool]], said_prob: Sequence[float]) -> Tuple[int, int]:
    """(ezafes the label asks for and the classifier does not hear, ezafes heard where the label has none)."""
    missing = sum(1 for (_, said), p in zip(targets, said_prob) if said and p < 0.5)
    extra = sum(1 for (_, said), p in zip(targets, said_prob) if not said and p >= 0.5)
    return missing, extra

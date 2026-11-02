# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Build a voice prompt from a source clip, with the recipe of every prompt made so far.
Steps, each optional, in this order: mains notch -> stationary denoise -> disguise (WORLD, or Praat with
--method praat) -> LTAS match.
  python tools/make_prompt.py SRC.wav OUT.wav [--notch] [--denoise-floor -20] [--formant 1.06 --semitones 0.5 --ap-squared]
                                               [--ltas corpus_q/prompts/human_ltas_22050.npy]
Recipes: F2 = --formant 1.06 --semitones 0.5 --ap-squared --ltas; F3 = F2 + --notch; F4 = F3 + --denoise-floor -20;
F5 = --notch --denoise-floor -20 --formant 1.10 --semitones 1.0 --ap-squared --ltas; F6 = F5 with --method praat (no --ap-squared);
f5a (د۱) = F5 without --notch. --formant-fade LO HI fades the formant shift out between LO and HI Hz and
--fricative-formant F gives the frames whose envelope lives above 2.5 kHz (ش, س) their own factor, --fricative-below HZ
only those whose hiss is centred below HZ (ش, not س) (the ×1.10 shift had lifted د۱'s ش to 5.5 kHz, close
to its س, a whistle; a typical ش sits near 3.8 kHz). f5v3 (د۴) = f5a + --formant-fade 3000 4000
--fricative-formant 0.80. Moving only the ش (--fricative-below 6500) did not lower the output's ش: the model follows
the prompt's whole hiss band.
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ganj_voice.anonymise import ltas_match, praat_shift, world_shift  # noqa: E402
from ganj_voice.dehum import denoise_stationary, estimate_mains, notch_mains  # noqa: E402


def build(y: np.ndarray, sr: int, notch: bool = False, denoise_floor=None, formant=None, semitones: float = 0.0,
          ap_squared: bool = False, ltas_ref=None, method: str = "world", formant_fade=None,
          fricative_formant=None, fricative_below=None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if notch:
        y = notch_mains(y, sr, base_hz=estimate_mains(y, sr))
    if denoise_floor is not None:
        y = denoise_stationary(y, sr, floor_db=denoise_floor)
    if formant is not None and method == "praat":
        y = praat_shift(y, sr, formant=formant, semitones=semitones)
    elif formant is not None:
        y = world_shift(y, sr, formant=formant, semitones=semitones, ap_squared=ap_squared,
                        fade_hz=tuple(formant_fade) if formant_fade else None, fricative_formant=fricative_formant,
                        fricative_below_hz=fricative_below)
    if ltas_ref is not None:
        y = ltas_match(y, sr, *ltas_ref)
    return np.asarray(y, dtype=np.float32)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--notch", action="store_true")
    ap.add_argument("--denoise-floor", type=float, default=None, help="dB, e.g. -20 (the gain never goes lower)")
    ap.add_argument("--formant", type=float, default=None)
    ap.add_argument("--semitones", type=float, default=0.0)
    ap.add_argument("--ap-squared", action="store_true")
    ap.add_argument("--method", choices=["world", "praat"], default="world")
    ap.add_argument("--ltas", default=None, help="npy of (freqs, dB) to tone-match to")
    ap.add_argument("--formant-fade", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="Hz: the formant shift fades out between LO and HI (WORLD only)")
    ap.add_argument("--fricative-formant", type=float, default=None, help="own factor for fricative frames (WORLD only)")
    ap.add_argument("--fricative-below", type=float, default=None,
                    help="Hz: only fricatives whose hiss is centred below this (6500: ش moves, س stays)")
    return ap.parse_args(argv)


def main():
    a = parse_args()
    y, sr = sf.read(a.src, dtype="float32")
    ref = tuple(np.load(a.ltas)) if a.ltas else None
    sf.write(a.out, build(y, sr, a.notch, a.denoise_floor, a.formant, a.semitones, a.ap_squared, ref, a.method,
                          a.formant_fade, a.fricative_formant, a.fricative_below), sr, subtype="PCM_16")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

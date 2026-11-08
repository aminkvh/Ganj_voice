# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Read the lines of a reading list with sherpa-onnx's ZipVoice, as the phone will (an environment with sherpa-onnx
2.0.1): the phone folder of recipes/zipvoice/export_distill.sh, its phone.json settings, the token-strings path. The list
is read as given, so glue it first (ganj_voice.glue) as synth_distill.sh does. Writes OUT_DIR/KEY.wav (sherpa's
output, before the app's clean-up: ganj_voice.phonepost) and timing.tsv (seconds of audio, seconds taken).
A line is given as one token string per character, named as in the folder's tokens.txt; sherpa drops a token it does not
know without a word, so a character missing from tokens.txt stops the run here instead.
  python tools/sherpa_read.py PHONE_DIR LIST.tsv OUT_DIR [--int8] [--steps N]
  [--guidance G] [--threads 4] [--keys k1,k2]"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf


def token_names(path: Path) -> set:
    """The names in a tokens.txt ("name id" or "name<TAB>id"; the space token's line starts with the space)."""
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    return {line[:len(line) - len(line.split()[-1]) - 1] for line in lines}   # the id and its one separator off


def as_tokens(phonemes: str, names: set, key: str) -> list:
    missing = sorted(set(phonemes) - names)
    if missing:
        raise SystemExit(f"{key}: characters not in tokens.txt (sherpa would drop them): {missing}")
    return list(phonemes)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phone_dir", type=Path)
    ap.add_argument("list", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--steps", type=int, default=None, help="default phone.json's")
    ap.add_argument("--guidance", type=float, default=None, help="default phone.json's")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--keys", default="")
    a = ap.parse_args(argv)
    d = a.phone_dir
    cfg = json.load(open(d / "phone.json", encoding="utf-8"))["sherpa"]
    steps = a.steps or cfg["num_steps"]
    guidance = a.guidance if a.guidance is not None else cfg["guidance_scale"]
    suffix = "_int8" if a.int8 else ""
    zcfg = sherpa_onnx.OfflineTtsZipvoiceModelConfig(
        tokens=str(d / "tokens.txt"), encoder=str(d / f"text_encoder{suffix}.onnx"),
        decoder=str(d / f"fm_decoder{suffix}.onnx"), vocoder=str(d / "vocos_24khz_despike.onnx"),
        feat_scale=cfg["feat_scale"], t_shift=cfg["t_shift"], target_rms=cfg["target_rms"], guidance_scale=guidance)
    tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(zipvoice=zcfg, num_threads=a.threads, provider="cpu")))
    prompt, sr = sf.read(d / "prompt.wav", dtype="float32")
    if sr != 24000 or prompt.ndim != 1:
        raise SystemExit(f"{d / 'prompt.wav'}: need mono 24 kHz, got {sr} Hz, {prompt.ndim} dims")
    prompt_text = (d / "prompt_phonemes.txt").read_text(encoding="utf-8").strip("\n")
    names = token_names(d / "tokens.txt")
    prompt_tokens = as_tokens(prompt_text, names, "prompt")
    keys = set(filter(None, a.keys.split(",")))
    a.out.mkdir(parents=True, exist_ok=True)
    timing = ["key\taudio_s\ttaken_s"]
    for row in open(a.list, encoding="utf-8"):
        key, _, _, phonemes = row.rstrip("\n").split("\t")
        if keys and key not in keys:
            continue
        g = sherpa_onnx.GenerationConfig()
        g.tokens = [prompt_tokens, as_tokens(phonemes, names, key)]
        g.extra = cfg.get("extra", {})   # min_words_in_sentence: the line is not split at ; (ژ) or ?
        g.reference_audio = prompt.tolist()
        g.reference_sample_rate = sr
        g.reference_text = prompt_text   # unused on the tokens path, but Generate returns nothing without it
        g.num_steps = steps
        g.speed = cfg["speed"]
        g.silence_scale = cfg["silence_scale"]
        t0 = time.time()
        audio = tts.generate("", g)
        taken = time.time() - t0
        y = np.asarray(audio.samples, dtype=np.float32)
        if not len(y):
            raise SystemExit(f"{key}: sherpa returned no audio")
        sf.write(a.out / f"{key}.wav", y, audio.sample_rate, subtype="FLOAT")
        timing.append(f"{key}\t{len(y) / audio.sample_rate:.3f}\t{taken:.3f}")
    (a.out / "timing.tsv").write_text("\n".join(timing) + "\n", encoding="utf-8")
    t = np.array([[float(x) for x in l.split("\t")[1:]] for l in timing[1:]])
    print(f"{len(t)} lines, {t[:, 0].sum():.1f} s of audio in {t[:, 1].sum():.1f} s (RTF {t[:, 1].sum() / t[:, 0].sum():.3f}), "
          f"{steps} steps, guidance {guidance}, {'int8' if a.int8 else 'fp32'}, {a.threads} threads -> {a.out}")


if __name__ == "__main__":
    main()

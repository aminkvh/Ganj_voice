# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Regenerate real recordings from part-way along the flow (ganj_voice.regen): each line's own mel is noised to
flow time T and the shipped Euler schedule (64 steps, t_shift 0.5, guidance as given) finishes it, conditioned on the
line's first --cond-frac of real frames (as in training) and its own transcript, at its real length. T = 1 is
copy-synthesis, T = 0 a whole generation in the reciter's voice and timing. The same noise for every T of a line.
--refines splits each remaining step (regen.tail_times: 4 = four times finer, 0 = one jump to the end; 1 = shipped).
Decoded with infer_despike's vocoder (despike, GUIDE_TMIN / GUIDE_TMAX as there); writes RES_DIR/t<T*100>[r<refine>]/<key>.wav.
Run from $ZIPVOICE_DIR/egs/zipvoice with PYTHONPATH=../../ in the ZipVoice environment (recipes/zipvoice/regen_partial.sh)."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages", "ganj-voice"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_despike  # noqa: E402,F401  (patches iz.get_vocoder: despike before Vocos; guidance interval)
import zipvoice.bin.infer_zipvoice as iz  # noqa: E402
from ganj_voice.regen import condition_mask, finish, noised, paste_condition, tail_times  # noqa: E402
from zipvoice.models.zipvoice import ZipVoice  # noqa: E402
from zipvoice.tokenizer.tokenizer import SimpleTokenizer  # noqa: E402
from zipvoice.utils.checkpoint import load_checkpoint  # noqa: E402
from zipvoice.utils.feature import VocosFbank  # noqa: E402
from zipvoice.utils.infer import load_prompt_wav, rms_norm  # noqa: E402

SR, TARGET_RMS, FEAT_SCALE = 24000, 0.1, 0.1


def load_model(model_dir: Path, ckpt: str, device: torch.device):
    tok = SimpleTokenizer(token_file=model_dir / "tokens.txt")
    cfg = json.load(open(model_dir / "model.json"))
    model = ZipVoice(**cfg["model"], vocab_size=tok.vocab_size, pad_id=tok.pad_id)
    load_checkpoint(filename=model_dir / ckpt, model=model, strict=True)
    return model.to(device).eval(), tok


def line_texts(test_list: Path) -> dict:
    """key -> the line's label (4th column of a ZipVoice test list: the transcript as the model was trained on)."""
    rows = (l.rstrip("\n").split("\t") for l in open(test_list, encoding="utf-8") if l.strip())
    return {r[0]: r[3] for r in rows}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-name", required=True)
    ap.add_argument("--test-list", type=Path, required=True)
    ap.add_argument("--set", type=Path, required=True, help="eval set jsonl (key -> human clip)")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--t-starts", default="0.0,0.5,0.8,1.0")
    ap.add_argument("--refines", default="1", help="comma list; 1 = the shipped steps")
    ap.add_argument("--guidance-scale", type=float, default=2.0)
    ap.add_argument("--num-step", type=int, default=64)
    ap.add_argument("--t-shift", type=float, default=0.5)
    ap.add_argument("--cond-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--res-dir", type=Path, required=True)
    a = ap.parse_args()

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model, tok = load_model(a.model_dir, a.checkpoint_name, device)
    vocoder = iz.get_vocoder(None).to(device).eval()
    fbank = VocosFbank()
    texts = line_texts(a.test_list)
    human = {r["key"]: r["human"] for r in map(json.loads, open(a.set, encoding="utf-8"))}
    for key in a.keys.split(","):
        wav, rms = rms_norm(load_prompt_wav(str(a.data / human[key]), sampling_rate=SR), TARGET_RMS)
        feat = (fbank.extract(wav, sampling_rate=SR).to(device) * FEAT_SCALE).unsqueeze(0)   # (1, T, F)
        lens = torch.tensor([feat.size(1)], device=device)
        text_cond, pad = model.forward_text_train(tokens=tok.texts_to_token_ids([texts[key]]), features_lens=lens)
        mask = condition_mask(feat.size(1), a.cond_frac).to(device)
        speech_cond = torch.where(mask.unsqueeze(-1), torch.zeros_like(feat), feat)
        noise = torch.randn(feat.shape, generator=torch.Generator().manual_seed(a.seed), dtype=feat.dtype).to(device)

        def step(t, x):
            return model.solver.model(t=t, x=x, text_condition=text_cond, speech_condition=speech_cond,
                                      padding_mask=pad, guidance_scale=a.guidance_scale)

        for T, r in ((T, r) for T in map(float, a.t_starts.split(",")) for r in map(int, a.refines.split(","))):
            times = tail_times(T, a.num_step, a.t_shift, r).to(device)
            out = paste_condition(finish(step, noised(feat, noise, float(times[0])), times, 0), feat, mask)
            y = vocoder.decode(out.permute(0, 2, 1) / FEAT_SCALE).squeeze(1).clamp(-1, 1)
            if rms < TARGET_RMS:
                y = y * rms / TARGET_RMS
            d = a.res_dir / (f"t{round(T * 100):03d}" + ("" if r == 1 else f"r{r}"))
            d.mkdir(parents=True, exist_ok=True)
            torchaudio.save(str(d / f"{key}.wav"), y.cpu(), sample_rate=SR)
            print(f"{key} T={T} start={float(times[0]):.3f} steps={len(times) - 1} -> {d.name}/{key}.wav", flush=True)
    print("REGEN_DONE", flush=True)


if __name__ == "__main__":
    main()

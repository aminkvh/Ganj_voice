# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Vocos fine-tune pairs from the flow model's own output (ganj_voice.regen; the laser forms in the model's
last steps). Cuts of the model's own training manifest; per batch the real mels are noised to a random flow time
T ~ U[--t-lo, --t-hi] and finished on the shipped schedule (64 steps, t_shift 0.5, guidance --guidance-scale, guided
only for t <= one of --guide-tmax at random: the recipe that ships decides which texture the vocoder must learn),
conditioned on a random first 0 .. --cond-max of real frames (pasted back), with each clip's own label at its real
length. Writes OUT/<cut id>.npz: gen and real log-mels in the Vocos domain (float16, (100, n)), the cut's audio (int16,
24 kHz) and T, tmax, cond (frames). Held out: every eval set line and --exclude ids.
Run from $ZIPVOICE_DIR/egs/zipvoice with PYTHONPATH=../../ in the ZipVoice environment (recipes/zipvoice/regen_pairs.sh)."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from lhotse import load_manifest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages", "ganj-voice"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ganj_voice.guidance import interval_scale  # noqa: E402
from ganj_voice.regen import condition_masks, finish, frame_batches, noised, paste_condition, tail_times  # noqa: E402
from regen_partial import FEAT_SCALE, load_model  # noqa: E402


def pick_cuts(manifest: Path, held_out: set, n: int, max_sec: float, seed: int) -> list:
    cuts = [c for c in load_manifest(manifest) if c.id not in held_out and c.duration <= max_sec]
    rng = np.random.default_rng(seed)
    return [cuts[i] for i in sorted(rng.choice(len(cuts), size=min(n, len(cuts)), replace=False))]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--checkpoint-name", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--set", type=Path, required=True, help="eval set jsonl: its keys are held out")
    ap.add_argument("--exclude", default="", help="comma list of more held-out cut ids (voice prompt clips)")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--max-sec", type=float, default=15.0)
    ap.add_argument("--max-frames", type=int, default=8000, help="padded frames per batch")
    ap.add_argument("--t-lo", type=float, default=0.5)
    ap.add_argument("--t-hi", type=float, default=0.9)
    ap.add_argument("--guide-tmax", default="0.6,1.0")
    ap.add_argument("--guidance-scale", type=float, default=2.0)
    ap.add_argument("--cond-max", type=float, default=0.3)
    ap.add_argument("--num-step", type=int, default=64)
    ap.add_argument("--t-shift", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if {"GUIDE_TMIN", "GUIDE_TMAX"} & set(os.environ):
        sys.exit("GUIDE_TMIN / GUIDE_TMAX would patch the guidance (infer_despike) on top of --guide-tmax: unset them")

    device = torch.device("cuda", 0)
    model, tok = load_model(a.model_dir, a.checkpoint_name, device)
    held_out = {json.loads(l)["key"] for l in open(a.set, encoding="utf-8")} | set(filter(None, a.exclude.split(",")))
    cuts = pick_cuts(a.manifest, held_out, a.n, a.max_sec, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    rng, tmaxes = np.random.default_rng(a.seed), [float(x) for x in a.guide_tmax.split(",")]
    batches = frame_batches([c.num_frames for c in cuts], a.max_frames)
    print(f"{len(cuts)} cuts, {sum(c.duration for c in cuts) / 3600:.1f} h, {len(batches)} batches", flush=True)
    t0, done = time.time(), 0
    for bi, idx in enumerate(batches):
        bc = [cuts[i] for i in idx]
        T, tmax, frac = rng.uniform(a.t_lo, a.t_hi), float(rng.choice(tmaxes)), rng.uniform(0.0, a.cond_max)
        if all((a.out / f"{c.id}.npz").exists() for c in bc):
            continue                                                     # resumable: same seed, same draws
        feats = [torch.from_numpy(c.load_features()) * FEAT_SCALE for c in bc]
        lens = torch.tensor([f.size(0) for f in feats], device=device)
        x1 = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True).to(device)
        text_cond, pad = model.forward_text_train(tokens=tok.texts_to_token_ids([c.supervisions[0].text for c in bc]),
                                                  features_lens=lens)
        mask = condition_masks(lens, frac, x1.size(1))
        speech_cond = torch.where(mask.unsqueeze(-1), torch.zeros_like(x1), x1)
        noise = torch.randn(x1.shape, generator=torch.Generator().manual_seed(a.seed + bi)).to(device)

        def step(t, x):
            return model.solver.model(t=t, x=x, text_condition=text_cond, speech_condition=speech_cond, padding_mask=pad,
                                      guidance_scale=interval_scale(float(t), a.guidance_scale, 0.0, tmax))

        times = tail_times(T, a.num_step, a.t_shift, 1).to(device)
        out = paste_condition(finish(step, noised(x1, noise, float(times[0])), times, 0), x1, mask)
        for j, c in enumerate(bc):
            n = int(lens[j])
            audio = np.clip(c.load_audio()[0] * 32768, -32768, 32767).astype(np.int16)
            np.savez(a.out / f"{c.id}.npz", gen=(out[j, :n] / FEAT_SCALE).T.half().cpu().numpy(),
                     real=(x1[j, :n] / FEAT_SCALE).T.half().cpu().numpy(), audio=audio, T=float(times[0]), tmax=tmax,
                     cond=int((~mask[j, :n]).sum()))
        done += len(bc)
        if bi % 20 == 0:
            el = time.time() - t0
            print(f"batch {bi + 1}/{len(batches)}: {done} cuts, {el / 60:.1f} min, {done / max(el, 1e-9) * 60:.0f} cuts/min",
                  flush=True)
    print(f"PAIRS_DONE {done} cuts in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()

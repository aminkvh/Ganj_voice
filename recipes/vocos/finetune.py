# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Fine-tune Vocos (charactr/vocos-mel-24khz, MIT) on the flow model's own mels (the laser forms in the model's
last steps; copy-synthesis is clean). Pairs from recipes/zipvoice/regen_pairs.py: input the regenerated log-mel (the real
one at --real-rate), despiked as at inference (ganj_voice.melsmooth); target the real audio under it
(ganj_voice.vocos_data). The generator starts from the pretrained weights; the discriminators (MPD + MRD,
vocos.discriminators) start fresh and train alone for --disc-warmup steps, then both with Vocos's own losses (hinge
GAN, feature matching, mel L1 x 45, MRD x 0.1). Saves OUT/step-<k>/{config.yaml, pytorch_model.bin} (the format of
infer_zipvoice's --vocoder-path) and OUT/state.pt (resume).
Run in the ZipVoice environment (recipes/vocos/finetune.sh)."""
import argparse
import glob
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from vocos import Vocos
from vocos.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from vocos.loss import DiscriminatorLoss, FeatureMatchingLoss, GeneratorLoss, MelSpecReconstructionLoss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages", "ganj-voice"))
from ganj_voice.melsmooth import despike_mel  # noqa: E402
from ganj_voice.vocos_data import load_crop  # noqa: E402

HOP, CROP = 256, 65            # 65 frames -> 16384 samples, Vocos's own training length
MEL_COEFF, MRD_COEFF = 45.0, 0.1


class Pairs(Dataset):
    def __init__(self, files, real_rate, seed):
        self.files, self.real_rate, self.seed = files, real_rate, seed

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i, int(time.time_ns() % 2**31)))
        while True:
            got = load_crop(self.files[i], CROP, HOP, self.real_rate, rng)
            if got is not None:
                return torch.from_numpy(got[0]), torch.from_numpy(got[1])
            i = int(rng.integers(len(self.files)))


def synth(gen, mel):
    """Vocos.decode without its inference_mode (training needs the graph)."""
    return gen.head(gen.backbone(mel))


def batches(loader):
    while True:
        yield from loader


def export(gen, config, d: Path):
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(config, d / "config.yaml")
    torch.save(gen.state_dict(), d / "pytorch_model.bin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--disc-warmup", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-gen", type=float, default=3e-5)
    ap.add_argument("--lr-disc", type=float, default=1e-4)
    ap.add_argument("--real-rate", type=float, default=0.25)
    ap.add_argument("--save-every", type=int, default=2500)
    ap.add_argument("--val-files", type=int, default=100)
    ap.add_argument("--seed", type=int, default=666)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device("cuda", 0)
    files = sorted(glob.glob(str(a.pairs / "*.npz")))
    val, train = files[:a.val_files], files[a.val_files:]
    loader = DataLoader(Pairs(train, a.real_rate, a.seed), batch_size=a.batch, shuffle=True, num_workers=4,
                        drop_last=True, persistent_workers=True)
    val_set = [load_crop(f, CROP, HOP, 0.0, np.random.default_rng(k)) for k, f in enumerate(val)]
    val_set = [v for v in val_set if v is not None]

    from huggingface_hub import hf_hub_download
    config = hf_hub_download("charactr/vocos-mel-24khz", "config.yaml")
    gen = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(dev).train()
    mpd, mrd = MultiPeriodDiscriminator().to(dev), MultiResolutionDiscriminator().to(dev)
    opt_g = torch.optim.AdamW(gen.parameters(), lr=a.lr_gen, betas=(0.8, 0.9))
    opt_d = torch.optim.AdamW(list(mpd.parameters()) + list(mrd.parameters()), lr=a.lr_disc, betas=(0.8, 0.9))
    d_loss, g_loss, fm_loss = DiscriminatorLoss(), GeneratorLoss(), FeatureMatchingLoss()
    mel_loss = MelSpecReconstructionLoss().to(dev)
    start = 0
    if (a.out / "state.pt").exists():
        s = torch.load(a.out / "state.pt", map_location=dev, weights_only=False)
        gen.load_state_dict(s["gen"]); mpd.load_state_dict(s["mpd"]); mrd.load_state_dict(s["mrd"])
        opt_g.load_state_dict(s["opt_g"]); opt_d.load_state_dict(s["opt_d"]); start = s["step"]
        print(f"resumed at step {start}", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)

    def val_mel():
        gen.eval()
        with torch.no_grad():
            ls = [float(mel_loss(synth(gen, despike_mel(torch.from_numpy(m)[None].to(dev))),
                                 torch.from_numpy(y)[None].to(dev))) for m, y in val_set]
        gen.train()
        return float(np.mean(ls))

    print(f"{len(train)} train / {len(val_set)} val pairs; val mel L1 at start {val_mel():.4f}", flush=True)
    t0, it = time.time(), batches(loader)
    for step in range(start, a.steps):
        mel, y = next(it)
        mel, y = despike_mel(mel.to(dev, non_blocking=True)), y.to(dev, non_blocking=True)
        joint = step >= a.disc_warmup
        with torch.set_grad_enabled(joint):
            y_hat = synth(gen, mel)
        r_mp, g_mp, _, _ = mpd(y=y, y_hat=y_hat.detach())
        r_mr, g_mr, _, _ = mrd(y=y, y_hat=y_hat.detach())
        l_mp, lr_mp, _ = d_loss(r_mp, g_mp)
        l_mr, lr_mr, _ = d_loss(r_mr, g_mr)
        ld = l_mp / len(lr_mp) + MRD_COEFF * l_mr / len(lr_mr)
        opt_d.zero_grad(set_to_none=True)
        ld.backward()
        opt_d.step()
        lg = lm = torch.zeros(())
        if joint:
            _, g_mp, f_rmp, f_gmp = mpd(y=y, y_hat=y_hat)
            _, g_mr, f_rmr, f_gmr = mrd(y=y, y_hat=y_hat)
            lg_mp, lgl_mp = g_loss(g_mp)
            lg_mr, lgl_mr = g_loss(g_mr)
            lm = mel_loss(y_hat, y)
            lg = (lg_mp / len(lgl_mp) + MRD_COEFF * lg_mr / len(lgl_mr) + fm_loss(f_rmp, f_gmp) / len(f_rmp)
                  + MRD_COEFF * fm_loss(f_rmr, f_gmr) / len(f_rmr) + MEL_COEFF * lm)
            opt_g.zero_grad(set_to_none=True)
            lg.backward()
            opt_g.step()
        if (step + 1) % 100 == 0:
            el = time.time() - t0
            print(f"step {step + 1}: disc {float(ld):.3f} gen {float(lg):.3f} mel {float(lm):.4f} "
                  f"({(step + 1 - start) / el:.2f} it/s)", flush=True)
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            export(gen, config, a.out / f"step-{step + 1}")
            torch.save({"gen": gen.state_dict(), "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                        "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "step": step + 1}, a.out / "state.pt")
            print(f"saved step-{step + 1}; val mel L1 {val_mel():.4f}", flush=True)
    print("VOCOS_FT_DONE", flush=True)


if __name__ == "__main__":
    main()

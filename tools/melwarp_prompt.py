# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Build a voice prompt with the mel-domain disguise (ganj_voice.melwarp). Runs in the ZipVoice
environment (Vocos):
  python tools/melwarp_prompt.py SRC.wav OUT.wav --alpha 1.10 [--fade 3000 4000] [--fricative 0.80]
SRC is resampled to 24 kHz, turned into Vocos's log-mel (100 bins, hop 256), warped, decoded by the shipped Vocos
(charactr/vocos-mel-24khz, MIT; weights loaded with weights_only) and written as 24 kHz 16-bit wav. Denoise the source
first with tools/make_prompt.py --denoise-floor -20 (as د۱) when it needs it."""
import argparse
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "packages", "ganj-voice"))
from ganj_voice.melwarp import mel_centres, warp_logmel  # noqa: E402

SR = 24000


def vocos():
    from vocos import Vocos
    snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--charactr--vocos-mel-24khz/snapshots/*/"))[0]
    voc = Vocos.from_hparams(os.path.join(snap, "config.yaml"))
    voc.load_state_dict(torch.load(os.path.join(snap, "pytorch_model.bin"), weights_only=True, map_location="cpu"))
    return voc.eval()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--alpha", type=float, required=True, help="frequency-axis factor, e.g. 1.10")
    ap.add_argument("--fade", type=float, nargs=2, default=None, metavar=("LO", "HI"), help="Hz: the warp fades out")
    ap.add_argument("--fricative", type=float, default=None, help="own factor for fricative frames, e.g. 0.80")
    a = ap.parse_args(argv)
    y, sr = sf.read(a.src, dtype="float32", always_2d=True)
    y = torch.from_numpy(y.mean(1))
    if sr != SR:
        y = torchaudio.functional.resample(y, sr, SR)
    voc = vocos()
    with torch.no_grad():
        lm = voc.feature_extractor(y[None]).squeeze(0).numpy()
        w = warp_logmel(lm, mel_centres(SR, lm.shape[0]), a.alpha, a.fade, a.fricative)
        z = voc.decode(torch.from_numpy(w).float()[None]).squeeze().numpy()
    z = z * np.sqrt(np.mean(y.numpy() ** 2) / max(np.mean(z ** 2), 1e-12))   # keep the input loudness
    sf.write(a.out, np.clip(z, -1, 1), SR, subtype="PCM_16")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

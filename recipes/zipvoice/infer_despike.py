# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""ZipVoice inference with the generated mel despiked before Vocos (ganj_voice.melsmooth).
Same CLI as zipvoice.bin.infer_zipvoice; DESPIKE_START (bin, default 60) and DESPIKE_TIME=1 (3x3 instead of 3x1);
TSMOOTH=3|5 also averages the bins below DESPIKE_START over that many frames (melsmooth.smooth_time; off by default).
GUIDE_TMIN / GUIDE_TMAX guide only flow times in that interval (ganj_voice.guidance; off by default).
TAIL_REFINE=k splits every sampling step from flow time TAIL_FROM (default 0.8) on into k (ganj_voice.schedule;
off by default; 64 steps with k = 4 run 57 + 28)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages", "ganj-voice"))
from ganj_voice.guidance import limit_guidance  # noqa: E402
from ganj_voice.melsmooth import START_BIN, despike_mel, smooth_time  # noqa: E402
from ganj_voice.schedule import refine_sampler  # noqa: E402
import zipvoice.bin.infer_zipvoice as iz  # noqa: E402
import zipvoice.models.modules.solver as solver  # noqa: E402

if "GUIDE_TMIN" in os.environ or "GUIDE_TMAX" in os.environ:
    limit_guidance(solver.DiffusionModel, float(os.environ.get("GUIDE_TMIN", 0.0)), float(os.environ.get("GUIDE_TMAX", 1.0)))
if int(os.environ.get("TAIL_REFINE", 1)) != 1:
    refine_sampler(solver.EulerSolver, float(os.environ.get("TAIL_FROM", 0.8)), int(os.environ["TAIL_REFINE"]))

_get_vocoder = iz.get_vocoder


def get_vocoder(*a, **k):
    voc = _get_vocoder(*a, **k)
    decode = voc.decode
    start, time = int(os.environ.get("DESPIKE_START", START_BIN)), os.environ.get("DESPIKE_TIME") == "1"
    frames = int(os.environ.get("TSMOOTH", 0))
    clean = (lambda m: smooth_time(despike_mel(m, start, time), frames, start)) if frames else (lambda m: despike_mel(m, start, time))
    voc.decode = lambda feats, *x, **y: decode(clean(feats), *x, **y)
    return voc


iz.get_vocoder = get_vocoder
if __name__ == "__main__":
    import logging
    logging.basicConfig(format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO, force=True)
    iz.main()

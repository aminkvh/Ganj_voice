# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import json

import numpy as np
import soundfile as sf


def test_postclean_writes_every_clip_cleaned_at_its_own_rate_and_records_the_settings(tmp_path):
    from tools.postclean import postclean_dir
    sr = 24000
    t = np.arange(sr * 2) / sr
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    for k in ("1-0", "2-3"):
        sf.write(str(src / f"{k}.wav"), (0.05 * np.sin(2 * np.pi * 48 * t) + 0.1 * np.sin(2 * np.pi * 110 * t)), sr,
                 subtype="PCM_16")
    n = postclean_dir(src, dst, highpass_hz=70.0, denoise_floor_db=None)
    assert n == 2
    for k in ("1-0", "2-3"):
        y, r = sf.read(str(dst / f"{k}.wav"))
        assert r == sr and sf.info(str(dst / f"{k}.wav")).subtype == "PCM_16" and len(y) == len(t)
        spec = np.abs(np.fft.rfft(y))
        assert spec[96] < 0.05 * spec[220]          # 48 Hz gone, 110 Hz kept (bins at 0.5 Hz)
    assert json.loads((dst / "postclean.json").read_text()) == {"source": str(src), "highpass_hz": 70.0,
                                                                "denoise_floor_db": None, "deess_db": None}


def test_postclean_cli_cleans_a_synthesis_dir_in_place_when_given_paths(tmp_path):
    # synth_voice.sh high-passes its own output dir
    from tools.postclean import main
    sr = 24000
    t = np.arange(sr) / sr
    sf.write(str(tmp_path / "k.wav"), 0.05 * np.sin(2 * np.pi * 48 * t) + 0.1 * np.sin(2 * np.pi * 110 * t), sr,
             subtype="PCM_16")
    main(["--paths", str(tmp_path), str(tmp_path)])
    spec = np.abs(np.fft.rfft(sf.read(str(tmp_path / "k.wav"))[0]))   # 1 Hz bins
    assert spec[48] < 0.05 * spec[110]
    assert json.loads((tmp_path / "postclean.json").read_text())["highpass_hz"] == 70.0


def test_postclean_cli_de_esses_when_given_a_depth_and_records_it(tmp_path):
    # the voices' س/ش are 5-11 dB too loud in 5-9 kHz (they whistle)
    from tools.postclean import main
    sr = 24000
    rng = np.random.default_rng(1)
    hiss = (0.05 * rng.normal(0, 1, sr)).astype(np.float32)
    sf.write(str(tmp_path / "k.wav"), hiss, sr, subtype="PCM_16")
    main(["--paths", str(tmp_path), str(tmp_path / "out"), "--highpass", "0", "--deess", "6"])
    y = sf.read(str(tmp_path / "out" / "k.wav"))[0]
    f = np.fft.rfftfreq(len(y), 1 / sr)
    before, after = (np.abs(np.fft.rfft(x)) ** 2 for x in (hiss, y))
    band = (f > 5500) & (f < 9000)
    assert 10 * np.log10(after[band].sum() / before[band].sum()) < -4.0
    assert json.loads((tmp_path / "out" / "postclean.json").read_text())["deess_db"] == 6.0

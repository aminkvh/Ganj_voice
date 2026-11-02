# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np

from ganj_corpus.clean import clipping_fraction, loudness_normalize, reciter_genders, summarize_ina


def test_summarize_ina(tmp_path):
    p = tmp_path / "1.csv"
    p.write_text("label,start,stop\nmusic,0,5\nmale,5,15\nnoEnergy,15,16\nfemale,16,18\n")
    s = summarize_ina(p)
    assert abs(s["music_fraction"] - 5 / 18) < 1e-6 and s["male_s"] == 10 and s["female_s"] == 2


def test_reciter_genders_majority_and_override(tmp_path):
    (tmp_path / "1.csv").write_text("label,start,stop\nmale,0,10\n")
    (tmp_path / "2.csv").write_text("label,start,stop\nfemale,0,3\n")
    (tmp_path / "3.csv").write_text("label,start,stop\nfemale,0,30\n")
    rows = [{"id": 1, "audioArtist": "A"}, {"id": 2, "audioArtist": "A"}, {"id": 3, "audioArtist": "B"}]
    assert reciter_genders(rows, tmp_path) == {"A": "male", "B": "female"}
    ov = tmp_path / "ov.csv"
    ov.write_text("audioArtist,gender\nA,female\n")
    assert reciter_genders(rows, tmp_path, overrides_csv=ov)["A"] == "female"


def test_loudness_normalize_hits_target_and_clamps_peak():
    sr = 22050
    y = (0.01 * np.sin(2 * np.pi * 220 * np.arange(sr * 3) / sr)).astype(np.float32)
    z = loudness_normalize(y, sr, target=-23.0)
    import pyloudnorm as pyln
    assert abs(pyln.Meter(sr).integrated_loudness(z) + 23.0) < 1.0
    assert np.abs(z).max() <= 10 ** (-1 / 20) + 1e-6


def test_clipping_fraction():
    y = np.zeros(1000, dtype=np.float32)
    y[:10] = 1.0
    assert abs(clipping_fraction(y) - 0.01) < 1e-9

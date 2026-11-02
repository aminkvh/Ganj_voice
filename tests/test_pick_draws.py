# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import json

from ganj_eval.vowelcheck import VowelModel

ROW = {"key": "22672-3", "lines": ["آبش کردم"], "phonemes": "@ab/š k/rd/m"}


class FakeChecker:
    """analyse() as ganj_eval.labelcheck.LineChecker returns it, the vowels given per file name."""

    def __init__(self, vowels):
        self.vowels = vowels

    def analyse(self, path, lines, label):
        return {"words": ["آبش", "کردم"], "present": [True, True], "fit": -1.0, "targets": [], "features": [],
                "vowels": self.vowels[path.rsplit("/", 1)[-1]]}


def _v(f1, f2, vowel="/", word=0):
    return {"word": word, "i": 1, "j": 2, "vowel": vowel, "f1": f1, "f2": f2}


def test_score_counts_the_short_vowels_the_voice_model_calls_wrong():
    from tools.pick_draws import score
    m = VowelModel({"/": [6.55, 7.32], "e": [6.12, 7.61], "o": [6.18, 6.87]}, [0.08, 0.08])
    checker = FakeChecker({"d8.wav": [_v(405.0, 2040.0)], "d1.wav": [_v(630.0, 1780.0)]})
    s8 = score(checker, None, "x/d8.wav", ROW, vowels=m)
    assert s8["wrong_vowel"] == 1 and s8["vowels_wrong"] == [{"word": "آبش", "vowel": "/", "f1": 405.0, "f2": 2040.0}]
    assert score(checker, None, "x/d1.wav", ROW, vowels=m)["wrong_vowel"] == 0
    assert score(checker, None, "x/d8.wav", ROW)["wrong_vowel"] == 0        # no vowel model: nothing is wrong


def test_fit_vowels_learns_a_voice_from_every_render_of_a_known_line(tmp_path):
    from tools.pick_draws import fit_vowels
    for name in ("22672-3_d1.wav", "22672-3_d2.wav", "99999-0.wav"):
        (tmp_path / name).write_bytes(b"")
    pts = {"22672-3_d1.wav": [_v(700, 1500), _v(450, 2000, "e"), _v(480, 950, "o")],
           "22672-3_d2.wav": [_v(720, 1550), _v(470, 2050, "e"), _v(500, 1000, "o")],
           "99999-0.wav": [_v(300, 3000)]}                                  # not in the set: not used
    out = tmp_path / "vowels.json"
    counts = fit_vowels(FakeChecker(pts), {"22672-3": ROW}, [tmp_path], out)
    assert counts == {"/": 2, "e": 2, "o": 2}
    m = VowelModel.load(out)
    assert m.wrong("/", 455, 2020) and not m.wrong("/", 710, 1520)
    assert set(json.loads(out.read_text())["centres"]) == {"/", "e", "o"}

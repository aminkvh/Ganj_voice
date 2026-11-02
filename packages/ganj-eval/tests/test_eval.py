# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np

from ganj_eval.listen import render_html
from ganj_eval.metrics import f0_logstd, line_end_recall, norm, tiny_word_recall, wer
from ganj_eval.set_builder import build_set, classify


def row(i, kind="verse", lines=("الف ب", "پ ت"), langs=(None, None), title="غزل", poem=None):
    return {"rec_id": i, "unit": 0, "poem_id": poem if poem is not None else i, "kind": kind, "lines": list(lines), "langs": list(langs),
            "title": title, "phonemes": "x", "clip": f"clips/{i}/0.wav"}


def test_classify_priority():
    assert classify(row(1, kind="prose")) == "prose"
    assert classify(row(2, langs=("ar-AE", None))) == "arabic"
    assert classify(row(3, lines=("بِشنَو", "ب"))) == "vowelled"
    assert classify(row(4, title="مولانا » مثنوی معنوی » دفتر اول")) == "masnavi"
    assert classify(row(5)) == "plain"


def test_build_set_uses_only_val_poems_and_caps_per_class():
    rows = [row(i, poem=i) for i in range(40)] + [row(100 + i, kind="prose", poem=100 + i) for i in range(5)]
    s = build_set(rows, val_poems={*range(0, 20), 100, 101}, n_per_class=3)
    assert all(r["rec_id"] < 20 or r["rec_id"] in (100, 101) for r in s)
    assert sum(r["cls"] == "plain" for r in s) == 3 and sum(r["cls"] == "prose" for r in s) == 2
    assert {r["key"] for r in s} == {f"{r['rec_id']}-{r['unit']}" for r in s}


def test_norm_and_wer():
    assert norm("بِشنَو از نی‌ چون") == "بشنو از نی چون"
    assert wer("بشنو از نی", "بشنو از نی") == 0.0 and wer("بشنو از نی", "بشنو نی") > 0


def test_probes():
    assert tiny_word_recall("ز کوی یار و از در", "کوی یار از در") == 0.0
    assert tiny_word_recall("ز کوی یار و از در", "ز کوی یار و از در") == 1.0
    assert line_end_recall(["بشنو از نی", "چون حکایت می‌کند"], "بشنو از نی چون حکایت") == 0.5


def test_f0_logstd_is_near_zero_for_a_steady_tone():
    sr = 22050
    y = (0.3 * np.sin(2 * np.pi * 150 * np.arange(sr * 2) / sr)).astype(np.float32)
    assert f0_logstd(y, sr) < 0.05


def test_render_html_lists_runs_and_keys():
    html = render_html([{"key": "1-0", "lines": ["الف", "ب"], "human": "human/1-0.wav"}], ["gyro", "v1"])
    assert "runs/gyro/1-0.wav" in html and "runs/v1/1-0.wav" in html and "human/1-0.wav" in html and "الف" in html


def test_build_set_derives_unit_from_clip_name_when_missing():
    r = {k: v for k, v in row(7, poem=7).items() if k != "unit"}
    r["clip"] = "clips/7/3.wav"
    s = build_set([r], val_poems={7}, n_per_class=1)
    assert s[0]["key"] == "7-3" and s[0]["unit"] == "3"


def test_asr_cache_entry_is_dropped_when_a_wav_is_rewritten(tmp_path):
    from ganj_eval.metrics import PersianASR
    asr = PersianASR.__new__(PersianASR)
    asr.cache_path = None
    asr.cache = {str(tmp_path / "a.wav"): "stale"}
    asr.forget(tmp_path / "a.wav")
    assert str(tmp_path / "a.wav") not in asr.cache


def test_ganjoor_language_ids_separate_arabic_from_azeri_turkish():
    assert classify(row(6, langs=(2, None))) == "arabic"      # Ganjoor languageId 2 = Arabic
    assert classify(row(7, langs=(3, 3))) == "turkish"        # languageId 3 = Azeri Turkish
    assert classify(row(8, langs=("az", None))) == "turkish"


def test_turkish_rows_never_enter_a_persian_eval_set():
    rows = [row(i, poem=i, langs=(3, 3)) for i in range(5)] + [row(10 + i, poem=10 + i) for i in range(3)]
    s = build_set(rows, val_poems=set(range(20)), n_per_class=10)
    assert {r["cls"] for r in s} == {"plain"}


def test_persian_summary_leaves_out_arabic_and_turkish_rows():
    from ganj_eval.run import persian_summary
    rows = [{"cls": "plain", "wer_gap": 0.1, "mos": 3.0, "tiny": 0.5, "line_end": 1.0},
            {"cls": "turkish", "wer_gap": 0.9, "mos": 1.0, "tiny": None, "line_end": 0.0},
            {"cls": "arabic", "wer_gap": 0.7, "mos": 1.0, "tiny": 0.0, "line_end": 0.0}]
    s = persian_summary(rows)
    assert s == {"n": 1, "wer_gap": 0.1, "mos": 3.0, "tiny": 0.5, "line_end": 1.0}


def test_ecapa_embed_resamples_from_the_clip_rate_not_an_assumed_22050():
    # ZipVoice writes 24 kHz; reading it as 22.05 kHz showed ECAPA every output 8 % slower and lower (2026-10-29)
    from ganj_eval.identity import Ecapa
    seen = []

    class FakeEnc:
        def encode_batch(self, x):
            seen.append(x.shape[-1])
            import torch
            return torch.zeros(1, 1, 4)

    ec = Ecapa.__new__(Ecapa)
    ec.enc = FakeEnc()
    ec.embed(np.zeros(24000, dtype=np.float32), 24000)
    ec.embed(np.zeros(22050, dtype=np.float32))          # the corpus rate stays the default
    assert seen == [16000, 16000]


def test_mos_scores_a_70hz_high_passed_copy_so_mains_hum_cannot_buy_naturalness(tmp_path):
    # UTMOSv2 scores a clip +0.35..0.56 higher with a steady 48/56 Hz hum added, human recordings too (2026-10-29)
    import soundfile as sf
    from ganj_eval.metrics import Mos
    sr = 24000
    t = np.arange(2 * sr) / sr
    p = tmp_path / "a.wav"
    sf.write(str(p), 0.05 * np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 200 * t), sr, subtype="PCM_16")
    seen = {}

    class FakeUtmos:
        def predict(self, **kw):
            seen.update(kw)
            return np.array([3.25])

    m = Mos.__new__(Mos)
    m.device, m.model = "cpu", FakeUtmos()
    assert m.score(p) == 3.25
    assert seen["sr"] == 16000 and seen["device"] == "cpu" and len(seen["data"]) == 32000
    spec = np.abs(np.fft.rfft(seen["data"]))           # 0.5 Hz bins
    assert spec[100] < 0.01 * spec[400]                 # 50 Hz gone, 200 Hz kept
    # a clip already high-passed by postclean.py is scored as it is: UTMOSv2 moves with what is left below 100 Hz
    # (F2 filtered once 2.74, twice 2.91), so every clip is scored as shipped, filtered exactly once
    assert m.score(p, prefiltered=True) == 3.25
    spec = np.abs(np.fft.rfft(seen["data"]))
    assert spec[100] > 0.3 * spec[400]


def test_rescore_mos_replaces_each_rows_mos_keeps_the_first_raw_value_and_rewrites_summary_and_report(tmp_path):
    import json
    from ganj_eval.run import rescore_mos
    run = tmp_path / "eval" / "runs" / "v"
    run.mkdir(parents=True)
    rows = [{"key": "a", "cls": "plain", "wer_gap": -0.1, "wer_synth": 0.3, "wer_human": 0.4, "mos": 3.0, "tiny": 0.5,
             "line_end": 1.0, "f0_ratio": 1.0, "seconds_ratio": 0.9},
            {"key": "b", "cls": "arabic", "wer_gap": 0.1, "wer_synth": 0.5, "wer_human": 0.4, "mos": 2.0, "tiny": None,
             "line_end": 0.0, "f0_ratio": 1.2, "seconds_ratio": 1.1}]
    summary = {"tag": "v", "n": 2, "wer_gap": 0.0, "wer_synth": 0.4, "wer_human": 0.4, "mos": 2.5, "line_end": 0.5,
               "f0_ratio": 1.1, "seconds_ratio": 0.94, "tiny": 0.5,  # length fixed in the summary only (rate fix)
               "persian": {"n": 1, "wer_gap": -0.1, "mos": 3.0, "tiny": 0.5, "line_end": 1.0},
               "by_class": {"plain": {"wer_gap": -0.1, "mos": 3.0, "line_end": 1.0},
                            "arabic": {"wer_gap": 0.1, "mos": 2.0, "line_end": 0.0}},
               "identity_max_cos": 0.5, "identity_closest": "x", "rate_fix_note": "n"}
    (run / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}))

    flags = []

    class FakeMos:
        def score(self, path, prefiltered=False):
            flags.append(prefiltered)
            return {"a.wav": 2.0, "b.wav": 1.0}[path.name]

    rescore_mos(tmp_path, "v", FakeMos())
    (run / "postclean.json").write_text("{}")
    rescore_mos(tmp_path, "v", FakeMos())           # a second pass must not overwrite the first raw value
    assert flags == [False, False, True, True]      # a post-cleaned run is not filtered again
    r = json.loads((run / "results.json").read_text())
    assert [x["mos"] for x in r["rows"]] == [2.0, 1.0] and [x["mos_raw"] for x in r["rows"]] == [3.0, 2.0]
    s = r["summary"]
    assert s["mos"] == 1.5 and s["mos_raw"] == 2.5 and s["persian"]["mos"] == 2.0
    assert s["by_class"]["arabic"]["mos"] == 1.0 and s["wer_gap"] == 0.0 and s["identity_max_cos"] == 0.5
    assert s["seconds_ratio"] == 0.94                 # only MOS fields change, never one fixed in the summary alone
    rep = (run / "report.md").read_text()
    assert "MOS=1.50" in rep and "identity: max cos 0.500 to x (2026-10-29 rate fix" in rep


def test_a_run_counts_as_post_cleaned_when_its_dir_or_its_wav_source_has_postclean_json(tmp_path):
    # `--wav-dir` copies only the wavs; a synth_voice.sh output dir scored that way must not be high-passed twice
    from ganj_eval.run import is_prefiltered
    out, src = tmp_path / "run", tmp_path / "synth"
    out.mkdir()
    src.mkdir()
    assert not is_prefiltered(out, src) and not is_prefiltered(out)
    (src / "postclean.json").write_text("{}")
    assert is_prefiltered(out, src) and not is_prefiltered(out)
    (out / "postclean.json").write_text("{}")
    assert is_prefiltered(out)


def test_length_ratio_compares_seconds_not_samples():
    from ganj_eval.run import seconds_ratio
    assert seconds_ratio(np.zeros(24000), 24000, np.zeros(22050), 22050) == 1.0

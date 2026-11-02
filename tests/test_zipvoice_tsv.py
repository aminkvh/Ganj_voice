# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from recipes.zipvoice.make_tsv import tsv_line


def test_tsv_line_uses_phonemes_and_absolute_path():
    assert tsv_line({"rec_id": 5, "unit": 2, "phonemes": "abš", "clip": "clips/5/2.wav"}, "/d") == "5-2\tabš\t/d/clips/5/2.wav"


def test_tsv_line_derives_unit_from_clip_name_when_missing():
    assert tsv_line({"rec_id": 5, "phonemes": "ab", "clip": "clips/5/2.wav"}, "/d") == "5-2\tab\t/d/clips/5/2.wav"


def test_training_rows_drop_narrow_band_like_the_piper_corpus():
    from recipes.zipvoice.make_tsv import training_rows
    rows = [{"clip": "a", "narrow": False}, {"clip": "b", "narrow": True}, {"clip": "c"}]
    assert [r["clip"] for r in training_rows(rows)] == ["a", "c"]


def test_test_line_pairs_eval_phonemes_with_the_prompt():
    from recipes.zipvoice.make_test_q import test_line
    item = {"key": "984-30", "phonemes": "goft"}
    assert test_line(item, ("p ph", "/x/p.wav")) == "984-30\tp ph\t/x/p.wav\tgoft"


def test_prompt_is_read_from_the_existing_test_list(tmp_path):
    from recipes.zipvoice.make_test_q import read_prompt
    (tmp_path / "t.tsv").write_text("a\tpt\t/w.wav\tx\nb\tpt\t/w.wav\ty\n", encoding="utf-8")
    assert read_prompt(tmp_path / "t.tsv") == ("pt", "/w.wav")

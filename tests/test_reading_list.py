# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import json
import types

from ganj_corpus.pauses import PAUSE_LONG, PAUSE_SHORT
from ganj_reading import tokens as T
from tools.reading_list import build, voice_has_pauses


def test_a_voice_has_pauses_when_its_token_list_has_the_pause_tokens():
    assert voice_has_pauses(f"_\t0\na\t1\n{PAUSE_SHORT}\t93\n{PAUSE_LONG}\t94\n")
    assert not voice_has_pauses("_\t0\na\t1\n")


def test_each_line_gets_a_list_row_with_the_voices_prompt_and_a_text_row_for_the_draw_checks():
    rows = [{"key": "p1", "lines": ["بردند و سعی کردند: یکی"], "kind": "prose"}]
    labels = [T.PROSE + "bord/nd o s/@y k/rd/nd yeki"]
    tsv, text = build(rows, labels, "prompt ph", pauses=True, va=True)
    key, prompt, wav, target = tsv[0].rstrip("\n").split("\t")
    assert (key, prompt, wav) == ("p1", "prompt ph", "prompt.wav")
    assert target == T.PROSE + f"bord/nd v/ s/@y k/rd/nd {PAUSE_SHORT} yeki"
    assert json.loads(text[0]) == {"key": "p1", "lines": rows[0]["lines"], "phonemes": T.PROSE + "bord/nd v/ s/@y k/rd/nd yeki"}


def test_a_line_the_g2p_could_not_read_is_left_out_and_named():
    rows = [{"key": "a", "lines": ["x"], "kind": "verse"}, {"key": "b", "lines": ["جان و دل"], "kind": "verse"}]
    tsv, text = build(rows, [None, T.VERSE + "jan o del"], "p", pauses=False, va=False)
    assert [l.split("\t")[0] for l in tsv] == ["b"] and len(text) == 1


def test_a_reviewers_reading_for_a_line_replaces_that_word_before_the_rules():
    from tools.reading_list import with_overrides
    rows = [{"key": "18347-45", "lines": ["بار خود از آب برون میکشد"], "kind": "verse"}]
    labels = [T.VERSE + "bare1 xod @/z @abe1 borun mikeš/d"]
    o = [{"id": "18347-45", "line": 0, "word": 3, "text": "آب", "label": "@ab"}]
    assert with_overrides(rows, labels, o) == [T.VERSE + "bare1 xod @/z @ab borun mikeš/d"]
    assert with_overrides(rows, labels, []) == labels


def test_a_voice_with_its_own_g2p_reads_verse_with_it_and_prose_with_the_base_model(tmp_path):
    from tools.reading_list import g2p_for

    class FakeG2P:
        def __init__(self, model_id="base", lexicon=None, device="cpu"):
            self.model_id = model_id

        def phonemize_units(self, units):
            return [self.model_id for _ in units]

    G = types.SimpleNamespace(__file__=str(tmp_path / "g2p.py"), load_lexicon=lambda p: {}, G2P=FakeG2P)
    units = [(["x"], "verse", None), (["y"], "prose", None)]
    assert g2p_for(tmp_path, G).phonemize_units(units) == ["base", "base"]
    (tmp_path / "g2p").mkdir()
    assert g2p_for(tmp_path, G).phonemize_units(units) == [str(tmp_path / "g2p"), "base"]


def test_a_non_persian_hemistich_is_dropped_before_the_g2p_and_named():
    # Ruled 2026-11-08: the voice skips Arabic and other non-Persian hemistichs (the app shows them and pauses)
    from tools.reading_list import drop_non_persian
    rows = [{"key": "2130-1", "kind": "verse", "langs": [2, None],
             "lines": ["اَلا یا اَیُّهَا السّاقی اَدِرْ کَأساً و ناوِلْها", "که عشق آسان نمود اوّل ولی افتاد مشکل‌ها"]},
            {"key": "2130-7", "kind": "verse",   # untagged: the detector finds the Arabic half
             "lines": ["حضوری گر همی خواهی از او غایب مشو حافظ", "مَتیٰ ما تَلْقَ مَنْ تَهْویٰ دَعِ الدُّنْیا وَ اَهْمِلْها"]},
            {"key": "x-1", "kind": "verse", "langs": [2, 2], "lines": ["الف", "ب"]}]
    kept, skipped = drop_non_persian(rows)
    assert [(r["key"], r["lines"], r["skipped"]) for r in kept] == [
        ("2130-1", ["که عشق آسان نمود اوّل ولی افتاد مشکل‌ها"], [0]),
        ("2130-7", ["حضوری گر همی خواهی از او غایب مشو حافظ"], [1])]
    assert [(r["key"], r["skipped"]) for r in skipped] == [("x-1", [0, 1])]


def test_a_unit_with_nothing_to_skip_is_passed_on_unchanged_and_prose_is_skipped_only_by_its_tag():
    from tools.reading_list import drop_non_persian
    rows = [{"key": "a", "kind": "verse", "lines": ["جان و دل", "دل و جان"]},
            {"key": "p", "kind": "prose", "lines": ["قال النبی علیه السلام"]}]
    kept, skipped = drop_non_persian(rows)
    assert kept == rows and skipped == []


def test_the_text_row_names_the_hemistichs_the_voice_left_out():
    rows = [{"key": "k", "lines": ["که عشق آسان نمود اوّل ولی افتاد مشکل‌ها"], "kind": "verse", "skipped": [0]}]
    _, text = build(rows, [T.VERSE + "ke @ešq"], "p", pauses=False, va=False)
    assert json.loads(text[0])["skipped"] == [0]

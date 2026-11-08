# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from ganj_corpus.pauses import PAUSE_LONG, PAUSE_SHORT
from ganj_reading import tokens as T
from ganj_voice.glue import glue_list_line, glue_vav

S, L = PAUSE_SHORT, PAUSE_LONG


def test_a_standalone_o_after_a_consonant_is_written_onto_that_word():
    assert glue_vav(T.PROSE + "bord/nd o s/@y") == T.PROSE + "bord/ndo s/@y"
    assert glue_vav("@/nduxt o n/xor/d o dig/r") == "@/nduxto n/xor/do dig/r"


def test_after_a_vowel_an_ezafe_or_a_pause_the_o_stays_apart():
    assert glue_vav("k/rdi o dig/r") == "k/rdi o dig/r"
    assert glue_vav("peyro o g/r") == "peyro o g/r"
    assert glue_vav(f"k/rd/nd {S} o yeki") == f"k/rd/nd {S} o yeki"
    assert glue_vav("dele1 o") == "dele1 o"


def test_only_a_standalone_o_is_moved_and_the_hemistich_mark_is_kept():
    ph = T.VERSE + f"jan o del{L}" + T.HEMI + "ham/n to"
    assert glue_vav(ph) == T.VERSE + f"jano del{L}" + T.HEMI + "ham/n to"


def test_a_list_line_has_only_its_target_glued():
    line = "k1\tdol/t o @asayeš\tprompt.wav\tbord/nd o s/@y\n"
    assert glue_list_line(line) == "k1\tdol/t o @asayeš\tprompt.wav\tbord/ndo s/@y\n"


def test_a_voice_with_pause_tokens_gets_a_long_pause_before_the_line_after_the_style_marks():
    from ganj_voice.glue import LONG, lead_pause
    assert LONG == PAUSE_LONG
    assert lead_pause(T.PROSE + f"goft {S} sahebe1") == T.PROSE + f"{L} goft {S} sahebe1"
    assert lead_pause(T.PROSE + f"{L} goft") == T.PROSE + f"{L} goft"   # once only


def test_the_list_gets_the_leading_pause_only_when_asked_and_keeps_the_prompt():
    line = "k1\tdol/t o\tprompt.wav\tbord/nd o s/@y\n"
    assert glue_list_line(line, lead=True) == f"k1\tdol/t o\tprompt.wav\t{L} bord/ndo s/@y\n"
    assert glue_list_line(line) == "k1\tdol/t o\tprompt.wav\tbord/ndo s/@y\n"


def test_the_cli_adds_the_leading_pause_when_the_voices_token_list_has_it(tmp_path):
    from ganj_voice.glue import main
    lst, tok = tmp_path / "list.tsv", tmp_path / "tokens.txt"
    lst.write_text("k1\tp\tprompt.wav\tbord/nd o s/@y\n", encoding="utf-8")
    tok.write_text(f"_\t0\n{S}\t93\n{L}\t94\n", encoding="utf-8")
    main([str(lst), "--tokens", str(tok)])
    assert lst.read_text(encoding="utf-8") == f"k1\tp\tprompt.wav\t{L} bord/ndo s/@y\n"
    tok.write_text("_\t0\n", encoding="utf-8")
    main([str(lst), "--tokens", str(tok)])   # no pause tokens: glue only (already glued, unchanged)
    assert lst.read_text(encoding="utf-8") == f"k1\tp\tprompt.wav\t{L} bord/ndo s/@y\n"


def test_a_word_ending_in_zhe_is_a_consonant_end_too():
    assert glue_vav("d/; o b/ru") == "d/;o b/ru"

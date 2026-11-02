# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from ganj_corpus.pauses import PAUSE_LONG, PAUSE_SHORT
from ganj_corpus.rules_check import check_label, missing_overrides, stale_keys
from ganj_reading import tokens as T

V = T.VERSE


def test_a_written_ezafe_must_be_read():
    # «شرحِ داغِ دلِ پروانه» — the kasre after شرح is in the text, so the label must carry the ezafe
    lines = ["شرحِ داغِ دلِ پروانه چو گفتم", "با شمع"]
    good = V + "š/rhe1 daqe1 dele1 p/rvane co goft/m" + T.HEMI + "ba š/m@"
    bad = V + "š/rh daqe1 dele1 p/rvane co goft/m" + T.HEMI + "ba š/m@"
    assert check_label(lines, good) == []
    assert [(v.word, v.rule) for v in check_label(lines, bad)] == [("شرحِ", "written ezafe")]


def test_a_kasra_before_a_shadda_is_a_written_ezafe_too():
    # «حقِّ ما» (Unicode puts the kasra before the shadda): h/qqe1 is right, h/qq drops the written ezafe
    assert check_label(["حقِّ ما"], "h/qqe1 ma") == []
    assert [v.rule for v in check_label(["حقِّ ما"], "h/qq ma")] == ["written ezafe"]


def test_pause_tokens_do_not_hide_a_rule():
    lines = ["شرحِ داغ"]
    assert check_label(lines, f"š/rh {PAUSE_SHORT} daq") != []
    assert check_label(lines, f"š/rhe1 {PAUSE_SHORT} daq{PAUSE_LONG}") == []


def test_tashdid_sukun_and_written_vowels_are_checked():
    assert [v.rule for v in check_label(["شکّر"], "š/k/r")] == ["written vowels, tashdid or sukun"]
    assert check_label(["شکّر"], "š/kk/r") == []
    assert [v.rule for v in check_label(["مَرْدُم"], "m/r/dom")] == ["written vowels, tashdid or sukun"]
    assert [v.rule for v in check_label(["دستْ بر"], "d/ste1 b/r")] == ["sukun: no ezafe"]


def test_a_label_that_does_not_line_up_is_reported_not_guessed():
    assert [v.rule for v in check_label(["یک‌تو برآورم"], "yek to b/ravor/m")] == ["label words do not line up"]


def test_every_per_line_reading_must_survive_into_the_labels():
    ov = [{"id": "c1", "line": 1, "word": 0, "text": "بود", "label": "bov/d"}]
    rows = {"c1": ["دل", "بود جان"]}
    ok = {"c1": V + "del" + T.HEMI + f"bov/d {PAUSE_SHORT} jan"}
    lost = {"c1": V + "del" + T.HEMI + "bud jan"}
    assert missing_overrides(ok, rows, ov) == []
    assert missing_overrides(lost, rows, ov) == [("c1", "بود", "bov/d", "bud")]


def test_a_reading_list_built_from_old_labels_is_caught():
    current = {"k1": "a b", "k2": "c d"}
    assert stale_keys({"k1": "a b", "k2": "c x"}, current) == ["k2"]
    assert stale_keys({"k1": f"a {PAUSE_SHORT} b"}, current) == []      # pause tokens are not a label change


def test_the_kasra_of_ze_is_its_own_vowel_not_an_ezafe():
    # 182 of the 191 q9e "written ezafe" findings were «زِ» read ze: the G2P is right, the check was too strict
    assert check_label(["زِ تو"], "ze to") == []
    assert [v.rule for v in check_label(["دلِ تو"], "del to")] == ["written ezafe"]


def test_arabic_and_turkish_findings_are_reported_and_a_waived_word_does_not_fail():
    from ganj_corpus.rules_check import Violation, triage
    vs = [Violation(0, "مردیّ", "m/rdi", "written vowels, tashdid or sukun"), Violation(0, "نَبُرد", "n/b/r/d", "x")]
    fail, report = triage("4606-5", "vowelled", vs, [{"id": "4606-5", "word": "مردیّ", "why": "asked"}])
    assert [v.word for v in fail] == ["نَبُرد"] and [v.word for v in report] == ["مردیّ"]
    fail, report = triage("35843-7", "arabic", vs, [])
    assert fail == [] and len(report) == 2


def test_a_waiver_can_name_a_whole_line_with_its_reason():
    # 35372-6: a Quranic quotation inside a prose line, read as it is like the Arabic lines
    from ganj_corpus.rules_check import Violation, triage
    vs = [Violation(0, "صَدَقَ", "sedqe1", "written vowels, tashdid or sukun")]
    assert triage("35372-6", "prose", vs, [{"id": "35372-6", "word": "*", "why": "Arabic quotation"}]) == ([], vs)

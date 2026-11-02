# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from tools import check_rules


def test_a_list_checked_without_the_eval_set_passes_with_every_line_unchecked(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(check_rules, "DATA", tmp_path)
    lst = tmp_path / "list.tsv"
    lst.write_text("my-1\tprompt ph\tprompt.wav\tjan o del\n", encoding="utf-8")
    assert check_rules.check_list(lst, tmp_path / "no_current.jsonl", tmp_path / "no_waivers.json") == 0
    assert "1 lines not in the eval set (unchecked)" in capsys.readouterr().out

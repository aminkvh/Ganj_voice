# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from ganj_reading import tokens as T


def chars(w):
    return "".join(ch for ch in w.replace("‌", "") if ch.isalpha())


def test_a_clip_is_markable_when_its_label_has_one_word_per_text_word_in_every_hemistich():
    from tools.pause_labels import plan_cut
    words, why = plan_cut(["سال‌ها نرد", "این گره"], T.VERSE + "s/lha n/rd" + T.HEMI + "@in gereh", chars)
    assert why is None and words == ["سال‌ها", "نرد", "این", "گره"]


def test_a_split_compound_or_a_word_without_letters_is_skipped_with_the_reason():
    from tools.pause_labels import plan_cut
    assert plan_cut(["یک‌تو برآورم"], "yek to b/ravor/m", chars)[1] == "word count"
    assert plan_cut(["دل ۱۲"], "del d/vazdah", chars)[1] == "no letters"
    assert plan_cut(["دل", "جان"], "del", chars)[1] == "line count"


def test_the_extended_token_list_keeps_every_id_and_appends_the_pause_tokens(tmp_path):
    from ganj_corpus.pauses import PAUSE_LONG, PAUSE_SHORT
    from tools.pause_labels import extend_tokens
    src = tmp_path / "tokens.txt"
    src.write_text("_\t0\n \t1\na\t2\n", encoding="utf-8")
    extend_tokens(src, tmp_path / "out.txt")
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == f"_\t0\n \t1\na\t2\n{PAUSE_SHORT}\t3\n{PAUSE_LONG}\t4\n"


def test_the_checkpoint_gets_two_embedding_rows_copied_from_the_space_token(tmp_path):
    import torch
    from tools.pause_labels import extend_checkpoint
    emb = torch.arange(12, dtype=torch.float32).view(3, 4)
    torch.save({"model": {"embed.weight": emb, "other": torch.ones(2)}, "batch_idx_train": 7}, tmp_path / "a.pt")
    extend_checkpoint(tmp_path / "a.pt", tmp_path / "b.pt", rows=2, init_id=1)
    c = torch.load(tmp_path / "b.pt", weights_only=False)
    w = c["model"]["embed.weight"]
    assert w.shape == (5, 4) and torch.equal(w[:3], emb) and torch.equal(w[3], emb[1]) and torch.equal(w[4], emb[1])
    assert torch.equal(c["model"]["other"], torch.ones(2)) and c["batch_idx_train"] == 7


def test_a_cut_that_contradicts_a_written_mark_is_held_back_unless_it_carries_a_per_line_reading():
    from tools.pause_labels import held_back
    rows = {"a": ["شرحِ داغ"], "b": ["شرحِ داغ"], "c": ["دل جان"]}
    labels = {"a": "š/rh daq", "b": "š/rh daq", "c": "del jan"}
    held = held_back(labels, rows, keep={"b"})
    assert set(held) == {"a"} and held["a"][0].rule == "written ezafe"


def test_a_reading_copied_to_a_recording_that_writes_the_word_apart_is_not_counted_missing():
    from tools.pause_labels import unapplied
    rows = {"r1": ["درآورد سر"], "r2": ["در آورد سر"]}
    labels = {"r1": "d/rav/rd s/r", "r2": "d/r @av/rd s/r"}
    copied = {"id": "r2", "line": 0, "word": 0, "text": "درآورد", "label": "d/rav/rd", "of": "r1"}
    own = dict(copied, of="r2")
    wrong = {"id": "r1", "line": 0, "word": 1, "text": "سر", "label": "ser"}
    assert unapplied(labels, rows, [copied]) == []
    assert unapplied(labels, rows, [own, wrong]) == [("r2", "درآورد", "d/rav/rd", "word not found"), ("r1", "سر", "ser", "s/r")]


def test_the_eval_set_is_read_at_the_input_tag_and_written_at_the_output_tag(tmp_path, monkeypatch):
    import json
    import tools.pause_labels as pl
    monkeypatch.setattr(pl, "DATA", tmp_path)
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "set_q_qa.jsonl").write_text(json.dumps({"key": "k", "lines": ["دل جان"], "phonemes": "del jan"}) + "\n",
                                                      encoding="utf-8")
    ov = tmp_path / "ov.json"
    ov.write_text(json.dumps([{"id": "k", "line": 0, "word": 1, "text": "جان", "label": "jane1", "key": "x"}]), encoding="utf-8")
    pl.eval_set(tag_in="_qa", tag="_qb", overrides_path=ov)
    out = [json.loads(l) for l in open(tmp_path / "eval" / "set_q_qb.jsonl", encoding="utf-8")]
    assert out == [{"key": "k", "lines": ["دل جان"], "phonemes": "del jane1"}]


def test_the_tags_and_readings_files_reach_the_cuts_step(monkeypatch):
    from pathlib import Path
    import tools.pause_labels as pl
    got = {}
    monkeypatch.setattr(pl, "write_cuts", lambda *a, **k: got.update(args=a, kw=k))
    pl.main(["--tag-in", "_q9g", "--tag", "_q9h", "cuts", "--overrides", "all.json", "--keep", "listener.json"])
    assert got == {"args": (0.0,), "kw": {"tag_in": "_q9g", "tag": "_q9h", "overrides_path": Path("all.json"),
                                          "keep_path": Path("listener.json")}}


def test_a_reading_marks_its_pauses_where_the_reader_was_silent():
    # a line with a reference reading takes its pauses from that recording
    import numpy as np
    from ganj_corpus.pauses import PAUSE_LONG
    from tools.pause_labels import measured_label
    sr, rng = 16000, np.random.default_rng(0)

    def part(secs, amp):
        return amp * rng.standard_normal(int(sr * secs))
    y = np.concatenate([part(0.4, 0.1), part(0.05, 1e-5), part(0.4, 0.1), part(0.6, 1e-5), part(0.4, 0.1)]).astype(np.float32)

    class Aligner:
        chars = staticmethod(chars)

        def spans(self, y, words):
            return [(0.0, 0.4), (0.45, 0.85), (1.45, 1.85)], 1.0
    lab = T.VERSE + "del jan" + T.HEMI + "t/n"
    assert measured_label(["دل جان", "تن"], lab, y, Aligner()) == T.VERSE + "del jan" + PAUSE_LONG + T.HEMI + "t/n"
    assert measured_label(["یک‌تو برآورم"], "yek to b/ravor/m", y, Aligner()) is None   # words do not line up

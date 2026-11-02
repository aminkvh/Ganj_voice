# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import json

from ganj_corpus.relabel import relabel_cut_line, relabel_tsv_line


def test_relabel_cut_line_replaces_only_the_supervision_text():
    cut = {"id": "5-0a", "duration": 1.0, "supervisions": [{"id": "5-0a", "text": "old"}], "features": {"x": 1}}
    out = json.loads(relabel_cut_line(json.dumps(cut), {"5-0a": "new"}))
    assert out["supervisions"][0]["text"] == "new" and out["features"] == {"x": 1}


def test_relabel_cut_line_keeps_cuts_without_a_new_label():
    line = json.dumps({"id": "6-1", "supervisions": [{"id": "6-1", "text": "old"}]})
    assert json.loads(relabel_cut_line(line, {}))["supervisions"][0]["text"] == "old"


def test_relabel_tsv_line_swaps_target_text_by_key():
    assert relabel_tsv_line("k1\tprompt\t/p.wav\told\n", {"k1": "new"}) == "k1\tprompt\t/p.wav\tnew\n"
    assert relabel_tsv_line("k2\tprompt\t/p.wav\told\n", {"k1": "new"}) == "k2\tprompt\t/p.wav\told\n"


class FakeG2P:
    def __init__(self):
        self.seen = []

    def phonemize_units(self, units, batch=32):
        self.seen += [u[0][0] for u in units]
        return [None if u[0][0] == "bad" else "ph-" + u[0][0] for u in units]


def test_label_units_resumes_skips_failures_and_writes_whole_lines(tmp_path):
    from ganj_corpus.relabel import label_units
    out = tmp_path / "labels.jsonl"
    out.write_text(json.dumps({"id": "a", "phonemes": "old-a"}) + "\n")
    rows = [{"id": i, "lines": [t], "kind": "prose"} for i, t in (("a", "x"), ("b", "y"), ("c", "bad"), ("d", "z"))]
    g = FakeG2P()
    labels = label_units(g, rows, out, wanted={"a", "b", "c"}, chunk=2)
    assert labels == {"a": "old-a", "b": "ph-y"}
    assert g.seen == ["y", "bad"]                          # done and unwanted rows are not recomputed
    assert [json.loads(l)["id"] for l in out.read_text().splitlines()] == ["a", "b"]


def test_relabel_tsv_line_also_swaps_a_known_prompt_text():
    line = "k1\told prompt\t/p.wav\told\n"
    assert relabel_tsv_line(line, {"k1": "new"}, {"old prompt": "new prompt"}) == "k1\tnew prompt\t/p.wav\tnew\n"


def test_source_tsvs_skip_every_relabelled_set(tmp_path):
    from ganj_corpus.relabel import source_tsvs
    for n in ("test_u0r.tsv", "test_u0r_q9b.tsv", "test_u0r_q9c.tsv", "test_b1f.tsv"):
        (tmp_path / n).write_text("")
    assert sorted(t.name for t in source_tsvs(tmp_path)) == ["test_b1f.tsv", "test_u0r.tsv"]


def test_overrides_pin_one_word_in_one_line_and_keep_the_style_prefix():
    # a reviewer's reading for that word in that line only, never for the word everywhere
    from ganj_corpus.relabel import apply_overrides
    from ganj_reading import tokens as T
    labels = {"7-1": T.VERSE + "dele1 m/n" + T.HEMI + "b/nuše1 baz", "8-0": T.VERSE + "b/nuše1 m/y"}
    rows = {"7-1": ["دل من", "بنوش باز"], "8-0": ["بنوش می"]}
    out, applied, skipped = apply_overrides(labels, rows, [{"id": "7-1", "line": 1, "word": 0, "text": "بنوش",
                                                             "label": "benuš"}])
    assert out["7-1"] == T.VERSE + "dele1 m/n" + T.HEMI + "benuš baz"
    assert out["8-0"] == labels["8-0"]                          # the same word elsewhere is untouched
    assert (applied, skipped) == (1, [])
    assert labels["7-1"].endswith("b/nuše1 baz")                # input not mutated


def test_an_override_whose_word_moved_is_found_by_its_text_or_skipped():
    from ganj_corpus.relabel import apply_overrides
    labels = {"7-1": "a b c"}
    rows = {"7-1": ["آنکه دین او"]}
    ov = [{"id": "7-1", "line": 0, "word": 0, "text": "دین", "label": "din"},     # index moved: found by its text
          {"id": "7-1", "line": 0, "word": 2, "text": "سر", "label": "s/r"}]      # text not in the line: skipped
    out, applied, skipped = apply_overrides(labels, rows, ov)
    assert out["7-1"] == "a din c" and applied == 1 and len(skipped) == 1


class _Fake:
    def __init__(self, tag):
        self.tag, self.seen = tag, []

    def phonemize_units(self, units):
        self.seen += [u[0][0] for u in units]
        return [f"{self.tag}:{u[0][0]}" for u in units]


def test_verse_units_go_to_the_verse_g2p_and_prose_units_to_the_prose_g2p_in_order():
    # a model fine-tuned on verse labels verse; prose (ManaTTS, Golestan) keeps the base model
    from ganj_corpus.relabel import ByKind
    v, p = _Fake("v"), _Fake("p")
    out = ByKind(verse=v, prose=p).phonemize_units([(["a"], "verse", "M"), (["b"], "prose", None), (["c"], "verse", None)])
    assert out == ["v:a", "p:b", "v:c"]
    assert v.seen == ["a", "c"] and p.seen == ["b"]

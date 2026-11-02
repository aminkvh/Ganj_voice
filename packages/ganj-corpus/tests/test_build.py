# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from ganj_corpus.build import assign_speakers, csv_row, metre_of


def test_metre_of_reads_first_section_or_empty():
    assert metre_of({"sections": [{"ganjoorMetre": {"rhythm": "مفاعیلن مفاعیلن فعولن"}}]}) == "مفاعیلن مفاعیلن فعولن"
    assert metre_of({"sections": [{"ganjoorMetre": None}]}) == ""
    assert metre_of({}) == ""


def test_csv_row_uses_speaker_name_and_escapes_separator():
    row = {"clip": "clips/5/0.wav", "lines": ["الف", "ب | پ"], "ids": [1, 0, 7, 0, 2]}
    assert csv_row(row, "عندلیب") == "clips/5/0.wav|عندلیب|الف ب ، پ|1 0 7 0 2"


def test_assign_speakers_reserves_blend_slot_and_applies_minimum():
    rows = [{"artist": "A", "seconds": 400}] * 2 + [{"artist": "B", "seconds": 100}] + [{"artist": "C", "seconds": 900}]
    sp = assign_speakers(rows, min_minutes=10)
    assert set(sp) == {"A", "C", "blend"} and sp["blend"] == 2 and {sp["A"], sp["C"]} == {0, 1}


def test_csv_row_collapses_newlines_inside_text():
    row = {"clip": "c.wav", "lines": ["الف\nب", "پ"], "ids": [1, 2]}
    assert csv_row(row, "x") == "c.wav|x|الف ب پ|1 2"


def test_clip_name_uses_unit_and_part():
    from ganj_corpus.build import clip_name
    from ganj_corpus.cut import Cut
    assert clip_name(Cut(4, 0, 1, ["x"], "verse", [9], "a")) == "4a"
    assert clip_name(Cut(4, 0, 1, ["x"], "verse", [9])) == "4"


def test_write_csvs_keeps_an_existing_speaker_map(tmp_path):
    import json
    from ganj_corpus.build import write_csvs
    rows = [{"clip": f"clips/{i}/1.wav", "artist": a, "poem_id": i, "seconds": 700.0, "ids": [1, 5, 2], "phonemes": "x",
             "lines": ["بشنو"], "text": "بشنو"} for i, a in enumerate(["ب", "ا"])]
    (tmp_path / "clips.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    trained = {"ب": 0, "ا": 1, "blend": 2}  # Piper's own map from a finished run: the authority
    (tmp_path / "speakers.json").write_text(json.dumps(trained, ensure_ascii=False), encoding="utf-8")
    write_csvs(tmp_path)
    assert json.loads((tmp_path / "speakers.json").read_text(encoding="utf-8")) == trained


def test_assign_speakers_pools_small_reciters_into_chorus():
    from ganj_corpus.build import CHORUS
    rows = [{"artist": "A", "seconds": 1900}, {"artist": "B", "seconds": 600}, {"artist": "C", "seconds": 300}]
    sp = assign_speakers(rows, min_minutes=30, pool=True)
    assert set(sp) == {"A", CHORUS, "blend"} and len(set(sp.values())) == 3


def test_speaker_for_maps_small_reciters_to_chorus():
    from ganj_corpus.build import CHORUS, speaker_for
    sp = {"A": 0, CHORUS: 1, "blend": 2}
    assert speaker_for("A", sp) == "A" and speaker_for("B", sp) == CHORUS


def test_write_csvs_drops_narrow_band_and_pools(tmp_path):
    import json
    from ganj_corpus.build import CHORUS, write_csvs
    rows = [{"clip": f"clips/{i}/1.wav", "artist": a, "poem_id": i, "seconds": s, "ids": [1, 5, 2], "phonemes": "x",
             "lines": ["بشنو"], "text": "بشنو", "narrow": n}
            for i, (a, s, n) in enumerate([("A", 2000.0, False), ("B", 100.0, False), ("C", 5000.0, True)])]
    (tmp_path / "clips.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_csvs(tmp_path, min_minutes=30, pool=True)
    speakers = [l.split("|")[1] for l in (tmp_path / "metadata.csv").read_text(encoding="utf-8").splitlines()]
    assert sorted(speakers) == sorted(["A", CHORUS])
    assert "C" not in json.loads((tmp_path / "speakers.json").read_text(encoding="utf-8"))


def test_training_manifest_skips_eval_poems(tmp_path):
    from ganj_corpus.build import load_eval_poems, training_manifest
    (tmp_path / "eval_poems.txt").write_text("7\n9\n", encoding="utf-8")
    ev = load_eval_poems(tmp_path / "eval_poems.txt")
    manifest = [{"id": 1, "poemId": 7}, {"id": 2, "poemId": 8}, {"id": 3, "poemId": 9}]
    keep, skipped = training_manifest(manifest, ev)
    assert [m["id"] for m in keep] == [2] and [m["id"] for m in skipped] == [1, 3]


def test_load_eval_poems_missing_file_is_an_error(tmp_path):
    import pytest
    from ganj_corpus.build import load_eval_poems
    with pytest.raises(FileNotFoundError):
        load_eval_poems(tmp_path / "absent.txt")


def test_shards_are_disjoint_and_cover_the_manifest():
    from ganj_corpus.build import shard_of
    m = list(range(10))
    parts = [shard_of(m, k, 3) for k in range(3)]
    assert sorted(x for p in parts for x in p) == m and not set(parts[0]) & set(parts[1])


def test_merge_shards_joins_clips_and_rejects_then_writes_csvs(tmp_path):
    import json
    from ganj_corpus.build import merge_shards
    row = lambda i: {"clip": f"clips/{i}/1.wav", "artist": "A", "poem_id": i, "seconds": 2000.0, "ids": [1, 2],
                     "phonemes": "x", "lines": ["ب"], "text": "ب", "gender": "f", "kind": "verse"}
    for k in range(2):
        (tmp_path / f"clips.{k}.jsonl").write_text(json.dumps(row(k), ensure_ascii=False) + "\n", encoding="utf-8")
        (tmp_path / f"rejects.{k}.jsonl").write_text(json.dumps({"clip": f"{k}", "reason": "timing"}) + "\n", encoding="utf-8")
    merge_shards(tmp_path, 2)
    assert len((tmp_path / "clips.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert len((tmp_path / "rejects.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    assert (tmp_path / "metadata.csv").exists() and "timing: 2" in (tmp_path / "corpus_report.md").read_text(encoding="utf-8")


def test_merge_shards_refuses_a_missing_shard(tmp_path):
    import pytest
    from ganj_corpus.build import merge_shards
    (tmp_path / "clips.0.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "rejects.0.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        merge_shards(tmp_path, 2)

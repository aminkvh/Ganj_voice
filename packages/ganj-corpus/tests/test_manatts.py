# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from ganj_corpus.manatts import clip_id, keep


def row(**k):
    return {"file_name": "1213_0007.wav", "transcript": "این یک جمله است.", "duration": 6.0, "match_quality": "HIGH", **k}


def test_keeps_a_high_quality_clip_of_trainable_length():
    assert keep(row())


def test_drops_middle_quality_alignments():
    assert not keep(row(match_quality="MIDDLE"))


def test_drops_clips_outside_the_training_length_window():
    assert not keep(row(duration=0.6)) and not keep(row(duration=14.0))


def test_drops_text_with_digits_or_latin_letters_the_g2p_cannot_read():
    assert not keep(row(transcript="سال ۱۴۰۲ آمد")) and not keep(row(transcript="سال 1402")) and not keep(row(transcript="مدل GPT"))


def test_clip_ids_are_namespaced_away_from_ganjoor_ids():
    assert clip_id("1213_0007.wav") == "mana-1213_0007"


def test_parquet_rows_with_spaced_column_names_are_normalised():
    from ganj_corpus.manatts import from_parquet
    r = from_parquet({"file name": "1_2.wav", "transcript": " متن ", "duration": 2.5, "match quality": "HIGH", "samplerate": 44100.0})
    assert r == {"file_name": "1_2.wav", "transcript": "متن", "duration": 2.5, "match_quality": "HIGH", "sample_rate": 44100}


def test_a_row_without_a_transcript_is_dropped_not_fatal():
    from ganj_corpus.manatts import from_parquet
    r = from_parquet({"file name": "1_3.wav", "transcript": None, "duration": 2.5, "match quality": "HIGH", "samplerate": 44100.0})
    assert not keep(r)


def test_merge_drops_every_copy_of_a_clip_id_that_occurs_twice(tmp_path):
    # ManaTTS reuses file names across parquet parts; the converter wrote one wav per name, so a duplicated id's audio
    # may belong to the other copy's transcript — neither copy can be trusted
    from ganj_corpus.manatts import merge
    (tmp_path / "a.tsv").write_text("mana-1-1\tx\tw1\nmana-1-2\ty\tw2\n", encoding="utf-8")
    (tmp_path / "b.tsv").write_text("mana-1-2\tz\tw2\nmana-3-1\tq\tw3\n", encoding="utf-8")
    kept, dropped = merge(tmp_path, tmp_path / "all.tsv")
    assert (kept, dropped) == (2, 2)
    assert sorted(l.split("\t")[0] for l in (tmp_path / "all.tsv").read_text(encoding="utf-8").splitlines()) == ["mana-1-1", "mana-3-1"]


def test_transcripts_are_read_back_by_clip_id_and_a_name_in_two_parts_is_left_out(tmp_path):
    # the converter kept only phonemes; a relabel with a newer G2P needs the text again (2026-10-29)
    import pyarrow as pa
    import pyarrow.parquet as pq
    from ganj_corpus.manatts import transcripts
    pq.write_table(pa.table({"file name": ["1-1.wav", "1-2.wav"], "transcript": [" سلام ", "دوست"]}), tmp_path / "p1.parquet")
    pq.write_table(pa.table({"file name": ["1-2.wav", "2-1.wav"], "transcript": ["دیگر", "باران"]}), tmp_path / "p2.parquet")
    assert transcripts(tmp_path) == {"mana-1-1": "سلام", "mana-2-1": "باران"}

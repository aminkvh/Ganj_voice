# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from ganj_corpus.clean_subset import is_clean, cut_rec_unit


def test_cut_rec_unit_splits_a_cut_id_into_recording_and_unit():
    assert cut_rec_unit("10001-0a") == (10001, "0a")
    assert cut_rec_unit("984-30") == (984, "30")


def test_is_clean_rejects_separated_narrow_or_poorly_aligned_clips():
    row = {"rec_id": 5, "bandwidth": 12000, "align_score": 0.9}
    assert is_clean(row, demucs_ids=set())
    assert not is_clean(row, demucs_ids={5})                       # music was separated out
    assert not is_clean({**row, "bandwidth": 9000}, demucs_ids=set())  # low-bitrate mp3
    assert not is_clean({**row, "align_score": 0.7}, demucs_ids=set())  # text/audio disagree

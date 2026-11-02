# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import gzip
import json

from ganj_corpus.oversample import repeats, rare_score


def test_a_clip_with_a_rare_phoneme_trigram_scores_low_and_common_ones_high():
    counts = {"xam": 13, "amy": 13, "ab/": 5000, "b/d": 9000}
    assert rare_score("xamy", counts, n=3) == 13
    assert rare_score("ab/d", counts, n=3) == 5000


def test_rare_clips_repeat_up_to_the_cap_and_common_ones_once():
    assert repeats(13, threshold=50, max_repeat=3) == 3
    assert repeats(40, threshold=50, max_repeat=3) == 2
    assert repeats(5000, threshold=50, max_repeat=3) == 1


def test_oversample_manifest_adds_copies_with_new_ids(tmp_path):
    from ganj_corpus.oversample import oversample
    src, out = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    cuts = [{"id": "r1", "supervisions": [{"text": "xamyaze"}]}] + [{"id": f"c{i}", "supervisions": [{"text": "b/d b/d"}]} for i in range(60)]
    with gzip.open(src, "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(c) + "\n" for c in cuts)
    added = oversample(src, out, threshold=50, max_repeat=3)
    ids = [json.loads(l)["id"] for l in gzip.open(out, "rt", encoding="utf-8")]
    assert added == 2 and ids.count("r1") == 1 and "r1_rep1" in ids and "r1_rep2" in ids and len(ids) == 63

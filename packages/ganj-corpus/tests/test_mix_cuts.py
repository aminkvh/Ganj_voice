# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import gzip

from ganj_corpus.mix_cuts import mix


def write(p, ids):
    with gzip.open(p, "wt", encoding="utf-8") as f:
        f.writelines(f'{{"id": "{i}"}}\n' for i in ids)


def test_mix_keeps_every_cut_once_and_interleaves_the_sources(tmp_path):
    a, b, out = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz", tmp_path / "m.jsonl.gz"
    write(a, [f"g{i}" for i in range(200)]); write(b, [f"mana-{i}" for i in range(200)])
    n = mix([a, b], out, seed=0)
    ids = [l.strip() for l in gzip.open(out, "rt", encoding="utf-8")]
    assert n == 400 and len(set(ids)) == 400
    first_half = ids[:200]
    assert 50 < sum("mana" in i for i in first_half) < 150   # shuffled, not concatenated

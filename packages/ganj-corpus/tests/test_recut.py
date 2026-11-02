# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np

from ganj_corpus.align import Alignment
from ganj_corpus.cut import Cut
from ganj_corpus.recut import align_spans, recut_bounds, MAX_POST, MAX_PRE, PRE, POST


def test_cut_between_words():
    b = recut_bounds([(1000, 3000), (3400, 6000)], contiguous=[True])
    (s0, e0), (s1, e1) = b
    assert 3000 < e0 < 3400 and s1 >= e0 and s1 < 3400
    assert s0 == 1000 - PRE and e1 == 6000 + POST


def test_unaligned_clip_is_dropped_and_its_neighbour_keeps_fixed_pads():
    b = recut_bounds([None, (3400, 6000)], contiguous=[True])
    assert b[0] is None and b[1] == (3400 - PRE, 6000 + POST)


def test_overlapping_alignments_drop_both():
    assert recut_bounds([(1000, 3500), (3480, 6000)], contiguous=[True]) == [None, None]


def test_long_pause_caps_padding():
    (s0, e0), (s1, e1) = recut_bounds([(1000, 3000), (5000, 7000)], contiguous=[True])
    assert e0 == 3000 + MAX_POST and s1 == 5000 - MAX_PRE


def test_non_contiguous_neighbours_use_fixed_pads():
    assert recut_bounds([(1000, 3000), (3300, 6000)], contiguous=[False]) == [(1000 - PRE, 3000 + POST), (3300 - PRE, 6000 + POST)]


def test_start_never_negative():
    assert recut_bounds([(50, 900)], contiguous=[])[0][0] == 0


class FakeAligner:
    """Words sit at fixed offsets inside whatever window it is given; 'bad' text fails the gate."""
    def __init__(self):
        self.calls = []

    def align(self, seg, text):
        self.calls.append(text)
        if "bad" in text:
            return Alignment(0.1, [], 0.0, 0, 1.0)
        words = text.split()
        spans = [(w, 1.0 + 1.0 * i, 1.8 + 1.0 * i) for i, w in enumerate(words)]
        return Alignment(0.9, spans, spans[-1][2], 0, 0.0)


def test_neighbours_are_aligned_jointly_so_the_pause_is_consistent():
    y = np.zeros(22050 * 30, dtype=np.float32)
    cuts = [Cut(0, 2000, 5000, ["a b"], "verse", [1]), Cut(1, 5000, 9000, ["c d"], "verse", [3])]
    fa = FakeAligner()
    spans, contiguous = align_spans(fa, y, 22050, cuts, widen_ms=1000)
    assert "a b c d" in fa.calls  # one alignment over both lines
    ws = 2000 - 1000
    assert spans[0][1] == ws + 2800 and spans[1][0] == ws + 3000  # end of "b", start of "c" from the joint alignment
    assert contiguous == [True]
    assert spans[0][1] < spans[1][0]


def test_failed_joint_alignment_falls_back_to_single_lines_with_fixed_pads():
    y = np.zeros(22050 * 30, dtype=np.float32)
    cuts = [Cut(0, 2000, 5000, ["a b"], "verse", [1]), Cut(1, 5000, 9000, ["bad"], "verse", [3])]
    spans, contiguous = align_spans(FakeAligner(), y, 22050, cuts, widen_ms=1000)
    assert spans[0] is not None and spans[1] is None and contiguous == [False]


def test_recut_returns_new_cut_or_none_per_original():
    from ganj_corpus.recut import recut
    y = np.zeros(22050 * 30, dtype=np.float32)
    cuts = [Cut(1, 5000, 9000, ["c d"], "verse", [3]), Cut(0, 2000, 5000, ["a b"], "verse", [1])]
    pairs = recut(FakeAligner(), y, 22050, cuts)
    assert [old.unit for old, _ in pairs] == [0, 1]
    assert all(new is not None and new.unit == old.unit and new.lines == old.lines for old, new in pairs)
    assert pairs[0][1].end_ms <= pairs[1][1].start_ms


def test_clip_length_limits():
    from ganj_corpus.build import length_ok
    assert length_ok(1000, 1600) and not length_ok(1000, 1500) and not length_ok(0, 12001)

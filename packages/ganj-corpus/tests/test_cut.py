# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np
import pytest
import soundfile as sf

from ganj_corpus.cut import Cut, CutError, plan_cuts, write_clip


def verses(*rows):
    return [{"vOrder": i + 1, "position": p, "text": t, "coupletIndex": c} for i, (p, t, c) in enumerate(rows)]


def timings(*pairs):
    return [{"verseOrder": v, "audioStartMilliseconds": ms, "verseText": ""} for v, ms in pairs]


GHAZAL = verses((0, "الف", 0), (1, "ب", 0), (0, "پ", 1), (1, "ت", 1))


def test_couplets_become_units_ending_at_next_start():
    cuts = plan_cuts(timings((0, 0), (1, 5000), (2, 8000), (3, 11000), (4, 14000)), GHAZAL)
    assert cuts[0] == Cut(unit=0, start_ms=5000, end_ms=11000, lines=["الف", "ب"], kind="verse", vorders=[1, 2])
    assert cuts[1].start_ms == 11000 and cuts[1].end_ms is None  # last unit: aligner decides


def test_title_and_intro_are_not_clips():
    cuts = plan_cuts(timings((0, 0), (1, 9000), (2, 12000), (3, 15000), (4, 18000)), GHAZAL)
    assert all(c.start_ms >= 9000 for c in cuts) and all(0 not in c.vorders for c in cuts)


def test_non_monotonic_timings_rejected():
    with pytest.raises(CutError):
        plan_cuts(timings((1, 5000), (2, 4000), (3, 8000), (4, 9000)), GHAZAL)
    with pytest.raises(CutError):
        plan_cuts(timings((1, 5000), (2, 5000), (3, 8000), (4, 9000)), GHAZAL)


def test_prose_paragraph_is_its_own_unit_and_comments_skipped():
    v = verses((-1, "نثر", 0), (5, "یادداشت", 1), (0, "الف", 2), (1, "ب", 2))
    cuts = plan_cuts(timings((1, 1000), (2, 4000), (3, 6000), (4, 9000)), v)
    assert cuts[0].kind == "prose" and cuts[0].lines == ["نثر"] and cuts[0].end_ms == 4000
    assert [c.vorders for c in cuts] == [[1], [3, 4]]


def test_untimed_first_verse_skips_unit_and_short_units_dropped():
    cuts = plan_cuts(timings((2, 1000), (3, 1400), (4, 9000)), GHAZAL)
    assert [c.vorders for c in cuts] == [[3, 4]]  # unit 0 lacks vOrder 1 timing; 1000-1400 would be too short anyway


def test_long_units_split_into_hemistichs():
    cuts = plan_cuts(timings((1, 0), (2, 7000), (3, 15000), (4, 20000)), GHAZAL)
    assert [(c.start_ms, c.end_ms, c.lines) for c in cuts[:2]] == [(0, 7000, ["الف"]), (7000, 15000, ["ب"])]


def test_write_clip_slices_samples(tmp_path):
    sr = 22050
    x = np.zeros(sr * 3, dtype=np.float32)
    x[sr:sr * 2] = 0.5
    src = tmp_path / "src.wav"
    sf.write(src, x, sr, subtype="PCM_16")
    out = tmp_path / "clip.wav"
    seconds = write_clip(src, 1000, 2000, out)
    y, _ = sf.read(out)
    assert abs(seconds - 1.0) < 1e-3 and len(y) == sr and abs(y.mean() - 0.5) < 1e-3


def test_live_api_key_versePosition_is_accepted():
    v = [{"vOrder": 1, "versePosition": 0, "text": "الف", "coupletIndex": 0}, {"vOrder": 2, "versePosition": 1, "text": "ب", "coupletIndex": 0}]
    cuts = plan_cuts(timings((1, 0), (2, 3000)), v)
    assert [c.vorders for c in cuts] == [[1, 2]] and cuts[0].end_ms is None


def test_split_hemistichs_get_distinct_parts():
    cuts = plan_cuts(timings((1, 0), (2, 7000), (3, 15000), (4, 20000)), GHAZAL)
    assert cuts[0].unit == cuts[1].unit and cuts[0].part == "a" and cuts[1].part == "b"
    assert cuts[0].name != cuts[1].name and cuts[2].part == "" and cuts[2].name == str(cuts[2].unit)

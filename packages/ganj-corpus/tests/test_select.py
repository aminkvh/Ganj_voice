# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from ganj_corpus.select import eligible, estimate_seconds, select, split_by_poem


def rec(i, artist="a", poem=None, size=8000 * 60, rtype=0, sync=True, mistakes=""):
    return {"id": i, "poemId": poem if poem is not None else i, "audioArtist": artist, "mp3SizeInBytes": size,
            "recitationType": rtype, "inSyncWithText": sync, "mistakes": mistakes, "mp3Url": f"https://i.ganjoor.net/{i}.mp3"}


def test_eligible_filters_commentary_unsynced_and_mistakes():
    assert eligible(rec(1))
    assert not eligible(rec(2, rtype=1))
    assert not eligible(rec(3, sync=False))
    assert not eligible(rec(4, mistakes="line 3 skipped"))
    assert not eligible({**rec(5), "mp3Url": None})


def test_estimate_seconds_assumes_64kbps():
    assert estimate_seconds(rec(1, size=8000 * 90)) == 90.0


def test_select_caps_dominant_reciter_by_hours():
    rows = [rec(i, artist="big") for i in range(100)] + [rec(1000 + i, artist=f"s{i % 20}") for i in range(400)]
    chosen = select(rows, target_hours=2.0, cap_fraction=0.10)
    big = [r for r in chosen if r["audioArtist"] == "big"]
    total = sum(estimate_seconds(r) for r in chosen) / 3600
    assert sum(estimate_seconds(r) for r in big) / 3600 <= 0.10 * total + 1 / 60
    assert abs(total - 2.0) < 0.1


def test_select_drops_speakers_below_minimum_minutes():
    rows = [rec(i, artist="tiny") for i in range(5)] + [rec(100 + i, artist="ok") for i in range(60)]
    chosen = select(rows, target_hours=1.0, min_speaker_minutes=10)
    assert all(r["audioArtist"] != "tiny" for r in chosen)


def test_select_excludes_eval_poems():
    rows = [rec(i, poem=i) for i in range(30)]
    chosen = select(rows, target_hours=0.5, exclude_poems=frozenset({1, 2, 3}))
    assert not {r["poemId"] for r in chosen} & {1, 2, 3}


def test_select_balances_genders_when_known():
    rows = [rec(i, artist="m1") for i in range(60)] + [rec(100 + i, artist="f1") for i in range(60)] + [rec(200 + i, artist="m2") for i in range(60)]
    chosen = select(rows, target_hours=1.0, cap_fraction=0.5, genders={"m1": "male", "m2": "male", "f1": "female"})
    hours = {}
    for r in chosen:
        g = {"m1": "male", "m2": "male", "f1": "female"}[r["audioArtist"]]
        hours[g] = hours.get(g, 0) + estimate_seconds(r) / 3600
    assert abs(hours["male"] - hours["female"]) / sum(hours.values()) <= 0.10


def test_split_by_poem_keeps_poems_whole():
    rows = [rec(i, poem=i % 10) for i in range(100)]
    train, val = split_by_poem(rows, val_fraction=0.2, seed=3)
    assert not {r["poemId"] for r in train} & {r["poemId"] for r in val}
    assert len(train) + len(val) == 100


def test_eval_poems_excluded():
    from ganj_corpus.select import select
    rows = [{"id": i, "poemId": p, "audioArtist": "x", "mp3SizeInBytes": 8000 * 700, "mp3Url": "u", "inSyncWithText": True,
             "recitationType": 0, "mistakes": ""} for i, p in enumerate([1, 2, 3, 4])]
    chosen = select(rows, target_hours=10, cap_fraction=1.0, exclude_poems=frozenset({2, 4}))
    assert sorted(r["poemId"] for r in chosen) == [1, 3]

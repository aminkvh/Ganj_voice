# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Census filtering and balanced selection of recitations."""
import argparse
import csv
import json
import random
from collections import defaultdict
from typing import Iterable, Optional

BYTES_PER_SECOND = 8000.0  # 64 kbps MP3 until the file has been decoded and measured


def eligible(row: dict) -> bool:
    return (
        row.get("recitationType", 0) == 0
        and bool(row.get("inSyncWithText", False))
        and not (row.get("mistakes") or "").strip()
        and bool(row.get("mp3Url"))
    )


def estimate_seconds(row: dict) -> float:
    if row.get("seconds"):
        return float(row["seconds"])
    return float(row.get("mp3SizeInBytes") or 0) / BYTES_PER_SECOND


def select(rows: Iterable[dict], target_hours: float, cap_fraction: float = 0.10, genders: Optional[dict] = None,
           min_speaker_minutes: float = 10.0, exclude_poems=frozenset(), seed: int = 1) -> list:
    rows = [r for r in rows if eligible(r) and r["poemId"] not in exclude_poems]
    by_artist = defaultdict(list)
    for r in rows:
        by_artist[r["audioArtist"]].append(r)
    by_artist = {a: rs for a, rs in by_artist.items() if sum(estimate_seconds(r) for r in rs) >= min_speaker_minutes * 60}
    rng = random.Random(seed)
    for rs in by_artist.values():
        rng.shuffle(rs)
    target = target_hours * 3600
    cap = cap_fraction * target
    per_gender_target = None
    if genders:
        per_gender_target = {"male": target / 2, "female": target / 2}
    chosen, used, per_artist, per_gender = [], 0.0, defaultdict(float), defaultdict(float)
    cursors = {a: 0 for a in by_artist}
    artists = sorted(by_artist)
    progress = True
    while used < target and progress:
        progress = False
        for a in artists:  # round-robin keeps any single reciter from dominating
            i = cursors[a]
            if i >= len(by_artist[a]):
                continue
            r = by_artist[a][i]
            s = estimate_seconds(r)
            g = (genders or {}).get(a, "unknown")
            if per_artist[a] + s > cap:
                cursors[a] = len(by_artist[a])
                continue
            if per_gender_target and g in per_gender_target and per_gender[g] + s > per_gender_target[g]:
                cursors[a] = len(by_artist[a])
                continue
            cursors[a] += 1
            chosen.append(r)
            used += s
            per_artist[a] += s
            per_gender[g] += s
            progress = True
            if used >= target:
                break
    return chosen


def split_by_poem(rows: list, val_fraction: float = 0.02, seed: int = 1):
    poems = sorted({r["poemId"] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(poems)
    n_val = max(1, int(len(poems) * val_fraction))
    val_poems = set(poems[:n_val])
    return [r for r in rows if r["poemId"] not in val_poems], [r for r in rows if r["poemId"] in val_poems]


def load_reciters(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8", newline="") as f:
        return {row["audioArtist"]: row["gender"] for row in csv.DictReader(f) if row.get("gender") in ("male", "female")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True)
    ap.add_argument("--hours", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reciters")
    ap.add_argument("--exclude-poems", help="file with one poem id per line")
    ap.add_argument("--cap", type=float, default=0.10)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.census, encoding="utf-8")]
    excl = frozenset(int(x) for x in open(a.exclude_poems)) if a.exclude_poems else frozenset()
    chosen = select(rows, a.hours, cap_fraction=a.cap, genders=load_reciters(a.reciters) or None, exclude_poems=excl)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hours = sum(estimate_seconds(r) for r in chosen) / 3600
    print(f"selected {len(chosen)} recitations, ~{hours:.1f} h, {len({r['audioArtist'] for r in chosen})} reciters")


if __name__ == "__main__":
    main()

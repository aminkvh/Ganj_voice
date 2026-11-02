# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Turn Ganjoor's per-hemistich start times into per-beyt / per-paragraph clips.

Timings are start-only. A unit ends where the next timed verse starts; the last unit's end is None here and
is filled by the aligner — never by the file end, which is usually outro music.
"""
from dataclasses import dataclass
from typing import List, Optional

import soundfile as sf

from ganj_reading.tokens import MAX_SECONDS

MIN_MS = 600
MAX_MS = int(MAX_SECONDS * 1000)


class CutError(ValueError):
    pass


@dataclass
class Cut:
    unit: int
    start_ms: int
    end_ms: Optional[int]
    lines: List[str]
    kind: str
    vorders: List[int]
    part: str = ""  # "a"/"b" when a long couplet was split into hemistichs (both halves used to share a name)

    @property
    def name(self) -> str:
        return f"{self.unit}{self.part}"


def _units(verses):
    """Group verses into units: couplet (0+1 / 2+3), single (4), paragraph (-1); comments (5) are skipped."""
    units = []
    for v in sorted(verses, key=lambda v: v["vOrder"]):
        p = v["versePosition"] if "versePosition" in v else v["position"]  # live API key is versePosition
        if p == 5:
            continue
        if p in (1, 3) and units and units[-1]["positions"][-1] == p - 1 and len(units[-1]["verses"]) == 1:
            units[-1]["verses"].append(v)
            units[-1]["positions"].append(p)
        else:
            units.append({"verses": [v], "positions": [p], "kind": "prose" if p == -1 else "verse"})
    return units


def plan_cuts(timings, verses) -> List[Cut]:
    starts = {}
    last = -1
    for t in sorted(timings, key=lambda t: t["verseOrder"]):
        if t["verseOrder"] == 0:
            continue  # title read aloud: never a clip
        ms = int(t["audioStartMilliseconds"])
        if ms <= last:
            raise CutError(f"non-monotonic timing at verseOrder {t['verseOrder']}: {ms} <= {last}")
        last = ms
        starts[t["verseOrder"]] = ms
    ordered = sorted(starts.items(), key=lambda kv: kv[0])
    next_start = {}
    for (v, ms), nxt in zip(ordered, ordered[1:] + [(None, None)]):
        next_start[v] = nxt[1]
    cuts = []
    for i, u in enumerate(_units(verses)):
        vo = [v["vOrder"] for v in u["verses"]]
        if vo[0] not in starts:
            continue
        start = starts[vo[0]]
        end = next_start.get(vo[-1]) if vo[-1] in starts else next_start.get(vo[0])
        lines = [v["text"] for v in u["verses"]]
        if end is not None and end - start < MIN_MS:
            continue
        if end is not None and end - start > MAX_MS and len(vo) == 2 and all(v in starts for v in vo):
            a, b = starts[vo[0]], starts[vo[1]]
            for part, (s, e, line, v) in zip("ab", ((a, b, lines[0], vo[0]), (b, next_start.get(vo[1]), lines[1], vo[1]))):
                if e is None or MIN_MS <= e - s <= MAX_MS:
                    cuts.append(Cut(i, s, e, [line], u["kind"], [v], part))
            continue
        if end is not None and end - start > MAX_MS:
            continue
        cuts.append(Cut(i, start, end, lines, u["kind"], vo))
    return cuts


def write_clip(wav_path, start_ms: int, end_ms: int, out_path, sr: int = 22050) -> float:
    info = sf.info(str(wav_path))
    if info.samplerate != sr:
        raise CutError(f"{wav_path} is {info.samplerate} Hz, expected {sr}")
    a, b = int(start_ms * sr / 1000), int(end_ms * sr / 1000)
    y, _ = sf.read(str(wav_path), start=a, stop=b, dtype="float32")
    sf.write(str(out_path), y, sr, subtype="PCM_16")
    return len(y) / sr

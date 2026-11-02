# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Clean subset for the quality fine-tune: no Demucs-separated (music-bearing) recordings, wide-band
sources only, and good text/audio alignment. Filters the existing ZipVoice fbank cut manifest, so no features are
recomputed. Usage: python -m ganj_corpus.clean_subset <corpus_q/clips.jsonl> <demucs dir> <cuts in> <cuts out>
"""
import gzip
import json
import os
import sys

MIN_BW_HZ = 11000    # 31 % of the hours come from mp3s band-limited below 10 kHz
MIN_ALIGN = 0.8


def cut_rec_unit(cut_id: str):
    rec, unit = cut_id.split("-", 1)
    return int(rec), unit


def is_clean(row: dict, demucs_ids: set) -> bool:
    return (row["rec_id"] not in demucs_ids and row.get("bandwidth", 0) >= MIN_BW_HZ
            and row.get("align_score", 0) >= MIN_ALIGN)


def main(clips_jsonl, demucs_dir, cuts_in, cuts_out):
    dem = {int(f[:-4]) for f in os.listdir(demucs_dir) if f[:-4].isdigit()}
    keep = set()
    for line in open(clips_jsonl, encoding="utf-8"):
        r = json.loads(line)
        if is_clean(r, dem):
            keep.add((r["rec_id"], r["unit"]))
    n = k = secs = 0
    with gzip.open(cuts_in, "rt", encoding="utf-8") as fi, gzip.open(cuts_out, "wt", encoding="utf-8") as fo:
        for line in fi:
            n += 1
            c = json.loads(line)
            if cut_rec_unit(c["id"]) in keep:
                k += 1
                secs += c["duration"]
                fo.write(line)
    print(f"kept {k}/{n} cuts, {secs / 3600:.1f} h -> {cuts_out}")


if __name__ == "__main__":
    main(*sys.argv[1:])

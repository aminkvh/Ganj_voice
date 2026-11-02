# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Corpus builder: manifest → verified, cleaned, labelled, phonemized clips + Piper CSVs."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from ganj_corpus.align import Aligner, accepts
from ganj_corpus.recut import recut
from ganj_corpus.clean import (bandwidth_hz, clipping_fraction, loudness_normalize, reciter_genders,
                                    separate_vocals, summarize_ina)
from ganj_corpus.cut import CutError, plan_cuts, write_clip
from ganj_corpus.select import split_by_poem
from ganj_reading import tokens as T
from ganj_reading.g2p import G2P, PhonemizeError, load_lexicon

MUSIC_SEPARATE = 0.05
CLIP_MAX = 0.001
NARROW_HZ = 7000.0
CHORUS = "chorus"  # pooled id for reciters with too little audio for their own
Q_MIN_MINUTES = 30.0


def metre_of(poem: dict) -> str:
    for s in poem.get("sections") or []:
        m = s.get("ganjoorMetre") or {}
        if m.get("rhythm"):
            return m["rhythm"]
    return ""


def clip_name(c) -> str:
    """Unique clip name within a recitation: unit index plus hemistich part."""
    return f"{c.unit}{getattr(c, 'part', '')}"


def csv_row(row: dict, speaker: str) -> str:
    text = " ".join(" ".join(row["lines"]).replace("|", "،").split())  # '|' is the CSV separator; newlines inside verse text would split the row
    return f"{row['clip']}|{speaker}|{text}|{' '.join(map(str, row['ids']))}"


def assign_speakers(rows, min_minutes: float = 10.0, pool: bool = False) -> dict:
    """Reciters with >= min_minutes get their own id; the rest are dropped, or with pool share the CHORUS id."""
    secs = defaultdict(float)
    for r in rows:
        secs[r["artist"]] += r["seconds"]
    artists = sorted(a for a, s in secs.items() if s >= min_minutes * 60)
    if pool and len(artists) < len(secs):
        artists.append(CHORUS)
    sp = {a: i for i, a in enumerate(artists)}
    sp["blend"] = len(artists)
    return sp


def speaker_for(artist: str, sp: dict) -> str:
    return artist if artist in sp else CHORUS


def load_eval_poems(path: Path) -> set:
    """Poem ids held out for evaluation; a missing list is an error, never an empty exclusion (Review Focus 5)."""
    return {int(l) for l in Path(path).read_text(encoding="utf-8").split()}


def training_manifest(manifest, eval_poems: set):
    keep = [m for m in manifest if int(m["poemId"]) not in eval_poems]
    skipped = [m for m in manifest if int(m["poemId"]) in eval_poems]
    return keep, skipped


def shard_of(manifest, k: int, n: int):
    return manifest[k::n]


def merge_shards(out: Path, n: int):
    """Join the per-shard clips/rejects of a sharded build, then write the CSVs and the report as an unsharded build."""
    out = Path(out)
    for name in ("clips", "rejects"):
        parts = [out / f"{name}.{k}.jsonl" for k in range(n)]
        missing = [p for p in parts if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing shard outputs: {missing}")
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for p in parts:
                f.write(p.read_text(encoding="utf-8"))
    rejects = Counter(json.loads(l)["reason"] for l in open(out / "rejects.jsonl", encoding="utf-8") if l.strip())
    write_csvs(out, Q_MIN_MINUTES, pool=True)  # the ZipVoice corpus: wideband clips only, small reciters pooled
    report(out, rejects)


def build(data: Path, lexicon_path, device: str, limit: int = 0, out: Path = None, shard=None):
    """Inputs (manifest, wav, ina, demucs, census) come from `data`; the corpus (clips, labels, CSVs) goes to `out`."""
    out = Path(out) if out else data
    out.mkdir(parents=True, exist_ok=True)
    sfx = f".{shard[0]}" if shard else ""  # a shard (k, n) writes its own files; merge_shards joins them
    g2p = G2P(lexicon=load_lexicon(lexicon_path), device="cpu", log_path=out / f"g2p_diacritised{sfx}.tsv")
    aligner = Aligner(device=device)
    manifest = [json.loads(l) for l in open(data / "manifest.jsonl", encoding="utf-8")]
    manifest, held_out = training_manifest(manifest, load_eval_poems(data / "census" / "eval_poems.txt"))
    if shard:
        manifest = shard_of(manifest, *shard)
    if limit:
        manifest = manifest[:limit]
    genders = reciter_genders(manifest, data / "ina", data / "census" / "reciter_overrides.csv")
    rejects = Counter()
    clips_out = open(out / f"clips{sfx}.jsonl", "w", encoding="utf-8")
    rej_out = open(out / f"rejects{sfx}.jsonl", "w", encoding="utf-8")

    def reject(key, reason, **extra):
        rejects[reason] += 1
        rej_out.write(json.dumps({"clip": key, "reason": reason, **extra}, ensure_ascii=False) + "\n")

    for m in held_out:
        reject(str(m["id"]), "eval_poem")

    for n, m in enumerate(manifest, 1):
        rec = m["id"]
        try:
            timings = json.load(open(m["verses_json"], encoding="utf-8"))
            poem = json.load(open(m["poem_json"], encoding="utf-8"))
            cuts = plan_cuts(timings, poem["verses"])
        except CutError as e:
            reject(str(rec), "timing", detail=str(e))
            continue
        except (KeyError, TypeError, ValueError) as e:
            reject(str(rec), "metadata", detail=str(e))
            continue
        try:  # a bad file must not kill a multi-hour build
            wav_path = Path(m["wav"])
            ina = data / "ina" / f"{rec}.csv"
            music = summarize_ina(ina)["music_fraction"] if ina.exists() else 0.0
            if music > MUSIC_SEPARATE:
                wav_path = separate_vocals(wav_path, data / "demucs")
            bw = bandwidth_hz(m["mp3"])
            y, sr = sf.read(str(wav_path), dtype="float32")
        except Exception as e:  # noqa: BLE001
            reject(str(rec), "io_error", detail=f"{type(e).__name__}: {str(e)[:120]}")
            continue
        metre = metre_of(poem)
        lang_of = {v["vOrder"]: v.get("languageId") for v in poem["verses"]}
        title = poem.get("fullTitle") or ""
        rec_dir = out / "clips" / str(rec)
        rec_dir.mkdir(parents=True, exist_ok=True)
        try:
            pairs = recut(aligner, y, sr, cuts)  # cut at the aligned pause, not at Ganjoor's marks
        except Exception as e:  # noqa: BLE001
            reject(str(rec), "recut_error", detail=f"{type(e).__name__}: {str(e)[:120]}")
            continue
        for old, c in pairs:
            key = f"{rec}/{clip_name(old)}"
            if c is None:
                reject(key, "recut_unaligned")
                continue
            try:
                _one_cut(c, key, y, sr, aligner, g2p, rec, m, genders, bw, metre, lang_of, title, rec_dir, reject, clips_out)
            except Exception as e:  # noqa: BLE001
                reject(key, "exception", detail=f"{type(e).__name__}: {str(e)[:120]}")
        if n % 20 == 0:
            print(f"{n}/{len(manifest)} rejects={sum(rejects.values())}", flush=True)
    clips_out.close()
    rej_out.close()
    if shard:
        return
    write_csvs(out, Q_MIN_MINUTES, pool=True)  # the ZipVoice corpus: wideband clips only, small reciters pooled
    report(out, rejects)


def length_ok(start_ms: int, end_ms: int) -> bool:
    return 600 <= end_ms - start_ms <= T.MAX_SECONDS * 1000


def _one_cut(c, key, y, sr, aligner, g2p, rec, m, genders, bw, metre, lang_of, title, rec_dir, reject, clips_out):
    cname = clip_name(c)
    end = c.end_ms
    if not length_ok(c.start_ms, end):
        reject(key, "length")
        return
    seg = y[int(c.start_ms * sr / 1000):int(end * sr / 1000)]
    if clipping_fraction(seg) > CLIP_MAX:
        reject(key, "clipping")
        return
    al = aligner.align(seg, " ".join(c.lines))
    if not accepts(al):
        reject(key, "alignment", score=round(al.score, 3), low_frac=round(al.low_frac, 3))
        return
    try:
        ph = g2p.phonemize_unit(c.lines, c.kind, metre)
        ids = T.encode(ph)
    except (PhonemizeError, ValueError) as e:
        reject(key, "phonemize", detail=str(e)[:80])
        return
    if len(ids) > T.MAX_IDS:
        reject(key, "too_many_ids")
        return
    out = rec_dir / f"{cname}.wav"
    sf.write(str(out), loudness_normalize(seg, sr), sr, subtype="PCM_16")
    row = {"clip": f"clips/{rec}/{cname}.wav", "unit": cname, "rec_id": rec, "poem_id": m["poemId"], "artist": m["audioArtist"],
           "artist_url": m.get("audioArtistUrl"), "gender": genders.get(m["audioArtist"], "unknown"), "kind": c.kind,
           "lines": c.lines, "vorders": c.vorders, "seconds": len(seg) / sr, "phonemes": ph, "ids": ids,
           "align_score": round(al.score, 3), "bandwidth": round(bw), "narrow": bw < NARROW_HZ, "metre": metre,
           "langs": [lang_of.get(v) for v in c.vorders], "title": title,
           "words": [[w, round(s, 3), round(e, 3)] for w, s, e in al.words]}
    clips_out.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csvs(data: Path, min_minutes: float = 10.0, pool: bool = False):
    rows = [json.loads(l) for l in open(data / "clips.jsonl", encoding="utf-8")]
    if pool:  # quality corpus: wideband only, small reciters pooled
        rows = [r for r in rows if not r.get("narrow")]
    sp = assign_speakers(rows, min_minutes, pool)
    rows = [dict(r, artist=speaker_for(r["artist"], sp)) for r in rows if pool or r["artist"] in sp]
    for r in rows:
        r["poemId"] = r["poem_id"]
    train, val = split_by_poem(rows, val_fraction=0.02)
    for name, part in (("metadata", rows), ("metadata_train", train), ("metadata_val", val)):
        with open(data / f"{name}.csv", "w", encoding="utf-8") as f:
            for r in part:
                f.write(csv_row(r, r["artist"]) + "\n")  # Piper maps names to ids itself; its training writes the map back
    # Provisional map only: once Piper's training has written its own speaker_id_map here, that map is the authority.
    if not (data / "speakers.json").exists():
        (data / "speakers.json").write_text(json.dumps(sp, ensure_ascii=False, indent=1), encoding="utf-8")


def report(data: Path, rejects: Counter):
    rows = [json.loads(l) for l in open(data / "clips.jsonl", encoding="utf-8")]
    h = defaultdict(float)
    for r in rows:
        h[("artist", r["artist"])] += r["seconds"] / 3600
        h[("gender", r["gender"])] += r["seconds"] / 3600
        h[("kind", r["kind"])] += r["seconds"] / 3600
    lines = [f"# Corpus report\n\nclips: {len(rows)}, hours: {sum(r['seconds'] for r in rows) / 3600:.1f}\n", "## rejects\n"]
    lines += [f"- {k}: {v}" for k, v in rejects.most_common()]
    for group in ("gender", "kind", "artist"):
        lines.append(f"\n## hours by {group}\n")
        lines += [f"- {k[1]}: {v:.2f}" for k, v in sorted(h.items(), key=lambda kv: -kv[1]) if k[0] == group]
    (data / "corpus_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--lexicon", default="packages/ganj-reading/ganj_reading/lexicon.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", help="corpus output folder (default: --data)")
    ap.add_argument("--shard", help="K/N: build every N-th recording from K; join with --merge N")
    ap.add_argument("--merge", type=int, help="join N shard outputs in --out and write the CSVs and report")
    a = ap.parse_args()
    out = Path(a.out) if a.out else None
    if a.merge:
        merge_shards(out or Path(a.data), a.merge)
        return
    shard = tuple(int(x) for x in a.shard.split("/")) if a.shard else None
    build(Path(a.data), a.lexicon, a.device, a.limit, out, shard)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""ManaTTS (HF MahtaFetrat/Mana-TTS, CC0-1.0; 114 h, 44.1 kHz, one reader, Persian prose) as extra clean training data.

Kept: HIGH-quality alignments, 1-12 s (the fine-tune's --max-len is 12), text without digits or Latin letters (the G2P
does not read them). Audio is resampled to 24 kHz 16-bit wav; transcripts are phonemised as prose units (prose token,
no metre). Writes a ZipVoice custom TSV per parquet part (resumable: a part whose TSV exists is skipped).

python -m ganj_corpus.manatts RAW_DIR OUT_DIR [--device cuda]
"""
import argparse
import re
from pathlib import Path

MIN_S, MAX_S = 1.0, 12.0
UNREADABLE = re.compile(r"[0-9۰-۹٠-٩A-Za-z]")


def keep(row: dict) -> bool:
    return (bool(row["transcript"]) and row["match_quality"] == "HIGH" and MIN_S <= row["duration"] <= MAX_S
            and not UNREADABLE.search(row["transcript"]))


def clip_id(file_name: str) -> str:
    return "mana-" + Path(file_name).stem


META = ["file name", "transcript", "duration", "match quality", "samplerate"]


def from_parquet(r: dict) -> dict:
    return {"file_name": r["file name"], "transcript": (r["transcript"] or "").strip(), "duration": r["duration"],
            "match_quality": r["match quality"], "sample_rate": int(r["samplerate"])}


def convert_part(part: Path, out: Path, g2p, sr_out: int = 24000) -> int:
    """One parquet part -> 24 kHz wavs under out/clips and out/tsv/<part>.tsv. Returns the number of clips kept."""
    import librosa
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    f = pq.ParquetFile(part)
    meta = [from_parquet(r) for r in f.read(columns=META).to_pylist()]
    idx = [i for i, r in enumerate(meta) if keep(r)]
    phon = g2p.phonemize_units([([meta[i]["transcript"]], "prose", None) for i in idx]) if idx else []
    audio = f.read(columns=["audio"]).column("audio")
    (out / "clips").mkdir(parents=True, exist_ok=True)
    lines = []
    for i, ph in zip(idx, phon):
        if not ph:
            continue
        cid = clip_id(meta[i]["file_name"])
        wav = out / "clips" / f"{cid}.wav"
        if not wav.exists():
            y = np.asarray(audio[i].as_py(), dtype=np.float32)
            y = librosa.resample(y, orig_sr=meta[i]["sample_rate"], target_sr=sr_out)
            sf.write(str(wav), y, sr_out, subtype="PCM_16")
        lines.append(f"{cid}\t{ph}\t{wav}\n")
    tsv = out / "tsv" / (part.stem + ".tsv")
    tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv.with_suffix(".tmp").write_text("".join(lines), encoding="utf-8")
    tsv.with_suffix(".tmp").rename(tsv)          # a part counts as done only when its whole TSV is written
    return len(lines)


def merge(tsv_dir, out) -> tuple:
    """All part TSVs into one, without any clip whose id occurs more than once. Returns (kept, dropped)."""
    import collections
    lines = [l for t in sorted(Path(tsv_dir).glob("*.tsv")) if t.resolve() != Path(out).resolve()
             for l in open(t, encoding="utf-8") if l.strip()]
    n = collections.Counter(l.split("\t", 1)[0] for l in lines)
    kept = [l for l in lines if n[l.split("\t", 1)[0]] == 1]
    Path(out).write_text("".join(kept), encoding="utf-8")
    return len(kept), len(lines) - len(kept)


def transcripts(raw_dir) -> dict:
    """clip id -> transcript, read back from the raw parquet parts (only two columns, so it is quick); a file name that
    occurs in more than one part is left out, as `merge` drops its clips (the relabel needs the text)."""
    import collections

    import pyarrow.parquet as pq
    rows = [(clip_id(f), (t or "").strip()) for p in sorted(Path(raw_dir).rglob("*.parquet"))
            for f, t in zip(*pq.read_table(p, columns=["file name", "transcript"]).to_pydict().values())]
    n = collections.Counter(k for k, _ in rows)
    return {k: t for k, t in rows if n[k] == 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", help="RAW_DIR, or 'merge' to write OUT_DIR/all.tsv from OUT_DIR/tsv")
    ap.add_argument("out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lexicon", default="packages/ganj-reading/ganj_reading/lexicon.json")
    a = ap.parse_args()
    if a.raw == "merge":
        print("MANA_MERGED kept=%d dropped=%d" % merge(Path(a.out) / "tsv", Path(a.out) / "all.tsv"))
        return
    from ganj_reading.g2p import G2P, load_lexicon
    g2p = G2P(lexicon=load_lexicon(a.lexicon), device=a.device)
    out = Path(a.out)
    for part in sorted(Path(a.raw).glob("dataset/*.parquet")):
        if (out / "tsv" / (part.stem + ".tsv")).exists():
            continue
        print(part.stem, convert_part(part, out, g2p), flush=True)
    print("MANA_CONVERT_DONE")


if __name__ == "__main__":
    main()

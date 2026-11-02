# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Download selected recitations (max 4 workers), decode to 22.05 kHz mono WAV, fetch verse timings and poem JSON."""
import argparse
import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import soundfile as sf

from ganj_corpus.ganjoor import HEADERS, Ganjoor


def _requests_fetch(url, headers):
    r = requests.get(url, headers=headers, timeout=300)
    return r.status_code, r.content


def download_one(row: dict, mp3_dir: Path, fetch=None, tries: int = 6) -> Path:
    fetch = fetch or _requests_fetch
    mp3_dir = Path(mp3_dir)
    mp3_dir.mkdir(parents=True, exist_ok=True)
    out = mp3_dir / f"{row['id']}.mp3"
    expected = int(row.get("mp3SizeInBytes") or 0)
    if out.exists() and (expected == 0 or out.stat().st_size == expected):
        return out
    delay = 1.0
    status, body = None, b""
    for _ in range(tries):
        try:
            status, body = fetch(row["mp3Url"], HEADERS)
        except requests.RequestException:
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
            continue
        if status == 200 and (expected == 0 or len(body) == expected):
            part = out.with_suffix(".part")
            part.write_bytes(body)
            part.rename(out)
            return out
        if status == 429 or status >= 500 or (status == 200 and len(body) != expected):
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
            continue
        break
    raise RuntimeError(f"download failed for {row['id']}: status {status}, {len(body)} bytes, expected {expected}")


def decode_wav(mp3_path: Path, wav_path: Path, sr: int = 22050) -> float:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if not wav_path.exists():
        part = wav_path.with_suffix(".part")
        try:
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(mp3_path), "-ac", "1", "-ar", str(sr),
                            "-sample_fmt", "s16", "-f", "wav", str(part)], check=True)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.rename(wav_path)
    info = sf.info(str(wav_path))
    return info.frames / info.samplerate


def _atomic_write(path: Path, text: str):
    part = path.with_suffix(".part")
    try:
        part.write_text(text, encoding="utf-8")
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.rename(path)


def fetch_meta(g: Ganjoor, row: dict, data_dir: Path) -> tuple:
    verses = data_dir / "verses" / f"{row['id']}.json"
    poem = data_dir / "poems" / f"{row['poemId']}.json"
    verses.parent.mkdir(parents=True, exist_ok=True)
    poem.parent.mkdir(parents=True, exist_ok=True)
    if not verses.exists():
        _atomic_write(verses, json.dumps(g.recitation_verses(row["id"]), ensure_ascii=False))
    if not poem.exists():
        _atomic_write(poem, json.dumps(g.poem(row["poemId"]), ensure_ascii=False))
    return verses, poem


def process(row: dict, data_dir: Path, g: Ganjoor) -> dict:
    mp3 = download_one(row, data_dir / "mp3")
    wav = data_dir / "wav" / f"{row['id']}.wav"
    seconds = decode_wav(mp3, wav)
    verses, poem = fetch_meta(g, row, data_dir)
    return {"id": row["id"], "poemId": row["poemId"], "audioArtist": row["audioArtist"], "audioArtistUrl": row.get("audioArtistUrl"),
            "mp3": str(mp3), "wav": str(wav), "seconds": seconds, "verses_json": str(verses), "poem_json": str(poem)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    data_dir = Path(a.data)
    manifest = data_dir / "manifest.jsonl"
    done = set()
    if manifest.exists():
        done = {json.loads(l)["id"] for l in open(manifest, encoding="utf-8")}
    rows = [json.loads(l) for l in open(a.selected, encoding="utf-8")]
    rows = [r for r in rows if r["id"] not in done]
    g = Ganjoor()
    with open(manifest, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=min(4, a.workers)) as ex:
        futures = {ex.submit(process, r, data_dir, g): r for r in rows}
        for i, fut in enumerate(as_completed(futures), 1):
            r = futures[fut]
            try:
                out.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
                out.flush()
            except Exception as e:  # noqa: BLE001 — log and continue; the manifest is the resume point
                print(f"FAILED {r['id']}: {e}")
            if i % 50 == 0:
                print(f"{i}/{len(rows)}")


if __name__ == "__main__":
    main()

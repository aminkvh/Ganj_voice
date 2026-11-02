# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""ZipVoice custom-dataset TSVs (egs/zipvoice/run_custom.sh format) with phoneme strings as text."""
import json
import sys
from pathlib import Path

from ganj_corpus.select import split_by_poem


def tsv_line(row: dict, data_root: str) -> str:
    unit = row["unit"] if "unit" in row else Path(row["clip"]).stem  # clips.jsonl carries the unit in the clip name
    return f"{row['rec_id']}-{unit}\t{row['phonemes']}\t{data_root}/{row['clip']}"


def training_rows(rows):
    """Wideband clips only, the same rule as the Piper quality corpus."""
    return [r for r in rows if not r.get("narrow")]


def main(data: Path, out_dir: Path, hours: float):
    rows = training_rows([json.loads(l) for l in open(data / "clips.jsonl", encoding="utf-8")])
    for r in rows:
        r["poemId"] = r["poem_id"]
    train, dev = split_by_poem(rows, 0.02)
    out_dir.mkdir(parents=True, exist_ok=True)
    used = 0.0
    with open(out_dir / "custom_train.tsv", "w", encoding="utf-8") as f:
        for r in train:
            f.write(tsv_line(r, str(data)) + "\n")
            used += r["seconds"]
            if used >= hours * 3600:
                break
    with open(out_dir / "custom_dev.tsv", "w", encoding="utf-8") as f:
        for r in dev[:500]:
            f.write(tsv_line(r, str(data)) + "\n")
    print(f"train {used / 3600:.1f} h, dev {min(500, len(dev))}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]))

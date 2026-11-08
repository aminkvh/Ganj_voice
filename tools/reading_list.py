# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Text to a reading list for recipes/zipvoice/synth_voice.sh.

  python tools/reading_list.py IN.jsonl OUT_DIR --voice VOICE_DIR [--va] [--overrides FILE.json] [--device cuda]
      [--read-all]

Non-Persian hemistichs are left out (drop_non_persian; units may carry "langs", Ganjoor's languageId per line) unless
--read-all, which the eval lists use (eval/distill/list50_d4h.tsv scores the Arabic lines too).

IN.jsonl: one unit per line, {"key", "lines": [hemistichs, or one prose line], "kind": "verse" | "prose", "metre"
(optional, the poem's metre name)}. Writes OUT_DIR/list.tsv (key, the voice's prompt phonemes, prompt.wav, target
phonemes) and OUT_DIR/text.jsonl ({"key", "lines", "phonemes"}: what the draws are checked against), then
  recipes/zipvoice/synth_voice.sh OUT_DIR/list.tsv WAV_DIR OUT_DIR/text.jsonl
The label is the G2P's (metre, written marks, lexicon; verse through the voice's own model in VOICE_DIR/g2p when it
has one, prose through the base model); --va reads a standalone «و» as v/ in prose, at a line start and
after punctuation (g2p.read_vav); a voice whose tokens.txt has the pause tokens gets the reading-time pauses
(pauses.rule_label). The voice dir holds prompt_phonemes.txt (the prompt's label, with its pauses for a pause voice).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ganj_corpus.pauses import PAUSE_LONG, PAUSE_SHORT, reading_label, strip_pauses  # noqa: E402


def voice_has_pauses(tokens_txt: str) -> bool:
    chars = {l.split("\t")[0] for l in tokens_txt.splitlines() if l}
    return PAUSE_SHORT in chars and PAUSE_LONG in chars


def with_overrides(rows: list, labels: list, overrides: list) -> list:
    """Reviewed readings (audit_g2p/review/overrides_*.json: id = the unit's key, line, word, text, label) pinned
    into the G2P's labels, one word each (relabel.apply_overrides)."""
    from ganj_corpus.relabel import apply_overrides
    by_key = {r["key"]: ph for r, ph in zip(rows, labels) if ph is not None}
    new, _, _ = apply_overrides(by_key, {r["key"]: r["lines"] for r in rows}, overrides)
    return [new.get(r["key"], ph) if ph is not None else None for r, ph in zip(rows, labels)]


def g2p_for(voice: Path, G, device: str = "cpu"):
    """The voice's G2P: verse through the fine-tuned model in VOICE/g2p when the voice has one, prose through the base
    model (as the voice's training labels were made; relabel.ByKind); the base model for both otherwise."""
    from ganj_corpus.relabel import ByKind
    lexicon = G.load_lexicon(Path(G.__file__).parent / "lexicon.json")
    base = G.G2P(lexicon=lexicon, device=device)
    if not (voice / "g2p").is_dir():
        return base
    return ByKind(verse=G.G2P(model_id=str(voice / "g2p"), lexicon=lexicon, device=device), prose=base)


def drop_non_persian(rows: list) -> tuple:
    """(units with their non-Persian hemistichs taken out, units with none left). Since 2026-11-08 the voice does not
    read Arabic or other non-Persian hemistichs (ganj_reading.language: Ganjoor's tag in the unit's optional "langs",
    one languageId or null per line, else the Arabic detector; prose lines by their tag only). A unit that lost a line
    carries "skipped": the indices of the lines left out."""
    from ganj_reading.language import skip_hemistich
    kept, skipped = [], []
    for r in rows:
        langs = r.get("langs") or [None] * len(r["lines"])
        prose = r.get("kind") == "prose"
        out = [i for i, (line, lid) in enumerate(zip(r["lines"], langs))
               if (lid is not None and skip_hemistich(line, lid)) or (not prose and skip_hemistich(line, lid))]
        if not out:
            kept.append(r)
        elif len(out) == len(r["lines"]):
            skipped.append({**r, "skipped": out})
        else:
            kept.append({**r, "lines": [l for i, l in enumerate(r["lines"]) if i not in out], "skipped": out})
    return kept, skipped


def build(rows: list, labels: list, prompt_ph: str, pauses: bool, va: bool):
    """(list.tsv lines, text.jsonl lines); a unit the G2P could not read (label None) is left out."""
    tsv, text = [], []
    for r, ph in zip(rows, labels):
        if ph is None:
            print(f"{r['key']}: the G2P could not read it, left out", file=sys.stderr)
            continue
        prose = r.get("kind") == "prose"
        target = reading_label(ph, r["lines"], prose, pauses=pauses, va=va)
        tsv.append("\t".join([r["key"], prompt_ph, "prompt.wav", target]) + "\n")
        row = {"key": r["key"], "lines": r["lines"], "phonemes": strip_pauses(target)}
        if r.get("skipped"):
            row["skipped"] = r["skipped"]
        text.append(json.dumps(row, ensure_ascii=False) + "\n")
    return tsv, text


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--voice", required=True)
    ap.add_argument("--va", action="store_true", help="«و» read v/ in prose, at a line start and after punctuation")
    ap.add_argument("--overrides", action="append", default=[], help="reviewed per-line readings (JSON list)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--read-all", action="store_true",
                    help="keep the non-Persian hemistichs (the eval lists, which score the Arabic lines too)")
    a = ap.parse_args(argv)
    from ganj_reading import g2p as G
    voice = Path(a.voice)
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()]
    rows, skipped = (rows, []) if a.read_all else drop_non_persian(rows)
    labels = g2p_for(voice, G, a.device).phonemize_units([(r["lines"], r.get("kind", "verse"), r.get("metre")) for r in rows])
    labels = with_overrides(rows, labels, [o for p in a.overrides for o in json.load(open(p, encoding="utf-8"))])
    pauses = voice_has_pauses((voice / "tokens.txt").read_text(encoding="utf-8"))
    prompt_ph = (voice / "prompt_phonemes.txt").read_text(encoding="utf-8").strip("\n")
    tsv, text = build(rows, labels, prompt_ph, pauses, a.va)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "list.tsv").write_text("".join(tsv), encoding="utf-8")
    (out / "text.jsonl").write_text("".join(text), encoding="utf-8")
    (out / "skipped.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in skipped), encoding="utf-8")
    cut = sum(len(r.get("skipped", [])) for r in rows + skipped)
    print(f"wrote {out}/list.tsv and text.jsonl: {len(tsv)} of {len(rows) + len(skipped)} units; {cut} non-Persian "
          f"hemistichs left out ({len(skipped)} units with nothing left, in skipped.jsonl); pauses "
          f"{'on' if pauses else 'off'}, «و» rule {'on' if a.va else 'off'}")


if __name__ == "__main__":
    main()

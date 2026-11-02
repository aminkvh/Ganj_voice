# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Re-phonemise corpus_q with the current G2P and rewrite the ZipVoice cut manifests and eval test lists
in place of re-extracting features: only the supervision text changes.

python -m ganj_corpus.relabel --data $GANJ_DATA --cuts-dir <data_q/fbank> --subset clean [--tag _q9b] [--device cuda]
    [--overrides audit_g2p/review/overrides_q9e.json]
Writes corpus_q/labels<tag>.jsonl (id -> phonemes, resumable), <cuts><tag>.jsonl.gz for train_clean and dev,
eval/set_q<tag>.jsonl and corpus_q/prompts/test_<name><tag>.tsv (each tag, e.g. _q9b or _q9c, names one label set).
--overrides pins single words in single lines to a reviewer's reading (apply_overrides); --mana-raw also relabels the
ManaTTS cuts (custom_cuts_mana_{train,dev}<tag>.jsonl.gz, labels in corpus_q/labels<tag>_mana.jsonl). --verse-model
labels the verse units with another model of the G2P (ByKind); prose units keep the base model.
"""
import argparse
import gzip
import json
import re
from pathlib import Path

from ganj_reading import tokens as T
from ganj_reading.g2p import G2P, load_lexicon, normalize_fa, strip_harakat


PREFIX_CHARS = "".join(chr(c) for c in range(0xE000, 0xE100))


def _same_word(a: str, b: str) -> bool:
    return normalize_fa(strip_harakat(a)) == normalize_fa(strip_harakat(b))


def apply_overrides(labels: dict, rows: dict, overrides: list):
    """Pin single words to a reviewer's reading (a hand review of the labels): each override names a clip id, a
    line and a word index and is applied to that word only. If the word index no longer points at the override's text
    (a G2P change moved it), the one word with that text in the line is used; otherwise the override is skipped.
    rows: id -> text lines. Returns (new labels, number applied, skipped overrides with the reason)."""
    out, applied, skipped = dict(labels), 0, []
    for o in overrides:
        ph, lines = out.get(o["id"]), [l for l in rows.get(o["id"], []) if l.strip()]
        if ph is None or o["line"] >= len(lines):
            skipped.append([o, "no label or line"])
            continue
        k = len(ph) - len(ph.lstrip(PREFIX_CHARS))  # style and metre tokens (private-use codepoints)
        prefix, parts = ph[:k], ph[k:].split(T.HEMI)
        words = G2P.words(lines[o["line"]])
        labs = parts[o["line"]].split(" ") if o["line"] < len(parts) else []
        if len(labs) != len(words):
            skipped.append([o, "label not aligned with the line"])
            continue
        i = o["word"] if o["word"] < len(words) and _same_word(words[o["word"]], o["text"]) else None
        if i is None:
            hits = [j for j, w in enumerate(words) if _same_word(w, o["text"])]
            i = hits[0] if len(hits) == 1 else None
        if i is None:
            skipped.append([o, "word not found in the line"])
            continue
        labs[i] = o["label"]
        parts[o["line"]] = " ".join(labs)
        out[o["id"]] = prefix + T.HEMI.join(parts)
        applied += 1
    return out, applied, skipped


class ByKind:
    """Two G2Ps as one: verse units to `verse`, prose units to `prose`, in the units' order (a model fine-tuned on verse
    labels verse; prose keeps the base model when nothing has measured the other on prose)."""

    def __init__(self, verse, prose):
        self.verse, self.prose = verse, prose

    def phonemize_units(self, units):
        out = [None] * len(units)
        for g, kind in ((self.verse, "verse"), (self.prose, "prose")):
            idx = [i for i, u in enumerate(units) if (u[1] == "prose") == (kind == "prose")]
            for i, ph in zip(idx, g.phonemize_units([units[i] for i in idx]) if idx else []):
                out[i] = ph
        return out


def relabel_cut_line(line: str, labels: dict) -> str:
    c = json.loads(line)
    new = labels.get(c["id"])
    if new is not None:
        for s in c["supervisions"]:
            s["text"] = new
    return json.dumps(c, ensure_ascii=False)


def relabel_tsv_line(line: str, labels: dict, prompts: dict = None) -> str:
    key, prompt, wav, text = line.rstrip("\n").split("\t")
    return "\t".join([key, (prompts or {}).get(prompt, prompt), wav, labels.get(key, text)]) + "\n"


def prompt_labels(g2p, tsvs, sources):
    """New labels for the prompt texts of the test lists: each old prompt string is matched to a clip with the same old
    phonemes (corpus_q or the first corpus build) and that clip's text is phonemised again."""
    want = {l.split("\t")[1] for t in tsvs for l in open(t, encoding="utf-8") if l.strip()}
    found = {}
    for src in sources:
        for l in open(src, encoding="utf-8"):
            r = json.loads(l)
            if r["phonemes"] in want and r["phonemes"] not in found:
                found[r["phonemes"]] = (r["lines"], r["kind"], r.get("metre") or None)
    keys = list(found)
    return {k: ph for k, ph in zip(keys, g2p.phonemize_units([found[k] for k in keys])) if ph}


LABEL_TAG = re.compile(r"_q9[a-z]$")


def source_tsvs(prompts_dir):
    """The original test lists: every relabelled set (test_<name>_q9b, _q9c, ...) is an output, never a source."""
    return [t for t in Path(prompts_dir).glob("test_*.tsv") if not LABEL_TAG.search(t.stem)]


def _cut_ids(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return {json.loads(l)["id"] for l in f}


def label_units(g2p, rows, out_path, wanted, chunk: int = 512):
    """Labels for the wanted rows, resuming from out_path; batched G2P, one complete JSON line per label."""
    done = {}
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue  # a torn last line from an interrupted run
            done[r["id"]] = r["phonemes"]
    todo = [r for r in rows if r["id"] in wanted and r["id"] not in done]
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), chunk):
            part = todo[i:i + chunk]
            phs = g2p.phonemize_units([(r["lines"], r["kind"], r.get("metre") or None) for r in part])
            f.write("".join(json.dumps({"id": r["id"], "phonemes": ph}, ensure_ascii=False) + "\n"
                            for r, ph in zip(part, phs) if ph is not None))
            f.flush()
            done.update({r["id"]: ph for r, ph in zip(part, phs) if ph is not None})
            print(f"{len(done)} labelled ({i + len(part)}/{len(todo)} this run)", flush=True)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--cuts-dir", required=True)
    ap.add_argument("--subset", default="clean")
    ap.add_argument("--lexicon", default="packages/ganj-reading/ganj_reading/lexicon.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="_q9b")
    ap.add_argument("--overrides", default=None, help="per-line word readings (JSON list; see apply_overrides)")
    ap.add_argument("--mana-raw", default=None, help="ManaTTS raw parquet dir: also relabel custom_cuts_mana_{train,dev}")
    ap.add_argument("--verse-model", default=None, help="the G2P's model for verse units (a fine-tuned copy's directory); prose keeps the base")
    a = ap.parse_args()
    data, cuts = Path(a.data), Path(a.cuts_dir)
    train = cuts / f"custom_cuts_train_{a.subset}.jsonl.gz"
    dev = cuts / "custom_cuts_dev.jsonl.gz"
    wanted = _cut_ids(train) | _cut_ids(dev)
    rows = []
    for l in open(data / "corpus_q" / "clips.jsonl", encoding="utf-8"):
        r = json.loads(l)
        rows.append({"id": f"{r['rec_id']}-{r['unit']}", "lines": r["lines"], "kind": r["kind"], "metre": r.get("metre")})
    evals = [json.loads(l) for l in open(data / "eval" / "set_q.jsonl", encoding="utf-8")]
    erows = [{"id": e["key"], "lines": e["lines"], "kind": "prose" if e["cls"] == "prose" else "verse", "metre": e.get("metre")}
             for e in evals]
    g2p = G2P(lexicon=load_lexicon(a.lexicon), device=a.device)
    if a.verse_model:
        g2p = ByKind(verse=G2P(model_id=a.verse_model, lexicon=load_lexicon(a.lexicon), device=a.device), prose=g2p)
    labels = label_units(g2p, erows + rows, data / "corpus_q" / f"labels{a.tag}.jsonl", wanted | {e["id"] for e in erows})
    if a.overrides:  # applied to the written cuts and lists only; labels<tag>.jsonl stays the plain G2P output
        overrides = json.load(open(a.overrides, encoding="utf-8"))
        labels, applied, skipped = apply_overrides(labels, {r["id"]: r["lines"] for r in erows + rows}, overrides)
        print(f"overrides applied: {applied} of {len(overrides)}", flush=True)
        for o, why in skipped:
            print("override skipped:", o["id"], o["text"], o["label"], why, flush=True)
    for src in (train, dev):
        dst = src.with_name(src.name.replace(".jsonl.gz", f"{a.tag}.jsonl.gz"))
        with gzip.open(src, "rt", encoding="utf-8") as fi, gzip.open(dst, "wt", encoding="utf-8") as fo:
            for line in fi:
                fo.write(relabel_cut_line(line, labels) + "\n")
        print("wrote", dst)
    with open(data / "eval" / f"set_q{a.tag}.jsonl", "w", encoding="utf-8") as f:
        for e in evals:
            f.write(json.dumps({**e, "phonemes": labels.get(e["key"], e["phonemes"])}, ensure_ascii=False) + "\n")
    if a.mana_raw:  # ManaTTS kept only phonemes; its text is read back from the parquet parts
        from ganj_corpus.manatts import transcripts
        text = transcripts(a.mana_raw)
        msrc = [cuts / f"custom_cuts_mana_{s}.jsonl.gz" for s in ("train", "dev")]
        mids = set().union(*(_cut_ids(m) for m in msrc))
        mrows = [{"id": i, "lines": [text[i]], "kind": "prose", "metre": None} for i in sorted(mids) if i in text]
        mlabels = label_units(g2p, mrows, data / "corpus_q" / f"labels{a.tag}_mana.jsonl", mids)
        for src in msrc:
            dst = src.with_name(src.name.replace(".jsonl.gz", f"{a.tag}.jsonl.gz"))
            with gzip.open(src, "rt", encoding="utf-8") as fi, gzip.open(dst, "wt", encoding="utf-8") as fo:
                for line in fi:
                    fo.write(relabel_cut_line(line, mlabels) + "\n")
            print("wrote", dst, flush=True)
    tsvs = source_tsvs(data / "corpus_q" / "prompts")
    prompts = prompt_labels(g2p, tsvs, [data / "corpus_q" / "clips.jsonl", data / "clips.jsonl"])
    print(f"prompt texts relabelled: {len(prompts)}")
    for tsv in tsvs:
        out = tsv.with_name(tsv.stem + f"{a.tag}.tsv")
        out.write_text("".join(relabel_tsv_line(l, labels, prompts) for l in open(tsv, encoding="utf-8") if l.strip()),
                       encoding="utf-8")
    print("RELABEL_DONE", len(labels))


if __name__ == "__main__":
    main()

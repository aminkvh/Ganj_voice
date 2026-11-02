# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Pause tokens in the training labels (ganj_corpus.pauses).

  python tools/pause_labels.py [--tag-in _q9e] [--tag _q9p] COMMAND ...
      every file read is at the input tag, every file written at the output tag (below: the defaults)
  python tools/pause_labels.py label [--device cuda] [--limit N]
      align every cut of the q9e train/dev mixes with its text and write corpus_q/labels_q9p.jsonl (resumable):
      {"id", "phonemes" with pause tokens, "pauses" in s per word boundary, "conf"} or {"id", "skip": reason}
  python tools/pause_labels.py cuts [--min-conf 0.0] [--overrides ALL.json] [--keep LISTENER.json]
      write custom_cuts_{train,dev}_mix_q9p.jsonl.gz (marked cuts only, minus those whose label contradicts a written
      mark, listed in corpus_q/held_back_q9p.jsonl; fails if any per-line reading is missing) and tokens_custom_q9p.txt.
      --keep: the readings whose cuts are never held back (default: all of --overrides)
  python tools/pause_labels.py evalset [--overrides EVAL.json]
      eval/set_q_q9p.jsonl: the q9e eval labels with the eval-line readings (audit_g2p/review/overrides_eval_q9p.json)
  python tools/pause_labels.py testlist SRC.tsv DST.tsv --prompt-wav W.wav --prompt-id 18347-15
      a reading list from the current eval labels (set_q_q9p.jsonl; SRC gives only the keys and the prompt text): every
      target gets rule_label pauses, the prompt text its own measured pauses
  python tools/pause_labels.py ckpt SRC.pt DST.pt
      the fine-tune base: embed.weight gets one row per pause token, each a copy of the space token's row

A cut is marked only when its label has one word per text word in every hemistich (G2P.words) and every word has
letters the aligner knows; ≈ 2 % of the q9e mix (split compounds such as یک‌تو → yek to) is left out.
"""
import argparse
import gzip
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ganj_corpus.pauses import PAUSE_LONG, PAUSE_SHORT, mark_pauses, pause_token, rule_label, split_label  # noqa: E402
from ganj_reading.g2p import G2P  # noqa: E402

from tools._env import env_path  # noqa: E402
DATA = env_path("GANJ_DATA")
DQ = env_path("ZIPVOICE_DIR") / "egs/zipvoice/data_q"
TAG_IN, TAG = "_q9e", "_q9p"
SETS = ("train", "dev")


def plan_cut(lines, label, chars):
    """(the text's words in order, None) when the cut can be marked, else (None, reason)."""
    lines = [l for l in lines if l.strip()]
    _, parts = split_label(label)
    if len(parts) != len(lines):
        return None, "line count"
    words = []
    for part, line in zip(parts, lines):
        ws = G2P.words(line)
        if len(ws) != len(part):
            return None, "word count"
        words += ws
    if not all(chars(w) for w in words):
        return None, "no letters"
    return words, None


def extend_tokens(src, dst):
    rows = [l.rstrip("\n").split("\t") for l in open(src, encoding="utf-8") if l.strip("\n")]
    have = {r[0] for r in rows}
    if PAUSE_SHORT in have or PAUSE_LONG in have:
        raise ValueError("token list already has the pause tokens")
    nxt = max(int(r[1]) for r in rows) + 1
    with open(dst, "w", encoding="utf-8") as f:
        f.write("".join(f"{t}\t{i}\n" for t, i in rows) + f"{PAUSE_SHORT}\t{nxt}\n{PAUSE_LONG}\t{nxt + 1}\n")


def extend_checkpoint(src, dst, rows: int = 2, init_id: int = 1):
    import torch
    c = torch.load(src, map_location="cpu", weights_only=False)
    w = c["model"]["embed.weight"]
    c["model"]["embed.weight"] = torch.cat([w, w[init_id:init_id + 1].repeat(rows, 1)])
    torch.save(c, dst)


def cuts_in(s, tag_in=TAG_IN):
    with gzip.open(DQ / "fbank" / f"custom_cuts_{s}_mix{tag_in}.jsonl.gz", "rt", encoding="utf-8") as f:
        for l in f:
            yield json.loads(l)


def texts():
    from ganj_corpus.manatts import transcripts
    out = {f"{r['rec_id']}-{r['unit']}": r["lines"]
           for r in map(json.loads, open(DATA / "corpus_q" / "clips.jsonl", encoding="utf-8"))}
    out.update({k: [v] for k, v in transcripts(DATA / "manatts" / "raw").items()})
    return out


def label(device: str, limit: int, out_path=None, tag_in=TAG_IN, tag=TAG):
    import librosa
    import soundfile as sf
    from ganj_eval.align import SR, Aligner, boundary_pauses
    out_path = Path(out_path or DATA / "corpus_q" / f"labels{tag}.jsonl")
    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["id"])
            except json.JSONDecodeError:
                pass   # a torn last line from an interrupted run
    aligner, text = Aligner(device), texts()
    jobs = [(c["id"], text.get(c["id"]), c["supervisions"][0]["text"], c["recording"]["sources"][0]["source"])
            for s in SETS for c in cuts_in(s, tag_in) if c["id"] not in done]
    jobs = jobs[:limit] if limit else jobs
    q = queue.Queue(maxsize=64)

    def load():
        for j in jobs:
            try:
                y, sr = sf.read(j[3], dtype="float32", always_2d=True)
                q.put((j, librosa.resample(y.mean(axis=1), orig_sr=sr, target_sr=SR)))
            except Exception as e:  # noqa: BLE001 - an unreadable file is recorded as a skip, not a crash
                q.put((j, e))
        q.put(None)

    threading.Thread(target=load, daemon=True).start()
    n, skips, t0 = 0, {}, time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        while (item := q.get()) is not None:
            (cid, lines, lab, _), y = item
            rec = {"id": cid}
            words, why = plan_cut(lines or [], lab, aligner.chars) if lines else (None, "no text")
            if isinstance(y, Exception):
                why = f"audio: {y}"
            if why is None:
                try:
                    spans, conf = aligner.spans(y, words)
                    pauses = boundary_pauses(spans, y)
                    rec.update(phonemes=mark_pauses(lab, [pause_token(p) for p in pauses]),
                               pauses=[round(p, 3) for p in pauses], conf=round(conf, 3))
                except (ValueError, RuntimeError) as e:
                    why = f"align: {e}"[:80]
            if why is not None:
                rec["skip"] = why
                skips[why.split(":")[0]] = skips.get(why.split(":")[0], 0) + 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                f.flush()
                print(f"{n}/{len(jobs)} {n / (time.time() - t0):.1f} cuts/s skips {skips}", flush=True)
    print(f"LABEL_DONE {n} skips {skips}", flush=True)


OVERRIDES = DATA / "audit_g2p" / "review" / "overrides_q9e.json"


def held_back(labels: dict, rows: dict, keep: set) -> dict:
    """Cuts whose label contradicts a mark written in their text (rules_check), id -> findings; the cuts in keep (those
    carrying per-line readings) are never held back: no generation trains on such a label."""
    from ganj_corpus.rules_check import UNALIGNED, check_label
    out = {}
    for k, lab in labels.items():
        if k in keep or k not in rows:
            continue
        found = [v for v in check_label(rows[k], lab) if v.rule != UNALIGNED]
        if found:
            out[k] = found
    return out


def unapplied(labels: dict, rows: dict, overrides: list) -> list:
    """The per-line readings missing from the labels (rules_check.missing_overrides), except a reading copied from
    another recording (`of`) whose word this one writes differently (در آورد for درآورد): relabel skips those."""
    from ganj_corpus.rules_check import missing_overrides
    return [m for o in overrides for m in missing_overrides(labels, rows, [o])
            if not (o.get("of", o["id"]) != o["id"] and m[3] == "word not found")]


def write_cuts(min_conf: float, tag_in=TAG_IN, tag=TAG, overrides_path=OVERRIDES, keep_path=None):
    labels = {}
    for l in open(DATA / "corpus_q" / f"labels{tag}.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if "phonemes" in r and r["conf"] >= min_conf:
            labels[r["id"]] = r["phonemes"]
    rows, overrides = texts(), json.load(open(overrides_path, encoding="utf-8"))
    lost = unapplied(labels, rows, overrides)
    if lost:
        raise SystemExit(f"per-line readings missing from the {tag} labels: {lost[:5]} ... ({len(lost)})")
    keep = json.load(open(keep_path, encoding="utf-8")) if keep_path else overrides
    held = held_back(labels, rows, {o["id"] for o in keep})
    print(f"held back for written marks not read: {len(held)} cuts "
          f"({sum(len(v) >= 4 for v in held.values())} with >= 4 findings, mostly Arabic lines)", flush=True)
    for s in SETS:
        dst = DQ / "fbank" / f"custom_cuts_{s}_mix{tag}.jsonl.gz"
        kept = total = 0
        with gzip.open(dst, "wt", encoding="utf-8") as fo:
            for c in cuts_in(s, tag_in):
                total += 1
                if c["id"] in labels and c["id"] not in held:
                    for sup in c["supervisions"]:
                        sup["text"] = labels[c["id"]]
                    fo.write(json.dumps(c, ensure_ascii=False) + "\n")
                    kept += 1
        print(f"wrote {dst}: {kept} of {total} cuts", flush=True)
    with open(DATA / "corpus_q" / f"held_back{tag}.jsonl", "w", encoding="utf-8") as f:
        for k, vs in held.items():
            f.write(json.dumps({"id": k, "findings": [[v.line, v.word, v.label, v.rule] for v in vs]}, ensure_ascii=False) + "\n")
    if not (DQ / f"tokens_custom{tag}.txt").exists():
        extend_tokens(DQ / "tokens_custom.txt", DQ / f"tokens_custom{tag}.txt")
    print("tokens", DQ / f"tokens_custom{tag}.txt")


def prompt_lines(prompt_id: str) -> list:
    """A prompt source's text: corpus_q first, then the first corpus build (where F2/F5's clip 18347/15 is)."""
    want = "clips/" + prompt_id.replace("-", "/", 1) + ".wav"
    for src in (DATA / "corpus_q" / "clips.jsonl", DATA / "clips.jsonl"):
        for l in open(src, encoding="utf-8"):
            r = json.loads(l)
            if r["clip"] == want:
                return r["lines"]
    raise KeyError(prompt_id)


EVAL_OVERRIDES = DATA / "audit_g2p" / "review" / "overrides_eval_q9p.json"


def eval_set(tag_in=TAG_IN, tag=TAG, overrides_path=EVAL_OVERRIDES):
    from ganj_corpus.relabel import apply_overrides
    evals = [json.loads(l) for l in open(DATA / "eval" / f"set_q{tag_in}.jsonl", encoding="utf-8")]
    labels, applied, skipped = apply_overrides({e["key"]: e["phonemes"] for e in evals}, {e["key"]: e["lines"] for e in evals},
                                               json.load(open(overrides_path, encoding="utf-8")))
    if skipped:
        raise SystemExit(f"eval-line readings not applied: {skipped}")
    with open(DATA / "eval" / f"set_q{tag}.jsonl", "w", encoding="utf-8") as f:
        for e in evals:
            f.write(json.dumps({**e, "phonemes": labels[e["key"]]}, ensure_ascii=False) + "\n")
    print(f"wrote set_q{tag}.jsonl: {applied} eval-line readings applied")


def measured_label(lines, label: str, y, aligner):
    """The label with a pause token at each word boundary where the recording y (16 kHz) is silent, as the training
    labels were marked; None when the label's words do not line up with the text (plan_cut)."""
    from ganj_eval.align import boundary_pauses
    words, why = plan_cut(lines, label, aligner.chars)
    if why is not None:
        return None
    spans, _ = aligner.spans(y, words)
    return mark_pauses(label, [pause_token(p) for p in boundary_pauses(spans, y)])


def test_list(src, dst, prompt_wav, prompt_id, device: str = "cpu", pauses_from=None, tag=TAG):
    """Targets get the reading-time rule (rule_label), or, with pauses_from, the pauses of a reading of that line
    (<dir>/<key>.wav: a reference reader's recordings); the prompt its measured pauses."""
    import librosa
    from ganj_eval.align import SR, Aligner
    rows = {json.loads(l)["key"]: json.loads(l) for l in open(DATA / "eval" / f"set_q{tag}.jsonl", encoding="utf-8")}
    aligner, plines = Aligner(device), prompt_lines(prompt_id)
    y, _ = librosa.load(prompt_wav, sr=SR, mono=True)
    out, read = [], 0
    for l in open(src, encoding="utf-8"):
        if not l.strip():
            continue
        key, ptext, _, _ = l.rstrip("\n").split("\t")
        text, lines = rows[key]["phonemes"], rows[key]["lines"]
        pmarked = measured_label(plines, ptext, y, aligner) or rule_label(ptext, plines)
        reading = Path(pauses_from) / f"{key}.wav" if pauses_from else None
        target = None
        if reading and reading.exists():
            target = measured_label(lines, text, librosa.load(reading, sr=SR, mono=True)[0], aligner)
            read += target is not None
        out.append("\t".join([key, pmarked, str(prompt_wav), target or rule_label(text, lines)]) + "\n")
    Path(dst).write_text("".join(out), encoding="utf-8")
    print(f"wrote {dst}: {len(out)} lines, {read} with the pauses of a reading; prompt: "
          + pmarked.replace(PAUSE_SHORT, " ⟨S⟩").replace(PAUSE_LONG, "⟨L⟩"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-in", default=TAG_IN, help="tag of the files read: the mixes, the eval set")
    ap.add_argument("--tag", default=TAG, help="tag of the files written")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("label")
    a1.add_argument("--device", default="cuda")
    a1.add_argument("--limit", type=int, default=0)
    a1.add_argument("--out", default=None, help="labels file (default corpus_q/labels<tag>.jsonl)")
    a2 = sub.add_parser("cuts")
    a2.add_argument("--min-conf", type=float, default=0.0)
    a2.add_argument("--overrides", type=Path, default=OVERRIDES, help="every per-line reading the labels must carry")
    a2.add_argument("--keep", type=Path, default=None, help="readings whose cuts are never held back (default: --overrides)")
    a5 = sub.add_parser("evalset")
    a5.add_argument("--overrides", type=Path, default=EVAL_OVERRIDES)
    a4 = sub.add_parser("testlist")
    a4.add_argument("src")
    a4.add_argument("dst")
    a4.add_argument("--prompt-wav", required=True)
    a4.add_argument("--prompt-id", required=True)
    a4.add_argument("--pauses-from", default=None, help="dir of readings <key>.wav whose pauses the targets take")
    a3 = sub.add_parser("ckpt")
    a3.add_argument("src")
    a3.add_argument("dst")
    a = ap.parse_args(argv)
    if a.cmd == "label":
        label(a.device, a.limit, a.out, tag_in=a.tag_in, tag=a.tag)
    elif a.cmd == "evalset":
        eval_set(tag_in=a.tag_in, tag=a.tag, overrides_path=a.overrides)
    elif a.cmd == "testlist":
        test_list(a.src, a.dst, a.prompt_wav, a.prompt_id, pauses_from=a.pauses_from, tag=a.tag)
    elif a.cmd == "cuts":
        write_cuts(a.min_conf, tag_in=a.tag_in, tag=a.tag, overrides_path=a.overrides, keep_path=a.keep)
    else:
        extend_checkpoint(a.src, a.dst)


if __name__ == "__main__":
    main()

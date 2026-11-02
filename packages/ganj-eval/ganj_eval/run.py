# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Score one voice (sherpa model dir + speaker id, or a directory of pre-synthesised <key>.wav) on the fixed set."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from ganj_eval.identity import Ecapa, novelty, reciter_embeddings
from ganj_eval.listen import render_listen
from ganj_eval.metrics import Mos, PersianASR, f0_logstd, line_end_recall, tiny_word_recall, wer
from ganj_voice.synth import Synth


NON_PERSIAN = ("arabic", "turkish")


def persian_summary(rows) -> dict:
    """Headline metrics over the Persian rows only (22 % of set_q is Arabic or Azeri Turkish, read by Persian G2P)."""
    fa = [r for r in rows if r["cls"] not in NON_PERSIAN]
    out = {"n": len(fa)}
    for k in ("wer_gap", "mos", "tiny", "line_end"):
        v = [r[k] for r in fa if r.get(k) is not None]
        out[k] = float(np.mean(v)) if v else None
    return out


def is_prefiltered(out_dir: Path, wav_dir=None) -> bool:
    """postclean.py (or synth_voice.sh through it) already high-passed these clips: its postclean.json is in the run dir
    or in the dir the wavs are copied from (every clip is scored filtered exactly once)."""
    return any((Path(d) / "postclean.json").exists() for d in (out_dir, wav_dir) if d)


def seconds_ratio(ys: np.ndarray, sr_s: int, yh: np.ndarray, sr_h: int) -> float:
    """Synth length over human length in seconds (the outputs are 24 kHz, the human clips 22.05 kHz)."""
    return (len(ys) / sr_s) / max(1e-9, len(yh) / sr_h)


def run(data: Path, tag: str, model_dir=None, sid: int = 0, wav_dir=None, device: str = "cpu", skip_identity: bool = False,
        set_name: str = "set.jsonl") -> dict:
    items = [json.loads(l) for l in open(data / "eval" / set_name, encoding="utf-8")]
    out_dir = data / "eval" / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    synth = Synth(model_dir, sid=sid) if model_dir else None
    asr = PersianASR(device, cache_path=data / "eval" / "asr_cache.json")
    mos = Mos(device)
    pre = is_prefiltered(out_dir, wav_dir)
    per = []
    for it in items:
        wav = out_dir / f"{it['key']}.wav"
        if synth:
            sf.write(str(wav), synth.say(it["phonemes"]), 22050, subtype="PCM_16")
            asr.forget(wav)
        elif not wav.exists():
            src = Path(wav_dir) / f"{it['key']}.wav"
            if not src.exists():
                continue
            y, sr = sf.read(str(src), dtype="float32")
            sf.write(str(wav), y, sr, subtype="PCM_16")
        ref = " ".join(it["lines"])
        hyp_s, hyp_h = asr.transcribe(wav), asr.transcribe(data / it["human"])
        ys, sr_s = sf.read(str(wav), dtype="float32")
        yh, sr_h = sf.read(str(data / it["human"]), dtype="float32")
        row = {"key": it["key"], "cls": it["cls"], "wer_synth": wer(ref, hyp_s), "wer_human": wer(ref, hyp_h),
               "mos": mos.score(wav, prefiltered=pre), "tiny": tiny_word_recall(ref, hyp_s), "line_end": line_end_recall(it["lines"], hyp_s),
               "f0_ratio": (f0_logstd(ys, sr_s) + 1e-6) / (f0_logstd(yh, sr_h) + 1e-6), "hyp": hyp_s,
               "seconds_ratio": seconds_ratio(ys, sr_s, yh, sr_h)}
        row["wer_gap"] = row["wer_synth"] - row["wer_human"]
        per.append(row)
    summary = {"tag": tag, "n": len(per), **summarise(per)}
    if not skip_identity:
        ecapa = Ecapa(device)
        clips = [json.loads(l) for l in open(data / "clips.jsonl", encoding="utf-8")]
        reciters = reciter_embeddings(clips, data, ecapa)
        embs = [ecapa.embed(*sf.read(str(out_dir / f"{r['key']}.wav"), dtype="float32")) for r in per[:20]]
        top, who, table = novelty(embs, reciters)
        summary["identity_max_cos"], summary["identity_closest"] = top, who
        summary["identity_top5"] = table[:5]
    lines = write_results(out_dir, summary, per)
    render_listen(data)
    print("\n".join(lines))
    return summary


def summarise(per) -> dict:
    """Run-level means of the per-clip rows (overall, Persian only, per text class)."""
    by_cls = defaultdict(list)
    for r in per:
        by_cls[r["cls"]].append(r)
    out = {k: float(np.mean([r[k] for r in per])) if per else None
           for k in ("wer_gap", "wer_synth", "wer_human", "mos", "line_end", "f0_ratio", "seconds_ratio")}
    tiny = [r["tiny"] for r in per if r["tiny"] is not None]
    out["tiny"] = float(np.mean(tiny)) if tiny else None
    out["persian"] = persian_summary(per)
    out["by_class"] = {c: {k: float(np.mean([r[k] for r in rows])) for k in ("wer_gap", "mos", "line_end")} for c, rows in by_cls.items()}
    return out


def write_results(out_dir: Path, summary: dict, per: list) -> list:
    """results.json and report.md of one run; returns the report lines."""
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "rows": per}, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = [f"# {summary['tag']}", "", f"n={summary['n']}  WER gap={summary['wer_gap']:.3f} (synth {summary['wer_synth']:.3f} / human {summary['wer_human']:.3f})",
             f"MOS={summary['mos']:.2f}  tiny-word recall={summary['tiny']}  line-end recall={summary['line_end']:.2f}  f0 ratio={summary['f0_ratio']:.2f}  length ratio={summary['seconds_ratio']:.2f}"]
    fa = summary["persian"]
    if fa["n"] and None not in fa.values():
        lines.append(f"Persian only (n={fa['n']}): WER gap={fa['wer_gap']:.3f}  MOS={fa['mos']:.2f}  tiny-word recall={fa['tiny']:.2f}"
                     f"  line-end recall={fa['line_end']:.2f}")
    if "identity_max_cos" in summary:
        note = " (2026-10-29 rate fix; see results.json before_rate_fix)" if "rate_fix_note" in summary else ""
        lines.append(f"identity: max cos {summary['identity_max_cos']:.3f} to {summary['identity_closest']}{note}")
    if "mos_raw" in summary:
        lines.append(f"MOS re-scored as shipped, high-passed at 70 Hz once; first MOS was {summary['mos_raw']:.2f}")
    lines += ["", "| class | WER gap | MOS | line-end |", "|---|---|---|---|"]
    lines += [f"| {c} | {v['wer_gap']:.3f} | {v['mos']:.2f} | {v['line_end']:.2f} |" for c, v in summary["by_class"].items()]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def rescore_mos(data: Path, tag: str, mos) -> dict:
    """Re-score only MOS of an existing run with the current Mos, keeping the first unfiltered value as mos_raw."""
    out_dir = Path(data) / "eval" / "runs" / tag
    r = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    summary, per = r["summary"], r["rows"]
    summary.setdefault("mos_raw", summary["mos"])
    pre = is_prefiltered(out_dir)
    for row in per:
        row.setdefault("mos_raw", row["mos"])
        row["mos"] = mos.score(out_dir / f"{row['key']}.wav", prefiltered=pre)
    fresh = summarise(per)            # only the MOS fields are taken: some summaries carry fixes their rows lack (rate fix)
    summary["mos"], summary["persian"]["mos"] = fresh["mos"], fresh["persian"]["mos"]
    for c, v in summary["by_class"].items():
        v["mos"] = fresh["by_class"][c]["mos"]
    write_results(out_dir, summary, per)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--sid", type=int, default=0)
    ap.add_argument("--wav-dir")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-identity", action="store_true")
    ap.add_argument("--set", default="set.jsonl", help="eval set file under eval/ (set_q.jsonl: the current labels)")
    ap.add_argument("--mos-only", action="store_true", help="re-score MOS of an existing run; nothing else")
    a = ap.parse_args()
    if a.mos_only:
        s = rescore_mos(Path(a.data), a.tag, Mos(a.device))
        print(f"{a.tag} MOS {s['mos_raw']:.2f} -> {s['mos']:.2f}  Persian {s['persian']['mos']:.2f}")
        return
    run(Path(a.data), a.tag, a.model_dir, a.sid, a.wav_dir, a.device, a.skip_identity, a.set)


if __name__ == "__main__":
    main()

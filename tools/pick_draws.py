# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Pick, for each line, the noise draw that says its label (a dropped word or ezafe belongs to one draw):
every word (CTC, ganj_eval.labelcheck) and every ezafe (its classifier, trained here on a native reader's recordings and
the corpus recordings of the eval set: the reader says each ezafe of the corrected labels).

  python tools/pick_draws.py fit MODEL.joblib                       train the ezafe classifier; prints its accuracy on
                                                                     held-out lines of the reader's recordings
  python tools/pick_draws.py pick MODEL.joblib DRAWS_DIR OUT_DIR    DRAWS_DIR/KEY_d<k>.wav (recipes/zipvoice/draws.sh)
                                                                     -> OUT_DIR/KEY.wav and OUT_DIR/picks.json
  python tools/pick_draws.py check MODEL.joblib WAV_DIR             per file: missing words, ezafes unheard / unwanted
  python tools/pick_draws.py fit-vowels VOWELS.json DIR...          a voice's short-vowel centres from its own renders
                                                                     (KEY.wav or KEY_d<k>.wav of lines in --set)

--vowels VOWELS.json (pick, check): a short vowel of the label the voice says wrong counts like a wrong ezafe
(2026-11-02, «آبش» said abesh; ganj_eval.vowelcheck).
--set is the eval set whose text and labels the lines are checked against (default: the q9p labels, which carry
the per-line readings)."""
import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ganj_eval.labelcheck import LineChecker, ezafe_errors, pick_draw, vowel_errors  # noqa: E402
from ganj_eval.vowelcheck import VowelModel  # noqa: E402

from tools._env import env_path  # noqa: E402
DATA = env_path("GANJ_DATA")
C, FOLDS = 0.01, 5
DRAW = re.compile(r"^(.+)_d(\d+)\.wav$")


def rows_of(path: Path) -> dict:
    return {r["key"]: r for r in map(json.loads, open(path, encoding="utf-8"))}


def classifier():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


def fit(checker: LineChecker, rows: dict, out: Path) -> None:
    import joblib
    import numpy as np
    from sklearn.model_selection import GroupKFold
    sets = {"reader": [], "human": []}
    for p in sorted(glob.glob(str(DATA / "readings/wav24/*.wav"))):
        key = Path(p).stem
        if key in rows:
            sets["reader"].append((key, p))
            sets["human"].append((key, str(DATA / rows[key]["human"])))
    data = {}
    for name, pairs in sets.items():
        X, y, g = [], [], []
        for key, p in pairs:
            r = checker.analyse(p, rows[key]["lines"], rows[key]["phonemes"])
            X += r["features"]; y += [s for _, s in r["targets"]]; g += [key] * len(r["targets"])
        data[name] = (np.array(X), np.array(y), np.array(g))
    Xa, ya, ga = data["reader"]
    p = np.zeros(len(ya))
    for tr, te in GroupKFold(FOLDS).split(Xa, ya, ga):
        p[te] = classifier().fit(Xa[tr], ya[tr]).predict_proba(Xa[te])[:, 1]
    acc = {"accuracy": float(((p > 0.5) == ya).mean()), "ezafe_heard": float((p[ya] > 0.5).mean()),
           "none_heard_none": float((p[~ya] < 0.5).mean()), "boundaries": int(len(ya)), "ezafes": int(ya.sum())}
    X = np.concatenate([d[0] for d in data.values()])
    y = np.concatenate([d[1] for d in data.values()])
    joblib.dump({"model": classifier().fit(X, y), "layers": LineChecker.LAYERS, "C": C, "held_out": acc}, out)
    print(f"held-out lines of the reader's recordings: {json.dumps(acc)}; saved {out}")


def score(checker: LineChecker, model, path: str, row: dict, vowels: VowelModel = None) -> dict:
    r = checker.analyse(path, row["lines"], row["phonemes"])
    wrong = [{"word": r["words"][m["word"]], "vowel": m["vowel"], "f1": m["f1"], "f2": m["f2"]}
             for m in vowel_errors(vowels, r.get("vowels", []))]
    probs = model.predict_proba(r["features"])[:, 1].tolist() if r["features"] else []
    unheard, unwanted = ezafe_errors(r["targets"], probs)
    words = [w for w, ok in zip(r["words"], r["present"]) if not ok]
    ez = [{"word": r["words"][i] if i < len(r["words"]) else str(i), "said": s, "p": round(q, 3)}
          for (i, s), q in zip(r["targets"], probs)]
    return {"missing_words": len(words), "missing": words, "wrong_ezafe": unheard + unwanted, "ezafe_unheard": unheard,
            "ezafe_unwanted": unwanted, "fit": r["fit"], "ezafe": ez, "wrong_vowel": len(wrong), "vowels_wrong": wrong}


def key_of(path: str) -> str:
    m = DRAW.match(os.path.basename(path))
    return m.group(1) if m else Path(path).stem


def fit_vowels(checker: LineChecker, rows: dict, dirs, out: Path) -> dict:
    """Fit and save a voice's VowelModel on every measured short vowel of its renders in `dirs`; counts per vowel."""
    import numpy as np
    points = []
    for d in dirs:
        for p in sorted(glob.glob(str(Path(d) / "*.wav"))):
            row = rows.get(key_of(p))
            if row:
                points += [(m["vowel"], m["f1"], m["f2"]) for m in checker.analyse(p, row["lines"], row["phonemes"])["vowels"]]
    model = VowelModel.fit(points)
    model.save(out)
    counts = {v: sum(1 for q in points if q[0] == v) for v in sorted({q[0] for q in points})}
    hz = {v: [round(float(x)) for x in np.exp(c)] for v, c in model.centres.items()}
    print(f"vowels measured {counts}; centres F1/F2 Hz {hz}; saved {out}")
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fit", "pick", "check", "fit-vowels"])
    ap.add_argument("model", type=Path)
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--set", type=Path, default=DATA / "eval/set_q_q9p.jsonl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vowels", type=Path, default=None, help="the voice's VowelModel (fit-vowels); off when not given")
    a = ap.parse_args(argv)
    rows, checker = rows_of(a.set), LineChecker(a.device)
    if a.cmd == "fit":
        return fit(checker, rows, a.model)
    if a.cmd == "fit-vowels":
        return fit_vowels(checker, rows, a.dirs, a.model)
    vowels = VowelModel.load(a.vowels) if a.vowels else None
    import joblib
    model = joblib.load(a.model)["model"]
    if a.cmd == "check":
        for p in sorted(glob.glob(str(a.dirs[0] / "*.wav"))):
            key = DRAW.match(os.path.basename(p)).group(1) if DRAW.match(os.path.basename(p)) else Path(p).stem
            if key in rows:
                s = score(checker, model, p, rows[key], vowels)
                bad = ", ".join(f"{w['word']} {w['vowel']} ({w['f1']:.0f}/{w['f2']:.0f})" for w in s["vowels_wrong"])
                print(f"{os.path.basename(p)}: missing {s['missing'] or '-'}; ezafe unheard {s['ezafe_unheard']}, "
                      f"unwanted {s['ezafe_unwanted']}; vowels wrong {bad or '-'}")
        return
    src, out = a.dirs
    out.mkdir(parents=True, exist_ok=True)
    groups, unchecked = {}, {}
    for p in sorted(glob.glob(str(src / "*.wav"))):
        m = DRAW.match(os.path.basename(p))
        if m:
            (groups if m.group(1) in rows else unchecked).setdefault(m.group(1), {})[f"d{m.group(2)}"] = p
    for key, draws in unchecked.items():                 # no text to check against: keep the first draw
        first = min(draws, key=lambda d: int(d[1:]))
        shutil.copy(draws[first], out / f"{key}.wav")
        print(f"{key}: not in {a.set.name}, kept {first} unchecked", flush=True)
    report = {}
    for key, draws in groups.items():
        scores = {d: score(checker, model, p, rows[key], vowels) for d, p in draws.items()}
        best = pick_draw(scores)
        shutil.copy(draws[best], out / f"{key}.wav")
        report[key] = {"picked": best, "draws": scores}
        print(f"{key}: {best} (missing words {scores[best]['missing_words']}, ezafe wrong {scores[best]['wrong_ezafe']}, "
              f"vowels wrong {scores[best]['wrong_vowel']}; first draw: {scores['d1']['missing_words']}, "
              f"{scores['d1']['wrong_ezafe']}, {scores['d1']['wrong_vowel']})", flush=True)
    json.dump(report, open(out / "picks.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

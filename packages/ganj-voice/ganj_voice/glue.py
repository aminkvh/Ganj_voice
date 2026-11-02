# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""What the model reads. Only the target the model reads changes; labels, pause rules and the draw checks keep one label
word per text word.

- A standalone «و» (label o) after a consonant-final word is written onto that word (bord/nd o s/@y → bord/ndo s/@y).
  Written apart, the voice drops it in some contexts in nearly every draw (after «بردند», «آموخت», «اوست»: 6–8 of 8),
  with the same model, guidance, speed or a v/ label; glued, rarely.
- A voice trained with the pause tokens gets a long pause before the line: straight after the prompt, the line's first
  word is often lost («گفت صاحب…» in 8 of 8 draws; with the pause, 1 of 8).

  python -m ganj_voice.glue LIST.tsv [--tokens VOICE/tokens.txt]   (rewrites the targets of a reading list in place)
"""
import argparse
from pathlib import Path

from ganj_reading import tokens as T

CONSONANTS = set("bcdfghjklmnpqrstvwxyzš@")
LONG = ""   # the long pause token (ganj_corpus.pauses.PAUSE_LONG)
PREFIX = set(T.STYLE + T.METRE_TOKENS)


def glue_vav(label: str) -> str:
    out = []
    for tok in label.split(" "):
        if tok == "o" and out and out[-1] and out[-1][-1] in CONSONANTS:
            out[-1] += "o"
        else:
            out.append(tok)
    return " ".join(out)


def lead_pause(label: str) -> str:
    """The long pause token before the first word, after the style and metre marks."""
    k = len(label) - len(label.lstrip("".join(PREFIX)))
    return label if label[k:].startswith(LONG) else label[:k] + LONG + " " + label[k:]


def glue_list_line(line: str, lead: bool = False) -> str:
    """A reading-list line (key, prompt phonemes, prompt wav, target) with its target as the model reads it."""
    cols = line.rstrip("\n").split("\t")
    cols[3] = glue_vav(cols[3])
    if lead:
        cols[3] = lead_pause(cols[3])
    return "\t".join(cols) + "\n"


def has_pause_tokens(tokens_txt: str) -> bool:
    return LONG in {l.split("\t")[0] for l in tokens_txt.splitlines() if l}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("list")
    ap.add_argument("--tokens", default=None, help="the voice's tokens.txt: a long pause before each line if it has one")
    a = ap.parse_args(argv)
    lead = bool(a.tokens) and has_pause_tokens(Path(a.tokens).read_text(encoding="utf-8"))
    path = Path(a.list)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    path.write_text("".join(glue_list_line(l, lead) for l in lines if l.strip()), encoding="utf-8")


if __name__ == "__main__":
    main()

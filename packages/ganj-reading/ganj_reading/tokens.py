# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""The token contract: one character per token, fixed for the life of the voice.

Ids 0-3 follow Piper's convention so the gyro warm start keeps the meaning of pad/bos/eos/space.
GE2PE writes š as '$', which collides with Piper's EOS, so GE2PE output is remapped before encoding.
Style tokens are private-use codepoints so sherpa-onnx's character frontend sees one token each.
"""
import json
from pathlib import Path

PAD, BOS, EOS, SPACE = "_", "^", "$", " "
VERSE, PROSE, HEMI, BEYT = "", "", "", ""
STYLE = [VERSE, PROSE, HEMI, BEYT]
SH = "š"
MAX_IDS = 720        # ~350 symbols after pad interleaving; longer clips are split or dropped at cut time
MAX_SECONDS = 12.0

_ASCII = [chr(c) for c in range(0x21, 0x7F) if chr(c) not in (PAD, BOS, EOS)]
# One token per metre, private-use codepoints from U+E010, appended after every older symbol so
# no existing id moves. The metre list is fixed before training and only ever appended to.
METRE_BASE = 0xE010
METRES = json.loads((Path(__file__).with_name("metres.json")).read_text(encoding="utf-8"))
METRE_TOKENS = [chr(METRE_BASE + i) for i in range(len(METRES))]
SYMBOLS = [PAD, BOS, EOS, SPACE] + _ASCII + [SH] + STYLE + METRE_TOKENS
_METRE_INDEX = {m: i for i, m in enumerate(METRES)}


def metre_token(metre: str) -> str:
    """The token for a poem's metre (the bracketed family name is ignored); "" for a metre not in the list."""
    i = _METRE_INDEX.get((metre or "").split("(")[0].strip())
    return METRE_TOKENS[i] if i is not None else ""
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for s, i in SYMBOL_TO_ID.items()}


def remap_ge2pe(s: str) -> str:
    return s.replace("$", SH)


def encode(phonemes: str, add_specials: bool = True) -> list:
    if not phonemes.strip():
        raise ValueError("empty phoneme string")
    ids = []
    for ch in phonemes:
        if ch not in SYMBOL_TO_ID:
            raise ValueError(f"unknown symbol {ch!r} (U+{ord(ch):04X}); remap GE2PE output first")
        if ch in (PAD, BOS, EOS):
            raise ValueError(f"special symbol {ch!r} inside phoneme string")
        ids.append(SYMBOL_TO_ID[ch])
    if not add_specials:
        return ids
    out = [SYMBOL_TO_ID[BOS], SYMBOL_TO_ID[PAD]]
    for i in ids:
        out.extend((i, SYMBOL_TO_ID[PAD]))
    out.append(SYMBOL_TO_ID[EOS])
    return out


def decode(ids) -> str:
    return "".join(ID_TO_SYMBOL[i] for i in ids if i not in (0, 1, 2))


def write_tokens_txt(path) -> None:
    Path(path).write_text("".join(f"{s} {i}\n" for s, i in SYMBOL_TO_ID.items()), encoding="utf-8")

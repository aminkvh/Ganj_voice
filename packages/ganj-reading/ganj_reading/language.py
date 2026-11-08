# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Which hemistichs the voice skips: the non-Persian ones (the voice reads Arabic badly; the app shows
a skipped hemistich and pauses instead). Ganjoor's own tag decides when a hemistich has one (verse languageId: 1 Persian
is read; 2 Arabic, 3 Azeri Turkish or any other language is skipped). Untagged hemistichs (Ganjoor leaves many Arabic ones untagged: Saadi's molamma',
Hafez 302 / 307, Mowlana) go through looks_arabic, built to read a Persian hemistich rather than skip it:
  1. normalise: harakat and tatweel off, ي -> ی, ك -> ک; words = runs of Arabic-script letters and ZWNJ;
  2. read as Persian when there are fewer than 2 words, a letter پ چ ژ گ, a Persian word (PERSIAN_WORDS), a ZWNJ inside a
     word, or no sign of Arabic grammar (an ال word, an Arabic function word, tanwin, a final ة or ى);
  3. letter n-grams (1-4 within <word>, ZWNJ as _): mean weight (arabic_ngrams.json, naive Bayes log ratios trained by
     tools/train_arabic_detector.py on Ganjoor's tags) of every word >= -WORD_FLOOR (no word that is clearly Persian),
     and the mean over the whole hemistich > LINE_MIN.
On poems held out of training it finds about half of the Arabic hemistichs and about 19 in 20 of those it flags are
Arabic (2026-11-08)."""
import json
import re
from functools import lru_cache
from pathlib import Path

PERSIAN_ID = 1   # Ganjoor's languageId; 2 Arabic, 3 Azeri Turkish
LINE_MIN, WORD_FLOOR = 0.05, 0.7
HARAKAT = re.compile("[ً-ٰٟـ]")
WORD = re.compile("[؀-ۿ‌]+")
PERSIAN_LETTERS = re.compile("[پچژگ]")
PERSIAN_WORDS = frozenset(
    "را که از در است نیست هست چو چون چه با هر تو اگر گر همه بود کرد دل جان تا ز نه کز وز خود باشد شود کند گفت مرا ترا "
    "اینک آنک کاین زان زین بدان بدین وین وان آنکه اینکه کو چند هیچ دگر دیگر نیز همی یک هزار آن ولیکن".split())
ARABIC_FUNCTION_WORDS = frozenset(
    "فی علی الی إلی عن لا لم لن قد إن ان أن إذا اذا إذ اذ هذا هذه ذلک ذاک الذی التی کل ثم حتی یا ایها أیها انت أنت انا "
    "أنا هو هی نحن ما من لی له لها لهم لنا بها بی بک فیه فیها منه منها علیه علیها الیه الیک علیک منک فیک عنه انی إنی "
    "لعل لیت کان کانت لیس متی ذا".split())
AL = ("ال", "وال", "بال", "فال", "کال", "لل")


def normalise(text: str) -> str:
    return HARAKAT.sub("", text).replace("ي", "ی").replace("ك", "ک")


def word_grams(word: str) -> list:
    w = "<" + word.replace("‌", "_") + ">"
    return [w[i:i + n] for n in (1, 2, 3, 4) for i in range(len(w) - n + 1)]


@lru_cache(maxsize=1)
def weights() -> dict:
    return json.loads((Path(__file__).parent / "arabic_ngrams.json").read_text(encoding="utf-8"))


def _mean(grams: list, w: dict) -> float:
    return sum(w.get(g, 0.0) for g in grams) / max(len(grams), 1)


def arabic_grammar(text: str, words: list) -> int:
    return (sum(x.startswith(AL) and len(x) > 3 for x in words) + sum(x in ARABIC_FUNCTION_WORDS for x in words)
            + len(re.findall("[ً-ٍ]", text)) + sum(x.endswith(("ة", "ى")) for x in words))


def looks_arabic(text: str, w: dict = None) -> bool:
    """w: n-gram weights (default the shipped arabic_ngrams.json)."""
    w = weights() if w is None else w
    words = WORD.findall(normalise(text))
    if len(words) < 2 or PERSIAN_LETTERS.search("".join(words)):
        return False
    if any(x in PERSIAN_WORDS or "‌" in x for x in words) or arabic_grammar(text, words) < 1:
        return False
    if min(_mean(word_grams(x), w) for x in words) < -WORD_FLOOR:
        return False
    return _mean([g for x in words for g in word_grams(x)], w) > LINE_MIN


def skip_hemistich(text: str, language_id=None) -> bool:
    """True for a hemistich the voice does not read: Ganjoor's tag when it has one (1 Persian is read, any other
    language is skipped), else looks_arabic."""
    if language_id is not None:
        return language_id != PERSIAN_ID
    return looks_arabic(text)

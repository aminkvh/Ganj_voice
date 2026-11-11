# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Homo-GE2PE front-end (MIT) with a word lexicon applied first and a log of diacritised words.

Ganjoor's partial vowel marks are honoured as constraints: among the model's n best readings the one that agrees
with the written marks wins (harakat_agrees), and every diacritised word can be logged with the chosen reading so
wrong ones get a lexicon entry.
"""
import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

from ganj_reading import tokens as T
from ganj_reading.scansion import metre_patterns, scans

HARAKAT = re.compile("[ً-ٰٟ]")
TATWEEL = "ـ"
PUNCT = re.compile("[،؛؟.,;:!?()«»\"'‐-―\\-]")
# No-break, thin and other typographic spaces are word breaks (1,222 U+00A0 in the corpus text glued words together).
# ZWNJ (U+200C, the Persian half-space) is NOT one of them: it joins the parts of one word (می‌روم).
SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")


class PhonemizeError(ValueError):
    pass


def normalize_fa(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = t.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک").replace("ۀ", "ه").replace(TATWEEL, "")
    t = PUNCT.sub(" ", SPACES.sub(" ", t))
    return re.sub(r"[ \t]+", " ", t).strip()


def strip_harakat(text: str) -> str:
    return HARAKAT.sub("", text)


NBEST = 8  # candidates tried against the metre and the written vowel marks
# From a 140-line hand audit: words that never carry ezafe — the model put one on ز in 11 of 15 cases.
NO_EZAFE = frozenset({"ز", "زان", "زانک", "زانکه", "کاین", "وین", "ور", "کز", "وز", "کو", "کآن", "گرت", "کای", "زین", "وان", "درین"})
# A finite verb cannot be an ezafe dependent, so the word before one takes no ezafe (eval line 5112-0 "خدمت سلطان کردی").
# Closed list of light/copular verbs; stems that are also common nouns (داد، دید، ساخت، گفت) are left out.
FINITE_VERB = re.compile("^(ن|نمی‌?|می‌?|ب)?(کرد|شد|بود|گشت|نمود|داشت|گردید|آمد|رفت|افتاد)(م|ی|یم|ید|ند)?$"
                         "|^(ن|نمی‌?|می‌?|ب)?(باش|شو|کن|گرد)(م|ی|د|یم|ید|ند)$|^(است|نیست|هست)$")
# Written vowel marks: fatha -> /, kasra -> e, damma (zamma) -> o. Sukun (U+0652) is NOT a vowel: it marks a letter
# with no vowel after it, checked by harakat_agrees; a final sukun means no ezafe.
MARK_VOWEL = {"\u064e": "/", "\u0650": "e", "\u064f": "o"}
SUKUN = "\u0652"
SHORT_VOWELS = "/eo"
HEH_HAMZA, HAMZA_ABOVE, KASRA, EZAFE = "ۀ", "\u0654", "\u0650", "1"
VOWEL_LETTERS = tuple("هاویء") + (HEH_HAMZA, HAMZA_ABOVE, KASRA)
VOWEL_END = tuple("aeiou/")


def _with_ezafe(ph: str, ezafe: bool, plain: str = "") -> str:
    """Add the ezafe (e after a consonant, ye after a vowel, then GE2PE's marker 1) unless it is already there.

    Keeps the model's ezafe when a lexicon entry replaces a word, and pronounces a written ezafe (ۀ, هٔ, or a final kasra).
    A final e on a word not written with a final ه is the model's bare ezafe, so it only gets its marker (روزِ: ruze ->
    ruze1, not ruzeye1; about 1,000 such labels). On a word with a final ه the model's bare ezafe is ye:
    when the reading already ends in e + ye, and that ye cannot be the word's own (no ی before the ه, or a second ye
    after it), it only gets its marker (گوشهٔ: gušeye -> gušeye1, not gušeyeye1; 199 q9e labels)."""
    if not ezafe or ph.endswith(EZAFE):
        return ph
    if ph.endswith("e") and plain and not plain.endswith("ه"):
        return ph + EZAFE
    if plain.endswith("ه") and ph.endswith("ye"):
        stem = ph[:-2]
        if stem.endswith("e") and (plain[-2:-1] != "ی" or stem.endswith("ye")):
            return ph + EZAFE
    return ph + ("ye1" if ph.endswith(VOWEL_END) else "e1")


def _strip_ezafe(ph: str, word: str = "") -> str:
    if ph.endswith("ye1"):
        # روی: ruye1 -> ruy (consonant y); ماهی: mahiye1 -> mahi; روزیِّ: ruziyye1 -> ruziyy (a doubled y takes e1)
        return ph[:-2] if word.endswith(("ای", "وی")) or ph.endswith("yye1") else ph[:-3]
    return ph[:-2] if ph.endswith("e1") else ph


# Tashdid / shadda (U+0651): the letter before it is doubled (gemination), e.g. shakkar -> š/kk/r; see geminate().
SHADDA = "\u0651"
LETTER_PHONE = {"ب": "b", "پ": "p", "ت": "t", "ط": "t", "ث": "s", "س": "s", "ص": "s", "ج": "j", "چ": "c", "ح": "h", "ه": "h",
                "خ": "x", "د": "d", "ذ": "z", "ز": "z", "ض": "z", "ظ": "z", "ر": "r", "ژ": ";", "ش": "š", "غ": "q", "ق": "q",
                "ف": "f", "ک": "k", "گ": "g", "ل": "l", "م": "m", "ن": "n", "و": "v", "ی": "y", "ع": "@", "ء": "@"}


SMALL_ALEF = "ٰ"
_NOT_SMALL_ALEF = re.compile("[ً-ٟ]")  # the harakat except the small alef


def _final_small_alef(word: str) -> bool:
    return _NOT_SMALL_ALEF.sub("", word).replace("‌", "").endswith(("ی" + SMALL_ALEF, "ى" + SMALL_ALEF))


def small_alef(word: str, ph: str) -> str:
    """A final ی carrying the small alef (یٰ: کسریٰ, عُقبیٰ, مصطفیٰ) is read a."""
    if not _final_small_alef(word):
        return ph
    if ph.endswith("iye" + EZAFE):
        return ph[:-len("iye" + EZAFE)] + "aye" + EZAFE
    return ph[:-1] + "a" if ph.endswith("i") else ph


TANWIN = {"ً": "/", "ٌ": "o", "ٍ": "e"}  # ً -an, ٌ -on, ٍ -en
TA_MARBUTA = "ة"


def _tanwin_vowel(word: str):
    """The vowel of a tanwin on the word's last letter (ً written before a final alef counts: حتمًا), else None."""
    w = word.replace("‌", "")
    i = len(w)
    while i and HARAKAT.search(w[i - 1]):
        i -= 1
    marks = w[i:]
    if not marks and w.endswith("ا"):
        k = len(w) - 1
        while k and HARAKAT.search(w[k - 1]):
            k -= 1
        marks = w[k:-1]
    return next((TANWIN[c] for c in marks if c in TANWIN), None)


def tanwin(word: str, ph: str) -> str:
    """A written tanwin is read with its n and no ezafe (فرضاً f/rz/n, قَلیلٌ q/lilon, خیرٍ xeyren); the alef that
    carries a tanwin fath is not a long a."""
    v = _tanwin_vowel(word)
    if not v:
        return ph
    body = _strip_ezafe(ph, strip_harakat(word))
    if v == "/" and body.endswith("a"):
        body = body[:-1]
    if body.endswith(v + "n"):
        return body
    return (body if body.endswith(v) else body + v) + "n"


def _read_ta_marbuta(word: str) -> str:
    """ة as the model can read it: ت when a mark follows (رحمةٌ: r/hm/ton), else a final ه (لیلة: leyle). Passed as
    is, the model read it k (88 of 95 Persian-line training labels)."""
    return re.sub(TA_MARBUTA + "(?=[ً-ْ])", "ت", word).replace(TA_MARBUTA, "ه")


def long_a_written(word: str, ph: str) -> bool:
    """No more long a in the reading than letters that can write one (ا آ أ إ, the small alef, a final ی as in موسی,
    the لله of الله): Persian writes a long a. «مهتر» read mehtar in all 27 training labels (meht/r or m/ht/r)."""
    base = _NOT_SMALL_ALEF.sub("", word).replace("‌", "").rstrip(SMALL_ALEF)
    n = sum(word.count(c) for c in "اآأإ" + SMALL_ALEF) + base.endswith(("ی", "ى")) + ("لله" in word)
    return _strip_ezafe(ph, base).count("a") <= n


YE_AST = "ی‌است"


def contract_ast(words: list, pw: list, metre: Optional[str] = None) -> list:
    """Ruled 2026-11-09: a word written with ی‌است is read -ist (زبانی‌است z/banist, not z/bani/st),
    unless the line then fails a metre it scans in. A one-letter stem keeps its ast (نی‌است, کی‌است)."""
    def one(word, ph):
        plain = strip_harakat(word)
        if plain.endswith(YE_AST) and len(plain) - len(YE_AST) >= 2:
            for tail in ("i/st", "i@/st"):
                if ph.endswith(tail):
                    return ph[: -len(tail)] + "ist"
        return ph
    out = [one(w, r) for w, r in zip(words, pw)]
    if out != pw and metre and scans(" ".join(pw), metre) and not scans(" ".join(out), metre):
        return pw
    return out


def written_ezafe(word: str) -> bool:
    """A final kasra or هٔ; Unicode writes a kasra before a shadda on the same letter (حقِّ = ح ق ِ ّ)."""
    return word.endswith(("ه" + HAMZA_ABOVE, KASRA, KASRA + SHADDA))


def harakat_agrees(word: str, ph: str) -> bool:
    """The written short vowels (a final kasra is the ezafe) occur in order among the phonemes' vowels, each as a vowel
    that reads it (mark_options: مَی mey, کَرِیم k/rim), and every letter carrying a shadda appears doubled (شکّر →
    š/kk/r)."""
    body = _strip_ezafe(ph)
    word = word[:-2] + SHADDA if word.endswith(KASRA + SHADDA) else word   # حقِّ: the kasra is the ezafe
    lm = _letters(word)
    it = iter(c for c in body if c in VOWELS)
    if not all(any(c in mark_options(lm, i, v) for c in it) for i, v in _marked(lm)):
        return False
    letters = [c for c in word if not HARAKAT.search(c) or c == SHADDA]
    doubled = [LETTER_PHONE.get(letters[i - 1]) for i, c in enumerate(letters) if c == SHADDA and i > 0]
    v = _tanwin_vowel(word)
    return (all(p is None or p * 2 in body for p in doubled) and _sukun_agrees(word, ph)
            and (not _final_small_alef(word) or body.endswith("a")) and (not v or body.endswith(v + "n")))


def _sukun_agrees(word: str, ph: str) -> bool:
    """No short vowel right after the sound of a letter written with a sukun (مَرْدُم: m/rdom, not m/r/dom). The letter is
    found at its occurrence counted among same-sounding letters, as in geminate; و ی ه ع are too ambiguous to place."""
    base, quiet = [], []
    for c in word:
        if c == SUKUN:
            if base:
                quiet.append(len(base) - 1)  # a sukun typed after a ZWNJ still belongs to the letter before it
        elif not HARAKAT.search(c) and c not in "‌‍":
            base.append(c)
    for i in quiet:
        p = LETTER_PHONE.get(base[i])
        if not p or p in NO_GEMINATE:
            continue
        k = sum(1 for x in base[:i] if LETTER_PHONE.get(x) == p)
        pos = [j for j, x in enumerate(ph) if x == p]
        if k >= len(pos):
            continue
        j = pos[k] + (1 if ph[pos[k] + 1:pos[k] + 2] == p else 0)  # a doubled consonant is one letter
        if ph[j + 1:j + 2] in tuple(SHORT_VOWELS):
            return False
    return True


NO_GEMINATE = {"v", "y", "h", "@"}  # و ی ه ع/ء: often vowels or silent in writing, too ambiguous to place

# Reading the written marks (ruled 2026-11-11): each letter with its marks, the vowels that read a mark, and the repair
ZWNJ, ZWJ = "\u200c", "\u200d"
ALEFS, YE, HE = "اأإآ", "ی", "ه"
VOWELS = SHORT_VOWELS + "aiu"
MEY = "م"  # مِی is mey (the kasra is written: never mi); کِی دِی نِی are read either way, فِی لِی هِی are Arabic i


def _letters(word: str) -> list:
    """[letter, its marks, starts a morpheme, ends a morpheme] for each letter of the word; a ZWNJ ends a morpheme."""
    out, start = [], True
    for c in word:
        if c == ZWNJ:
            if out:
                out[-1][3] = True
            start = True
        elif c == ZWJ:
            continue
        elif unicodedata.category(c) == "Mn":
            if out:
                out[-1][1] += c
        else:
            out.append([c, "", start, False])
            start = False
    if out:
        out[-1][3] = True
    return out


def _mark_vowel(marks: str):
    return next((MARK_VOWEL[c] for c in marks if c in MARK_VOWEL), None)


def _marked(lm: list) -> list:
    """(letter index, vowel) for each letter written with a short vowel; a kasra on the last letter is the ezafe."""
    out = [(i, _mark_vowel(m)) for i, (_, m, _, _) in enumerate(lm) if _mark_vowel(m)]
    return out[:-1] if out and out[-1] == (len(lm) - 1, "e") else out


def _bare(lm: list, i: int) -> bool:
    """Letter i has no vowel mark, or only the ezafe's kasra on the last letter (مَیِ)."""
    v = _mark_vowel(lm[i][1])
    return v is None or (v == "e" and i == len(lm) - 1)


def _semivowel(lm: list, i: int, letter: str) -> bool:
    """Letter i + 1 is a bare ی/و (letter) followed by no ا and no final ی/و: the ey of هَیجا and حَیوان, the ow of دَولت
    (not the consonant of بَیان، بِشَوی)."""
    if i + 1 >= len(lm) or lm[i + 1][0] != letter or not _bare(lm, i + 1):
        return False
    if i + 2 >= len(lm) or lm[i + 1][3]:
        return True
    n = lm[i + 2]
    return n[0] not in "اآ" and not (n[0] in YE + VAV and n[3])


def _mey(lm: list, i: int) -> bool:
    """مِی as a whole morpheme (مِی، مِی‌پرستی)."""
    return lm[i][2] and lm[i][0] == MEY and _semivowel(lm, i, YE) and lm[i + 1][3]


def mark_options(lm: list, i: int, v: str) -> set:
    """The vowels that read the mark v on letter i of _letters(word): fatha / (ey before a bare ی, the ow of a bare و as
    o, a before ا or the final ی of لَدَی, the e of a final ه as in کَندَه), kasra e (the long i of a bare ی after it, as
    in کَرِیم; مِی is ey), damma o (u before a bare و); a small alef on the letter writes a."""
    after = lm[i + 1][0] if i + 1 < len(lm) else ""
    final = i + 1 < len(lm) and lm[i + 1][3]
    out = {v, "a"} if SMALL_ALEF in lm[i][1] else {v}
    if v == "/":
        if after in ALEFS:
            out.add("a")
        elif _semivowel(lm, i, YE):
            out |= {"e", "a"} if final else {"e"}
        elif _semivowel(lm, i, VAV):
            out.add("o")
        elif after == HE and final:
            out.add("e")
    elif v == "e" and after == YE and _bare(lm, i + 1) and not _mey(lm, i):
        out.add("i")
    elif v == "o" and after == VAV and _bare(lm, i + 1):
        out.add("u")
    return out


def _place(lm: list, i: int, ph: str):
    """The index in ph of letter i's consonant (the second of a doubled one), or None: the word's first letter is ph[0]
    when it sounds as written (an alef as @); the others are found at their occurrence counted among same-sounding
    letters, as in geminate(), and و ی ه ع/ء cannot be placed."""
    if i == 0:
        p = "@" if lm[0][0] in ALEFS else LETTER_PHONE.get(lm[0][0])
        if not p or not ph.startswith(p):
            return None
        j = 0
    else:
        p = LETTER_PHONE.get(lm[i][0])
        if not p or p in NO_GEMINATE:
            return None
        k = sum(1 for x in lm[:i] if LETTER_PHONE.get(x[0]) == p)
        pos = [n for n, x in enumerate(ph) if x == p]
        if k >= len(pos):
            return None
        j = pos[k]
    return j + 1 if ph[j + 1:j + 2] == ph[j] else j


def _read_mark(lm: list, i: int, v: str, ph: str, j: int, ezafe: bool) -> str:
    """ph with the vowel after letter i (its consonant at j) read as the mark v."""
    nxt = ph[j + 1:j + 2]
    after = lm[i + 1][0] if i + 1 < len(lm) else ""
    final = i + 1 < len(lm) and lm[i + 1][3]
    if nxt and nxt in mark_options(lm, i, v):
        return ph
    if v == "o" and after == HE and final and nxt == "e":
        return ph  # سَعدُه: the Arabic -uh, not a vowel to swap
    if nxt and nxt in SHORT_VOWELS:
        return ph[:j + 1] + v + ph[j + 2:]
    if nxt in ("a", "i", "u"):
        if nxt == "i" and _semivowel(lm, i, YE) and (_mey(lm, i) or (v == "/" and (not final or lm[i][2]))):
            return ph[:j + 1] + ("e" if ph[j + 2:j + 3] == "y" else "ey") + ph[j + 2:]  # مِی mey, هَیجا heyja
        if nxt == "u" and v == "/" and _semivowel(lm, i, VAV):
            return ph[:j + 1] + "o" + ph[j + 2:]  # أَو: the ow of a bare و
        written = {"a": after in ALEFS or SMALL_ALEF in lm[i][1] or (after == YE and final),
                   "i": after == YE, "u": after == VAV}[nxt]
        return ph if written else ph[:j + 1] + v + ph[j + 2:]  # بِپْسَندی bepsandi -> beps/ndi
    if i < len(lm) - 1 or not ezafe:
        return ph[:j + 1] + v + ph[j + 1:]  # no vowel read after the letter
    return ph


def follow_marks(word: str, ph: str, last_vowel: bool = True) -> str:
    """Ruled 2026-11-11 (training labels and test couplets went against their marks when no candidate read them):
    the vowel after each marked letter becomes the mark (a short vowel swapped, a missing one
    added, a long vowel no letter writes shortened), a sukun drops the vowel after its letter, مِی is mey and وَ is va.
    A letter that cannot be placed (_place) is left, and a word that still does not agree keeps its reading.
    last_vowel=False leaves the last letter's mark to the next word («خَطُّ و خال», «گُلَ انْدَر»); the لله of Allah
    keeps its long a."""
    if harakat_agrees(word, ph) or "لله" in strip_harakat(word):
        return ph
    if strip_harakat(word) == VAV and _mark_vowel(word) == "/":
        return "v/"
    body = _strip_ezafe(ph)
    lm = _letters(word)
    vowels = dict(_marked(lm))
    out = body
    for i, (_, marks, _, _) in enumerate(lm):
        quiet, v = SUKUN in marks, vowels.get(i)
        if not quiet and (not v or (i == len(lm) - 1 and not last_vowel)):
            continue
        j = _place(lm, i, out)
        if j is None:
            continue
        if quiet:
            out = out[:j + 1] + out[j + 2:] if out[j + 1:j + 2] in tuple(SHORT_VOWELS) else out
        else:
            out = _read_mark(lm, i, v, out, j, body != ph)
    new = out + ph[len(body):]
    return new if harakat_agrees(word, new) else ph


def _n_words(raw) -> int:
    return len([w for w in T.remap_ge2pe(raw or "").split(" ") if w])


def _needs_nbest(raw, words) -> bool:
    """A top answer worth looking past: the wrong word count (a looping or merging answer) or a long a its word does
    not write (long_a_written)."""
    pw = [w for w in T.remap_ge2pe(raw or "").split(" ") if w]
    return len(pw) != len(words) or not all(long_a_written(w, r) for w, r in zip(words, pw))


# The model sometimes slips into English-style letter pairs for one Persian sound (پیشه = pishe for piše; 893 "sh", 100
# "gh", 76 "kh", 3 "zh" in the q9c training labels). pair -> (its one sound, the letters that write that sound, the
# letter pairs that rightly give the two sounds, as in اصحاب = @/shab or روزها = ruzha).
LETTER_PAIRS = {"sh": (T.SH, "ش", "سصث", "هح"), "kh": ("x", "خ", "ک", "هح"),
                "gh": ("q", "غق", "گ", "هح"), "zh": (";", "ژ", "زذضظ", "هح")}


def fix_letter_pairs(plain: str, ph: str) -> str:
    """Write a pair as its one sound when the word has that sound's letter and no two-letter source for the pair."""
    w = plain.replace("\u200c", "")
    for pair, (one, letters, first, second) in LETTER_PAIRS.items():
        if pair in ph and any(c in w for c in letters) and not any(a + b in w for a in first for b in second):
            ph = ph.replace(pair, one)
    return ph


def _max_run(seq: list, k: int) -> int:
    """The most times any k-long block of seq occurs back to back."""
    best = 0
    for i in range(len(seq) - k + 1):
        r = 1
        while seq[i + r * k:i + (r + 1) * k] == seq[i:i + k]:
            r += 1
        best = max(best, r)
    return best


def deloop(pw: list, plain: list):
    """A looping seq2seq answer (صد گونه بجوشیدم twice, به که ماند ×5 for ×4): drop extra back-to-back copies of a block
    the label repeats more times in a row than the text repeats any block of that length, while the label is longer
    than the text. Returns the shortened label only if it then has the text's word count, else None (no guessing)."""
    pw = list(pw)
    while len(pw) > len(plain):
        cut = None
        for k in range(1, len(pw) - len(plain) + 1):
            allowed = _max_run(plain, k)
            for i in range(len(pw) - 2 * k + 1):
                r = 1
                while pw[i + r * k:i + (r + 1) * k] == pw[i:i + k]:
                    r += 1
                if r > max(1, allowed):
                    cut = (i + (r - 1) * k, k)  # the last copy of the run
                    break
            if cut:
                break
        if cut is None:
            return None
        del pw[cut[0]:cut[0] + cut[1]]
    return pw if len(pw) == len(plain) else None


def _geminate_y(letters: list, i: int, ph: str) -> str:
    """A shadda on ی, doubled like any other («ز مردیّ و زور» = m/rdiyy o zur): a final ی after the
    vowel i gets yy (with its ezafe: ruziye1 -> ruziyye1); elsewhere the consonant y is doubled when no ی comes before it,
    since an earlier ی may be the vowel i and the y could not be placed."""
    if all(c in "‌‍" for c in letters[i + 1:]):
        if ph.endswith("i"):
            return ph + "yy"
        if ph.endswith("iye" + EZAFE):
            return ph[:-len("ye" + EZAFE)] + "yye" + EZAFE
    if "y" in ph and not any(LETTER_PHONE.get(x) == "y" for x in letters[:i - 1]):
        j = ph.index("y")
        return ph[:j] + "y" + ph[j:]
    return ph


def geminate(word: str, ph: str) -> str:
    """Double each shadda consonant that the phonemes leave single, at its occurrence counted among same-sounding letters."""
    letters = [c for c in word if not HARAKAT.search(c) or c == SHADDA]
    out = ph
    for i, c in enumerate(letters):
        if c != SHADDA or i == 0:
            continue
        p = LETTER_PHONE.get(letters[i - 1])
        if p == "y" and p * 2 not in out:
            out = _geminate_y(letters, i, out)
            continue
        if not p or p in NO_GEMINATE or p * 2 in out:
            continue
        k = sum(1 for x in letters[:i - 1] if LETTER_PHONE.get(x) == p)  # earlier letters with the same sound
        pos = [j for j, x in enumerate(out) if x == p]
        if k < len(pos):
            out = out[:pos[k]] + p + out[pos[k]:]
    return out


def model_words(text: str):
    """(the line's words, the same words as the model reads them: vowel marks dropped, ة readable). The written ezafe
    mark (ۀ) is kept on the word through normalisation."""
    words = G2P.words(text.replace(HEH_HAMZA, "ه" + HAMZA_ABOVE))
    return words, [strip_harakat(_read_ta_marbuta(w)) for w in words]


def load_lexicon(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path and Path(path).exists() else {}


class _HFModel:
    def __init__(self, model_id: str, device: str):
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = T5ForConditionalGeneration.from_pretrained(model_id).to(device).eval()
        self.device = device

    def __call__(self, words):
        import torch
        out = []
        for i in range(0, len(words), 64):
            batch = words[i:i + 64]
            enc = self.tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, num_beams=5, max_length=512)
            out.extend(self.tok.batch_decode(gen, skip_special_tokens=True))
        return [o.strip() for o in out]

    def nbest_batch(self, texts, n: int):
        import torch
        enc = self.tok(list(texts), return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(**enc, num_beams=n, num_return_sequences=n, max_length=512)
        dec = [o.strip() for o in self.tok.batch_decode(gen, skip_special_tokens=True)]
        return [dec[i * n:(i + 1) * n] for i in range(len(texts))]

    def nbest(self, text: str, n: int):
        """The model's n best outputs for one line, best first (metre-guided choice)."""
        import torch
        enc = self.tok([text], return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            gen = self.model.generate(**enc, num_beams=n, num_return_sequences=n, max_length=512)
        return [o.strip() for o in self.tok.batch_decode(gen, skip_special_tokens=True)]


class G2P:
    def __init__(self, model_id: str = "mehdi-hf/Homo-GE2PE-Persian-HF", lexicon: Optional[dict] = None,
                 device: str = "cpu", log_path=None):
        self.model = _HFModel(model_id, device)
        self.lexicon = lexicon or {}
        self.log_path = Path(log_path) if log_path else None
        self.mismatches = 0  # lines where the model's word count differed from the text's (lexicon skipped)

    @staticmethod
    def words(text: str) -> list:
        out = []
        for w in normalize_fa(text).split(" "):
            w = w.strip("\u200c\u200d\u200e\u200f")  # a trailing ZWNJ hid words from NO_EZAFE, the lexicon and the e-rule
            if not w:
                continue
            if out and not strip_harakat(w):  # a vowel mark typed as its own token belongs to the previous word
                out[-1] += w
            else:
                out.append(w)
        return out

    def _lex(self, word: str, plain: str):
        """Lexicon entry unless the word's written vowel marks contradict it (وَ is va, not the entry's o)."""
        e = self.lexicon.get(word) or self.lexicon.get(plain)
        return e if e and harakat_agrees(word, e) else None

    def phonemize_line(self, text: str, metre: Optional[str] = None) -> str:
        """Homo-GE2PE is a sentence-level model. Called word by word it duplicates frequent words
        ("که" → "ke ke", "نی" → "nin ni"), which corrupted 47 % of the first build's clips. The whole line goes through the
        model once; the lexicon is applied by word position only when the model kept the text's word count.

        With the poem's metre, the model's NBEST outputs are tried best first and the first one
        that scans in the metre is used; if none scans, the top output is."""
        words, plain, joined, metred, guided = self._prepare(text, metre)
        raws = self.model.nbest(joined, NBEST) if guided else [self.model([joined])[0]]
        if not guided and hasattr(self.model, "nbest") and _needs_nbest(raws[0], words):
            raws = self.model.nbest(joined, NBEST)  # a looping, merging or unwritten-long-a answer: look further down
        return self._choose(text, words, plain, metre, metred, raws)

    def _nbest_many(self, texts: list) -> list:
        return self.model.nbest_batch(texts, NBEST) if hasattr(self.model, "nbest_batch") else [self.model.nbest(t, NBEST) for t in texts]

    def _prepare(self, text: str, metre: Optional[str]):
        words, plain = model_words(text)
        if not words:
            raise PhonemizeError(f"no words in {text!r}")
        metred = bool(metre) and bool(metre_patterns(metre))
        guided = (metred or any(HARAKAT.search(w) for w in words)) and hasattr(self.model, "nbest")
        return words, plain, " ".join(plain), metred, guided

    def _choose(self, text, words, plain, metre, metred, raws) -> str:
        best, best_key = None, None
        for raw in raws:
            pw = [w for w in T.remap_ge2pe(raw or "").split(" ") if w]
            if not pw:
                continue
            if len(pw) > len(words):
                pw = deloop(pw, plain) or pw
            aligned = len(pw) == len(words)
            if aligned:
                pw = [fix_letter_pairs(p, r) for p, r in zip(plain, pw)]
                lexed = [self._lex(w, p) for w, p in zip(words, plain)]
                # a lexicon reading's or a function word's final e is its own vowel (ز = ze, زانک = zanke), never a bare ezafe
                pw = [_with_ezafe(lx or r, r.endswith(EZAFE) or written_ezafe(w),
                                  "" if lx or p in NO_EZAFE else p)
                      for w, p, r, lx in zip(words, plain, pw, lexed)]
                last = len(pw) - 1  # the model often writes ezafe as a bare e; 1 is its own token
                pw = [r + EZAFE if i < last and p not in NO_EZAFE and r.endswith("e") and not p.endswith(VOWEL_LETTERS) else r
                      for i, (p, r) in enumerate(zip(plain, pw))]
                pw = [_strip_ezafe(r, p) if p in NO_EZAFE else r for p, r in zip(plain, pw)]
                # a written ezafe (final kasra, هٔ) outranks the finite-verb guess: «عقلِ کُندِ», «جهانِ هست»
                pw = [_strip_ezafe(r, p) if i < last and FINITE_VERB.match(plain[i + 1])
                      and not written_ezafe(w) else r
                      for i, (w, p, r) in enumerate(zip(words, plain, pw))]
                pw = [_strip_ezafe(r, p) if w.endswith(SUKUN) else r for w, p, r in zip(words, plain, pw)]  # دستْ: no ezafe
                pw = no_ezafe_before_pause(text, words, plain, pw)  # «به خواب، دو چشم»: xab
                pw[-1] = _strip_ezafe(pw[-1], plain[-1])  # a hemistich ends a phrase: no ezafe across the line break
                pw = contract_ast(words, pw, metre if metred else None)
            # rank: aligned, scans the metre, agrees with the written vowel marks, then (when none aligns) the word count
            # closest to the text's — a refrain written 7 times for 4 loses to one written 5 times; ties keep the model's order
            key = (aligned, aligned and metred and scans(" ".join(pw), metre),
                   aligned and all(harakat_agrees(w, r) for w, r in zip(words, pw)),
                   aligned and all(long_a_written(w, r) for w, r in zip(words, pw)), -abs(len(pw) - len(words)))
            if best_key is None or key > best_key:
                best, best_key = (pw, aligned), key
        if best is None:
            raise PhonemizeError(f"empty phonemes for {text!r}")
        pw, aligned = best
        if aligned:  # last resort, after ranking: a candidate that already doubles the shadda letter wins above
            pw = [tanwin(w, small_alef(w, geminate(w, r) if SHADDA in w else r)) for w, r in zip(words, pw)]
            # and one that reads the marks; a mark on the last letter before و or an alef runs into that word
            last = [i == len(words) - 1 or not (plain[i + 1] == VAV or plain[i + 1].startswith(tuple(ALEFS)))
                    for i in range(len(words))]
            pw = [follow_marks(w, r, lv) for w, r, lv in zip(words, pw, last)]
        else:
            self.mismatches += 1
        logged = zip(words, pw) if aligned else ((w, "?") for w in words)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                for w, r in logged:
                    if HARAKAT.search(w):
                        f.write(f"{w}\t{r}\n")
        return " ".join(pw)

    def phonemize_unit(self, lines: list, kind: str, metre: Optional[str] = None) -> str:
        lines = [l for l in lines if l.strip()]
        if not lines:
            raise PhonemizeError("unit has no text")
        prefix = T.VERSE + T.metre_token(metre) if kind == "verse" else T.PROSE
        return prefix + T.HEMI.join(self.phonemize_line(l, metre if kind == "verse" else None) for l in lines)

    def phonemize_units(self, units, batch: int = 32) -> list:
        """phonemize_unit over many (lines, kind, metre) units with batched model calls; None where a unit fails."""
        jobs, out = [], [None] * len(units)  # job: (unit index, line index, text, metre, prepared or error)
        for ui, (lines, kind, metre) in enumerate(units):
            lines = [l for l in lines if l.strip()]
            if not lines:
                continue
            m = metre if kind == "verse" else None
            for li, l in enumerate(lines):
                try:
                    jobs.append((ui, li, l, m, self._prepare(l, m)))
                except PhonemizeError:
                    jobs.append((ui, li, l, m, None))
        guided = [j for j in jobs if j[4] and j[4][4]]
        plain = [j for j in jobs if j[4] and not j[4][4]]
        raws = {}
        for i in range(0, len(guided), batch):
            chunk = guided[i:i + batch]
            raws.update({(j[0], j[1]): r for j, r in zip(chunk, self._nbest_many([j[4][2] for j in chunk]))})
        for i in range(0, len(plain), batch):
            chunk = plain[i:i + batch]
            raws.update({(j[0], j[1]): [r] for j, r in zip(chunk, self.model([j[4][2] for j in chunk]))})
        retry = [j for j in plain if _needs_nbest(raws[(j[0], j[1])][0], j[4][0])] if hasattr(self.model, "nbest") else []
        for i in range(0, len(retry), batch):  # as in phonemize_line: a top answer worth looking past gets the n-best
            chunk = retry[i:i + batch]
            raws.update({(j[0], j[1]): r for j, r in zip(chunk, self._nbest_many([j[4][2] for j in chunk]))})
        lines_out = {}
        for ui, li, l, m, prep in jobs:
            try:
                if prep is None:
                    raise PhonemizeError(f"no words in {l!r}")
                words, pl, _, metred, _ = prep
                lines_out.setdefault(ui, {})[li] = self._choose(l, words, pl, m, metred, raws[(ui, li)])
            except PhonemizeError:
                lines_out.setdefault(ui, {})[li] = None
        for ui, (lines, kind, metre) in enumerate(units):
            got = lines_out.get(ui)
            if not got or any(v is None for v in got.values()):
                continue
            prefix = T.VERSE + T.metre_token(metre) if kind == "verse" else T.PROSE
            out[ui] = prefix + T.HEMI.join(got[i] for i in sorted(got))
        return out


SHORT_MARKS, LONG_MARKS = "،,؛;:", ".!?؟"  # the punctuation a reader pauses at (short, long)
VAV = "و"
VAV_MID_VERSE = ("o", "vo")  # the G2P's readings of a standalone و: the metre's -o (vo after a vowel)


def punctuation_after_words(line: str) -> list:
    """For each of G2P.words(line), the pause-bearing punctuation typed after it ("" if none); a mark typed apart
    ("شرع ، مرا") belongs to the word before it."""
    out = []
    for tok in line.split():
        p = "".join(ch for ch in tok if ch in SHORT_MARKS + LONG_MARKS)
        n = len(G2P.words(tok))
        if n:
            out += [""] * (n - 1) + [p]
        elif out and p:
            out[-1] += p
    return out


def no_ezafe_before_pause(line: str, words: list, plain: list, pw: list) -> list:
    """Ruled 2026-11-10 («تا کی ز قدیم و مُحدَث، اومیدم و بیم»): a word right before a typed pause mark ends a phrase
    and takes no ezafe; a written kasra keeps it. Training labels had one there («به خواب، دو چشم»
    xabe1, «مَرد، اَحوَل» m/rde1). A mark typed against the next word («گفت !هر») belongs to the word before it. Returns a
    new list; a line whose marks do not line up with its words is left as it is."""
    marks = punctuation_after_words("".join(f" {ch} " if ch in SHORT_MARKS + LONG_MARKS else ch for ch in line))
    if len(marks) != len(pw):
        return list(pw)
    return [_strip_ezafe(r, p) if m and not written_ezafe(w) else r for w, p, r, m in zip(words, plain, pw, marks)]


def read_vav(line: str, pw: list, prose: bool) -> list:
    """A standalone «و» is read v/ (va) in prose, at the start of a line and after a pause mark, as Persian reads it
    there; inside a line of verse it stays the metre's o. The voice drops a prose «و» labelled o far more often than
    it drops one in verse. A written vowel on the و decides, and a label that does not line up with the text's words
    is left as it is. Returns a new list."""
    words = G2P.words(line)
    if len(words) != len(pw):
        return list(pw)
    marks = punctuation_after_words(line)
    after_mark = [False] + ([bool(m) for m in marks[:-1]] if len(marks) == len(words) else [False] * (len(words) - 1))
    return ["v/" if w == VAV and r in VAV_MID_VERSE and (prose or i == 0 or after_mark[i]) else r
            for i, (w, r) in enumerate(zip(words, pw))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lexicon")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    g = G2P(lexicon=load_lexicon(a.lexicon), device=a.device)
    with open(a.inp, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as out:
        for line in f:
            out.write(g.phonemize_line(line.strip()) + "\n")


if __name__ == "__main__":
    main()

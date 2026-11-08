# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import json

from ganj_reading.language import looks_arabic, skip_hemistich

GHAZAL_1 = [  # Hafez, ghazal 1 (Ganjoor poem 2130); Ganjoor tags hemistichs 1 and 14 Arabic (languageId 2)
    "اَلا یا اَیُّهَا السّاقی اَدِرْ کَأساً و ناوِلْها", "که عشق آسان نمود اوّل ولی افتاد مشکل‌ها",
    "به بویِ نافه‌ای کآخِر صبا زان طُرّه بُگشاید", "ز تابِ جَعدِ مشکینش چه خون افتاد در دل‌ها",
    "مرا در منزلِ جانان چه امنِ عیش، چون هر دَم", "جَرَس فریاد می‌دارد که بَربندید مَحمِل‌ها",
    "به مِی سجّاده رنگین کُن گَرَت پیرِ مُغان گوید", "که سالِک بی‌خبر نَبْوَد ز راه و رسمِ منزل‌ها",
    "شبِ تاریک و بیمِ موج و گِردابی چنین هایل", "کجا دانند حالِ ما سبک‌بارانِ ساحل‌ها؟",
    "همهْ کارم ز خودکامی به بدنامی کشید آخر", "نهان کِی مانَد آن رازی کَزو سازند مَحفل‌ها؟",
    "حضوری گر همی خواهی از او غایب مشو حافظ", "مَتیٰ ما تَلْقَ مَنْ تَهْویٰ دَعِ الدُّنْیا وَ اَهْمِلْها",
]


def test_ganjoors_language_tag_decides_when_it_is_there():
    assert skip_hemistich("که عشق آسان نمود اوّل ولی افتاد مشکل‌ها", language_id=2)   # Arabic
    assert skip_hemistich("که عشق آسان نمود اوّل ولی افتاد مشکل‌ها", language_id=3)   # Azeri Turkish
    assert not skip_hemistich(GHAZAL_1[0], language_id=1)                              # tagged Persian: read


def test_ghazal_1_without_tags_skips_its_two_arabic_hemistichs_and_reads_the_rest():
    assert [i for i, h in enumerate(GHAZAL_1) if skip_hemistich(h)] == [0, 13]


def test_untagged_arabic_hemistichs_are_skipped():
    for h in ["سَلِ المَصانِعَ رَکباً تَهیمُ في الفَلَواتِ",   # Saadi, untagged on Ganjoor
              "فَاسألوا حالَها عَنِ الاَطْلال",               # Hafez 302
              "مَرضیّةُ السَجایا مَحمودَةُ الخَصائل",         # Hafez 307
              "اَشهیٰ لَنا و اَحلیٰ مِن قُبلَةِ العَذارا"]:   # Hafez 5
        assert looks_arabic(h), h


def test_an_arabic_hemistich_with_a_word_that_is_also_common_persian_is_read_not_skipped():
    # the known cost of reading rather than skipping when unsure: برید is also Persian (cut)
    assert not looks_arabic("یا بَریدَ الْحِمی حَماکَ الله")


def test_persian_hemistichs_full_of_arabic_words_are_read():
    for h in ["یا بیامیز یا بیاموزم", "حساس تر شوند لطیفان علی الیقین", "طمطراق اجتهاد و بارنامه اعتقاد",
              "صالح و ناصالح و خوب و خشوک", "اظهار حروف اسم اعظم", "عجب عِلمیست عِلم هیئت عشق",
              "لامکانی فوق وهم سالکان", "من مصحف باطلم ولیکن", "اللّه اللّه غیرت آمد غیر سوز"]:
        assert not looks_arabic(h), h


def test_the_app_fixture_matches(tmp_path):
    from pathlib import Path
    fixture = Path(__file__).parent / "fixtures" / "hemistich_language.json"
    for case in json.loads(fixture.read_text(encoding="utf-8")):
        assert skip_hemistich(case["text"], case.get("language_id")) == case["skip"], case

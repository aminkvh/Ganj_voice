# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import pytest

from ganj_reading import tokens as T
from ganj_reading.g2p import G2P, PhonemizeError, normalize_fa, strip_harakat


class FakeModel:
    def __init__(self, table):
        self.table = table
        self.calls = []

    def __call__(self, words):
        self.calls.append(list(words))
        out = [self.table.get(w, "") for w in words]
        return [o[0] if isinstance(o, (list, tuple)) else o for o in out]  # the real model returns its top output

    def nbest(self, text, n):
        self.calls.append(("nbest", text, n))
        v = self.table.get(text, "")
        return list(v) if isinstance(v, (list, tuple)) else [v]


def g2p(table, lexicon=None, log_path=None):
    g = G2P.__new__(G2P)
    g.model = FakeModel(table)
    g.lexicon = lexicon or {}
    g.log_path = log_path
    g.mismatches = 0
    return g


def test_normalize_unifies_arabic_letters_and_keeps_zwnj():
    assert normalize_fa("كتاب‌ها يك") == "کتاب‌ها یک"
    assert strip_harakat("بِشنَو") == "بشنو"


def test_lexicon_wins_over_model_and_output_is_remapped():
    # the model is called once per LINE (it is a sentence-level G2P; per-word calls duplicate words).
    g = g2p({"ز شب بود": "z $/b bud"}, lexicon={"ز": "ze"})   # شب is š/b: a long a needs an alef (long_a_written)
    assert g.phonemize_line("ز شب بود") == "ze š/b bud"
    assert g.model.calls == [["ز شب بود"]]


def test_lexicon_is_skipped_when_model_word_count_differs():
    g = g2p({"همی گفتند که": "h/migoft/nd ke"}, lexicon={"که": "ke2"})
    assert g.phonemize_line("همی گفتند که") == "h/migoft/nd ke"  # 2 phoneme words vs 3 text words: output kept as is
    assert g.mismatches == 1


def test_unit_adds_style_prefix_and_hemistich_boundary():
    g = g2p({"الف": "alef", "ب": "be"})
    s = g.phonemize_unit(["الف", "ب"], "verse")
    assert s == T.VERSE + "alef" + T.HEMI + "be"
    assert g.phonemize_unit(["الف"], "prose").startswith(T.PROSE)
    assert T.encode(s)  # encodable under the contract


def test_empty_phonemes_raise():
    g = g2p({"۱۲۳": "", "   ": ""})
    with pytest.raises(PhonemizeError):
        g.phonemize_line("۱۲۳")
    with pytest.raises(PhonemizeError):
        g.phonemize_unit(["   "], "verse")


def test_diacritised_words_are_logged_for_lexicon_review(tmp_path):
    log = tmp_path / "diacritised.tsv"
    g = g2p({"بشنو": "be$no"}, log_path=log)
    g.phonemize_line("بِشنَو")
    assert log.read_text(encoding="utf-8").strip() == "بِشنَو\tbešno"


def test_lexicon_entry_keeps_the_models_ezafe():
    g = g2p({"چشم مست": "ceš/me1 m/st"}, lexicon={"چشم": "c/šm"})
    assert g.phonemize_line("چشم مست") == "c/šme1 m/st"


def test_lexicon_entry_ending_in_a_vowel_takes_ye_ezafe():
    g = g2p({"باده ناب": "badee1 nab"}, lexicon={"باده": "bade"})  # the model sees plain words
    assert g.phonemize_line("بادهٔ ناب") == "badeye1 nab"


def test_written_hamza_ezafe_is_pronounced():
    g = g2p({"باده ناب": "bade nab", "خانه دوست": "xane dust"})
    assert g.phonemize_line("بادهٔ ناب") == "badeye1 nab"
    assert g.phonemize_line("خانۀ دوست") == "xaneye1 dust"


def test_written_kasra_ezafe_is_pronounced():
    g = g2p({"من مسکین": "m/n meskin"})
    assert g.phonemize_line("منِ مسکین") == "m/ne1 meskin"


MUTAQARIB = "فعولن فعولن فعولن فعل (متقارب مثمن محذوف یا وزن شاهنامه)"


def test_metre_picks_the_best_ranked_candidate_that_scans():
    line = "توانا بود هر که دانا بود"
    g = g2p({line: ["t/vana bov/d h/r ke d/na bov/d", "t/vana bov/d h/r ke dana bov/d"]})
    assert g.phonemize_line(line, metre=MUTAQARIB) == "t/vana bov/d h/r ke dana bov/d"


def test_no_scanning_candidate_falls_back_to_the_top_one():
    line = "توانا بود هر که دانا بود"
    g = g2p({line: ["t/vana bov/d h/r ke d/na bov/d", "t/vana bov/d h/r d/na bov/d"]})
    assert g.phonemize_line(line, metre=MUTAQARIB) == "t/vana bov/d h/r ke d/na bov/d"


def test_without_a_metre_only_the_top_output_is_used():
    g = g2p({"ز شب بود": "z $/b bud"})
    g.phonemize_line("ز شب بود")
    assert not any(c[0] == "nbest" for c in g.model.calls if isinstance(c, tuple))


def test_verse_unit_carries_its_metre_token():
    from ganj_reading import tokens as T
    line = "توانا بود هر که دانا بود"
    g = g2p({line: ["t/vana bov/d h/r ke dana bov/d"]})
    ph = g.phonemize_unit([line], "verse", MUTAQARIB)
    tok = T.metre_token(MUTAQARIB)
    assert tok and ph.startswith(T.VERSE + tok)
    assert g2p({line: "x"}).phonemize_unit([line], "prose").startswith(T.PROSE + "x")


# --- label fixes from the 140-line audit ---

def test_function_words_never_take_ezafe():
    g = g2p({"ز شب بود": "zeye1 š/b bud", "کاین دل": "kine1 del"}, lexicon={"ز": "ze"})
    assert g.phonemize_line("ز شب بود") == "ze š/b bud"
    assert g.phonemize_line("کاین دل") == "kin del"


def test_hemistich_final_word_drops_ezafe():
    g = g2p({"دل من": "dele1 m/ne1"})
    assert g.phonemize_line("دل من") == "dele1 m/n"


def test_written_vowel_marks_pick_the_candidate_that_agrees():
    g = g2p({"گر بود": ["g/r bud", "g/r bov/d"]})
    assert g.phonemize_line("گر بوَد") == "g/r bov/d"


def test_unmarked_lines_without_metre_still_use_only_the_top_output():
    g = g2p({"گر بود": ["g/r bud", "g/r bov/d"]})
    assert g.phonemize_line("گر بود") == "g/r bud"


def test_metre_still_wins_over_vowel_marks_when_both_cannot_be_met():
    line = "توانا بوَد هر که دانا بود"
    g = g2p({line.replace("َ", ""): ["t/vana bud h/r ke dana bov/d", "t/vana bov/d h/r ke dana bov/d"]})
    assert g.phonemize_line(line, metre=MUTAQARIB) == "t/vana bov/d h/r ke dana bov/d"


def test_harakat_vowels_and_short_vowels_agree_as_a_subsequence():
    from ganj_reading.g2p import harakat_agrees
    assert harakat_agrees("نکُشد", "n/koš/d")
    assert not harakat_agrees("نکُشد", "n/keš/d")
    assert harakat_agrees("مُلکِ", "molke1")      # a final kasra is the ezafe, not a vowel inside the word
    assert harakat_agrees("ملک", "melke1")        # no marks: anything agrees


def test_a_bare_final_e_after_a_consonant_letter_is_marked_as_ezafe():
    g = g2p({"عمر جاودان": "@omre javedan", "خانه دوست": "xane dust"})
    assert g.phonemize_line("عمر جاودان") == "@omre1 javedan"
    assert g.phonemize_line("خانه دوست") == "xane dust"   # the e of خانه is the word's own vowel


# --- final-review fixes (I1-I4) ---

def test_a_trailing_zwnj_does_not_hide_a_word_from_the_rules():
    g = g2p({"به دل": "be del", "ز شب": "zeye1 š/b"})
    assert G2P.words("به‌ دل") == ["به", "دل"]
    assert g.phonemize_line("به‌ دل") == "be del"          # no false ezafe from the bare-e rule
    assert g.phonemize_line("ز‌ شب") == "ze š/b"


def test_lexicon_yields_to_written_vowel_marks_that_contradict_it():
    g = g2p({"و دل": "v/ del"}, lexicon={"و": "o"})
    assert g.phonemize_line("وَ دل") == "v/ del"
    assert g.phonemize_line("و دل") == "o del"


def test_hemistich_final_written_ye_survives_the_ezafe_strip():
    g = g2p({"این روی": "@in ruye1", "دل من": "dele1 m/ne1"})
    assert g.phonemize_line("این روی") == "@in ruy"
    assert g.phonemize_line("دل من") == "dele1 m/n"


def test_a_detached_vowel_mark_joins_the_previous_word():
    assert G2P.words("کفّ ِ دست") == ["کفِّ", "دست"] or G2P.words("کفّ ِ دست") == ["کفّ" + "ِ", "دست"]


def test_hemistich_final_vowel_ye_still_drops_the_whole_ezafe():
    g = g2p({"این ماهی": "@in mahiye1"})
    assert g.phonemize_line("این ماهی") == "@in mahi"


class BatchFake(FakeModel):
    def nbest_batch(self, texts, n):
        self.calls.append(("nbest_batch", len(texts)))
        return [self.nbest(t, n) for t in texts]


def test_batched_units_equal_unit_by_unit_and_use_one_model_call():
    table = {"توانا بود هر که دانا بود": ["t/vana bov/d h/r ke d/na bov/d", "t/vana bov/d h/r ke dana bov/d"],
             "ز شب بود": "zeye1 š/b bud", "گر بود": ["g/r bud", "g/r bov/d"]}
    units = [(["توانا بود هر که دانا بود", "ز شب بود"], "verse", MUTAQARIB), (["گر بوَد"], "prose", None), ([""], "prose", None)]
    one = g2p(table)
    expected = []
    for lines, kind, metre in units:
        try:
            expected.append(one.phonemize_unit(lines, kind, metre))
        except PhonemizeError:
            expected.append(None)
    batched = g2p(table)
    batched.model = BatchFake(table)
    assert batched.phonemize_units(units) == expected
    assert sum(1 for c in batched.model.calls if c[0] == "nbest_batch") == 1


def test_a_shadda_requires_the_letter_to_be_doubled():
    from ganj_reading.g2p import harakat_agrees
    assert harakat_agrees("شکّر", "š/kk/r")
    assert not harakat_agrees("شکّر", "šokr")
    assert harakat_agrees("بدرّید", "bed/rrid")
    assert not harakat_agrees("بدرّید", "bed/rid")


def test_shadda_picks_the_geminated_candidate():
    g = g2p({"شکر لب": ["šokr l/b", "š/kk/r l/b"]})
    assert g.phonemize_line("شکّر لب") == "š/kk/r l/b"


def test_geminate_doubles_the_shadda_consonant_in_place():
    from ganj_reading.g2p import geminate
    assert geminate("غمّاز", "q/maz") == "q/mmaz"
    assert geminate("طُرّه", "t/re") == "t/rre"
    assert geminate("شکّر", "š/kk/r") == "š/kk/r"          # already doubled: unchanged
    assert geminate("مُرَّه", "morr/") == "morr/"
    assert geminate("قوّت", "qov/t") == "qov/t"            # و is often a vowel: left alone


def test_shadda_is_doubled_when_no_candidate_has_it():
    g = g2p({"غماز دل": "q/maz del"})
    assert g.phonemize_line("غمّاز دل") == "q/mmaz del"


def test_a_shadda_on_ye_is_doubled_like_any_other():
    # «ز مردیّ و زور» is m/rdiyy o zur, like any other tashdid; 182 q9p labels left it single
    from ganj_reading.g2p import geminate
    assert geminate("مردیّ", "m/rdi") == "m/rdiyy"               # final, after the vowel i
    assert geminate("روزیِّ", "ruziye1") == "ruziyye1"            # with its ezafe
    assert geminate("نافه‌ایّ", "nafei") == "nafeiyy"             # a ZWNJ is not a letter
    assert geminate("رعیّت", "r/@y/t") == "r/@yy/t"              # the consonant y, no ی before it
    assert geminate("ایّام", "@/yyam") == "@/yyam"               # already doubled: unchanged
    assert geminate("زیبیّت", "zib/y/t") == "zib/y/t"             # an earlier ی (here the vowel i): too ambiguous, left
    from ganj_reading.g2p import harakat_agrees
    assert harakat_agrees("روزیِّ", "ruziyye1") and harakat_agrees("زاریّ", "zariyye1")   # yy + e1, not i + ye1
    g = g2p({"مردی و زور": "m/rdi o zur"})
    assert g.phonemize_line("مردیّ و زور") == "m/rdiyy o zur"


def test_a_small_alef_on_a_final_ye_reads_a():
    # a small alef is read a: «کسریٰ» is Kasrā; 26 of 42 such training words were read with a final i
    from ganj_reading.g2p import harakat_agrees, small_alef
    assert small_alef("کسریٰ", "k/sri") == "k/sra"
    assert small_alef("سَلْمیٰ", "s/lmiye1") == "s/lmaye1"
    assert small_alef("عُقبیٰ", "@oqba") == "@oqba"
    assert small_alef("کسری", "k/sri") == "k/sri"          # unmarked کسری (a shortfall) keeps its reading
    assert not harakat_agrees("کسریٰ", "k/sri") and harakat_agrees("کسریٰ", "k/sra")
    g = g2p({"کسری در": "k/sri d/r"})
    assert g.phonemize_line("کسریٰ در") == "k/sra d/r"


def test_tanwin_is_read_with_its_n():
    # Arabic harakat: ً -an, ٌ -on, ٍ -en. Persian-line training labels had the n in 40 of 75
    # tanwin fath words (فرضاً f/rza) and in 0 of 83 tanwin damm / kasr
    from ganj_reading.g2p import harakat_agrees, tanwin
    assert tanwin("فرضاً", "f/rza") == "f/rz/n"          # the alef carrying it is not a long a
    assert tanwin("طَیْرًا", "t/yra") == "t/yr/n"         # written before the alef
    assert tanwin("عجباً", "@/j/b/n") == "@/j/b/n"
    assert tanwin("خیرٍ", "xeyr") == "xeyren" and tanwin("قَلیلٌ", "q/lile1") == "q/lilon"
    assert tanwin("صُمٌّ", "somm") == "sommon"            # Unicode puts the tanwin before the shadda
    assert tanwin("اٌستاد", "@ostad") == "@ostad" and tanwin("دل", "del") == "del"
    assert not harakat_agrees("فرضاً", "f/rza") and harakat_agrees("فرضاً", "f/rz/n")


def test_ta_marbuta_is_read_t_before_a_mark_and_as_a_final_he_otherwise():
    # ة went to the model as is and came back k (88 of 95 Persian-line labels: روضةٌ ruz/k, لیلة leylek)
    g = g2p({"روضت": "rowz/t"})
    assert g.phonemize_line("روضةٌ") == "rowz/ton"
    g = g2p({"لیله": "leyle"})
    assert g.phonemize_line("لیلة") == "leyle"


def test_a_candidate_whose_long_a_is_written_wins():
    # «مهتر» is meht/r or m/ht/r — never a long a, which Persian writes with an alef;
    # all 27 training labels read mehtar. Ranking only: which of meh / mah stays per line.
    from ganj_reading.g2p import long_a_written
    assert not long_a_written("مهتر", "mehtar") and long_a_written("مهتر", "meht/r")
    assert long_a_written("موسی", "musa") and long_a_written("الله", "@llah") and long_a_written("کسریٰ", "k/sra")
    g = g2p({"مهتر ما": ["mehtar ma", "meht/r ma"]})
    assert g.phonemize_line("مهتر ما") == "meht/r ma"


def test_a_kasra_written_before_the_shadda_is_still_a_written_ezafe():
    # Unicode orders kasra before shadda (حقِّ = ح ق ِ ّ): 982 corpus words end so, and none counted as a written ezafe
    from ganj_reading.g2p import harakat_agrees, written_ezafe
    assert written_ezafe("حقِّ") and written_ezafe("دلِ") and written_ezafe("خانهٔ") and not written_ezafe("حقّ")
    assert harakat_agrees("حقِّ", "h/qqe1")
    g = g2p({"حق ما": "h/qq ma"})
    assert g.phonemize_line("حقِّ ما") == "h/qqe1 ma"


# --- label errors found by listening ---

def test_no_ezafe_on_the_word_before_a_finite_verb():
    # 5112-0 "خدمت سلطان کردی": the model wrote soltane1; a verb cannot be an ezafe dependent
    g = g2p({"یکی خدمت سلطان کردی": "yeki xedm/te1 soltane1 k/rdi"})
    assert g.phonemize_line("یکی خدمت سلطان کردی") == "yeki xedm/te1 soltan k/rdi"


def test_a_written_kasra_keeps_its_ezafe_even_before_a_word_that_looks_like_a_verb():
    # A mark in the text outranks a heuristic: «عقلِ کُندِ» (کند = dull, not a verb) and «جهانِ هست» lost their written ezafe to
    # the finite-verb rule in 9 q9e training labels.
    g = g2p({"عقل کند": ["@/qle1 kond"], "جهان هست": ["j/hane1 h/st"]})
    assert g.phonemize_line("عقلِ کُندِ") == "@/qle1 kond"
    assert g.phonemize_line("جهانِ هست") == "j/hane1 h/st"


def test_ezafe_before_a_noun_that_is_not_a_verb_survives():
    g = g2p({"خدمت سلطان عادل": "xedm/te1 soltane1 @adel"})
    assert g.phonemize_line("خدمت سلطان عادل") == "xedm/te1 soltane1 @adel"


def test_gar_with_a_clitic_never_takes_ezafe():
    # unheard u9602 "گرت مشاهده": garat is گر + ت, a contraction like کز
    g = g2p({"گرت مشاهده": "g/rte1 mošahede"}, lexicon={"گرت": "g/r/t"})
    assert g.phonemize_line("گرت مشاهده") == "g/r/t mošahede"


def test_contractions_with_a_demonstrative_or_vocative_never_take_ezafe():
    # Sonnet mining (1,500 lines): کای (که ای) 6×, زین (ز این) 5×, وان (و آن) 3×, درین (در این) 2× labelled with ezafe
    g = g2p({"کای دل": "kaye1 del", "زین سخن": "zine1 soxan", "وان دل": "vane1 del", "درین شهر": "d/rine1 š/hr"})
    assert g.phonemize_line("کای دل") == "kay del"
    assert g.phonemize_line("زین سخن") == "zin soxan"
    assert g.phonemize_line("وان دل") == "van del"
    assert g.phonemize_line("درین شهر") == "d/rin š/hr"


# --- label review: spaces, ezafe notation, sukun ---

def test_no_break_and_thin_spaces_separate_words_like_a_space():
    # 1,222 U+00A0 in the corpus text glued words together ("آنکه\xa0" missed NO_EZAFE)
    assert normalize_fa("آنکه دین") == "آنکه دین"
    assert normalize_fa("دل من تو") == "دل من تو"
    assert G2P.words("آنکه  دین") == ["آنکه", "دین"]


def test_a_bare_e_before_a_written_kasra_is_the_ezafe_not_a_vowel_to_extend():
    # روزِ → ruzeye1, برایِ → b/rayeye1: the model's bare e got a second ye1 appended
    g = g2p({"روز پسین": "ruze p/sin", "برای خدا": "b/raye xoda", "سایه سرو": "saye s/rv"})
    assert g.phonemize_line("روزِ پسین") == "ruze1 p/sin"
    assert g.phonemize_line("برایِ خدا") == "b/raye1 xoda"
    assert g.phonemize_line("سایهٔ سرو") == "sayeye1 s/rv"      # the e of سایه is the word's own vowel: ye1 is right


def test_a_written_heh_hamza_ezafe_the_model_already_said_as_ye_is_not_doubled():
    # گوشهٔ was read gušeyeye1; 199 q9e labels: سلسلهٔ selseleyeye1, عرصهٔ @/rseyeye1, نشانهٔ neš/neyeye1
    g = g2p({"گوشه دل": "gušeye del", "سلسله مو": "selseleye mu", "سایه سرو": "sayeye s/rv", "گریه شمع": "gerye š/m@"})
    assert g.phonemize_line("گوشهٔ دل") == "gušeye1 del"
    assert g.phonemize_line("سلسلۀ مو") == "selseleye1 mu"
    assert g.phonemize_line("سایهٔ سرو") == "sayeye1 s/rv"      # سایه's own ye, then the model's bare ezafe ye
    assert g.phonemize_line("گریهٔ شمع") == "geryeye1 š/m@"     # گریه = gerye: its ye is the word's, the ezafe is added


def test_a_letter_with_sukun_has_no_vowel_after_it():
    from ganj_reading.g2p import harakat_agrees
    assert harakat_agrees("مَرْدُم", "m/rdom")
    assert not harakat_agrees("مَرْدُم", "m/r/dom")
    assert harakat_agrees("خام‌ْطَمَع", "xamt/m/@")        # w87: the sukun after the ZWNJ belongs to the م before it


def test_a_final_sukun_means_no_ezafe():
    # "دستْ کِی بودی", "از دلْ خیالِ": a letter with a sukun takes no vowel after it, so no ezafe
    g = g2p({"دست کی": "d/ste1 key", "دل خیال": "dele1 x/yale1"})
    assert g.phonemize_line("دستْ کِی") == "d/st key"
    assert g.phonemize_line("دلْ خیال") == "del x/yal"


def test_a_function_words_own_final_e_is_not_taken_for_a_bare_ezafe():
    g = g2p({"ز دست": "ze d/st"}, lexicon={"ز": "ze"})
    assert g.phonemize_line("زِ دست") == "ze d/st"


LOOP = "به که ماند به که ماند"


def test_a_looping_answer_on_an_unguided_line_is_replaced_by_a_candidate_with_the_texts_word_count():
    # Refrains (به که ماند ×4) came back written ×5 or more: seq2seq looping on lines that had only the top answer
    g = g2p({LOOP: ["be ke manad be ke manad be ke manad", "be ke manad be ke manad"]})
    assert g.phonemize_line(LOOP) == "be ke manad be ke manad"
    assert g.mismatches == 0


def test_when_no_candidate_has_the_texts_word_count_the_closest_one_wins():
    g = g2p({"دل من": ["del m/n to va hal", "del m/n to"]})   # extra words that are no repeat, so nothing to de-loop
    assert g.phonemize_line("دلِ من") == "del m/n to"


def test_batched_units_also_retry_a_looping_unguided_line():
    table = {LOOP: ["be ke manad be ke manad be ke manad", "be ke manad be ke manad"]}
    one, batched = g2p(table), g2p(table)
    batched.model = BatchFake(table)
    assert batched.phonemize_units([([LOOP], "prose", None)]) == [one.phonemize_unit([LOOP], "prose", None)]
    assert one.phonemize_unit([LOOP], "prose", None) == T.PROSE + "be ke manad be ke manad"


def test_a_phrase_the_model_repeated_once_too_often_is_written_as_often_as_the_text_has_it():
    # 15134-7 صد گونه بجوشیدم came back twice although every n-best candidate loops the same way
    text = "صد جام بنوشیدم صد گونه بجوشیدم"
    g = g2p({text: "s/d jam benušid/m s/d gune bejušid/m s/d gune bejušid/m"})
    assert g.phonemize_line(text) == "s/d jam benušid/m s/d gune bejušid/m"
    g = g2p({LOOP: "be ke manad be ke manad be ke manad"})
    assert g.phonemize_line(LOOP) == "be ke manad be ke manad"


def test_a_repeat_the_text_itself_has_is_never_cut():
    from ganj_reading.g2p import deloop
    assert deloop(["del", "del", "del", "m/n"], ["دل", "دل", "من"]) == ["del", "del", "m/n"]   # ×3 for the text's ×2
    assert deloop(["del", "del", "m/n", "m/n"], ["دل", "دل", "من"]) is None   # every run is ≤ the text's: no guess


@pytest.mark.parametrize("word, raw, want", [
    ("پیشه", "pishe", "piše"), ("بیخودی", "bikhodi", "bixodi"), ("غوغا", "qogha", "qoqa"), ("ویژه", "vizhe", "vi;e"),
    ("اصحاب", "@/shab", "@/shab"), ("روزها", "ruzha", "ruzha"), ("رنگها", "r/ngha", "r/ngha"),   # two letters: kept
    ("اصحابش", "@/shab/š", "@/shab/š"),                                                         # both: no guess
])
def test_an_english_style_letter_pair_for_one_persian_sound_is_written_as_that_sound(word, raw, want):
    # 2026-10-29: 893 training words had "sh" for ش (پیشه = pishe), 76 "kh" for خ, 100 "gh" for غ/ق, 3 "zh" for ژ
    assert g2p({word: raw}).phonemize_line(word) == want


def test_every_lexicon_entry_writes_its_long_a():
    # the lexicon's «مهتر»: "mehtar" put a long a into all 27 training labels; it is meht/r, rarely m/ht/r
    import json
    from pathlib import Path
    from ganj_reading.g2p import long_a_written
    lex = json.loads((Path(__file__).parent.parent / "ganj_reading" / "lexicon.json").read_text(encoding="utf-8"))
    assert [f"{w}={ph}" for w, ph in lex.items() if not long_a_written(w, ph)] == []
    assert lex["مهتر"] == "meht/r"


# «و» standing alone: v/ (va) where Persian reads it so — in prose, at the start of a line, after a pause mark;
# the metre's o inside a line of verse.
def test_a_standalone_vav_is_read_va_in_prose():
    from ganj_reading.g2p import read_vav
    pw = ["do", "k/s", "r/nje1", "bihude", "bord/nd", "o", "s/@y", "bifayede", "k/rd/nd"]
    assert read_vav("دو کس رنج بیهوده بردند و سعی بی‌فایده کردند", pw, prose=True)[5] == "v/"
    assert pw[5] == "o"  # the input is not changed


def test_in_verse_a_vav_inside_the_line_stays_the_metres_o():
    from ganj_reading.g2p import read_vav
    assert read_vav("جان و دل", ["jan", "o", "del"], prose=False) == ["jan", "o", "del"]


def test_in_verse_a_vav_is_va_at_the_start_of_a_line_and_after_a_pause_mark():
    from ganj_reading.g2p import read_vav
    assert read_vav("و گر", ["o", "g/r"], prose=False) == ["v/", "g/r"]
    assert read_vav("ای دل، و جان", ["@ey", "del", "o", "jan"], prose=False) == ["@ey", "del", "v/", "jan"]


def test_a_vav_read_vo_after_a_vowel_is_va_in_prose():
    from ganj_reading.g2p import read_vav
    assert read_vav("تو و من", ["to", "vo", "m/n"], prose=True) == ["to", "v/", "m/n"]


def test_a_written_vowel_on_vav_words_containing_vav_and_a_misaligned_label_are_left_alone():
    from ganj_reading.g2p import read_vav
    assert read_vav("وُ دل", ["o", "del"], prose=True) == ["o", "del"]
    assert read_vav("وین دل", ["vin", "del"], prose=True) == ["vin", "del"]
    assert read_vav("بردند و سعی", ["bord/nd", "o"], prose=True) == ["bord/nd", "o"]


def test_punctuation_after_words_lives_with_the_reading_rules():
    from ganj_reading.g2p import punctuation_after_words
    assert punctuation_after_words("ای دل، و جان.") == ["", "،", "", "."]


def test_the_model_reads_a_line_as_its_words_without_vowel_marks_with_a_written_ezafe_kept_on_the_word():
    # one definition of the model's input, shared by the G2P and the data a fine-tune of its model is built from
    from ganj_reading.g2p import model_words
    words, plain = model_words("خانهٔ دلِ ما، رحمةٌ")
    assert words == ["خانهٔ", "دلِ", "ما", "رحمةٌ"]
    assert plain == ["خانه", "دل", "ما", "رحمت"]


def test_zhe_is_written_as_the_voices_phoneme_in_the_shadda_and_sukun_rules():
    from ganj_reading.g2p import geminate, harakat_agrees
    assert geminate("مژّه", "mo;e") == "mo;;e"
    assert harakat_agrees("مُژْده", "mo;de")
    assert not harakat_agrees("مُژْده", "mo;ode")


def test_the_shipped_lexicon_reads_ze_qabul_joined_as_zeq_bul():
    # Ruled 2026-11-07: «ما را زقبول و ردّ کس باکی نیست» is ze qabul; the models wrote zoqbul and z/qbul
    from pathlib import Path
    from ganj_reading import g2p as G
    lexicon = G.load_lexicon(Path(G.__file__).parent / "lexicon.json")
    assert g2p({"ما را زقبول و": "ma ra zoqbul o"}, lexicon=lexicon).phonemize_line("ما را زقبول و") == "ma ra zeq/bul o"


def test_the_shipped_lexicon_reads_bedravi_as_you_will_reap():
    # Ruled 2026-11-07: «هر چه کنی کشت، همان بدروی» is bedravi; the fine-tuned model wrote bedruy (all 8 uses of the
    # written word in the poems are "you will reap")
    from pathlib import Path
    from ganj_reading import g2p as G
    lexicon = G.load_lexicon(Path(G.__file__).parent / "lexicon.json")
    assert g2p({"همان بدروی": "h/man bedruy"}, lexicon=lexicon).phonemize_line("همان بدروی") == "h/man bedr/vi"


def test_the_shipped_lexicon_reads_bebod_as_was():
    # Ruled 2026-11-09: «ببد شاه یک ماه در نیمروز» is bebod; every fine-tune wrote beb/d
    from pathlib import Path
    from ganj_reading import g2p as G
    lexicon = G.load_lexicon(Path(G.__file__).parent / "lexicon.json")
    assert g2p({"ببد شاه": "beb/d $ah"}, lexicon=lexicon).phonemize_line("ببد شاه") == "bebod šah"


def test_a_word_written_with_ye_ast_is_read_ist():
    # Ruled 2026-11-09: «که هر خاری به تسبیحش زبانی‌است» is z/banist; the models wrote z/bani/st
    assert g2p({"زبانی‌است": "z/bani/st"}).phonemize_line("زبانی‌است") == "z/banist"
    assert g2p({"تسبیح‌خوانی‌است": "t/sbihxani@/st"}).phonemize_line("تسبیح‌خوانی‌است") == "t/sbihxanist"


def test_ye_ast_after_a_one_letter_stem_keeps_its_ast():
    # نی‌است, کی‌است: ney / key + ast; nist would be "is not"
    assert g2p({"نی‌است": "ni/st"}).phonemize_line("نی‌است") == "ni/st"
    assert g2p({"کی‌است": "key/st"}).phonemize_line("کی‌است") == "key/st"


HAZAJ = "مفاعیلن مفاعیلن فعولن (هزج مسدس محذوف یا وزن دوبیتی)"
MOJTASS = "مفاعلن فعلاتن مفاعلن فعلن (مجتث مثمن مخبون محذوف)"


def test_ye_ast_is_read_ist_in_a_metre_and_keeps_its_ast_where_only_that_scans():
    line = "که هر خاری به تسبیحش زبانی‌است"
    g = g2p({line: ["ke h/r xari be t/sbih/$ z/bani/st"]})
    assert g.phonemize_line(line, metre=HAZAJ) == "ke h/r xari be t/sbih/š z/banist"
    line = "زبانی‌است به مرا که می زرا به کما"
    g = g2p({line: ["z/bani/st be mara ke mi z/ra be k/ma"]})
    assert g.phonemize_line(line, metre=MOJTASS) == "z/bani/st be mara ke mi z/ra be k/ma"


def test_the_shipped_lexicon_reads_the_marked_mohdas_as_mohdas():
    # «تا کی ز قدیم و مُحدَث»: mohd/s, for the metre; the model wrote moh/dd/s
    from pathlib import Path
    from ganj_reading import g2p as G
    lexicon = G.load_lexicon(Path(G.__file__).parent / "lexicon.json")
    assert g2p({"جهان چه محدث": "j/han ce moh/dd/s"}, lexicon=lexicon).phonemize_line("جهان چه مُحدَث") == "j/han ce mohd/s"
    # only the marked spelling: an unmarked محدث stays the model's (mohaddes, the hadith narrator)
    assert g2p({"جهان چه محدث": "j/han ce moh/ddes"}, lexicon=lexicon).phonemize_line("جهان چه محدث") == "j/han ce moh/ddes"


def test_no_ezafe_right_before_a_typed_pause_mark():
    # a word before ، ؛ : ! ؟ ends a phrase, so it takes no ezafe
    assert g2p({"به خواب دو چشم": "be xabe1 do c/šm"}).phonemize_line("به خواب، دو چشم") == "be xab do c/šm"
    assert g2p({"نوشت این مرد آزادی": "nevešt @in m/rde1 @azadi"}).phonemize_line("نوشت این مرد: آزادی") == \
        "nevešt @in m/rd @azadi"
    assert g2p({"حیف استاد": "heyfe1 @ostad"}).phonemize_line("حیف! اُستاد") == "heyf @ostad"
    assert g2p({"کار این کار": "kare1 @in kar"}).phonemize_line("کار ، این کار") == "kar @in kar"  # typed apart
    assert g2p({"خواب دو چشم": "xabe1 do c/šm"}).phonemize_line("خواب،دو چشم") == "xab do c/šm"  # typed together
    assert g2p({"ماه روزه": "mahe1 ruze"}).phonemize_line("ماه روزه") == "mahe1 ruze"  # no mark: the ezafe stays


def test_a_mark_typed_at_the_start_of_the_next_word_belongs_to_the_word_before_it():
    assert g2p({"گفت هر کس": "gofte1 h/r k/s"}).phonemize_line("گفت !هر کس") == "goft h/r k/s"
    assert g2p({"گفت هر کس": "gofte1 hore1 k/s"}).phonemize_line("گفت !هر کس") == "goft hore1 k/s"


def test_a_written_kasra_keeps_its_ezafe_before_a_pause_mark():
    assert g2p({"دل من": "dele1 m/n"}).phonemize_line("دلِ، من") == "dele1 m/n"


def test_the_marked_mohdas_before_a_comma_is_read_without_ezafe():
    # 145660-c1: the lexicon keeps the model's ezafe; the pause mark then drops it
    from pathlib import Path
    from ganj_reading import g2p as G
    lexicon = G.load_lexicon(Path(G.__file__).parent / "lexicon.json")
    line = "تا کی ز قدیم و مُحدَث، اومیدم و بیم"
    g = g2p({"تا کی ز قدیم و محدث اومیدم و بیم": "ta key ze q/dim o mohd/se1 @umid/m o bim"}, lexicon=lexicon)
    assert g.phonemize_line(line) == "ta key ze q/dim o mohd/s @umid/m o bim"


def test_harakat_agrees_with_the_diphthongs_and_long_vowels_the_marks_write():
    from ganj_reading.g2p import harakat_agrees
    assert harakat_agrees("مَی", "mey") and harakat_agrees("هَیجا", "heyja") and harakat_agrees("حَیوان", "heyvan")
    assert harakat_agrees("دَولت", "dol/t")                     # the ow of a bare و is o
    assert harakat_agrees("کَندَه", "k/nde")                    # fatha before a final ه: its e
    assert harakat_agrees("کَرِیم", "k/rim") and harakat_agrees("نُور", "nur") and harakat_agrees("لَدَی", "l/da")
    assert not harakat_agrees("بَیان", "beyan") and harakat_agrees("بَیان", "b/yan")   # ی before ا is a consonant
    assert not harakat_agrees("مِی", "mi") and harakat_agrees("مِی", "mey")          # the kasra is written


def test_follow_marks_swaps_the_vowel_after_a_marked_letter():
    # Ruled 2026-11-11: when no candidate reads the marks, the chosen one is made to
    from ganj_reading.g2p import follow_marks
    assert follow_marks("انگِشت", "@/ngošt") == "@/ngešt"
    assert follow_marks("گُرسَنه", "gorsene") == "gors/ne"
    assert follow_marks("بُد", "b/d") == "bod"
    assert follow_marks("مَلِکِ", "molke1") == "m/leke1"        # the ezafe stays
    assert follow_marks("شَنگُل", "š/ng/l") == "š/ngol"
    assert follow_marks("شَمیرانُ", "š/miran") == "š/mirano"     # a missing vowel is added
    assert follow_marks("دَور", "dur") == "dor"


def test_follow_marks_drops_the_vowel_after_a_sukun_and_shortens_an_unwritten_long_vowel():
    from ganj_reading.g2p import follow_marks
    assert follow_marks("خِفتانْشْ", "x/ftan/š") == "xeftanš"
    assert follow_marks("نتْوان", "n/t/van") == "n/tvan"
    assert follow_marks("بِپْسَندی", "bepsandi") == "beps/ndi"
    assert follow_marks("بِرَفت", "beraft") == "ber/ft"
    assert follow_marks("مُلْکاً", "m/lek/n") == "molk/n"        # with its tanwin


def test_follow_marks_reads_mey_and_va():
    from ganj_reading.g2p import follow_marks
    assert follow_marks("مِی", "mi") == "mey"
    assert follow_marks("مِی‌پرستی", "mip/r/sti") == "meyp/r/sti"
    assert follow_marks("وَ", "o") == "v/"


def test_follow_marks_leaves_arabic_long_vowels_and_what_it_cannot_place():
    from ganj_reading.g2p import follow_marks
    assert follow_marks("لَدَی", "l/da") == "l/da" and follow_marks("إِلَی", "@ela") == "@ela"   # the final ی read a
    assert follow_marks("فِی", "fi") == "fi" and follow_marks("کِی", "ki") == "ki"
    assert follow_marks("کَندَه", "k/nde") == "k/nde"
    assert follow_marks("بِشَوی", "bešuy") == "bešuy"            # و ی inside a word cannot be placed


def test_follow_marks_reads_a_final_mark_unless_it_runs_into_the_next_word():
    from ganj_reading.g2p import follow_marks
    assert follow_marks("جَرَّبَ", "jorrbo") == "j/rr/b/"        # an Arabic case ending
    assert follow_marks("خَطُّ", "x/tt", last_vowel=False) == "x/tt"   # «خَطُّ و خال»: the و's o
    assert follow_marks("گُلَ", "gol", last_vowel=False) == "gol"      # «گُلَ انْدَر»: run into the alef


def test_a_line_whose_candidates_all_misread_a_mark_follows_the_mark():
    g = g2p({"فروزنده اخگر چون انگشت کرد": "foruz/nde @/xg/r con @/ngo$t k/rd"})
    assert g.phonemize_line("فروزنده اخگر چون انگِشت کرد") == "foruz/nde @/xg/r con @/ngešt k/rd"
    assert g2p({"گرسنه خفتد": "gorsene xoft/d"}).phonemize_line("گُرسَنه خفتَد") == "gors/ne xoft/d"


def test_a_final_mark_before_vav_or_an_alef_is_left_to_the_next_word():
    assert g2p({"خط و خال": "x/tt o xal"}).phonemize_line("خَطُّ و خال") == "x/tt o xal"
    assert g2p({"گل از دست": "gol @/z d/st"}).phonemize_line("گُلَ ازْ دست") == "gol @/z d/st"

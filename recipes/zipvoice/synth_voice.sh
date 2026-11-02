#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Synthesise with the shipped ZipVoice voice (recipes/zipvoice/voice_d4h.json): one wav per line of a test list.
# Usage: recipes/zipvoice/synth_voice.sh LIST.tsv OUT_DIR [TEXT.jsonl]
#   LIST: key<TAB>prompt phonemes<TAB>prompt wav<TAB>target phonemes ($GANJ_DATA/models/ganj-zip-d4h/example.tsv);
#         a relative prompt wav path is taken from the voice dir
#   TEXT: one JSON per line, {"key", "lines": [hemistichs], "phonemes"} (example.jsonl next to it; the eval set's format)
# The voice dir (the released model folder) holds the model, model.json, tokens.txt, prompt.wav and ezafe.joblib (the
# checker the draws are chosen by; EZAFE_MODEL overrides it) and, when there, vowels.json (the short-vowel check the
# draws are also chosen by, tools/pick_draws.py fit-vowels; VOWEL_MODEL overrides it, VOWEL_MODEL=none switches it off).
# DEESS=DB turns the hiss of س/ش frames down after picking (packages/ganj-voice/ganj_voice/deess.py; default 6, DEESS=0 = off).
# Recipe: 64 steps with every step from flow time 0.8 on split in four (the laser), guidance 2.0 only for t <= 0.6,
# mel despiked 3x1 before Vocos, speed 1.004, 70 Hz high-pass (the vocoder's hum lines). With TEXT, each line is drawn
# DRAWS times (default 8) and the draw that says every word and ezafe of its label is kept (tools/pick_draws.py), then
# every stop inside a hemistich is cut to at most TRIM s (default 0.30; tools/trim_pauses.py). Without TEXT: one draw,
# no trim. DRAWS=1 / TRIM=0 switch those off. A «و» after a consonant-final word is glued onto it in what the model
# reads, and a pause voice gets a long pause before each line (GLUE=0: neither). tools/reading_list.py makes LIST and TEXT from plain text.
# VOICE_DIR (a voice folder, relative or absolute) and SPEED override the voice. A voice whose tokens.txt has the pause
# tokens reads the pause marks tools/reading_list.py puts in the list.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
V=$(realpath "${VOICE_DIR:-$GANJ_DATA/models/ganj-zip-d4h}"); LIST=$(realpath "$1"); mkdir -p "$2"; OUT=$(realpath "$2")
TEXT=${3:+$(realpath "$3")}; DRAWS=${DRAWS:-8}; TRIM=${TRIM:-0.30}; DEESS=${DEESS:-6}
[ -n "$TEXT" ] || { DRAWS=1; TRIM=0; echo "no TEXT.jsonl: one draw per line, stops not trimmed"; }
# The rules every generation carries: a list built from stale labels or one that does not read a mark
# written in the text stops here (tools/check_rules.py; CURRENT= the eval labels the list was built from, default
# check_rules.py's); SKIP_RULES=1 only to reproduce an old run.
[ "${SKIP_RULES:-0}" = 1 ] || ${GANJ_PY:-python} $REPO/tools/check_rules.py list "$LIST" ${CURRENT:+--current "$CURRENT"}
EZ=${EZAFE_MODEL:-$V/ezafe.joblib}
[ "$DRAWS" -le 1 ] || [ -f "$EZ" ] || { echo "no ezafe checker at $EZ (set EZAFE_MODEL, or DRAWS=1)"; exit 1; }
VW=${VOWEL_MODEL:-$V/vowels.json}; VOWELS=(); [ ! -f "$VW" ] || VOWELS=(--vowels "$VW")
W=$OUT/.work; rm -rf "$W"; mkdir -p "$W"
# each line DRAWS times as KEY_d<k> (one draw: the key unchanged); relative prompt wavs taken from the voice dir
awk -F'\t' -v OFS='\t' -v v="$V" -v n="$DRAWS" '{ if ($3 !~ /^\//) $3 = v "/" $3
  if (n <= 1) { print; next }
  key = $1; for (k = 1; k <= n; k++) { $1 = key "_d" k; print } }' "$LIST" > "$W/list.tsv"
# what the model reads (packages/ganj-voice/ganj_voice/glue.py): a standalone «و» after a consonant-final word written onto it, and for
# a voice with the pause tokens a long pause before the line; GLUE=0 reads the targets as written
[ "${GLUE:-1}" = 0 ] || (cd $REPO && ${GANJ_PY:-python} -m ganj_voice.glue "$W/list.tsv" --tokens "$V/tokens.txt")
cd $ZIPVOICE_DIR/egs/zipvoice && GUIDE_TMAX=0.6 TAIL_FROM=0.8 TAIL_REFINE=4 PYTHONPATH=../../ \
  ${ZIPVOICE_PY:-python3} $REPO/recipes/zipvoice/infer_despike.py \
  --model-name zipvoice --model-dir $V --checkpoint-name iter-40000-avg-2.pt --tokenizer simple \
  --test-list "$W/list.tsv" --res-dir "$W/raw" --guidance-scale 2.0 --num-step 64 --speed ${SPEED:-1.004}
cd $REPO
SRC=$W/raw
if [ "${HIGHPASS:-70}" != 0 ]; then
  ${GANJ_PY:-python} tools/postclean.py --paths "$SRC" "$W/pch" --highpass ${HIGHPASS:-70}
  SRC=$W/pch
fi
if [ "$DRAWS" -gt 1 ]; then
  ${GANJ_PY:-python} tools/pick_draws.py pick "$EZ" \
    "$SRC" "$W/picked" --set "$TEXT" "${VOWELS[@]}"
  SRC=$W/picked
fi
if [ "$DEESS" != 0 ]; then
  ${GANJ_PY:-python} tools/postclean.py --paths "$SRC" "$W/deess" --highpass 0 --deess "$DEESS"
  SRC=$W/deess
fi
if [ "$TRIM" != 0 ]; then
  ${GANJ_PY:-python} tools/trim_pauses.py "$SRC" "$OUT" --cap "$TRIM" --set "$TEXT"
else
  cp "$SRC"/*.wav "$OUT"/
fi
echo "SYNTH_DONE $(ls "$OUT"/*.wav | wc -l) lines -> $OUT"

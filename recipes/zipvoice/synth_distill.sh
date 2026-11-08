#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Synthesise with a distilled student of the voice (recipes/zipvoice/distill_q9h.sh), in PyTorch, for quick comparisons:
# synth_voice.sh's chain (rules check, glue and lead pause, despike before Vocos, 70 Hz high-pass, de-ess, pause trim,
# best of DRAWS with TEXT) with --model-name zipvoice_distill, NUM_STEP (default 8) Euler steps and GUIDANCE (default
# 2.0) given to the student as its input, and none of the teacher-side sampler patches (GUIDE_TMAX, TAIL_REFINE): the
# student learned them. The phone chain (sherpa-onnx) is judged separately.
# Usage: recipes/zipvoice/synth_distill.sh LIST.tsv OUT_DIR [TEXT.jsonl]; VOICE_DIR = the student's folder (CKPT, default
# iter-<n>-avg-2.pt found there, model.json, tokens.txt, prompt.wav, ezafe.joblib, vowels.json); T_SHIFT default 0.5;
# SEED (default ZipVoice's 666) draws other noise, for the spread of one take.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
: "${VOICE_DIR:?set VOICE_DIR to the student folder}"
V=$(realpath "$VOICE_DIR"); LIST=$(realpath "$1"); mkdir -p "$2"; OUT=$(realpath "$2")
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
CKPT=${CKPT:-$(cd "$V" && ls iter-*-avg-*.pt | head -1)}
cd $ZIPVOICE_DIR/egs/zipvoice && PYTHONPATH=../../ \
  ${ZIPVOICE_PY:-python3} $REPO/recipes/zipvoice/infer_despike.py \
  --model-name zipvoice_distill --model-dir $V --checkpoint-name "$CKPT" --tokenizer simple \
  --test-list "$W/list.tsv" --res-dir "$W/raw" --guidance-scale ${GUIDANCE:-2.0} --num-step ${NUM_STEP:-8} --t-shift ${T_SHIFT:-0.5} --speed ${SPEED:-1.004} \
  ${SEED:+--seed $SEED}
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

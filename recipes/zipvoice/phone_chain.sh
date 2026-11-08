#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# The 50 eval lines through the phone chain, scored like the PyTorch runs: the rules check and glue of synth_distill.sh,
# sherpa-onnx (tools/sherpa_read.py, run by SHERPA_PY in an environment with sherpa-onnx 2.0.1) on a phone folder of recipes/zipvoice/export_distill.sh, the
# app's clean-up (ganj_voice.phonepost), then ganj_eval.run.
# Usage: recipes/zipvoice/phone_chain.sh PHONE_DIR TAG
#   INT8=1  STEPS / GUIDANCE (default phone.json's)  SKIP (clean-up steps to leave out, e.g. deess,trim)  THREADS (4)
#   VOICE_DIR (default ganj-zip-d4h, whose tokens.txt the glue reads)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
P=$(realpath "$1"); TAG=$2; V=$(realpath "${VOICE_DIR:-$GANJ_DATA/models/ganj-zip-d4h}")
D=$GANJ_DATA/eval/distill; R=$GANJ_DATA/eval/runs; OUT=$R/$TAG; W=$OUT/.work; PY=${GANJ_PY:-python}
cd $REPO
$PY tools/check_rules.py list $D/list50_d4h.tsv --current $GANJ_DATA/eval/set_q_q9g.jsonl
rm -rf "$W"; mkdir -p "$W"; cp $D/list50_d4h.tsv $W/list.tsv
$PY -m ganj_voice.glue $W/list.tsv --tokens $V/tokens.txt
${SHERPA_PY:-python} tools/sherpa_read.py "$P" $W/list.tsv $W/raw ${INT8:+--int8} \
  ${STEPS:+--steps $STEPS} ${GUIDANCE:+--guidance $GUIDANCE} --threads ${THREADS:-4}
$PY -m ganj_voice.phonepost $W/raw "$OUT" --phone "$P/phone.json" --set $D/text50_d4h.jsonl ${SKIP:+--skip $SKIP}
$PY -m ganj_eval.run --data $GANJ_DATA --tag $TAG --wav-dir "$OUT" --device cuda --set set_q.jsonl > $GANJ_DATA/logs/eval_$TAG.log 2>&1
echo "$TAG $(sed -n 3,6p $OUT/report.md | tr '\n' ' ')"

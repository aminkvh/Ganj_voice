#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# N noise draws of the same lines (ZipVoice seeds once and takes each line's noise from the same stream, so a line
# repeated N times in one list gets N different draws; the k-th copy of KEY is written as KEY_d<k>.wav). The voice and
# recipe are the newer sampler settings (ft_mana, f5a = د۱, guidance 2.0 only for t <= 0.6, tail x4 from t 0.8, despike 3x1, speed
# 1.004), post-cleaned (70 Hz). Is a dropped ezafe or word a property of the line or of one draw?
# Usage: KEYS=22672-3,18463-1 N=8 OUT=$GANJ_DATA/eval/runs/draws recipes/zipvoice/draws.sh   (only when the GPU is free)
# KEYS=all takes every line of the list; X / TEST / SPEED / GUIDE override the model dir, list, speed and guidance scale (the pause-token model:
# X=$GANJ_DATA/zipvoice_runs/zipvoice_q_ft_q9p TEST=$GANJ_DATA/corpus_q/prompts/test_f5as_q9p.tsv, rule pauses).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
X=${X:-$GANJ_DATA/zipvoice_runs/zipvoice_q_ft_mana}; TEST=${TEST:-$GANJ_DATA/corpus_q/prompts/test_f5a_q9pn.tsv}
mkdir -p "$OUT"; LIST=$OUT/draws.tsv; : > "$LIST"
[ "$KEYS" = all ] && KEYS=$(cut -f1 "$TEST" | paste -sd,)
for key in ${KEYS//,/ }; do
  row=$(grep -P "^$key\t" "$TEST") || { echo "no $key in $TEST"; exit 1; }
  for k in $(seq 1 "$N"); do printf '%s_d%s\t%s\n' "$key" "$k" "$(cut -f2- <<< "$row")" >> "$LIST"; done
done
# what the model reads, as synth_voice.sh does: a «و» after a consonant-final word glued onto it, and for a model with
# the pause tokens a long pause before the line (GLUE=0: the targets as written, as in the runs before 2026-11-01)
[ "${GLUE:-1}" = 0 ] || (cd $REPO && ${GANJ_PY:-python} -m ganj_voice.glue "$LIST" --tokens "$X/tokens.txt")
(cd $ZIPVOICE_DIR/egs/zipvoice && GUIDE_TMAX=0.6 TAIL_FROM=0.8 TAIL_REFINE=4 DESPIKE_TIME=0 PYTHONPATH=../../ \
  ${ZIPVOICE_PY:-python3} $REPO/recipes/zipvoice/infer_despike.py \
  --model-name zipvoice --model-dir $X --checkpoint-name iter-40000-avg-2.pt --tokenizer simple --test-list "$LIST" \
  --res-dir "$OUT/raw" --guidance-scale ${GUIDE:-2.0} --num-step 64 --speed ${SPEED:-1.004} > "$OUT/infer.log" 2>&1)
grep -m1 "refined sampler" "$OUT/infer.log"
${GANJ_PY:-python} $REPO/tools/postclean.py --paths "$OUT/raw" "$OUT/pch"
echo "DRAWS_DONE $(date -Is)"

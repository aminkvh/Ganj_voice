#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Follow-up to regen_partial.sh: the laser forms in the last 7 steps (t 0.80 -> 1). Step size or the model itself? The same three
# lines noised to the same t = 0.803 and finished in 28 steps (each shipped step split in four) and in one jump
# (x1 = x_t + (1 - t) v); the shipped 7 steps are regen_partial.sh's t080. Same model and recipe. recipes/zipvoice/regen_partial.py.
# Usage: recipes/zipvoice/regen_steps.sh   (only when the GPU is free)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; X=$GANJ_DATA/zipvoice_runs/zipvoice_q_ft_mana; R=$GANJ_DATA/eval/runs/regen-ft_mana
(cd $E && PYTHONPATH=../../ ${ZIPVOICE_PY:-python3} $REPO/recipes/zipvoice/regen_partial.py \
  --model-dir $X --checkpoint-name iter-40000-avg-2.pt --test-list $GANJ_DATA/corpus_q/prompts/test_f5a_q9pn.tsv \
  --set $GANJ_DATA/eval/set_q.jsonl --data $GANJ_DATA --keys 22672-3,18463-1,9466-0 --t-starts 0.8 --refines 4,0 \
  --res-dir $R)
for d in $R/t080r4 $R/t080r0; do
  ${GANJ_PY:-python} $REPO/tools/postclean.py --paths $d $d-pch
done
echo "REGEN_SH_DONE $(date -Is)"

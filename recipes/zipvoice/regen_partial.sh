#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Where in the flow does the laser form? Three held-out lines regenerated from their own real mel by
# the shipped model and recipe (ft_mana, 64 steps, guidance 2.0, despike, 70 Hz high-pass): T = 1 copy-synthesis,
# 0.8 (last 7 steps), 0.5 (last 21), 0 (all 64, the reciter's voice and timing). recipes/zipvoice/regen_partial.py.
# Usage: recipes/zipvoice/regen_partial.sh   (only when the GPU is free)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; X=$GANJ_DATA/zipvoice_runs/zipvoice_q_ft_mana; R=$GANJ_DATA/eval/runs/regen-ft_mana
(cd $E && PYTHONPATH=../../ ${ZIPVOICE_PY:-python3} $REPO/recipes/zipvoice/regen_partial.py \
  --model-dir $X --checkpoint-name iter-40000-avg-2.pt --test-list $GANJ_DATA/corpus_q/prompts/test_f5a_q9pn.tsv \
  --set $GANJ_DATA/eval/set_q.jsonl --data $GANJ_DATA --keys 22672-3,18463-1,9466-0 --t-starts 1.0,0.8,0.5,0.0 \
  --res-dir $R)
for d in $R/t*; do
  ${GANJ_PY:-python} $REPO/tools/postclean.py --paths $d $d-pch
done
echo "REGEN_SH_DONE $(date -Is)"

#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Vocos fine-tune pairs from ft_mana's own output (recipes/zipvoice/regen_pairs.py): N training cuts of its manifest,
# regenerated from T ~ U[0.5, 0.9] on the shipped schedule; held out: the eval set and the voice prompts' clips.
# Usage: N=10000 OUT=$GANJ_DATA/vocos_ft/pairs recipes/zipvoice/regen_pairs.sh   (only when the GPU is free; resumable)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; X=$GANJ_DATA/zipvoice_runs/zipvoice_q_ft_mana
(cd $E && PYTHONPATH=../../ ${ZIPVOICE_PY:-python3} $REPO/recipes/zipvoice/regen_pairs.py \
  --model-dir $X --checkpoint-name iter-40000-avg-2.pt --manifest $E/data_q/fbank/custom_cuts_train_mix.jsonl.gz \
  --set $GANJ_DATA/eval/set_q.jsonl --exclude mana-83-102,18347-15 --n ${N:-10000} --max-frames ${MAX_FRAMES:-8000} --out $OUT)
echo "REGEN_PAIRS_SH_DONE $(date -Is)"

#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Finish a fine-tune whose wrapper died after training: average, mark FT_TRAIN_DONE, score A and C, mark FT_CLEAN_DONE.
# Usage: recipes/zipvoice/finish_ft.sh <exp dir> <label suffix ('' or _q9b)>   (appends to that run's log via stdout)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
X=$1; LBL=${2:-}; E=$ZIPVOICE_DIR/egs/zipvoice
LAST=$(ls $X/checkpoint-*.pt | sed 's/.*checkpoint-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
(cd $E && PYTHONPATH=../../ ${ZIPVOICE_PY:-python3} -m zipvoice.bin.generate_averaged_model --iter $LAST --avg 2 --model-name zipvoice --exp-dir $X)
echo "FT_TRAIN_DONE $(date -Is) iter=$LAST"
cd $REPO
for p in p3_a08 u0r; do X=$X TEST=$GANJ_DATA/corpus_q/prompts/test_$p$LBL.tsv CKPT=iter-$LAST-avg-2.pt SUF=ft$LBL-$p- bash recipes/zipvoice/sweep_q.sh g2.0s32; done
echo FT_CLEAN_DONE

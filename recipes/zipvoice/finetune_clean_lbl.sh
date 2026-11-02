#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Quality fine-tune with a label set: ZipVoice-Q 200k -> clean subset (no Demucs, bandwidth >= 11 kHz, align >= 0.8; 152 h),
# ZipVoice fine-tune mode (fixed LR 1e-4, no warm-up dropout), own exp dir; then average and score prompts A (p3_a08) and C (u0r).
# LBL=_q9b (or _q9c, ...) trains and scores on that label set (same recipe otherwise, so the two runs differ only in labels).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; DQ=$E/data_q; SRC=$GANJ_DATA/zipvoice_runs/zipvoice_q; LBL=${LBL:-}; X=$GANJ_DATA/zipvoice_runs/zipvoice_q_ft_clean$LBL
ITERS=${FT_ITERS:-40000}; mkdir -p $X; cp $SRC/model.json $SRC/tokens.txt $X/
cd $E; export PYTHONPATH=../../:${PYTHONPATH:-}
${ZIPVOICE_PY:-python3} -m zipvoice.bin.train_zipvoice --world-size 1 --use-fp16 1 --num-iters $ITERS --max-duration ${ZIP_MAXDUR:-125} \
  --finetune 1 --base-lr ${FT_LR:-0.0001} --checkpoint $SRC/iter-200000-avg-2.pt --max-len 12 --model-config conf/zipvoice_base.json \
  --tokenizer simple --token-file $DQ/tokens_custom.txt --dataset custom --train-manifest $DQ/fbank/custom_cuts_train_clean$LBL.jsonl.gz \
  --dev-manifest $DQ/fbank/custom_cuts_dev$LBL.jsonl.gz --exp-dir $X
LAST=$(ls $X/checkpoint-*.pt | sed 's/.*checkpoint-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
${ZIPVOICE_PY:-python3} -m zipvoice.bin.generate_averaged_model --iter $LAST --avg 2 --model-name zipvoice --exp-dir $X
echo "FT_TRAIN_DONE $(date -Is) iter=$LAST"
cd $REPO
for p in p3_a08 u0r; do X=$X TEST=$GANJ_DATA/corpus_q/prompts/test_$p$LBL.tsv CKPT=iter-$LAST-avg-2.pt SUF=ft$LBL-$p- bash recipes/zipvoice/sweep_q.sh g2.0s32; done
echo FT_CLEAN_DONE

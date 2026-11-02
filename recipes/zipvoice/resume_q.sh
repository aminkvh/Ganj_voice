#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Continue ZipVoice-Q from the last saved epoch to ZIP_ITERS total iters, average, and score the frozen voice recipe
# (recipes/zipvoice/voice_q.json prompt, g2.0 s32) under the tag zip-q-q2-g2.0s32.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; DQ=$E/data_q; X=$GANJ_DATA/zipvoice_runs/zipvoice_q
H=$(sed 's/TRAIN_HOURS //' $DQ/hours.txt); LRH=$(python3 -c "print(round(1000 * ($H ** 0.3)))")
LASTEP=$(ls $X/epoch-*.pt | sed 's/.*epoch-\([0-9]*\)\.pt/\1/' | sort -n | tail -1); ITERS=${ZIP_ITERS:-200000}
cd $E; export PYTHONPATH=../../:${PYTHONPATH:-}
${ZIPVOICE_PY:-python3} -m zipvoice.bin.train_zipvoice --world-size 1 --use-fp16 1 --num-iters $ITERS --max-duration ${ZIP_MAXDUR:-125} \
  --lr-hours $LRH --max-len 12 --model-config conf/zipvoice_base.json --tokenizer simple --token-file $DQ/tokens_custom.txt \
  --dataset custom --train-manifest $DQ/fbank/custom_cuts_train.jsonl.gz --dev-manifest $DQ/fbank/custom_cuts_dev.jsonl.gz --exp-dir $X --start-epoch $((LASTEP + 1))
LAST=$(ls $X/checkpoint-*.pt | sed 's/.*checkpoint-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
${ZIPVOICE_PY:-python3} -m zipvoice.bin.generate_averaged_model --iter $LAST --avg 2 --model-name zipvoice --exp-dir $X
cd $REPO
TEST=$GANJ_DATA/corpus_q/prompts/test_p3_a08.tsv CKPT=iter-$LAST-avg-2.pt SUF=q2- bash recipes/zipvoice/sweep_q.sh "g2.0s32"
echo "ZIP_RESUME_DONE $(date -Is) iter=$LAST"

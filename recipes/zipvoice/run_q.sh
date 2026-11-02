#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# ZipVoice from scratch on corpus_q (after prep_q.sh). ~3,000 iters/h at max-duration 125 on the 4060 Ti (50 h
# trial: 20k iters in 6.7 h), so 120k iters ≈ 40 h ≈ 10 epochs at 400 h. Then average and synthesise the eval set.
set -euo pipefail
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; DQ=${DQ:-$E/data_q}; X=${X:-$GANJ_DATA/zipvoice_runs/zipvoice_q}; mkdir -p $X
cd $E; export PYTHONPATH=../../:${PYTHONPATH:-}
H=$(sed 's/TRAIN_HOURS //' $DQ/hours.txt); LRH=$(python3 -c "print(round(1000 * ($H ** 0.3)))")
ITERS=${ZIP_ITERS:-120000}
${ZIPVOICE_PY:-python3} -m zipvoice.bin.train_zipvoice --world-size 1 --use-fp16 1 --num-iters $ITERS --max-duration ${ZIP_MAXDUR:-125} \
  --lr-hours $LRH --max-len 12 --model-config conf/zipvoice_base.json --tokenizer simple --token-file $DQ/tokens_custom.txt \
  --dataset custom --train-manifest $DQ/fbank/custom_cuts_train.jsonl.gz --dev-manifest ${DEV:-$DQ/fbank/custom_cuts_dev.jsonl.gz} --exp-dir $X
LAST=$(ls $X/checkpoint-*.pt | sed 's/.*checkpoint-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
${ZIPVOICE_PY:-python3} -m zipvoice.bin.generate_averaged_model --iter $LAST --avg 2 --model-name zipvoice --exp-dir $X
${ZIPVOICE_PY:-python3} -m zipvoice.bin.infer_zipvoice --model-name zipvoice --model-dir $X \
  --checkpoint-name iter-$LAST-avg-2.pt --tokenizer simple --test-list $DQ/test_q.tsv --res-dir $X/results_q
echo "ZIP_Q_DONE $(date -Is) iter=$LAST"

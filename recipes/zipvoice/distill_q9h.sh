#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Distil the released voice's model (zipvoice_q_ft_q9h, the ganj-zip-d4h voice) into a few-step student for on-device
# reading, through recipes/zipvoice/train_distill_ganj.py (the teacher guides only t <= 0.6, finer teacher jumps from
# t 0.8, stage 1 guidance in [0, 3)). Same q9h manifests and tokens as the fine-tune.
#   STAGE=first|second (default first)  ITERS (default 60000; 2000 for second)  TEACHER (stage 2: the stage-1 average)
#   X (exp dir; default zipvoice_q_distill_1 / _2)  LR (default 0.0005 / 0.0001)  ZIP_MAXDUR (default 125)
# Ends with the averaged checkpoint iter-<last>-avg-<AVG>.pt in X (AVG default 2; stage 2 saves only 1000 and 2000, so AVG=1).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; DQ=$E/data_q; F=$DQ/fbank; R=$GANJ_DATA/zipvoice_runs
STAGE=${STAGE:-first}
if [ "$STAGE" = first ]; then
  ITERS=${ITERS:-60000}; LR=${LR:-0.0005}; X=${X:-$R/zipvoice_q_distill_1}; TEACHER=${TEACHER:-$R/zipvoice_q_ft_q9h/iter-40000-avg-2.pt}
  SAVE=(--save-every-n ${SAVE_N:-5000})
else
  ITERS=${ITERS:-2000}; LR=${LR:-0.0001}; X=${X:-$R/zipvoice_q_distill_2}; : "${TEACHER:?stage 2 needs TEACHER=the stage-1 average}"
  SAVE=(--save-every-n 1000)
fi
mkdir -p $X; cp $R/zipvoice_q_ft_q9h/model.json $X/; cp $DQ/tokens_custom_q9h.txt $X/tokens.txt
cd $E; export PYTHONPATH=../../:${PYTHONPATH:-}
echo "DISTILL_START stage=$STAGE iters=$ITERS teacher=$TEACHER $(date -Is)"
${ZIPVOICE_PY:-python3} $REPO/recipes/zipvoice/train_distill_ganj.py \
  --world-size 1 --use-fp16 1 --num-iters $ITERS --max-duration ${ZIP_MAXDUR:-125} --base-lr $LR --max-len 12 \
  --model-config conf/zipvoice_base.json --tokenizer simple --token-file $DQ/tokens_custom_q9h.txt --dataset custom \
  --train-manifest $F/custom_cuts_train_mix_q9h.jsonl.gz --dev-manifest $F/custom_cuts_dev_mix_q9h.jsonl.gz \
  --teacher-model $TEACHER --distill-stage $STAGE --exp-dir $X "${SAVE[@]}"
LAST=$( (ls $X/checkpoint-*.pt 2>/dev/null || true) | sed 's/.*checkpoint-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
if [ -n "$LAST" ]; then
  ${ZIPVOICE_PY:-python3} -m zipvoice.bin.generate_averaged_model --iter $LAST --avg ${AVG:-2} \
    --model-name zipvoice_distill --exp-dir $X
fi
echo "DISTILL_DONE stage=$STAGE iter=${LAST:-none} $(date -Is)"

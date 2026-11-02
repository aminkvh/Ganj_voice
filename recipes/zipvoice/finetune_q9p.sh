#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Pause-token fine-tune q9p: mark measured pauses in the q9e labels (tools/pause_labels.py: CTC forced
# alignment of every train/dev cut, silence >= 150 ms short, >= 450 ms long), extend the ft_mana model's token
# embedding by the two pause tokens, fine-tune 40k iters on the marked mix, then read the eval set with the د۱ prompt
# (f5a.wav) and rule-placed pauses, post-clean and score. Own exp dir zipvoice_q_ft_q9p.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; DQ=$E/data_q; F=$DQ/fbank; P=$GANJ_DATA/corpus_q/prompts
SRC=$GANJ_DATA/zipvoice_runs/zipvoice_q_ft_mana; X=$GANJ_DATA/zipvoice_runs/zipvoice_q_ft_q9p; ITERS=${FT_ITERS:-40000}
cd $REPO
if ! grep -q LABEL_DONE $GANJ_DATA/logs/pause_labels.log 2>/dev/null; then
  ${GANJ_PY:-python} tools/pause_labels.py label --device cuda >> $GANJ_DATA/logs/pause_labels.log 2>&1
fi
${GANJ_PY:-python} tools/pause_labels.py cuts
mkdir -p $X; cp $SRC/model.json $X/; cp $DQ/tokens_custom_q9p.txt $X/tokens.txt
[ -f $X/base_q9p.pt ] || ${GANJ_PY:-python} tools/pause_labels.py ckpt $SRC/iter-40000-avg-2.pt $X/base_q9p.pt
echo "LABELS_READY $(date -Is)"
cd $E; export PYTHONPATH=../../:${PYTHONPATH:-}
${ZIPVOICE_PY:-python3} -m zipvoice.bin.train_zipvoice --world-size 1 --use-fp16 1 --num-iters $ITERS --max-duration ${ZIP_MAXDUR:-125} \
  --finetune 1 --base-lr ${FT_LR:-0.0001} --checkpoint $X/base_q9p.pt --max-len 12 --model-config conf/zipvoice_base.json \
  --tokenizer simple --token-file $DQ/tokens_custom_q9p.txt --dataset custom --train-manifest $F/custom_cuts_train_mix_q9p.jsonl.gz \
  --dev-manifest $F/custom_cuts_dev_mix_q9p.jsonl.gz --exp-dir $X
LAST=$(ls $X/checkpoint-*.pt | sed 's/.*checkpoint-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
${ZIPVOICE_PY:-python3} -m zipvoice.bin.generate_averaged_model --iter $LAST --avg 2 --model-name zipvoice --exp-dir $X
echo "FT_TRAIN_DONE $(date -Is) iter=$LAST"
cd $REPO
t=ft_q9p-f5as-ds0-g2.0s64
SPEED=${Q9P_SPEED:-1.004} DESPIKE_TIME=0 X=$X TEST=$P/test_f5as_q9p.tsv CKPT=iter-$LAST-avg-2.pt SUF=ft_q9p-f5as-ds0- \
  bash recipes/zipvoice/sweep_despike_x.sh g2.0s64
${GANJ_PY:-python} tools/postclean.py zip-q-$t zip-q-$t-pch
${GANJ_PY:-python} -m ganj_eval.run --data $GANJ_DATA --tag zip-q-$t-pch \
  --wav-dir $GANJ_DATA/eval/runs/zip-q-$t-pch --device cuda --set set_q.jsonl > $GANJ_DATA/logs/eval_zip-q-$t-pch.log 2>&1
echo "$t-pch $(sed -n 3,6p $GANJ_DATA/eval/runs/zip-q-$t-pch/report.md | tr '\n' ' ')"
echo "FT_Q9P_DONE $(date -Is)"

#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Clean-texture fine-tune: the q9c model -> clean Ganjoor subset (q9c labels) + ManaTTS (CC0, clean prose,
# one reader), shuffled into one manifest; ZipVoice fine-tune mode, own exp dir; then average and score F2/F/C at 64 steps
# with the 3x1 despike (the shipped recipe). Needs $GANJ_DATA/manatts/tsv/*.tsv (ganj_corpus.manatts).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; DQ=$E/data_q; SRC=$GANJ_DATA/zipvoice_runs/zipvoice_q_ft_clean_q9c; X=$GANJ_DATA/zipvoice_runs/zipvoice_q_ft_mana
M=$GANJ_DATA/manatts; ITERS=${FT_ITERS:-40000}; mkdir -p $X $DQ/raw; cp $SRC/model.json $SRC/tokens.txt $X/
(cd $REPO && ${GANJ_PY:-python} -m ganj_corpus.manatts merge $M)
shuf --random-source=<(yes) $M/all.tsv -o $M/all.tsv
head -n 200 $M/all.tsv > $DQ/raw/custom_mana_dev.tsv; tail -n +201 $M/all.tsv > $DQ/raw/custom_mana_train.tsv
cd $E; export PYTHONPATH=../../:${PYTHONPATH:-}
for s in mana_train mana_dev; do
  [ -f $DQ/fbank/custom_cuts_$s.jsonl.gz ] && continue
  ${ZIPVOICE_PY:-python3} -m zipvoice.bin.prepare_dataset --tsv-path $DQ/raw/custom_$s.tsv --prefix custom --subset $s --num-jobs 6 --output-dir $DQ/manifests
  CUDA_VISIBLE_DEVICES='' ${ZIPVOICE_PY:-python3} -m zipvoice.bin.compute_fbank --source-dir $DQ/manifests --dest-dir $DQ/fbank --dataset custom --subset $s --num-jobs 6
done
cd $REPO
${GANJ_PY:-python} -m ganj_corpus.mix_cuts $DQ/fbank/custom_cuts_train_mix.jsonl.gz $DQ/fbank/custom_cuts_train_clean_q9c.jsonl.gz $DQ/fbank/custom_cuts_mana_train.jsonl.gz
${GANJ_PY:-python} -m ganj_corpus.mix_cuts $DQ/fbank/custom_cuts_dev_mix.jsonl.gz $DQ/fbank/custom_cuts_dev_q9c.jsonl.gz $DQ/fbank/custom_cuts_mana_dev.jsonl.gz
cd $E
${ZIPVOICE_PY:-python3} -m zipvoice.bin.train_zipvoice --world-size 1 --use-fp16 1 --num-iters $ITERS --max-duration ${ZIP_MAXDUR:-125} \
  --finetune 1 --base-lr ${FT_LR:-0.0001} --checkpoint $SRC/iter-40000-avg-2.pt --max-len 12 --model-config conf/zipvoice_base.json \
  --tokenizer simple --token-file $DQ/tokens_custom.txt --dataset custom --train-manifest $DQ/fbank/custom_cuts_train_mix.jsonl.gz \
  --dev-manifest $DQ/fbank/custom_cuts_dev_mix.jsonl.gz --exp-dir $X
LAST=$(ls $X/checkpoint-*.pt | sed 's/.*checkpoint-\([0-9]*\)\.pt/\1/' | sort -n | tail -1)
${ZIPVOICE_PY:-python3} -m zipvoice.bin.generate_averaged_model --iter $LAST --avg 2 --model-name zipvoice --exp-dir $X
echo "FT_TRAIN_DONE $(date -Is) iter=$LAST"
cd $REPO
for p in f2 b1f u0r; do DESPIKE_TIME=0 X=$X TEST=$GANJ_DATA/corpus_q/prompts/test_${p}_q9c.tsv CKPT=iter-$LAST-avg-2.pt SUF=ft_mana-$p-ds0- bash recipes/zipvoice/sweep_despike_x.sh g2.0s64; done
echo FT_MANA_DONE

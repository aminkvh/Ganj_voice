#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Relabel round q9e (with 133 per-line readings): relabel the clean Ganjoor subset, the
# dev set and ManaTTS with the current G2P, write the q9e test lists for F2/F5/F6 and the train/dev mixes. No training:
# finetune_q9p.sh trains on these labels once they are checked.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; F=$E/data_q/fbank; P=$GANJ_DATA/corpus_q/prompts
cd $REPO
${GANJ_PY:-python} -m ganj_corpus.relabel --data $GANJ_DATA --cuts-dir $F --subset clean --tag _q9e \
  --overrides $GANJ_DATA/audit_g2p/review/overrides_q9e.json --mana-raw $GANJ_DATA/manatts/raw
for p in f2 f5 f6; do awk -F'\t' -v OFS='\t' -v w="$P/$p.wav" '{$3=w;print}' $P/test_b1f_q9e.tsv > $P/test_${p}_q9e.tsv; done
${GANJ_PY:-python} -m ganj_corpus.mix_cuts $F/custom_cuts_train_mix_q9e.jsonl.gz $F/custom_cuts_train_clean_q9e.jsonl.gz $F/custom_cuts_mana_train_q9e.jsonl.gz
${GANJ_PY:-python} -m ganj_corpus.mix_cuts $F/custom_cuts_dev_mix_q9e.jsonl.gz $F/custom_cuts_dev_q9e.jsonl.gz $F/custom_cuts_mana_dev_q9e.jsonl.gz
echo "RELABEL_Q9E_DONE $(date -Is)"

#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# ZipVoice (from scratch) data prep on corpus_q, CPU only: TSVs, lhotse manifests, fbank, char tokens (the
# style and metre codepoints come through as tokens because they are in the phoneme strings), test list from set_q.
# Writes under egs/zipvoice/data_q so the 50 h trial's data stays as it was. Logs PREP_Q_DONE.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
Q=${Q:-$GANJ_DATA/corpus_q}; E=$ZIPVOICE_DIR/egs/zipvoice; DQ=${DQ:-$E/data_q}
mkdir -p $DQ/raw
cd $REPO
${GANJ_PY:-python} -m recipes.zipvoice.make_tsv $Q $DQ/raw 100000
${GANJ_PY:-python} -m recipes.zipvoice.make_test_q $GANJ_DATA/eval/set_q.jsonl $E/test.tsv $DQ/test_q.tsv
cd $E; export PYTHONPATH=../../:${PYTHONPATH:-}
for s in train dev; do
  ${ZIPVOICE_PY:-python3} -m zipvoice.bin.prepare_dataset --tsv-path $DQ/raw/custom_$s.tsv --prefix custom --subset $s --num-jobs 6 --output-dir $DQ/manifests
  CUDA_VISIBLE_DEVICES='' ${ZIPVOICE_PY:-python3} -m zipvoice.bin.compute_fbank --source-dir $DQ/manifests --dest-dir $DQ/fbank --dataset custom --subset $s --num-jobs 6
done
${ZIPVOICE_PY:-python3} ./local/prepare_token_file_char.py --manifest $DQ/manifests/custom_cuts_train.jsonl.gz --tokens $DQ/tokens_custom.txt
${ZIPVOICE_PY:-python3} -c "
from lhotse import load_manifest_lazy
d=sum(c.duration for c in load_manifest_lazy('$DQ/fbank/custom_cuts_train.jsonl.gz')); print(f'TRAIN_HOURS {d/3600:.1f}')" > $DQ/hours.txt
echo "PREP_Q_DONE $(date -Is) $(cat $DQ/hours.txt) tokens=$(wc -l < $DQ/tokens_custom.txt)"

#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Inference-only sweep on the trained ZipVoice-Q model: classifier-free guidance x sampling steps, each scored on set_q.
# Usage: recipes/zipvoice/sweep_q.sh "g1.5s16 g2.0s16 ..."  (gGs<steps>; optional TEST=<tsv> prompt list, CKPT=<name>, SPEED=<x>, SUF=<tag prefix>)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; X=${X:-$GANJ_DATA/zipvoice_runs/zipvoice_q}; TEST=${TEST:-$E/data_q/test_q.tsv}; CK=${CKPT:-iter-120000-avg-2.pt}
mkdir -p "$X/sweep"
for v in $1; do
  G=$(sed 's/g\([0-9.]*\)s.*/\1/' <<< $v); S=$(sed 's/.*s\([0-9]*\)/\1/' <<< $v); R=$X/sweep/${SUF:-}$v
  (cd $E && PYTHONPATH=../../ ${ZIPVOICE_PY:-python3} -m zipvoice.bin.infer_zipvoice --model-name zipvoice --model-dir $X \
    --checkpoint-name $CK --tokenizer simple --test-list $TEST --res-dir $R --guidance-scale $G --num-step $S --speed ${SPEED:-1.0} > $R.log 2>&1)
  (cd $REPO && ${GANJ_PY:-python} -m ganj_eval.run --data $GANJ_DATA --tag zip-q-${SUF:-}$v \
    --wav-dir $R --device cuda --set set_q.jsonl > $R.eval.log 2>&1)
  echo "$v $(sed -n 3,4p $GANJ_DATA/eval/runs/zip-q-${SUF:-}$v/report.md | tr '\n' ' ')"
done
echo SWEEP_DONE

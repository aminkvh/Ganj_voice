#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# sweep_q.sh with the mel despiked before Vocos (recipes/zipvoice/infer_despike.py; DESPIKE_START, DESPIKE_TIME), plus
# EXTRA (extra infer flags, e.g. "--t-shift 1.0").
# Inference-only sweep on the trained ZipVoice-Q model: classifier-free guidance x sampling steps, each scored on set_q.
# Usage: recipes/zipvoice/sweep_q.sh "g1.5s16 g2.0s16 ..."  (gGs<steps>; optional TEST=<tsv> prompt list, CKPT=<name>, SPEED=<x>, SUF=<tag prefix>)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
E=$ZIPVOICE_DIR/egs/zipvoice; X=${X:-$GANJ_DATA/zipvoice_runs/zipvoice_q}; TEST=${TEST:-$E/data_q/test_q.tsv}; CK=${CKPT:-iter-120000-avg-2.pt}
mkdir -p "$X/sweep"
# The rules every generation carries: a list built from stale labels or one that does not read a mark
# written in the text stops here (tools/check_rules.py; CURRENT= the eval labels the list was built from, default
# check_rules.py's); SKIP_RULES=1 only to reproduce an old run.
[ "${SKIP_RULES:-0}" = 1 ] || ${GANJ_PY:-python} $REPO/tools/check_rules.py list "$TEST" ${CURRENT:+--current "$CURRENT"}
for v in $1; do
  G=$(sed 's/g\([0-9.]*\)s.*/\1/' <<< $v); S=$(sed 's/.*s\([0-9]*\)/\1/' <<< $v); R=$X/sweep/${SUF:-}$v
  (cd $E && PYTHONPATH=../../ ${ZIPVOICE_PY:-python3} $REPO/recipes/zipvoice/infer_despike.py --model-name zipvoice --model-dir $X \
    --checkpoint-name $CK --tokenizer simple --test-list $TEST --res-dir $R --guidance-scale $G --num-step $S --speed ${SPEED:-1.0} ${EXTRA:-} > $R.log 2>&1)
  (cd $REPO && ${GANJ_PY:-python} -m ganj_eval.run --data $GANJ_DATA --tag zip-q-${SUF:-}$v \
    --wav-dir $R --device cuda --set set_q.jsonl > $R.eval.log 2>&1)
  echo "$v $(sed -n 3,4p $GANJ_DATA/eval/runs/zip-q-${SUF:-}$v/report.md | tr '\n' ' ')"
done
echo SWEEP_DONE

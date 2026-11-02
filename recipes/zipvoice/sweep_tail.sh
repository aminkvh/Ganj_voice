#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# A finer finish on full generation (TAIL_REFINE / TAIL_FROM in infer_despike.py, packages/ganj-voice/ganj_voice/schedule.py).
# The laser fades when the last 7 of the 64 steps (t 0.80 -> 1) are split in four, on a real mel noised to t 0.8. Here
# the shipped voice (ft_mana, f5a = د۱, guidance 2.0, despike 3x1, speed 1.004) from noise: tail x4 from t 0.8
# (57 + 28 steps), x4 from t 0.5 (43 + 84) and x4 from t 0.8 with guidance only for t <= 0.6 (the candidate).
# The same 50 eval lines and labels; each post-cleaned (70 Hz) and scored.
# Usage: recipes/zipvoice/sweep_tail.sh   (only when the GPU is free)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
X=$GANJ_DATA/zipvoice_runs/zipvoice_q_ft_mana; P=$GANJ_DATA/corpus_q/prompts; R=$GANJ_DATA/eval/runs
cd $REPO
for v in "tr4 0.8 4 -" "tf05tr4 0.5 4 -" "tr4-gi06 0.8 4 0.6"; do
  read -r tag from k tmax <<< "$v"
  t=ft_mana-f5a-q9pn-$tag-ds0-g2.0s64
  guide=(); [ "$tmax" = - ] || guide=(GUIDE_TMAX=$tmax)                # unguided-tail runs leave the forward unpatched
  env "${guide[@]}" TAIL_FROM=$from TAIL_REFINE=$k SPEED=1.004 DESPIKE_TIME=0 X=$X TEST=$P/test_f5a_q9pn.tsv \
    CKPT=iter-40000-avg-2.pt SUF=ft_mana-f5a-q9pn-$tag-ds0- bash recipes/zipvoice/sweep_despike_x.sh g2.0s64
  grep -m1 -h "refined sampler" $X/sweep/ft_mana-f5a-q9pn-$tag-ds0-g2.0s64.log     # stops here if the patch was not live
  ${GANJ_PY:-python} tools/postclean.py zip-q-$t zip-q-$t-pch
  ${GANJ_PY:-python} -m ganj_eval.run --data $GANJ_DATA --tag zip-q-$t-pch \
    --wav-dir $R/zip-q-$t-pch --device cuda --set set_q.jsonl > $GANJ_DATA/logs/eval_zip-q-$t-pch.log 2>&1
  echo "$t-pch $(sed -n 3,6p $R/zip-q-$t-pch/report.md | tr '\n' ' ')"
done
echo "TAIL_DONE $(date -Is)"

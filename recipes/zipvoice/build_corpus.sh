#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# After the download, build the re-cut, metre-guided corpus into $GANJ_DATA/corpus_q.
# 1. finish segmentation of the new WAVs (CPU), 2. freeze the metre list, 3. build (GPU aligner + demucs),
# 4. audit the clip edges and the scansion pass rate.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
cd $REPO
OUT=$GANJ_DATA/corpus_q
LOG=$GANJ_DATA/logs/build_q.log
step() { echo "STEP $1 $(date -Is)" >> $LOG; }

step ina
CUDA_VISIBLE_DEVICES='' ${INA_PY:-python} tools/ina_segment.py --wav-dir $GANJ_DATA/wav --out-dir $GANJ_DATA/ina >> $LOG 2>&1

step freeze_metres
${GANJ_PY:-python} -m tools.freeze_metres $GANJ_DATA/poems packages/ganj-reading/ganj_reading/metres.json >> $LOG 2>&1 || { echo "FAILED freeze_metres" >> $LOG; exit 1; }

step build
# Two shards in parallel: the build is CPU-bound (G2P, demucs, loudness), the GPU aligner is mostly idle.
NSH=${NSHARDS:-3}; PIDS=()
export OMP_NUM_THREADS=$(( $(nproc) / NSH )) MKL_NUM_THREADS=$(( $(nproc) / NSH ))  # unbounded torch threads thrash across shards
for k in $(seq 0 $((NSH - 1))); do
  ${GANJ_PY:-python} -m ganj_corpus.build --data $GANJ_DATA --out $OUT --device cuda --shard $k/$NSH > $GANJ_DATA/logs/build_q.shard$k.log 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait $p || { echo "FAILED build shard (pid $p)" >> $LOG; exit 1; }; done
step merge
${GANJ_PY:-python} -m ganj_corpus.build --data $GANJ_DATA --out $OUT --merge $NSH >> $LOG 2>&1 || { echo "FAILED merge" >> $LOG; exit 1; }

step audit
mkdir -p $OUT/audit
GANJ_DATA=$OUT ${GANJ_PY:-python} tools/boundary_audit.py $OUT/audit/boundaries.jsonl 0 >> $LOG 2>&1
${GANJ_PY:-python} -m tools.scansion_report $OUT/clips.jsonl $OUT/audit/scansion.json >> $LOG 2>&1
echo "BUILD_Q_DONE $(date -Is)" >> $LOG

#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# Vocos fine-tune on ft_mana's own regenerated mels (recipes/vocos/finetune.py; pairs from recipes/zipvoice/regen_pairs.sh).
# Usage: PAIRS=$GANJ_DATA/vocos_ft/pairs OUT=$GANJ_DATA/vocos_ft/run1 recipes/vocos/finetune.sh [extra finetune.py flags]
# (only when the GPU is free; resumes from OUT/state.pt)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
${ZIPVOICE_PY:-python3} $REPO/recipes/vocos/finetune.py --pairs $PAIRS --out $OUT "$@"
echo "VOCOS_FT_SH_DONE $(date -Is)"

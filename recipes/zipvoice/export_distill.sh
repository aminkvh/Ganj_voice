#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
# The phone folder of a distilled student (recipes/zipvoice/distill_q9h.sh) for sherpa-onnx 2.0.1, on the CPU:
#   text_encoder.onnx / fm_decoder.onnx (zipvoice.bin.onnx_export, model zipvoice_distill) and their _int8 copies
#   (dynamic int8 MatMul weights, one scale per output channel),
#   vocos_24khz_despike.onnx (sherpa's Vocos with the despike in front, packages/ganj-voice/ganj_voice/despike_onnx.py),
#   tokens.txt (the student's, as it is: sherpa reads its tab-separated lines and the space token), prompt.wav (the
#   voice's prompt at 24 kHz, loaded and cleaned as ZipVoice's generate_sentence does (edges, long silences, 0.2 s of
#   silence after it), float, cut by ganj_voice.sherpa_prompt so that sherpa reads as many prompt frames as
#   ZipVoice), prompt_phonemes.txt, and phone.json (the
#   settings the app gives sherpa and the clean-up after it). sherpa splits a token list after any token named
#   , . ! ? ; or : (ours: ; = ژ and ?) and reads the pieces apart, unless pieces are merged up to min_words_in_sentence
#   tokens: phone.json's extra sets it above any line's length, as the app does.
# Usage: recipes/zipvoice/export_distill.sh STUDENT_DIR CKPT OUT_DIR   (STUDENT_DIR holds CKPT, model.json, tokens.txt)
#   VOICE_DIR (default ganj-zip-d4h: prompt.wav, prompt_phonemes.txt)  NUM_STEP (default 8)  GUIDANCE (default 2.0)
#   B8=1 (block-wise 8-bit copies too, text_encoder_b8.onnx / fm_decoder_b8.onnx)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${GANJ_DATA:?set GANJ_DATA to the data directory (see recipes/README.md)}"
: "${ZIPVOICE_DIR:?set ZIPVOICE_DIR to a ZipVoice checkout (see recipes/README.md)}"
S=$(realpath "$1"); CKPT=$2; mkdir -p "$3"; OUT=$(realpath "$3")
V=$(realpath "${VOICE_DIR:-$GANJ_DATA/models/ganj-zip-d4h}"); VOC=$GANJ_DATA/models/sherpa_vocos/vocos_24khz.onnx
# Python is fed on stdin below: ZIPVOICE_PY must be the ZipVoice environment's python itself, not `conda run`
PYV=${GANJ_PY:-python}; PYZ=${ZIPVOICE_PY:-python3}
(cd $ZIPVOICE_DIR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=${THREADS:-4} PYTHONPATH=. $PYZ -m zipvoice.bin.onnx_export \
  --model-name zipvoice_distill --model-dir "$S" --checkpoint-name "$CKPT" --onnx-model-dir "$OUT") > "$OUT/export.log" 2>&1
# the int8 copies again with one weight scale per output channel (onnx_export uses one per tensor): same size and speed,
# fewer lost words (20k, 3 takes each through sherpa: WER gap -0.074 against -0.057 per tensor and -0.084 at fp32)
($PYZ -I - "$OUT" <<'EOF'
import sys
from onnxruntime.quantization import QuantType, quantize_dynamic
for name in ["text_encoder", "fm_decoder"]:
    quantize_dynamic(model_input=f"{sys.argv[1]}/{name}.onnx", model_output=f"{sys.argv[1]}/{name}_int8.onnx",
                     op_types_to_quantize=["MatMul"], weight_type=QuantType.QInt8, per_channel=True)
EOF
) >> "$OUT/export.log" 2>&1
# B8=1: block-wise 8-bit copies as well (_b8: MatMulNBits, 8 bits, block 32, symmetric, accuracy level 4; needs onnx-ir in
# the ZipVoice environment): larger and slower than per-channel int8, by ear between it and fp32
if [ -n "${B8:-}" ]; then
($PYZ -I - "$OUT" <<'EOF'
import sys
import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
for name in ["text_encoder", "fm_decoder"]:
    q = MatMulNBitsQuantizer(onnx.load(f"{sys.argv[1]}/{name}.onnx"), bits=8, block_size=32, is_symmetric=True, accuracy_level=4)
    q.process()
    q.model.save_model_to_file(f"{sys.argv[1]}/{name}_b8.onnx", use_external_data_format=False)
EOF
) >> "$OUT/export.log" 2>&1
fi
cd $REPO
$PYV -m ganj_voice.despike_onnx "$VOC" "$OUT/vocos_24khz_despike.onnx"
cp "$S/tokens.txt" "$OUT/tokens.txt"
cp "$V/prompt_phonemes.txt" "$OUT/"
(CUDA_VISIBLE_DEVICES= $PYZ -I - "$V/prompt.wav" "$OUT" "${NUM_STEP:-8}" "${GUIDANCE:-2.0}" "$S" "$CKPT" "$ZIPVOICE_DIR" "$REPO/packages/ganj-voice" <<'EOF'
import json, sys
import soundfile as sf, torch
sys.path[:0] = sys.argv[7:9]   # the ZipVoice checkout and packages/ganj-voice
from zipvoice.utils.infer import load_prompt_wav, remove_silence
from ganj_voice.sherpa_prompt import frame_matched
src, out, steps, guidance, student, ckpt = sys.argv[1:7]
# the prompt as ZipVoice's generate_sentence feeds it: edges trimmed, silences over 1 s shortened, 0.2 s of silence after
# it (without that sherpa ran each line straight on from the prompt: 8 of 24 takes of Hafez's ghazal 1 garbled the
# -lhā rhyme, 0 of 24 with it); then cut so that sherpa reads as many prompt frames as ZipVoice
w = remove_silence(load_prompt_wav(src, sampling_rate=24000), 24000, only_edge=False, trail_sil=200)
w = frame_matched(w[0])[None]
sf.write(f"{out}/prompt.wav", w[0].numpy(), 24000, subtype="FLOAT")
rms = float(torch.sqrt(torch.mean(w[0] ** 2)))
json.dump({
    "student": f"{student.split('/')[-1]}/{ckpt}",   # which student, by folder name: the app ignores it
    "sherpa": {"num_steps": int(steps), "guidance_scale": float(guidance), "t_shift": 0.5, "feat_scale": 0.1,
               "target_rms": 0.1, "speed": 1.004, "silence_scale": 1.0, "reference_text": "prompt_phonemes.txt",
               "extra": {"min_words_in_sentence": "100000"},
               "tokens": "one per character, as named in tokens.txt; first the prompt's, then the line's"},
    "after": {"gain": round(min(1.0, rms / 0.1), 4), "clamp": 1.0, "highpass_hz": 70, "highpass_order": 8,
              "highpass": "Butterworth, forward and backward", "deess_db": 6, "pause_cap_s": 0.30,
              "order": ["gain", "clamp", "highpass", "deess", "pause trim", "clamp"]},
    "prompt_rms": round(rms, 4),
}, open(f"{out}/phone.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
EOF
)
ls -l "$OUT"

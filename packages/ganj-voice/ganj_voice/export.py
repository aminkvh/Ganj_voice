# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Export a piper1-gpl checkpoint to ONNX and re-label it for sherpa-onnx's character frontend."""
import subprocess
import sys
from pathlib import Path

import onnx

from ganj_reading import tokens as T

META = {"comment": "piper ganj-voice (characters frontend; tokens.txt is the contract; keep the word piper: sherpa selects the scales-input layout by it)", "model_type": "vits", "add_blank": "1",
        "pad_id": "0", "blank_id": "0", "bos_id": "1", "eos_id": "2", "use_eos_bos": "1", "sample_rate": "22050",
        "language": "Persian", "voice": "ganj", "frontend": "characters", "has_espeak": "0"}


def read_metadata(onnx_path) -> dict:
    return {p.key: p.value for p in onnx.load(str(onnx_path), load_external_data=False).metadata_props}


def patch_metadata(onnx_path, n_speakers: int) -> None:
    m = onnx.load(str(onnx_path))
    keep = {p.key: p.value for p in m.metadata_props if p.key not in META and p.key != "n_speakers"}
    del m.metadata_props[:]
    for k, v in {**keep, **META, "n_speakers": str(n_speakers)}.items():
        m.metadata_props.add(key=k, value=v)
    onnx.save(m, str(onnx_path))


def export_voice(ckpt, out_dir, n_speakers: int) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = out_dir / "model.onnx"
    subprocess.run([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(ckpt), "--output-file", str(model)], check=True)
    patch_metadata(model, n_speakers)
    T.write_tokens_txt(out_dir / "tokens.txt")
    return model

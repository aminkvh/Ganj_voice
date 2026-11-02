# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import onnx
from onnx import TensorProto, helper

from ganj_voice.export import patch_metadata, read_metadata


def test_patch_metadata_sets_character_frontend_keys(tmp_path):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])])
    m = helper.make_model(g)
    m.metadata_props.add(key="comment", value="piper")
    p = tmp_path / "m.onnx"
    onnx.save(m, p)
    patch_metadata(p, n_speakers=7)
    md = read_metadata(p)
    assert "piper" in md["comment"] and md["n_speakers"] == "7" and md["add_blank"] == "1"
    assert md["bos_id"] == "1" and md["eos_id"] == "2" and md["pad_id"] == "0" and md["use_eos_bos"] == "1"
    assert md["frontend"] == "characters"  # sherpa-onnx compares to the plural form

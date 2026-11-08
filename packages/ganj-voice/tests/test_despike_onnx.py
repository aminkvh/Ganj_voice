# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper

from ganj_voice.despike_onnx import despike_prefix
from ganj_voice.melsmooth import despike_mel


def pass_through(name: str = "mels") -> onnx.ModelProto:
    """A stand-in vocoder: (B, 100, T) mels in, the same out as `mag`, with a vocoder's metadata."""
    graph = helper.make_graph(
        [helper.make_node("Identity", [name], ["mag"])], "voc",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch_size", 100, "time"])],
        [helper.make_tensor_value_info("mag", TensorProto.FLOAT, ["batch_size", 100, "time"])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    for key, value in {"model_type": "vocos", "hop_length": "256"}.items():
        meta = model.metadata_props.add()
        meta.key, meta.value = key, value
    return model


def run(model: onnx.ModelProto, mels: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, {session.get_inputs()[0].name: mels})[0]


def test_the_despiked_graph_feeds_the_vocoder_what_despike_mel_gives():
    mels = np.random.default_rng(0).normal(size=(2, 100, 37)).astype(np.float32)
    mels[0, 80, 5] = 9.0   # a spike in the high band: filtered
    mels[0, 10, 5] = 9.0   # one in the low band: kept
    out = run(despike_prefix(pass_through()), mels)
    assert np.array_equal(out, despike_mel(torch.from_numpy(mels)).numpy())
    assert out[0, 10, 5] == 9.0 and out[0, 80, 5] < 9.0


def test_the_start_bin_can_be_moved():
    mels = np.random.default_rng(1).normal(size=(1, 100, 9)).astype(np.float32)
    out = run(despike_prefix(pass_through(), start_bin=30), mels)
    assert np.array_equal(out, despike_mel(torch.from_numpy(mels), start_bin=30).numpy())


def test_the_despiked_graph_keeps_its_names_metadata_and_opset_and_leaves_the_original_alone():
    original = pass_through()
    model = despike_prefix(original)
    onnx.checker.check_model(model)
    assert [i.name for i in model.graph.input] == ["mels"] and [o.name for o in model.graph.output] == ["mag"]
    assert {p.key: p.value for p in model.metadata_props} == {"model_type": "vocos", "hop_length": "256"}
    assert (model.ir_version, [(o.domain, o.version) for o in model.opset_import]) == (8, [("", 17)])
    assert [n.op_type for n in original.graph.node] == ["Identity"]

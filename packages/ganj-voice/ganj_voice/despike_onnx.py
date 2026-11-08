# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""melsmooth.despike_mel built into an ONNX vocoder, for the phone chain (sherpa-onnx runs the vocoder graph it is given
and nothing between the flow model and it). The bins >= start_bin of the vocoder's mel input are replaced by the median
of 3 neighbouring bins, median(a, b, c) = max(min(a, b), min(max(a, b), c)), the top bin padded by repeating it, as
despike_mel does (3 bins x 1 frame). The vocoder's own nodes, input and output names, metadata (sherpa reads the
vocoder's type and STFT settings there), opset and IR version are kept.
Usage: python -m ganj_voice.despike_onnx IN.onnx OUT.onnx [--start-bin 60]"""
import argparse
import copy

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ganj_voice.melsmooth import START_BIN

PREFIX = "despike/"
TO_END = np.iinfo(np.int64).max


def despike_prefix(vocoder: onnx.ModelProto, start_bin: int = START_BIN) -> onnx.ModelProto:
    """A copy of `vocoder` (one input, (B, C, T) mels) with the despike in front of its first node."""
    model = copy.deepcopy(vocoder)
    graph = model.graph
    mels = graph.input[0].name
    out = PREFIX + "mels"
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name == mels:
                node.input[i] = out

    consts = {"bin1": [1], "lo": [start_bin - 1], "start": [start_bin], "top": [-1], "zero": [0], "one": [1],
              "two": [2], "minus2": [-2], "end": [TO_END]}
    graph.initializer.extend(numpy_helper.from_array(np.array(v, dtype=np.int64), PREFIX + k) for k, v in consts.items())

    def n(op, inputs, output, **attrs):
        return helper.make_node(op, [PREFIX + i if i != mels else i for i in inputs], [PREFIX + output], **attrs)

    def piece(src, start, end, name):
        return n("Slice", [src, start, end, "bin1"], name)

    nodes = [
        piece(mels, "lo", "end", "high"),               # bins start-1 .. C-1
        piece(mels, "top", "end", "top_bin"),           # bin C-1, repeated as the padding above the top
        n("Concat", ["high", "top_bin"], "padded", axis=1),
        piece("padded", "zero", "minus2", "below"),     # each bin's lower neighbour
        piece("padded", "one", "top", "centre"),
        piece("padded", "two", "end", "above"),         # and its upper one
        n("Min", ["below", "centre"], "min_ab"),
        n("Max", ["below", "centre"], "max_ab"),
        n("Min", ["max_ab", "above"], "min_max_c"),
        n("Max", ["min_ab", "min_max_c"], "median"),
        piece(mels, "zero", "start", "low"),
        helper.make_node("Concat", [PREFIX + "low", PREFIX + "median"], [out], axis=1),
    ]
    vocoder_nodes = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes + vocoder_nodes)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("vocoder")
    ap.add_argument("out")
    ap.add_argument("--start-bin", type=int, default=START_BIN)
    a = ap.parse_args()
    model = despike_prefix(onnx.load(a.vocoder), a.start_bin)
    onnx.checker.check_model(model)
    onnx.save(model, a.out)


if __name__ == "__main__":
    main()

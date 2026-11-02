# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import torch

from ganj_voice.blend import blend_checkpoint, candidates


def test_candidates_same_gender_min_hours_excluding_dominant():
    rows = []
    for a, n, s in (("big", 400, 0.9), ("m1", 100, 0.8), ("m2", 100, 0.7), ("m3", 10, 0.95), ("f1", 100, 0.9)):
        rows += [{"artist": a, "seconds": 90, "align_score": s}] * n
    g = {"big": "male", "m1": "male", "m2": "male", "m3": "male", "f1": "female"}
    assert candidates(rows, g, {"big": 0, "m1": 1, "m2": 2, "m3": 3, "f1": 4, "blend": 5}, min_hours=2.0) == ["m1", "m2"]


def test_blend_checkpoint_writes_weighted_row(tmp_path):
    W = torch.zeros(4, 3)
    W[1] = torch.tensor([1.0, 0, 0])
    W[2] = torch.tensor([0, 1.0, 0])
    src = tmp_path / "in.ckpt"
    torch.save({"state_dict": {"model_g.emb_g.weight": W}}, src)
    out = tmp_path / "out.ckpt"
    blend_checkpoint(src, out, {"a": 1, "b": 2, "blend": 3}, {"a": 0.75, "b": 0.25})
    row = torch.load(out, weights_only=False)["state_dict"]["model_g.emb_g.weight"][3]
    assert torch.allclose(row, torch.tensor([0.75, 0.25, 0.0]))

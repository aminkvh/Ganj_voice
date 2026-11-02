# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import torch

from ganj_voice.chorus import euler_mixed, mix_target, shared_target_len


def test_shared_target_len_is_the_mean_of_each_prompts_rate_prediction():
    # prompt k speaks prompt_tok[k] tokens in prompt_len[k] frames; target has 10 tokens
    assert shared_target_len([100, 200], [10, 10], 10, speed=1.0) == 150
    assert shared_target_len([100], [10], 10, speed=2.0) == 50


def test_mix_target_replaces_every_target_segment_with_the_weighted_mean():
    v = torch.zeros(2, 9, 1)
    v[0, 2:6] = 1.0   # prompt 0 is 2 frames, target 4 frames
    v[1, 5:9] = 3.0   # prompt 1 is 5 frames
    out = mix_target(v, [2, 5], 4, [0.5, 0.5])
    assert torch.allclose(out[0, 2:6], torch.full((4, 1), 2.0))
    assert torch.allclose(out[1, 5:9], torch.full((4, 1), 2.0))
    assert out[0, :2].abs().sum() == 0 and out[1, :5].abs().sum() == 0  # prompt regions untouched
    assert v[0, 2:6].mean() == 1.0  # input not mutated


def test_euler_mixed_keeps_one_shared_target_trajectory_that_follows_the_mean_velocity():
    # velocity field: each row is pulled toward its own constant c_k; the mixture should go to the weighted mean
    c = torch.tensor([1.0, 3.0]).view(2, 1, 1)
    vfn = lambda t, x: c - x
    x0 = torch.zeros(2, 9, 1)
    ts = torch.linspace(0, 1, 201)
    x1 = euler_mixed(vfn, x0, ts, [2, 5], 4, [0.25, 0.75])
    assert torch.allclose(x1[0, 2:6], x1[1, 5:9])
    expected = 2.5 * (1 - torch.exp(torch.tensor(-1.0)))  # dx/dt = 2.5 - x from 0, t = 1
    assert torch.allclose(x1[0, 2:6], torch.full((4, 1), float(expected)), atol=1e-2)

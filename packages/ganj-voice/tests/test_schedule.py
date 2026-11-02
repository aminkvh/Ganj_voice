# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import pytest
import torch

from ganj_voice.regen import shipped_times
from ganj_voice.schedule import refine_sampler, refined_times


def test_refined_times_with_refine_one_is_the_shipped_schedule():
    assert torch.allclose(refined_times(64, 0.5, 0.8, 1), shipped_times(64, 0.5))


def test_refined_times_keep_the_head_and_split_the_tail():
    times, shipped = refined_times(64, 0.5, 0.8, 4), shipped_times(64, 0.5)
    assert len(times) == 57 + 28 + 1                       # the shipped 57 steps before t 0.803, then 7 x 4
    assert torch.allclose(times[:58], shipped[:58])
    assert torch.allclose(times[57::4], shipped[57:])       # every 4th time of the tail is a shipped time
    assert bool((times[1:] > times[:-1]).all()) and float(times[-1]) == 1.0


def test_refined_times_from_the_end_change_nothing():
    assert torch.allclose(refined_times(64, 0.5, 1.0, 4), shipped_times(64, 0.5))


class FakeSolver:
    def __init__(self):
        self.calls = []

    def model(self, t, x, text_condition, speech_condition, padding_mask, guidance_scale, **kwargs):
        self.calls.append((float(t), guidance_scale, kwargs))
        return text_condition - x                             # a flow to text_condition


def _euler(x, target, times):
    for i in range(len(times) - 1):
        x = x + (target - x) * (times[i + 1] - times[i])
    return x


def test_refine_sampler_runs_the_euler_loop_on_the_refined_times():
    cls = type("S", (FakeSolver,), {})
    refine_sampler(cls, 0.8, 4)
    s, x, target = cls(), torch.zeros(1, 3, 2), torch.ones(1, 3, 2)
    out = s.sample(x=x, text_condition=target, speech_condition=None, padding_mask=None, num_step=64,
                   guidance_scale=2.0, t_shift=0.5, extra=7)
    times = refined_times(64, 0.5, 0.8, 4)
    assert len(s.calls) == 85
    assert [c[0] for c in s.calls] == pytest.approx(times[:-1].tolist())
    assert all(c[1] == 2.0 and c[2] == {"extra": 7} for c in s.calls)
    assert torch.allclose(out, _euler(x, target, times))


def test_refine_sampler_refuses_part_of_the_flow():
    cls = type("S", (FakeSolver,), {})
    refine_sampler(cls, 0.8, 4)
    with pytest.raises(ValueError):
        cls().sample(x=torch.zeros(1), text_condition=torch.ones(1), speech_condition=None, padding_mask=None,
                     num_step=8, t_start=0.5)

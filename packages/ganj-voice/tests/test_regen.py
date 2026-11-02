# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import torch

from ganj_voice.regen import (condition_mask, finish, noised, paste_condition, shipped_times, start_index,
                                   tail_times)


def test_shipped_times_run_from_noise_to_data():
    t = shipped_times(64, 0.5)
    assert t.shape == (65,)
    assert t[0] == 0.0 and t[-1] == 1.0
    assert bool((t[1:] > t[:-1]).all())


def test_shipped_times_apply_the_shift():
    t = shipped_times(2, 0.5)
    assert torch.isclose(t[1], torch.tensor(1 / 3))


def test_start_index_picks_the_first_time_at_or_after_the_target():
    t = shipped_times(64, 0.5)
    k = start_index(t, 0.8)
    assert k == 57 and t[k] >= 0.8 > t[k - 1]
    assert start_index(t, 0.5) == 43


def test_start_index_covers_the_whole_flow_and_none_of_it():
    t = shipped_times(64, 0.5)
    assert start_index(t, 0.0) == 0
    assert start_index(t, 1.0) == 64


def test_noised_mixes_data_and_noise_as_in_training():
    x, n = torch.full((1, 4, 2), 2.0), torch.zeros(1, 4, 2)
    assert torch.equal(noised(x, n, 1.0), x)
    assert torch.equal(noised(x, n, 0.0), n)
    assert torch.allclose(noised(x, n, 0.25), torch.full((1, 4, 2), 0.5))


def test_condition_mask_keeps_the_first_frames_as_condition():
    m = condition_mask(100, 0.25)
    assert m.shape == (1, 100)
    assert not m[0, :25].any() and m[0, 25:].all()
    assert condition_mask(10, 0.0).all()


def test_finish_runs_the_remaining_euler_steps_in_order():
    t = shipped_times(8, 0.5)
    seen = []

    def step(tt, x):
        seen.append(float(tt))
        return torch.ones_like(x)

    out = finish(step, torch.zeros(1, 3, 2), t, 5)
    assert seen == [float(v) for v in t[5:-1]]
    assert torch.allclose(out, torch.full((1, 3, 2), float(1 - t[5])))


def test_finish_from_the_end_changes_nothing():
    x = torch.randn(1, 3, 2)
    assert torch.equal(finish(lambda tt, y: torch.ones_like(y), x, shipped_times(8, 0.5), 8), x)


def test_paste_condition_puts_back_the_real_condition_frames():
    out, real = torch.zeros(1, 4, 2), torch.ones(1, 4, 2)
    m = condition_mask(4, 0.5)
    got = paste_condition(out, real, m)
    assert torch.equal(got[0, :2], real[0, :2]) and torch.equal(got[0, 2:], out[0, 2:])


def test_tail_times_unrefined_is_the_shipped_tail():
    t = shipped_times(64, 0.5)
    assert torch.equal(tail_times(0.8, 64, 0.5, 1), t[57:])


def test_tail_times_refined_keeps_the_start_and_splits_every_step():
    t = shipped_times(64, 0.5)
    r = tail_times(0.8, 64, 0.5, 4)
    assert len(r) == 4 * 7 + 1
    assert torch.isclose(r[0], t[57]) and r[-1] == 1.0
    assert bool((r[1:] > r[:-1]).all())


def test_tail_times_zero_refine_is_one_jump_from_the_same_start():
    t = shipped_times(64, 0.5)
    r = tail_times(0.8, 64, 0.5, 0)
    assert len(r) == 2 and torch.isclose(r[0], t[57]) and r[1] == 1.0


def test_condition_masks_follow_each_clips_own_length():
    from ganj_voice.regen import condition_masks
    m = condition_masks(torch.tensor([8, 4]), 0.25, 8)
    assert m.shape == (2, 8)
    assert m[0].tolist() == [False, False] + [True] * 6
    assert m[1].tolist() == [False] + [True] * 7


def test_frame_batches_group_by_length_under_the_budget():
    from ganj_voice.regen import frame_batches
    lengths = [500, 100, 300, 900, 120]
    batches = frame_batches(lengths, 1000)
    assert sorted(i for b in batches for i in b) == [0, 1, 2, 3, 4]
    assert all(max(lengths[i] for i in b) * len(b) <= 1000 or len(b) == 1 for b in batches)
    assert [lengths[b[0]] for b in batches] == sorted(lengths[b[0]] for b in batches)


def test_frame_batches_keep_an_overlong_clip_alone():
    from ganj_voice.regen import frame_batches
    assert frame_batches([2000, 10], 1000) == [[1], [0]]

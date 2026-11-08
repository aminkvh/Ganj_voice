# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import pytest

from ganj_voice.distill import guidance_range, teacher_num_step


def test_a_teacher_jump_that_reaches_the_tail_is_taken_in_finer_steps():
    assert teacher_num_step(0.79) == 1
    assert teacher_num_step(0.8) == 4
    assert teacher_num_step(1.0, tail_steps=8) == 8
    assert teacher_num_step(0.95, tail_steps=1) == 1


def test_stage_one_samples_guidance_up_to_three_so_the_voices_two_is_inside_the_range():
    assert guidance_range("first") == (0.0, 3.0)
    assert guidance_range("first", first_top=2.0) == (0.0, 2.0)
    assert guidance_range("second") == (1.0, 3.0)
    with pytest.raises(ValueError):
        guidance_range("third")

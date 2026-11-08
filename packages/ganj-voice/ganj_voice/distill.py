# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Two choices in ZipVoice's distillation (zipvoice.bin.train_zipvoice_distill) made for this voice; the training wrapper
is recipes/zipvoice/train_distill_ganj.py.

teacher_num_step: the shipped voice finishes the flow in finer steps from t 0.8 (ganj_voice.schedule: coarse final
steps make the "laser"). A few-step student cannot do that at reading time, so a stage-1 teacher jump that ends at or
past tail_from is taken in tail_steps Euler steps instead of one, and the student learns the finely finished flow.

guidance_range: stage 1 samples the guidance scale uniformly in [0, 2) upstream and stage 2 in [1, 3). The voice reads
at 2.0, the edge of stage 1's range, so stage 1's top is raised to 3."""
from typing import Tuple


def teacher_num_step(t_end: float, tail_from: float = 0.8, tail_steps: int = 4) -> int:
    """Euler steps for a teacher jump ending at flow time t_end (0 = noise, 1 = data)."""
    return tail_steps if t_end >= tail_from else 1


def guidance_range(stage: str, first_top: float = 3.0) -> Tuple[float, float]:
    """(low, high) of the uniform guidance scale a distillation stage trains on."""
    if stage == "first":
        return 0.0, first_top
    if stage == "second":
        return 1.0, 3.0
    raise ValueError(f"unknown distillation stage {stage!r}")

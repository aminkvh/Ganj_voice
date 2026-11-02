# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Finish a real recording's mel with the flow model from part-way along the flow (SDEdit-style regeneration).

The real mel is noised to a flow time t (training's mix: x_t = t * data + (1 - t) * noise; t = 0 is noise, t = 1 data)
and the shipped Euler schedule is run from that time to the end. Low t leaves the model most of the work, t near 1 only
the last steps. The output keeps the real recording's timing, so (regenerated mel, real audio) pairs are time-aligned:
the data a vocoder fine-tune on the model's own output needs, and a test of where in the flow a texture forms.

ZipVoice's solver applies its t_shift to linspace(t_start, t_end), so solver.sample(t_start=0.8, t_shift=0.5) starts
at 0.667, not 0.8; the schedule is therefore built here once over [0, 1] and entered at an index."""
from typing import Callable

import torch


def shipped_times(num_step: int = 64, t_shift: float = 0.5) -> torch.Tensor:
    """The num_step + 1 flow times of ZipVoice's get_time_steps(0, 1, num_step, t_shift)."""
    u = torch.linspace(0.0, 1.0, num_step + 1)
    return t_shift * u / (1 + (t_shift - 1) * u)


def start_index(times: torch.Tensor, t_start: float) -> int:
    """Index of the first time >= t_start (len(times) - 1 for t_start = 1: no steps left)."""
    return int(torch.nonzero(times >= t_start - 1e-6)[0, 0])


def tail_times(t_start: float, num_step: int = 64, t_shift: float = 0.5, refine: int = 1) -> torch.Tensor:
    """The shipped schedule from its first time >= t_start to 1, each step split into `refine` equal steps before the
    shift (refine = 1: the shipped steps; 0: one jump to 1, x1 = x_t + (1 - t) v). The start time never moves."""
    k = start_index(shipped_times(num_step, t_shift), t_start)
    if refine == 0:
        return torch.stack([shipped_times(num_step, t_shift)[k], torch.tensor(1.0)])
    return shipped_times(num_step * refine, t_shift)[k * refine:]


def noised(features: torch.Tensor, noise: torch.Tensor, t: float) -> torch.Tensor:
    return features * t + noise * (1 - t)


def condition_mask(num_frames: int, frac: float) -> torch.Tensor:
    """(1, num_frames) bool; False on the first round(frac * num_frames) frames (the speech condition, as in training's
    condition_time_mask), True where the model generates."""
    m = torch.ones(1, num_frames, dtype=torch.bool)
    m[0, : round(frac * num_frames)] = False
    return m


def condition_masks(lens: torch.Tensor, frac: float, max_len: int) -> torch.Tensor:
    """(B, max_len) bool for a padded batch: False on each clip's first round(frac * len) frames."""
    n = (lens.float() * frac).round().long()
    return torch.arange(max_len, device=lens.device)[None] >= n[:, None]


def frame_batches(lengths: list, max_frames: int) -> list:
    """Indices grouped shortest first so that a padded batch (longest x count) stays within max_frames; a clip longer
    than the budget goes alone."""
    batches, cur = [], []
    for i in sorted(range(len(lengths)), key=lambda j: lengths[j]):
        if cur and lengths[i] * (len(cur) + 1) > max_frames:
            batches.append(cur)
            cur = []
        cur.append(i)
    return batches + ([cur] if cur else [])


def finish(step: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: torch.Tensor, times: torch.Tensor,
           k: int) -> torch.Tensor:
    """Euler steps from times[k] to times[-1]; step(t, x) is the (guided) velocity."""
    for i in range(k, len(times) - 1):
        x = x + step(times[i], x) * (times[i + 1] - times[i])
    return x


def paste_condition(out: torch.Tensor, real: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """out, real: (B, T, F); the real frames where mask is False (the condition the model only saw)."""
    return torch.where(mask.unsqueeze(-1), out, real)

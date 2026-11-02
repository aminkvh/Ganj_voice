# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""A finer finish for ZipVoice's Euler sampler (the laser forms in the last 7 of the shipped 64 steps, t 0.80 -> 1,
and fades when those steps are split in four; same model, same start).

refined_times keeps the shipped schedule up to its first time >= t_from and splits each later step into `refine` equal
steps before the shift (ganj_voice.regen.tail_times), so the head is unchanged and every shipped time is still
visited. refine_sampler patches ZipVoice's EulerSolver.sample in place to run that schedule (ZipVoice itself is not
edited); only the full flow (t_start 0, t_end 1) is supported."""
import logging

import torch

from ganj_voice.regen import finish, shipped_times, start_index, tail_times

log = logging.getLogger(__name__)


def refined_times(num_step: int = 64, t_shift: float = 0.5, t_from: float = 0.8, refine: int = 4) -> torch.Tensor:
    """The shipped num_step schedule with every step from its first time >= t_from split in `refine`."""
    times = shipped_times(num_step, t_shift)
    return torch.cat([times[:start_index(times, t_from)], tail_times(t_from, num_step, t_shift, refine)])


def refine_sampler(cls: type, t_from: float = 0.8, refine: int = 4) -> None:
    """Replace cls.sample(x, text_condition, speech_condition, padding_mask, num_step, guidance_scale, t_start, t_end,
    t_shift, **kw) with the same Euler loop on refined_times(num_step, t_shift, t_from, refine). The first call logs
    the schedule it runs (proof in the run's log that the patch is live)."""
    logged = []

    def sample(self, x, text_condition, speech_condition, padding_mask, num_step=10, guidance_scale=0.0,
               t_start=0.0, t_end=1.0, t_shift=1.0, **kwargs):
        if (t_start, t_end) != (0.0, 1.0):
            raise ValueError(f"refined sampler runs the whole flow only, got t_start={t_start}, t_end={t_end}")
        times = refined_times(num_step, t_shift, t_from, refine).to(x.device)
        if not logged:
            k = start_index(shipped_times(num_step, t_shift), t_from)
            log.info("refined sampler: %d Euler steps, the shipped %d split x%d from t %.3f", len(times) - 1, num_step,
                     refine, float(shipped_times(num_step, t_shift)[k]))
            logged.append(True)

        def step(t, x_t):
            return self.model(t=t, x=x_t, text_condition=text_condition, speech_condition=speech_condition,
                              padding_mask=padding_mask, guidance_scale=guidance_scale, **kwargs)

        return finish(step, x, times, 0)

    cls.sample = sample

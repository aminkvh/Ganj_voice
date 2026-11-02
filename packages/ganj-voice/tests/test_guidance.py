# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import torch

from ganj_voice.guidance import interval_scale, limit_guidance


def test_scale_is_kept_inside_the_interval():
    assert interval_scale(0.5, 2.0, t_min=0.0, t_max=0.8) == 2.0


def test_scale_is_zero_after_the_interval():
    assert interval_scale(0.9, 2.0, t_min=0.0, t_max=0.8) == 0.0


def test_scale_is_zero_before_the_interval():
    assert interval_scale(0.1, 2.0, t_min=0.2, t_max=1.0) == 0.0


def test_the_full_interval_keeps_every_step():
    assert all(interval_scale(t, 2.0) == 2.0 for t in (0.0, 0.3, 0.99, 1.0))


class FakeDiffusion:
    def forward(self, t, x, text_condition, speech_condition, padding_mask=None, guidance_scale=0.0, **kwargs):
        return guidance_scale


def test_limit_guidance_drops_the_scale_outside_the_interval():
    cls = type("D", (FakeDiffusion,), {})
    limit_guidance(cls, t_min=0.0, t_max=0.8)
    d = cls()
    assert d.forward(torch.tensor(0.5), None, None, None, None, guidance_scale=2.0) == 2.0
    assert d.forward(torch.tensor(0.9), None, None, None, None, guidance_scale=2.0) == 0.0


def test_limit_guidance_passes_a_positional_scale_and_extra_arguments():
    seen = {}

    class D(FakeDiffusion):
        def forward(self, t, x, text_condition, speech_condition, padding_mask=None, guidance_scale=0.0, **kwargs):
            seen.update(kwargs, scale=guidance_scale)
            return x

    limit_guidance(D, t_min=0.0, t_max=0.8)
    assert D().forward(torch.tensor(0.95), "x", None, None, None, 2.0, extra=1) == "x"
    assert seen == {"extra": 1, "scale": 0.0}

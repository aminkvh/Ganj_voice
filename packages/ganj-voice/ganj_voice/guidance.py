# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Classifier-free guidance on part of the flow only (Kynkäänniemi et al. 2024, "Applying guidance in a limited
interval improves sample and distribution quality in diffusion models").

ZipVoice guides every sampling step. Guidance near the end of the flow (t close to 1, where the mel's fine texture is
set) is the suspect for the over-sharp "razor" in held vowels and the bright sibilants; switching it off there keeps
the guided words and timing from the earlier steps. limit_guidance patches ZipVoice's DiffusionModel.forward in place
(ZipVoice itself is not edited); interval_scale is the rule it applies at each step."""
from typing import Any


def interval_scale(t: float, scale: Any, t_min: float = 0.0, t_max: float = 1.0) -> Any:
    """The guidance scale for flow time t (0 = noise, 1 = data): unchanged inside [t_min, t_max], 0.0 outside."""
    return scale if t_min <= t <= t_max else 0.0


def limit_guidance(cls: type, t_min: float = 0.0, t_max: float = 1.0) -> None:
    """Wrap cls.forward(t, x, text_condition, speech_condition, padding_mask=None, guidance_scale=0.0, **kw) so that
    the scale is interval_scale(t, ...) at every call."""
    forward = cls.forward

    def limited(self, t, x, text_condition, speech_condition, padding_mask=None, guidance_scale=0.0, **kwargs):
        scale = interval_scale(float(t), guidance_scale, t_min, t_max)
        return forward(self, t, x, text_condition, speech_condition, padding_mask, scale, **kwargs)

    cls.forward = limited

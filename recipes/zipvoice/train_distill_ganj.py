# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""ZipVoice distillation (zipvoice.bin.train_zipvoice_distill, same CLI) with the voice's sampling built into the teacher,
so that a few-step student learns the flow the shipped voice reads with (ZipVoice itself is not edited):

- the teacher guides only flow times <= DISTILL_GUIDE_TMAX (default 0.6; ganj_voice.guidance). The student's
  DistillDiffusionModel has its own forward, so it is not limited: it learns the teacher's limited guidance.
- a stage-1 teacher jump ending at or past TEACHER_TAIL_FROM (default 0.8) is taken in TEACHER_TAIL_STEPS Euler steps
  (default 4; 1 = upstream) instead of one (ganj_voice.distill.teacher_num_step).
- stage 1 samples the guidance scale in [0, DISTILL_G1_TOP) (default 3; upstream 2); stage 2 as upstream, [1, 3).

The first teacher call at t <= the guidance limit, the first past it, and the first finer teacher jump are logged, so
the run's log shows the patches are live. compute_fbank_loss below is upstream's (ZipVoice 2f7326f, Apache-2.0,
Copyright 2024 Xiaomi Corp.) with the two changes marked CHANGED."""
import logging
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "packages", "ganj-voice"))
from ganj_voice.distill import guidance_range, teacher_num_step  # noqa: E402
from ganj_voice.guidance import limit_guidance  # noqa: E402
import torch  # noqa: E402
import zipvoice.bin.train_zipvoice_distill as td  # noqa: E402
import zipvoice.models.modules.solver as solver  # noqa: E402
from zipvoice.utils.common import MetricsTracker, condition_time_mask, make_pad_mask  # noqa: E402

GUIDE_TMAX = float(os.environ.get("DISTILL_GUIDE_TMAX", 0.6))
TAIL_FROM = float(os.environ.get("TEACHER_TAIL_FROM", 0.8))
TAIL_STEPS = int(os.environ.get("TEACHER_TAIL_STEPS", 4))
G1_TOP = float(os.environ.get("DISTILL_G1_TOP", 3.0))
_logged = set()


def _log_once(key, msg, *args):
    if key not in _logged:
        _logged.add(key)
        logging.info(msg, *args)


def _logging_forward(forward):
    """Wraps DiffusionModel.forward below limit_guidance: it sees the scale the limit let through."""
    def logged(self, t, x, text_condition, speech_condition, padding_mask=None, guidance_scale=0.0, **kwargs):
        on = torch.is_tensor(guidance_scale) and bool((guidance_scale != 0).any())
        side = "inside" if float(t) <= GUIDE_TMAX else "past"
        _log_once(side, "teacher guidance %s the limit %.2f: t %.3f, guidance %s", side, GUIDE_TMAX, float(t),
                  f"on (mean {float(guidance_scale.mean()):.2f})" if on else "off")
        return forward(self, t, x, text_condition, speech_condition, padding_mask, guidance_scale, **kwargs)
    return logged


solver.DiffusionModel.forward = _logging_forward(solver.DiffusionModel.forward)
limit_guidance(solver.DiffusionModel, 0.0, GUIDE_TMAX)


def compute_fbank_loss(params, model, teacher_model, features, features_lens, tokens, is_training):
    device = model.device if isinstance(model, td.DDP) else next(model.parameters()).device

    batch_size, num_frames, _ = features.shape

    noise = torch.randn_like(features)  # (B, T, F)

    # Sampling t and guidance_scale from uniform distribution

    t_value = random.random()
    t = torch.ones(batch_size, 1, 1, device=device) * t_value
    # CHANGED: stage 1's range [0, DISTILL_G1_TOP) instead of [0, 2)
    low, high = guidance_range(params.distill_stage, G1_TOP)
    guidance_scale = torch.rand(batch_size, 1, 1, device=device) * (high - low) + low
    xt = features * t + noise * (1 - t)
    t_delta_fix = random.uniform(0.0, min(0.3, 1 - t_value))
    t_delta_ema = random.uniform(0.0, min(0.3, 1 - t_value - t_delta_fix))
    t_dest = t_value + t_delta_fix + t_delta_ema

    with torch.no_grad():
        speech_condition_mask = condition_time_mask(
            features_lens=features_lens,
            mask_percent=(0.7, 1.0),
            max_len=features.size(1),
        )

        if params.distill_stage == "first":
            # CHANGED: a jump that reaches the tail is taken in TEACHER_TAIL_STEPS steps
            n_mid = teacher_num_step(t_value + t_delta_fix, TAIL_FROM, TAIL_STEPS)
            n_end = teacher_num_step(t_dest, TAIL_FROM, TAIL_STEPS)
            if max(n_mid, n_end) > 1:
                _log_once("tail", "finer teacher jump: t %.3f -> %.3f -> %.3f in %d + %d steps", t_value,
                          t_value + t_delta_fix, t_dest, n_mid, n_end)
            teacher_x_t_mid, _ = teacher_model.sample_intermediate(
                tokens=tokens,
                features=features,
                features_lens=features_lens,
                noise=xt,
                speech_condition_mask=speech_condition_mask,
                t_start=t_value,
                t_end=t_value + t_delta_fix,
                num_step=n_mid,
                guidance_scale=guidance_scale,
            )

            target_x1, _ = teacher_model.sample_intermediate(
                tokens=tokens,
                features=features,
                features_lens=features_lens,
                noise=teacher_x_t_mid,
                speech_condition_mask=speech_condition_mask,
                t_start=t_value + t_delta_fix,
                t_end=t_dest,
                num_step=n_end,
                guidance_scale=guidance_scale,
            )
        else:
            teacher_x_t_mid, _ = teacher_model(
                tokens=tokens,
                features=features,
                features_lens=features_lens,
                noise=xt,
                speech_condition_mask=speech_condition_mask,
                t_start=t_value,
                t_end=t_value + t_delta_fix,
                num_step=1,
                guidance_scale=guidance_scale,
            )

            target_x1, _ = teacher_model(
                tokens=tokens,
                features=features,
                features_lens=features_lens,
                noise=teacher_x_t_mid,
                speech_condition_mask=speech_condition_mask,
                t_start=t_value + t_delta_fix,
                t_end=t_dest,
                num_step=1,
                guidance_scale=guidance_scale,
            )

    with torch.set_grad_enabled(is_training):

        pred_x1, _ = model(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=xt,
            speech_condition_mask=speech_condition_mask,
            t_start=t_value,
            t_end=t_dest,
            num_step=1,
            guidance_scale=guidance_scale,
        )
        pred_v = (pred_x1 - xt) / (t_dest - t)

        padding_mask = make_pad_mask(features_lens, max_len=num_frames)  # (B, T)
        loss_mask = speech_condition_mask & (~padding_mask)

        target_v = (target_x1 - xt) / (t_dest - t)
        loss = torch.mean((pred_v[loss_mask] - target_v[loss_mask]) ** 2)

        ut = features - noise  # (B, T, F)

        ref_loss = torch.mean((pred_v[loss_mask] - ut[loss_mask]) ** 2)

    assert loss.requires_grad == is_training
    info = MetricsTracker()
    num_frames = features_lens.sum().item()
    info["frames"] = num_frames
    info["loss"] = loss.detach().cpu().item() * num_frames
    info["ref_loss"] = ref_loss.detach().cpu().item() * num_frames
    return loss, info


td.compute_fbank_loss = compute_fbank_loss

if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    td.main()

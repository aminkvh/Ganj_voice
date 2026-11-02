# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Chorus sampling for ZipVoice: one utterance generated under K prompts at once.

Each prompt k is its own batch row [prompt_k frames | target frames]. The target segment starts from one shared
noise and, at every Euler step, moves by the weighted mean of the K velocities, so a single trajectory is steered
by several voices the model knows well. The result sits between them and is none of them: the spec's "chorus".
Prompts from trained reciters are used raw; the model renders them cleanly.
"""
import math

import torch


def shared_target_len(prompt_lens, prompt_tok_lens, tgt_tok_len, speed=1.0) -> int:
    """Frames for the target: the mean of each prompt's own frames-per-token prediction."""
    preds = [p / pt * tgt_tok_len / speed for p, pt in zip(prompt_lens, prompt_tok_lens)]
    return int(math.ceil(sum(preds) / len(preds)))


def mix_target(v: torch.Tensor, prompt_lens, tgt_len: int, weights) -> torch.Tensor:
    """Copy of v whose target segment in every row is the weighted mean of all rows' target segments."""
    segs = torch.stack([v[k, p:p + tgt_len] for k, p in enumerate(prompt_lens)])
    w = torch.as_tensor(weights, dtype=v.dtype, device=v.device).view(-1, 1, 1)
    mean = (segs * w).sum(0) / w.sum()
    out = v.clone()
    for k, p in enumerate(prompt_lens):
        out[k, p:p + tgt_len] = mean
    return out


def euler_mixed(vfn, x: torch.Tensor, timesteps: torch.Tensor, prompt_lens, tgt_len: int, weights) -> torch.Tensor:
    """Euler ODE solve where the target segment follows the mixed velocity; vfn(t, x) -> velocity like x."""
    x = mix_target(x, prompt_lens, tgt_len, weights)  # one shared starting noise for the target
    for i in range(len(timesteps) - 1):
        v = mix_target(vfn(timesteps[i], x), prompt_lens, tgt_len, weights)
        x = x + v * (timesteps[i + 1] - timesteps[i])
    return x


def chorus_sample(model, tokens, prompt_tokens, prompt_features, weights, num_step=32, guidance_scale=2.0,
                  t_shift=0.5, speed=1.0):
    """ZipVoice features (1, T, F) for `tokens` under K prompts. prompt_features: list of (T_k, F), already scaled."""
    from zipvoice.models.modules.solver import get_time_steps
    from zipvoice.utils.common import make_pad_mask

    dev = prompt_features[0].device
    plens = [f.size(0) for f in prompt_features]
    tgt = shared_target_len(plens, [len(p) for p in prompt_tokens], len(tokens), speed)
    K = len(prompt_features)
    feats_lens = torch.full((K,), tgt, device=dev)  # target frames only: ZipVoice adds the prompt length itself
    text_cond, pad_mask = model.forward_text_inference_gt_duration(
        tokens=[tokens] * K, features_lens=feats_lens, prompt_tokens=prompt_tokens,
        prompt_features_lens=torch.tensor(plens, device=dev))
    T = text_cond.size(1)
    speech = torch.zeros(K, T, prompt_features[0].size(1), device=dev)
    for k, f in enumerate(prompt_features):
        speech[k, :f.size(0)] = f
    speech = torch.where(make_pad_mask(torch.tensor(plens, device=dev), T).unsqueeze(-1), torch.zeros_like(speech), speech)
    gs = torch.full((K,), float(guidance_scale), device=dev)

    def vfn(t, x):
        return model.solver.model(t=t, x=x, text_condition=text_cond, speech_condition=speech, padding_mask=pad_mask,
                                  guidance_scale=gs.view(K, 1, 1))

    ts = get_time_steps(t_start=0.0, t_end=1.0, num_step=num_step, t_shift=t_shift, device=dev)
    x1 = euler_mixed(vfn, torch.randn(K, T, speech.size(2), device=dev), ts, plens, tgt, weights)
    return x1[0:1, plens[0]:plens[0] + tgt]

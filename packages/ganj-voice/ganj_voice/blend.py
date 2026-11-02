# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""A voice that is nobody's: interpolate 2–3 same-gender reciters' learned embeddings into the reserved slot."""
from collections import defaultdict

import torch


def candidates(rows, genders: dict, speakers: dict, min_hours: float = 2.0, exclude_top: bool = True) -> list:
    hours, score, n = defaultdict(float), defaultdict(float), defaultdict(int)
    for r in rows:
        hours[r["artist"]] += r["seconds"] / 3600
        score[r["artist"]] += r["align_score"]
        n[r["artist"]] += 1
    by_gender = defaultdict(float)
    for a, h in hours.items():
        by_gender[genders.get(a, "unknown")] += h
    gender = max(("male", "female"), key=lambda g: by_gender[g])
    top = max(hours, key=hours.get) if exclude_top else None
    ok = [a for a in hours if a in speakers and genders.get(a) == gender and hours[a] >= min_hours and a != top]
    return sorted(ok, key=lambda a: -(score[a] / n[a]))


def blend_checkpoint(ckpt_in, ckpt_out, speakers: dict, parts: dict) -> None:
    ck = torch.load(ckpt_in, map_location="cpu", weights_only=False)
    W = ck["state_dict"]["model_g.emb_g.weight"]
    total = sum(parts.values())
    row = sum((w / total) * W[speakers[a]] for a, w in parts.items())
    W[speakers["blend"]] = row
    torch.save(ck, ckpt_out)

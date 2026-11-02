# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from collections import Counter

from tools.freeze_metres import merge_metres


def test_existing_metres_keep_their_order_and_new_ones_are_appended_by_frequency():
    existing = ["A", "B"]
    counts = Counter({"C": 5, "B": 100, "D": 9, "A": 1})
    assert merge_metres(existing, counts, limit=10) == ["A", "B", "D", "C"]


def test_the_list_is_capped_without_dropping_existing_metres():
    assert merge_metres(["A", "B"], Counter({"C": 5, "D": 9}), limit=3) == ["A", "B", "D"]

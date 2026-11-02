# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import os
from pathlib import Path

import pytest


@pytest.fixture
def data_dir():
    root = os.environ.get("GANJ_DATA")
    if not root:
        pytest.skip("GANJ_DATA not set")
    return Path(root)

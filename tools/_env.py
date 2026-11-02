# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Folders that come from the environment: GANJ_DATA (the data folder) and ZIPVOICE_DIR (a ZipVoice checkout).

env_path(NAME) is that folder as a Path. When NAME is not set, it is a stand-in that can still be joined with `/` (so
module constants such as DATA / "eval" can be defined), and any use of it stops with a message naming the variable."""
import os
from pathlib import Path

HINTS = {"GANJ_DATA": "the data folder (corpus, eval set, model runs)",
         "ZIPVOICE_DIR": "a ZipVoice checkout (https://github.com/k2-fsa/ZipVoice)"}


class Unset:
    """A path whose environment variable is not set."""

    def __init__(self, name: str):
        self.name = name

    def __truediv__(self, other):
        return self

    def _stop(self, *_args, **_kwargs):
        raise SystemExit(f"{self.name} is not set: point it at {HINTS.get(self.name, 'a folder')}")

    __fspath__ = __str__ = __format__ = _stop

    def __getattr__(self, attr):
        if attr.startswith("__"):
            raise AttributeError(attr)
        self._stop()

    def __repr__(self):
        return f"<{self.name} not set>"


def env_path(name: str):
    root = os.environ.get(name)
    return Path(root) if root else Unset(name)

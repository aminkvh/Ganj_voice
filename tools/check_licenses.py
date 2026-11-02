# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Fails if any installed package in this environment carries a non-commercial or unknown-copyleft licence
that must stay out of a GPL/AGPL-distributed pipeline. Run inside the environment of the ganj packages."""
import json
import subprocess
import sys

ALLOWED_WORDS = ("mit", "bsd", "apache", "gpl", "lgpl", "mpl", "psf", "python", "isc", "zlib", "unlicense", "public domain", "hpnd", "cc0", "ofl")
FORBIDDEN_WORDS = ("non-commercial", "noncommercial", "nc 4.0", "by-nc")
# Packages whose metadata lacks a licence string but whose repo licence is known.
KNOWN = {"utmosv2": "MIT", "sherpa-onnx": "Apache-2.0", "piper-tts": "GPL-3.0", "piper": "GPL-3.0"}


def main() -> int:
    raw = subprocess.run([sys.executable, "-m", "piplicenses", "--format=json"], capture_output=True, text=True, check=True).stdout
    bad = []
    for row in json.loads(raw):
        name = row["Name"].lower()
        lic = (row.get("License") or "").lower()
        if name in KNOWN:
            continue
        if any(w in lic for w in FORBIDDEN_WORDS):
            bad.append((name, lic))
    for name, lic in bad:
        print(f"FORBIDDEN {name}: {lic}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

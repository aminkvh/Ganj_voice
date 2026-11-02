# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
from pathlib import Path

import pytest

from ganj_corpus.download import download_one


def row(i=7, size=5):
    return {"id": i, "mp3Url": f"https://i.ganjoor.net/{i}.mp3", "mp3SizeInBytes": size, "poemId": 1}


def test_download_skips_existing_and_backs_off(tmp_path, monkeypatch):
    calls = []
    sleeps = []
    monkeypatch.setattr("ganj_corpus.download.time.sleep", lambda s: sleeps.append(s))

    def fetch(url, headers):
        calls.append(url)
        if len(calls) == 1:
            return 503, b""
        return 200, b"12345"

    p = download_one(row(), tmp_path, fetch=fetch)
    assert p.read_bytes() == b"12345" and len(calls) == 2 and sleeps == [1.0]
    download_one(row(), tmp_path, fetch=fetch)  # second call: file exists with expected size
    assert len(calls) == 2
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_short_body(tmp_path, monkeypatch):
    monkeypatch.setattr("ganj_corpus.download.time.sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        download_one(row(size=100), tmp_path, fetch=lambda u, h: (200, b"tiny"), tries=2)
    assert not (tmp_path / "7.mp3").exists()


def test_download_retries_network_errors(tmp_path, monkeypatch):
    import requests
    sleeps = []
    monkeypatch.setattr("ganj_corpus.download.time.sleep", lambda s: sleeps.append(s))
    calls = []

    def fetch(url, headers):
        calls.append(url)
        if len(calls) == 1:
            raise requests.ConnectionError("network drop")
        return 200, b"12345"

    assert download_one(row(), tmp_path, fetch=fetch).read_bytes() == b"12345"
    assert len(calls) == 2 and sleeps == [1.0]


def test_interrupted_decode_leaves_no_wav(tmp_path, monkeypatch):
    import subprocess
    from ganj_corpus.download import decode_wav

    def killed_ffmpeg(cmd, check):
        Path(cmd[-1]).write_bytes(b"RIFF-truncated")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr("ganj_corpus.download.subprocess.run", killed_ffmpeg)
    wav = tmp_path / "wav" / "7.wav"
    with pytest.raises(subprocess.CalledProcessError):
        decode_wav(tmp_path / "7.mp3", wav)
    assert not wav.exists()


def test_interrupted_meta_write_leaves_no_json(tmp_path, monkeypatch):
    from ganj_corpus.download import fetch_meta

    class G:
        def recitation_verses(self, i):
            return [{"verseOrder": 1}]

        def poem(self, i):
            return {"id": i}

    real = Path.write_text

    def half_write(self, text, *a, **k):
        real(self, text[: len(text) // 2], *a, **k)
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", half_write)
    with pytest.raises(OSError):
        fetch_meta(G(), row(), tmp_path)
    assert not (tmp_path / "verses" / "7.json").exists()

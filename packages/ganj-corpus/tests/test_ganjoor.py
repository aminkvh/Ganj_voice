# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
import json

import pytest

from ganj_corpus.ganjoor import Ganjoor, HEADERS, poem_url


class FakeFetch:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, params, headers):
        self.calls.append((url, dict(params or {}), headers))
        status, hdrs, body = self.routes[(url, params.get("PageNumber") if params else None)]
        return status, hdrs, json.dumps(body).encode()


def test_list_recitations_follows_paging_headers():
    page1 = [{"id": 1, "poemId": 10}, {"id": 2, "poemId": 11}]
    page2 = [{"id": 3, "poemId": 12}]
    base = "https://api.ganjoor.net/api/audio/published"
    fetch = FakeFetch({
        (base, 1): (200, {"paging-headers": json.dumps({"totalCount": 3, "hasNextPage": True})}, page1),
        (base, 2): (200, {"paging-headers": json.dumps({"totalCount": 3, "hasNextPage": False})}, page2),
    })
    got = list(Ganjoor(fetch=fetch).list_recitations(page_size=2))
    assert [r["id"] for r in got] == [1, 2, 3]
    assert all(c[2]["User-Agent"].startswith("Ganj-voice-beta") and c[2]["Accept"] == "*/*" for c in fetch.calls)


def test_poem_uses_lean_flags():
    url = "https://api.ganjoor.net/api/ganjoor/poem/2130"
    fetch = FakeFetch({(url, None): (200, {}, {"id": 2130, "verses": []})})
    Ganjoor(fetch=fetch).poem(2130)
    params = fetch.calls[0][1]
    assert params["verseDetails"] == "true" and params["catInfo"] == "true" and params["rhymes"] == "true"
    assert params["recitations"] == "false" and params["comments"] == "false"


def test_get_json_retries_on_503_then_succeeds(monkeypatch):
    url = "https://api.ganjoor.net/api/audio/verses/5"
    calls = {"n": 0}

    def fetch(u, params, headers):
        calls["n"] += 1
        if calls["n"] < 3:
            return 503, {}, b"busy"
        return 200, {}, b"[]"

    sleeps = []
    monkeypatch.setattr("ganj_corpus.ganjoor.time.sleep", lambda s: sleeps.append(s))
    body, _ = Ganjoor(fetch=fetch).get_json(url)
    assert body == [] and calls["n"] == 3 and sleeps == [1.0, 2.0]


def test_poem_url():
    assert poem_url(2130) == "https://api.ganjoor.net/api/ganjoor/poem/2130"


def test_get_json_retries_network_errors_and_honours_retry_after(monkeypatch):
    import requests
    sleeps = []
    monkeypatch.setattr("ganj_corpus.ganjoor.time.sleep", lambda s: sleeps.append(s))
    seq = [requests.Timeout("slow"), (429, {"Retry-After": "7"}, b"{}"), (200, {}, b'{"ok": 1}')]

    def fetch(url, params, headers):
        item = seq.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    body, _ = Ganjoor(fetch=fetch).get_json("https://api.ganjoor.net/x")
    assert body == {"ok": 1} and sleeps == [1.0, 7.0]

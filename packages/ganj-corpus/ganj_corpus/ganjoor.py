# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Thin, polite client for the Ganjoor API (anonymous reads only).

The API sends no gzip, no ETag and no Cache-Control; i.ganjoor.net requires Accept: */*.
Keep concurrency <= 4 (enforced by the caller) and back off on 429/5xx.
"""
import json
import time
from typing import Callable, Iterator, Optional

import requests

BASE = "https://api.ganjoor.net"
UA = "Ganj-voice-beta/0.1 (+https://github.com/aminkvh/Ganj)"
HEADERS = {"User-Agent": UA, "Accept": "*/*"}
LEAN_POEM = {
    "verseDetails": "true", "catInfo": "true", "rhymes": "true", "navigation": "false",
    "recitations": "false", "images": "false", "songs": "false", "comments": "false",
    "catPoems": "false", "quoteds": "false",
}

FetchFn = Callable[[str, Optional[dict], dict], tuple]


def poem_url(poem_id: int) -> str:
    return f"{BASE}/api/ganjoor/poem/{poem_id}"


def _requests_fetch(url, params, headers):
    r = requests.get(url, params=params, headers=headers, timeout=120)
    return r.status_code, dict(r.headers), r.content


class Ganjoor:
    def __init__(self, fetch: Optional[FetchFn] = None, tries: int = 6):
        self.fetch = fetch or _requests_fetch
        self.tries = tries

    def get_json(self, url: str, params: Optional[dict] = None):
        delay = 1.0
        last = None
        for attempt in range(self.tries):
            try:
                status, headers, body = self.fetch(url, params, HEADERS)
            except requests.RequestException as e:
                last = repr(e)
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            if status == 200:
                return json.loads(body.decode("utf-8")), {k.lower(): v for k, v in headers.items()}
            last = (status, body[:200])
            if status == 429 or status >= 500:
                retry_after = {k.lower(): v for k, v in headers.items()}.get("retry-after", "")
                time.sleep(min(float(retry_after), 300.0) if retry_after.isdigit() else delay)
                delay = min(delay * 2, 60.0)
                continue
            break
        raise RuntimeError(f"GET {url} failed: {last}")

    def list_recitations(self, page_size: int = 200) -> Iterator[dict]:
        page = 1
        while True:
            rows, headers = self.get_json(f"{BASE}/api/audio/published", {"PageNumber": page, "PageSize": page_size})
            yield from rows
            paging = json.loads(headers.get("paging-headers", "{}"))
            if not rows or not paging.get("hasNextPage", False):
                return
            page += 1

    def recitation_verses(self, rec_id: int) -> list:
        rows, _ = self.get_json(f"{BASE}/api/audio/verses/{rec_id}")
        return rows

    def poem(self, poem_id: int) -> dict:
        body, _ = self.get_json(poem_url(poem_id), dict(LEAN_POEM))
        return body

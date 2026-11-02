# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Amin Akbari
"""Listening page across runs (same look as beta/tts-samples/listen.html)."""
import html
import json
from pathlib import Path

HEAD = """<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><title>آوای گنج · ارزیابی</title>
<style>body{font-family:Vazirmatn,Tahoma,sans-serif;background:#fbf6ea;color:#17130f;padding:24px}
.card{background:#fffdf7;border:2px solid #b88828;border-radius:18px;padding:14px 20px;margin:14px 0}
pre{font-family:inherit;font-size:1.1rem;line-height:2;white-space:pre-wrap;margin:0 0 8px}
.v{display:flex;gap:10px;align-items:center;margin:4px 0}.v b{width:9em;color:#635543}audio{width:100%}</style></head><body>
<h1>آوای گنج · نمونه‌های ارزیابی</h1>"""


def render_html(items, tags) -> str:
    parts = [HEAD]
    for it in items:
        parts.append(f'<div class="card"><small>{html.escape(it["key"])} · {html.escape(it.get("cls", ""))}</small><pre>{html.escape(chr(10).join(it["lines"]))}</pre>')
        parts.append(f'<div class="v"><b>خوانش انسانی</b><audio controls preload="none" src="{html.escape(it["human"])}"></audio></div>')
        for t in tags:
            parts.append(f'<div class="v"><b>{html.escape(t)}</b><audio controls preload="none" src="runs/{html.escape(t)}/{html.escape(it["key"])}.wav"></audio></div>')
        parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def render_listen(data: Path) -> Path:
    items = [json.loads(l) for l in open(data / "eval" / "set.jsonl", encoding="utf-8")]
    for it in items:
        it["human"] = "../" + it["human"]
    tags = sorted(p.name for p in (data / "eval" / "runs").glob("*") if (p / "results.json").exists())
    out = data / "eval" / "listen.html"
    out.write_text(render_html(items, tags), encoding="utf-8")
    return out

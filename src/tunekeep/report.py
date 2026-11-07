"""`tunekeep report`: one self-contained HTML file with the migration evidence.

No scripts, no external requests (a strict Content-Security-Policy is set), and
every value from your data is HTML-escaped: prompts can contain anything.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path
from typing import Any

from . import __version__
from .compare import golden_baseline, load_golden, load_rows, summarize
from .record import TERMS_NOTE
from .score import message_text, tool_calls

MAX_DIFFS = 12
MAX_CHARS = 1500


def e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def pct(x: float | None) -> str:
    return "–" if x is None else f"{x * 100:.1f}%"


def _clip(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + f"… ({len(text) - MAX_CHARS:,} more characters)"


def show_message(msg: Any) -> str:
    """A message as readable text: content, or its tool calls."""
    calls = tool_calls(msg)
    if calls:
        return "\n".join(f"{name}({json.dumps(args, ensure_ascii=False) if not isinstance(args, str) else args})"
                         for name, args in calls)
    return message_text(msg)


def last_user(messages: list[dict[str, Any]]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return message_text(m)
    return ""


def meter(x: float | None) -> str:
    if x is None:
        return '<span class="muted">–</span>'
    w = max(0.0, min(1.0, x)) * 100
    return (f'<span class="meter" title="{pct(x)}"><span class="fill" style="width:{w:.1f}%"></span></span>'
            f'<span class="num">{pct(x)}</span>')


def build(
    golden_files: list[Path],
    compare_files: list[Path],
    inventory: Path | None = None,
    retrain_summaries: list[Path] | None = None,
) -> str:
    inv = json.loads(inventory.read_text()) if inventory and inventory.exists() else None
    golden: dict[tuple[str, str], dict[str, Any]] = {}
    baselines = []
    for g in golden_files:
        rows = load_golden(g)
        for r in rows:
            golden[(r.get("model"), r["id"])] = r
        b = golden_baseline(rows)
        if b:
            baselines.append(b)
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in compare_files:
        for label, rows in load_rows(c).items():
            groups.setdefault(label, []).extend(rows)
    summaries = [summarize(rows) for rows in groups.values()]
    summaries.sort(key=lambda s: -(s["vs_golden"].get("mean") or 0))
    trained = []
    for p in retrain_summaries or []:
        try:
            trained.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue

    parts = [_head(), '<main>', '<header><p class="eyebrow">tunekeep report</p>',
             '<h1>Can a replacement answer like your fine-tune?</h1>',
             f'<p class="muted">Generated {e(dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))} '
             f'by tunekeep {e(__version__)}.</p></header>']
    if inv:
        parts.append(_inventory(inv))
    if summaries or baselines:
        parts.append(_candidates(summaries, baselines))
    if trained:
        parts.append(_retrain(trained))
    for label, rows in groups.items():
        parts.append(_diffs(label, rows, golden))
    parts.append(_method())
    parts.append(f'<footer><p class="muted">{e(TERMS_NOTE)}</p></footer></main></body></html>')
    return "\n".join(parts)


def _inventory(inv: dict[str, Any]) -> str:
    models = [m for m in inv.get("models") or [] if not m.get("checkpoint_of")]
    alive = [m["days_left"] for m in models if isinstance(m.get("days_left"), int) and m["days_left"] >= 0]
    training_ends = inv.get("training_ends")
    tiles = [
        ("Fine-tuned models", str(len(models)), "found for this key"),
        ("Days left", str(min(alive)) if alive else "–", "until the first shutdown"),
        ("Training ends", training_ends or "–", "last day for new fine-tuning jobs"),
    ]
    rows = "".join(
        f"<tr><td class=\"mono\">{e(m['id'])}</td><td>{e(m.get('shutdown_date') or '–')}</td>"
        f"<td class=\"r\">{e(m.get('days_left') if m.get('days_left') is not None else '–')}</td>"
        f"<td>{e(m.get('shutdown_source'))}</td></tr>"
        for m in models
    )
    return (
        '<section><h2>Deadline</h2><div class="tiles">'
        + "".join(f'<div class="tile"><p class="label">{e(a)}</p><p class="big">{e(b)}</p><p class="muted">{e(c)}</p></div>'
                  for a, b, c in tiles)
        + '</div>'
        + (f'<div class="scroll"><table><thead><tr><th>Model</th><th>Shuts down</th><th class="r">Days left</th><th>Source</th></tr></thead>'
           f'<tbody>{rows}</tbody></table></div>' if rows else "")
        + '</section>'
    )


def _candidates(summaries: list[dict[str, Any]], baselines: list[dict[str, Any]]) -> str:
    def row(s: dict[str, Any], baseline: bool = False) -> str:
        g, r = s["vs_golden"], s["vs_reference"]
        lat = "–" if s.get("latency_p50_ms") is None else f"{s['latency_p50_ms']:,} / {s['latency_p95_ms']:,} ms"
        tokens = f"{s['prompt_tokens']:,} / {s['completion_tokens']:,}"
        if s.get("reasoning_tokens"):
            tokens += f" <span class=\"muted\">({s['reasoning_tokens']:,} reasoning)</span>"
        cls = ' class="baseline"' if baseline else ""
        return (
            f"<tr{cls}><th scope=\"row\">{e(s['candidate'])}</th><td class=\"r\">{s['rows']:,}</td>"
            f"<td class=\"m\">{meter(g.get('mean')) if g.get('n') else '<span class=muted>baseline</span>'}</td>"
            f"<td class=\"r\">{pct(g.get('match')) if g.get('n') else '–'}</td>"
            f"<td class=\"m\">{meter(r.get('mean')) if r.get('n') else '<span class=muted>–</span>'}</td>"
            f"<td class=\"r\">{pct(g.get('json_valid')) if g.get('json_valid') is not None else '–'}</td>"
            f"<td class=\"r\">{lat}</td><td class=\"r\">{tokens}</td></tr>"
        )

    body = "".join(row(s) for s in summaries) + "".join(row(b, True) for b in baselines)
    return (
        '<section><h2>Candidates</h2>'
        '<p class="muted">Score vs recorded: 100% means the same answers as the recorded fine-tune. '
        'Score vs training answers compares with the answers in your training file, when the prompts came from it; '
        'the recorded fine-tune\'s own row is the baseline for that column.</p>'
        '<div class="scroll"><table><thead><tr><th>Candidate</th><th class="r">Prompts</th><th>Score vs recorded</th>'
        '<th class="r">Same answer</th><th>Score vs training answers</th><th class="r">Valid JSON</th>'
        '<th class="r">Latency p50 / p95</th><th class="r">Tokens in / out</th></tr></thead>'
        f'<tbody>{body}</tbody></table></div></section>'
    )


def _retrain(trained: list[dict[str, Any]]) -> str:
    rows = []
    for t in trained:
        by_epoch = [x for x in t.get("validation_loss_by_epoch") or [] if x is not None]
        before = t.get("validation_loss_before")
        after = by_epoch[-1] if by_epoch else None
        loss = "–" if before is None or after is None else f"{before:.3f} → {after:.3f}"
        minutes = round((t.get("train_seconds") or 0) / 60, 1)
        rows.append(
            f"<tr><td class=\"mono\">{e(t.get('base'))}"
            + (f"<br><span class=\"muted\">rev {e(t['revision'][:12])}</span>" if t.get("revision") else "") + "</td>"
            f"<td>{e(t.get('device'))}</td><td class=\"r\">{t.get('train_samples', 0):,}</td>"
            f"<td class=\"r\">{t.get('epochs')}</td><td class=\"r\">{minutes}</td><td class=\"r\">{e(loss)}</td></tr>"
        )
    return (
        '<section><h2>Retrained open model</h2>'
        '<div class="scroll"><table><thead><tr><th>Base model</th><th>GPU</th><th class="r">Samples</th><th class="r">Epochs</th>'
        '<th class="r">Minutes</th><th class="r">Validation loss</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div></section>'
    )


def _diffs(label: str, rows: list[dict[str, Any]], golden: dict[tuple[str, str], dict[str, Any]]) -> str:
    worst = sorted(rows, key=lambda r: ((r.get("scores") or {}).get("golden") or {}).get("score", 0))[:MAX_DIFFS]
    worst = [r for r in worst if (((r.get("scores") or {}).get("golden") or {}).get("score", 0)) < 1]
    if not worst:
        return f'<section><h2>{e(label)}</h2><p>Every answer matched the recorded fine-tune.</p></section>'
    items = []
    for r in worst:
        g = golden.get((r.get("golden_model"), r["id"])) or {}
        sc = (r.get("scores") or {}).get("golden") or {}
        prompt = last_user((g.get("request") or {}).get("messages") or [])
        items.append(
            f'<details><summary><span class="num">{pct(sc.get("score"))}</span> {e(sc.get("kind"))} · line {e(r.get("line"))}'
            f' · <span class="muted">{e(_clip(prompt)[:120])}</span></summary>'
            f'<div class="pair"><div><p class="label">Prompt</p><pre>{e(_clip(prompt))}</pre></div></div>'
            f'<div class="pair two"><div><p class="label">Recorded fine-tune</p>'
            f'<pre>{e(_clip(show_message((g.get("response") or {}).get("message"))))}</pre></div>'
            f'<div><p class="label">{e(label)}</p><pre>{e(_clip(show_message((r.get("response") or {}).get("message"))))}</pre></div></div>'
            '</details>'
        )
    return (f'<section><h2>Biggest differences: {e(label)}</h2>'
            f'<p class="muted">The {len(worst)} lowest-scoring prompts. Read these before trusting any number above.</p>'
            + "".join(items) + '</section>')


def _method() -> str:
    return (
        '<section><h2>How answers are scored</h2><ul>'
        '<li><b>Tool calls</b>: matched by function name; arguments compared field by field as JSON.</li>'
        '<li><b>JSON answers</b>: the share of leaf fields with the same value. Invalid JSON scores 0.</li>'
        '<li><b>Text</b>: 100% when equal after normalising case and spacing, otherwise word overlap (F1). '
        'Overlap is a rough signal for free text: two good answers can share few words.</li>'
        '<li><b>Same answer</b>: the share of prompts with an exact or normalised match.</li>'
        '</ul></section>'
    )


def _head() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>tunekeep report</title>
<style>
:root{color-scheme:light;--bg:#fcfcfb;--surface:#f3f2ee;--fg:#0b0b0b;--muted:#52514e;--line:#dcdad3;--series-1:#2a78d6;--track:#e3e1da}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--bg:#1a1a19;--surface:#232321;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3936;--series-1:#3987e5;--track:#34332f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:clamp(1.6rem,4vw,2.4rem);line-height:1.15;margin:.2em 0 .3em;letter-spacing:-.01em}
h2{font-size:1.15rem;margin:0 0 .6em}
section{margin-top:40px}
.eyebrow,.label{text-transform:uppercase;letter-spacing:.08em;font-size:.72rem;color:var(--muted);margin:0}
.muted{color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:16px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.tile .big{font-size:2rem;font-weight:650;margin:.1em 0;font-variant-numeric:tabular-nums}
.tile p{margin:0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}
thead th{font-size:.78rem;color:var(--muted);font-weight:600}
.r{text-align:right}
td.mono{overflow-wrap:anywhere;min-width:12ch}
.mono,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:.85rem}
tr.baseline th,tr.baseline td{color:var(--muted)}
td.m{white-space:nowrap}
.meter{display:inline-block;width:90px;height:8px;border-radius:4px;background:var(--track);vertical-align:middle;margin-right:8px;overflow:hidden}
.meter .fill{display:block;height:100%;background:var(--series-1);border-radius:4px}
.num{font-variant-numeric:tabular-nums;font-weight:600}
details{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:var(--surface)}
summary{cursor:pointer;padding:10px 14px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pair{display:grid;gap:12px;padding:0 14px 12px}
.pair.two{grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
pre{white-space:pre-wrap;word-break:break-word;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;margin:4px 0 0}
footer{margin-top:48px;border-top:1px solid var(--line);padding-top:16px}
</style></head><body>"""

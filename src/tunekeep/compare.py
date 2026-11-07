"""`tunekeep compare`: does another model already answer like your fine-tune?

Sends each recorded prompt to a candidate model (gpt-6-luna, any
OpenAI-compatible endpoint, or your retrained model on llama-server or vLLM),
optionally with a few examples from your own training data, and scores the
answer against the recorded one (and against the training reference if any).
"""

from __future__ import annotations

import datetime as dt
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .client import Client
from .record import load_prompts
from .runner import RunResult, done_ids, iter_jsonl, run_parallel
from .score import message_text, score

EFFORTS = ("auto", "default", "none", "minimal", "low", "medium", "high", "xhigh", "max")
# Request fields carried over from the recorded request.
_CARRY = ("tools", "tool_choice", "parallel_tool_calls", "response_format")


class CompareError(ValueError):
    pass


# Chat Completions rules for OpenAI reasoning models (checked 2026-11-07):
#   https://developers.openai.com/api/docs/guides/latest-model
#   https://developers.openai.com/api/docs/models/gpt-6-luna
# - GPT-6 Luna/Sol/Terra and GPT-5.1+ accept reasoning_effort "none"; temperature, top_p and
#   function tools work in Chat Completions only at "none".
# - GPT-6 Astra and GPT-6.1 Sol have no "none" and no function tools in Chat Completions.
# - o-series: no "none", no temperature.
def family(model: str) -> str:
    m = model.lower()
    if m.startswith(("gpt-6-astra", "gpt-6.1-sol")):
        return "no-none"
    if re.match(r"^o\d", m):
        return "o-series"
    if m in ("gpt-5", "gpt-5-mini", "gpt-5-nano") or re.match(r"^gpt-5-(mini|nano)?-?\d{4}-", m):
        return "gpt-5"
    if m.startswith(("gpt-6", "gpt-5.")):
        return "none-ok"
    return "plain"


def plan_params(model: str, effort: str, temperature: float | None, has_tools: bool) -> tuple[dict[str, Any], list[str]]:
    """Request parameters for the candidate, and warnings about what was changed."""
    fam = family(model)
    notes: list[str] = []
    if effort == "auto":
        effort = {"none-ok": "none", "no-none": "low", "o-series": "low", "gpt-5": "minimal"}.get(fam, "default")
    params: dict[str, Any] = {}
    if effort != "default":
        params["reasoning_effort"] = effort
    sampling_ok = fam == "plain" or (fam == "none-ok" and effort == "none")
    if temperature is not None:
        if sampling_ok:
            params["temperature"] = temperature
        else:
            notes.append(f"temperature not sent: {model} accepts it only with reasoning_effort none")
    if has_tools:
        if fam == "no-none":
            raise CompareError(f"{model} does not support function tools in Chat Completions.")
        if fam == "none-ok" and effort != "none":
            raise CompareError(
                f"{model} supports function tools in Chat Completions only with --reasoning-effort none."
            )
    return params, notes


@dataclass
class Shot:
    id: str
    messages: list[dict[str, Any]]


def load_shots(path: Path) -> list[Shot]:
    """Examples from your own training file: single-turn, plain-text answers."""
    shots: list[Shot] = []
    for p in load_prompts(path):
        ref = p.reference
        if not isinstance(ref, dict) or ref.get("tool_calls") or not message_text(ref):
            continue
        turns = [m for m in p.messages if m.get("role") not in ("system", "developer")]
        if len(turns) != 1 or turns[0].get("role") != "user" or p.extra.get("tools"):
            continue
        shots.append(Shot(p.id, [turns[0], {"role": "assistant", "content": message_text(ref)}]))
    return shots


def with_shots(messages: list[dict[str, Any]], shots: list[Shot], n: int, exclude: str) -> tuple[list[dict[str, Any]], int]:
    chosen = [s for s in shots if s.id != exclude][:n]
    if not chosen:
        return messages, 0
    head = [m for m in messages[:1] if m.get("role") in ("system", "developer")]
    examples = [m for s in chosen for m in s.messages]
    return head + examples + messages[len(head):], len(chosen)


def load_golden(path: Path) -> list[dict[str, Any]]:
    rows, seen = [], set()
    for _, row in iter_jsonl(path):
        if row.get("id") and row["id"] not in seen and isinstance(row.get("request"), dict):
            seen.add(row["id"])
            rows.append(row)
    if not rows:
        raise CompareError(f"{path}: no recorded rows. Run `tunekeep record` first.")
    rows.sort(key=lambda r: r.get("line") if isinstance(r.get("line"), int) else 0)  # file order, not finish order
    return rows


def run(
    client: Client,
    golden: list[dict[str, Any]],
    out: Path,
    *,
    candidate: str,
    label: str,
    effort: str = "auto",
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
    shots: list[Shot] | None = None,
    n_shots: int = 0,
    extra_body: dict[str, Any] | None = None,
    workers: int = 4,
    progress=None,
) -> tuple[RunResult, list[str]]:
    has_tools = any(row["request"].get("tools") for row in golden)
    params, notes = plan_params(candidate, effort, temperature, has_tools)
    done = done_ids(out, "candidate", label)
    todo = [row for row in golden if row["id"] not in done]

    def one(row: dict[str, Any]) -> dict[str, Any]:
        req = row["request"]
        messages, used = with_shots(req.get("messages") or [], shots or [], n_shots, row["id"])
        body: dict[str, Any] = {"model": candidate, "messages": messages, **params}
        body.update({k: req[k] for k in _CARRY if k in req})
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        body.update(extra_body or {})
        start = time.monotonic()
        resp = client.chat(body)
        latency = round((time.monotonic() - start) * 1000)
        choice = (resp.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        ref = row.get("reference")
        return {
            "id": row["id"],
            "line": row.get("line"),
            "candidate": label,
            "model": candidate,
            "golden_model": row.get("model"),
            "shots": used,
            "params": {k: v for k, v in body.items() if k not in ("model", "messages", "tools")},
            "response": {"message": message, "finish_reason": choice.get("finish_reason")},
            "scores": {
                "golden": score((row.get("response") or {}).get("message"), message),
                "reference": score(ref, message) if isinstance(ref, dict) else None,
            },
            "usage": resp.get("usage"),
            "latency_ms": latency,
            "compared_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }

    result = run_parallel(todo, one, out, total=len(golden), workers=workers, progress=progress)
    return result, notes


# -- summaries ---------------------------------------------------------------------


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [s for s in scores if s]
    if not scores:
        return {"n": 0}
    kinds: dict[str, int] = {}
    for s in scores:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    json_rows = [s for s in scores if s["kind"] == "json"]
    return {
        "n": len(scores),
        "mean": round(statistics.fmean(s["score"] for s in scores), 4),
        "exact": round(sum(1 for s in scores if s.get("exact")) / len(scores), 4),
        "match": round(sum(1 for s in scores if s.get("exact") or s.get("normalized") or s["score"] >= 0.999)
                       / len(scores), 4),
        "json_valid": round(sum(1 for s in json_rows if s.get("json_valid")) / len(json_rows), 4) if json_rows else None,
        "kinds": kinds,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lat = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    usage = [r.get("usage") or {} for r in rows]
    details = [u.get("completion_tokens_details") or {} for u in usage]
    return {
        "candidate": rows[0]["candidate"] if rows else None,
        "model": rows[0].get("model") if rows else None,
        "rows": len(rows),
        "vs_golden": summarize_scores([(r.get("scores") or {}).get("golden") for r in rows]),
        "vs_reference": summarize_scores([(r.get("scores") or {}).get("reference") for r in rows]),
        "latency_p50_ms": _round(_pct(lat, 0.5)),
        "latency_p95_ms": _round(_pct(lat, 0.95)),
        "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
        "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
        "reasoning_tokens": sum(d.get("reasoning_tokens") or 0 for d in details),
    }


def golden_baseline(golden: list[dict[str, Any]]) -> dict[str, Any] | None:
    """How closely the fine-tune itself matched its training answers (no API calls)."""
    scored = [score(r["reference"], (r.get("response") or {}).get("message"))
              for r in golden if isinstance(r.get("reference"), dict)]
    if not scored:
        return None
    lat = [r["latency_ms"] for r in golden if isinstance(r.get("latency_ms"), (int, float))]
    usage = [r.get("usage") or {} for r in golden]
    return {
        "candidate": f"{golden[0].get('model')} (recorded)",
        "model": golden[0].get("model"),
        "rows": len(golden),
        "vs_golden": {"n": 0},
        "vs_reference": summarize_scores(scored),
        "latency_p50_ms": _round(_pct(lat, 0.5)),
        "latency_p95_ms": _round(_pct(lat, 0.95)),
        "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
        "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
        "reasoning_tokens": 0,
    }


def _round(x: float | None) -> int | None:
    return None if x is None else round(x)


def load_rows(path: Path) -> dict[str, list[dict[str, Any]]]:
    """Compare rows grouped by candidate label (a file can hold several runs)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for _, row in iter_jsonl(path):
        if row.get("candidate") and row.get("scores"):
            groups.setdefault(row["candidate"], []).append(row)
    return groups


def render_summary(s: dict[str, Any]) -> str:
    def pct(x: float | None) -> str:
        return "-" if x is None else f"{x * 100:.1f}%"

    g, r = s["vs_golden"], s["vs_reference"]
    lines = [f"{s['candidate']}: {s['rows']} prompts"]
    if g.get("n"):
        lines.append(f"  vs recorded fine-tune: score {pct(g['mean'])}, same answer {pct(g['match'])}, exact {pct(g['exact'])}"
                     + (f", valid JSON {pct(g['json_valid'])}" if g.get("json_valid") is not None else ""))
    if r.get("n"):
        lines.append(f"  vs training answers:   score {pct(r['mean'])}, same answer {pct(r['match'])}")
    if s["latency_p50_ms"] is not None:
        lines.append(f"  latency p50 {s['latency_p50_ms']} ms, p95 {s['latency_p95_ms']} ms; "
                     f"tokens in {s['prompt_tokens']:,}, out {s['completion_tokens']:,}"
                     + (f" (reasoning {s['reasoning_tokens']:,})" if s["reasoning_tokens"] else ""))
    return "\n".join(lines)


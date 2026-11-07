"""Score a candidate answer against an expected one. Standard library only.

Three kinds of answer, picked from the expected message:

- tool: the expected answer calls tools. Calls are matched by name, arguments are
  compared as parsed JSON (field by field).
- json: the expected text parses as JSON. Leaf fields are compared.
- text: anything else. Exact after normalising, otherwise token overlap (F1).

`score` is always in [0, 1]. For text, token F1 is a rough signal, not a verdict:
read the side-by-side diffs in the report before deciding.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from typing import Any

_THINK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
_WORD = re.compile(r"\w+", re.UNICODE)


def message_text(msg: Any) -> str:
    """Text content of a chat message (string or a list of text parts)."""
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in ("text", "output_text"))
    else:
        text = ""
    # Local reasoning models (Qwen3 and others) may prefix a thinking block.
    return _THINK.sub("", text, count=1)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = " ".join(text.split())
    return text.strip().rstrip(".!")


def token_f1(expected: str, got: str) -> float:
    a = _WORD.findall(normalize(expected))
    b = _WORD.findall(normalize(got))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    common = sum((Counter(a) & Counter(b)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(b), common / len(a)
    return 2 * precision * recall / (precision + recall)


def parse_json(text: str) -> tuple[bool, Any]:
    m = _FENCE.match(text)
    if m:
        text = m.group(1)
    text = text.strip()
    if not text or text[0] not in "{[":
        return False, None
    try:
        return True, json.loads(text)
    except ValueError:
        return False, None


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Leaf paths of a JSON value: {"a.b[0]": 1, ...}. Empty containers are leaves."""
    out: dict[str, Any] = {}
    if isinstance(value, dict) and value:
        for k, v in value.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, list) and value:
        for i, v in enumerate(value):
            out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = value
    return out


def json_field_score(expected: Any, got: Any) -> float:
    a, b = flatten(expected), flatten(got)
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    same = sum(1 for k in keys if k in a and k in b and _same_leaf(a[k], b[k]))
    return same / len(keys)


def _same_leaf(x: Any, y: Any) -> bool:
    if isinstance(x, str) and isinstance(y, str):
        return normalize(x) == normalize(y)
    if isinstance(x, bool) or isinstance(y, bool):
        return x is y
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return abs(x - y) <= 1e-9 * max(1.0, abs(x), abs(y))
    return x == y


def tool_calls(msg: Any) -> list[tuple[str, Any]]:
    """(name, parsed arguments) for each tool call in a message."""
    calls: list[tuple[str, Any]] = []
    raw = msg.get("tool_calls") if isinstance(msg, dict) else None
    for c in raw or []:
        fn = (c.get("function") if isinstance(c, dict) else None) or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            ok, parsed = parse_json(args) if args.strip() else (True, {})
            args = parsed if ok else args
        calls.append((fn.get("name") or "", args))
    return calls


def tool_score(expected: list[tuple[str, Any]], got: list[tuple[str, Any]]) -> tuple[float, bool]:
    """Greedy, order-insensitive matching. Returns (score, exact)."""
    if not expected and not got:
        return 1.0, True
    unused = list(range(len(got)))
    total = 0.0
    for name, args in expected:
        best, best_i = 0.0, None
        for i in unused:
            g_name, g_args = got[i]
            if g_name != name:
                continue
            s = json_field_score(args, g_args) if not isinstance(args, str) and not isinstance(g_args, str) else float(args == g_args)
            if best_i is None or s > best:
                best, best_i = s, i
        if best_i is not None:
            unused.remove(best_i)
            total += best
    score = total / max(len(expected), len(got))
    return score, score == 1.0


def score(expected: Any, got: Any) -> dict[str, Any]:
    """Compare two assistant messages. `expected` is the recorded or reference answer."""
    exp_calls, got_calls = tool_calls(expected), tool_calls(got)
    if exp_calls:
        s, exact = tool_score(exp_calls, got_calls)
        return {"kind": "tool", "score": round(s, 4), "exact": exact,
                "tool_names": sorted(n for n, _ in exp_calls) == sorted(n for n, _ in got_calls)}

    exp_text, got_text = message_text(expected), message_text(got)
    if got_calls and not got_text:
        # Expected text, got a tool call instead.
        return {"kind": "text", "score": 0.0, "exact": False, "normalized": False, "note": "answered with a tool call"}

    is_json, exp_json = parse_json(exp_text)
    if is_json:
        valid, got_json = parse_json(got_text)
        s = json_field_score(exp_json, got_json) if valid else 0.0
        return {"kind": "json", "score": round(s, 4), "exact": valid and exp_json == got_json, "json_valid": valid}

    exact = exp_text.strip() == got_text.strip()
    normalized = normalize(exp_text) == normalize(got_text)
    s = 1.0 if normalized else token_f1(exp_text, got_text)
    return {"kind": "text", "score": round(s, 4), "exact": exact, "normalized": normalized}

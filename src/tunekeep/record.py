"""`tunekeep record`: capture how a fine-tuned model answers, while it still answers.

The output is a golden set for evaluation. It is never used as training data:
OpenAI's Services Agreement (section 3.3(e)) restricts using Output to develop
models that compete with OpenAI. Retraining uses only your original data.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .client import APIError, Client
from .files import open_private_append

TERMS_NOTE = (
    "Recorded outputs are for evaluation only. tunekeep never trains on them "
    "(OpenAI Services Agreement, section 3.3(e))."
)

# Fields OpenAI's training format allows on messages that Chat Completions rejects.
_TRAINING_ONLY_KEYS = {"weight"}
_REQUEST_KEYS = ("tools", "parallel_tool_calls", "response_format")


class PromptError(ValueError):
    pass


@dataclass
class Prompt:
    id: str
    line: int
    messages: list[dict[str, Any]]
    extra: dict[str, Any]
    reference: Any


def _clean_message(msg: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in msg.items() if k not in _TRAINING_ONLY_KEYS}


def parse_line(obj: Any, line: int) -> Prompt:
    """Accept OpenAI SFT lines, plain prompt lines, or DPO lines."""
    if not isinstance(obj, dict):
        raise PromptError(f"line {line}: expected a JSON object")
    reference: Any = None
    if "messages" in obj:
        messages = obj["messages"]
        if not isinstance(messages, list) or not messages:
            raise PromptError(f"line {line}: 'messages' must be a non-empty list")
        if isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
            reference = _clean_message(messages[-1])
            messages = messages[:-1]
        extra_src = obj
    elif isinstance(obj.get("input"), dict) and "messages" in obj["input"]:
        messages = obj["input"]["messages"]
        reference = obj.get("preferred_output")
        extra_src = obj["input"]
    else:
        raise PromptError(f"line {line}: no 'messages' (or DPO 'input.messages') found")
    if not messages:
        raise PromptError(f"line {line}: nothing left to send after removing the assistant answer")
    clean = [_clean_message(m) for m in messages if isinstance(m, dict)]
    extra = {k: extra_src[k] for k in _REQUEST_KEYS if k in extra_src}
    digest = hashlib.sha256(json.dumps({"m": clean, "x": extra}, sort_keys=True).encode()).hexdigest()[:16]
    return Prompt(id=digest, line=line, messages=clean, extra=extra, reference=reference)


def load_prompts(path: Path) -> list[Prompt]:
    prompts: list[Prompt] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise PromptError(f"line {n}: invalid JSON ({e.msg})") from None
            p = parse_line(obj, n)
            if p.id not in seen:
                seen.add(p.id)
                prompts.append(p)
    return prompts


def already_recorded(out: Path, model: str) -> set[str]:
    done: set[str] = set()
    if not out.exists():
        return done
    with out.open("r", encoding="utf-8") as fh:
        for raw in fh:
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue  # a partial last line from an interrupted run
            if rec.get("model") == model and rec.get("id"):
                done.add(rec["id"])
    return done


def build_request(p: Prompt, model: str, temperature: float | None, max_tokens: int | None, seed: int | None) -> dict[str, Any]:
    body: dict[str, Any] = {"model": model, "messages": p.messages, **p.extra}
    if temperature is not None:
        body["temperature"] = temperature
    if max_tokens is not None:
        body["max_completion_tokens"] = max_tokens
    if seed is not None:
        body["seed"] = seed
    return body


def estimate_tokens(prompts: Iterable[Prompt]) -> int:
    return sum(len(json.dumps(p.messages)) // 4 for p in prompts)


@dataclass
class RecordResult:
    total: int
    skipped: int
    recorded: int
    failed: int
    first_error: str | None


def run(
    client: Client,
    model: str,
    prompts: list[Prompt],
    out: Path,
    *,
    workers: int = 4,
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
    seed: int | None = None,
    progress=sys.stderr,
) -> RecordResult:
    done = already_recorded(out, model)
    todo = [p for p in prompts if p.id not in done]
    lock = threading.Lock()
    recorded = failed = 0
    first_error: str | None = None

    def one(p: Prompt) -> dict[str, Any]:
        body = build_request(p, model, temperature, max_tokens, seed)
        start = time.monotonic()
        resp = client.chat(body)
        latency = round((time.monotonic() - start) * 1000)
        choice = (resp.get("choices") or [{}])[0]
        return {
            "id": p.id,
            "line": p.line,
            "model": model,
            "request": {k: v for k, v in body.items() if k != "model"},
            "response": {"message": choice.get("message"), "finish_reason": choice.get("finish_reason")},
            "reference": p.reference,
            "usage": resp.get("usage"),
            "system_fingerprint": resp.get("system_fingerprint"),
            "latency_ms": latency,
            "recorded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }

    with open_private_append(out) as fh, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, p): p for p in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                rec = fut.result()
            except APIError as e:
                failed += 1
                if first_error is None:
                    first_error = f"line {futures[fut].line}: {e}"
                if e.status in (401, 403, 404):
                    # Wrong key, no access or model gone: no point sending the rest.
                    for f in futures:
                        f.cancel()
                    break
                continue
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                recorded += 1
            if progress and (i % 25 == 0 or i == len(todo)):
                print(f"  {i}/{len(todo)} done", file=progress)
    return RecordResult(len(prompts), len(prompts) - len(todo), recorded, failed, first_error)

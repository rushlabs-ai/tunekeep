"""`tunekeep retrain`: turn your OpenAI training file into a LoRA run on an open model.

Converts OpenAI fine-tuning JSONL (chat format, tools, legacy functions,
per-message weights) into clean chat data, checks it, and writes a small
self-contained kit: the data, a config, a pinned training script and a Modal
app. Nothing is uploaded by this command.

Only your own training data is used. Recorded model outputs are never training
data (OpenAI Services Agreement, section 3.3(e)).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from . import __version__
from .files import ensure_private_dir, write_private

# Default base model, pinned to an exact Hugging Face revision (Apache-2.0).
DEFAULT_BASE = "Qwen/Qwen3-1.7B"
KNOWN_REVISIONS = {
    "Qwen/Qwen3-0.6B": "c1899de289a04d12100db370d81485cdf75e47ca",
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
}
KIT_FILES = ("README.md", "train_lora.py", "modal_app.py", "requirements-train.txt", "requirements-gguf.txt")
GPUS = ("T4", "L4", "A10G", "L40S", "A100-40GB", "A100-80GB", "H100")


class DataError(ValueError):
    pass


@dataclass
class Report:
    lines: int = 0
    examples: int = 0
    trained_turns: int = 0
    with_tools: int = 0
    chars: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def warn(self, text: str) -> None:
        if len(self.warnings) < 20:
            self.warnings.append(text)


def _text(content: Any) -> str | None:
    """Plain text from a content string or list of parts; None if it holds non-text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if not isinstance(p, dict) or p.get("type") not in ("text", "input_text", "output_text"):
                return None
            parts.append(p.get("text") or "")
        return "".join(parts)
    return str(content)


def _args(value: Any, line: int, report: Report) -> Any:
    """Tool arguments as a dict: chat templates (Llama, Gemma, Qwen2.5) serialise them with tojson."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            parsed = json.loads(value)
            return parsed
        except ValueError:
            report.warn(f"line {line}: tool arguments are not valid JSON; kept as a string")
    return value


def convert_line(obj: Any, line: int, report: Report) -> dict[str, Any] | None:
    """One OpenAI training line -> {"messages": [...], "tools": [...]} or None (skipped)."""
    if not isinstance(obj, dict):
        report.skip("not a JSON object")
        return None
    if "input" in obj and "preferred_output" in obj:
        report.skip("preference (DPO) line: not supported in v0")
        return None
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        report.skip("no messages")
        return None

    tools = obj.get("tools")
    if not tools and obj.get("functions"):
        tools = [{"type": "function", "function": f} for f in obj["functions"] if isinstance(f, dict)]
    out: list[dict[str, Any]] = []
    call_ids: list[str] = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("system", "developer", "user", "assistant", "tool", "function"):
            report.skip("unknown message role")
            return None
        role = m["role"]
        text = _text(m.get("content"))
        if text is None:
            report.skip("images or audio in content: not supported in v0")
            return None
        if role == "developer":
            role = "system"  # open chat templates have no developer role
        msg: dict[str, Any] = {"role": role, "content": text}
        if role == "assistant":
            calls = m.get("tool_calls") or []
            if not calls and isinstance(m.get("function_call"), dict):
                calls = [{"id": f"call_{len(call_ids)}", "type": "function", "function": m["function_call"]}]
            if calls:
                msg["tool_calls"] = []
                for c in calls:
                    fn = c.get("function") or {}
                    cid = c.get("id") or f"call_{len(call_ids)}"
                    call_ids.append(cid)
                    msg["tool_calls"].append({"id": cid, "type": "function", "function": {
                        "name": fn.get("name") or "", "arguments": _args(fn.get("arguments"), line, report)}})
            weight = m.get("weight", 1)
            if weight not in (0, 1):
                report.warn(f"line {line}: weight {weight!r} treated as 1")
                weight = 1
            msg["weight"] = weight
        elif role == "function":  # legacy function result
            msg = {"role": "tool", "content": text, "tool_call_id": call_ids[-1] if call_ids else "call_0"}
            if m.get("name"):
                msg["name"] = m["name"]
        elif role == "tool":
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
        if m.get("name") and role in ("user", "system"):
            msg["name"] = m["name"]
        out.append(msg)

    trained = sum(1 for m in out if m["role"] == "assistant" and m.get("weight", 1) == 1)
    if trained == 0:
        report.skip("no assistant message to train on")
        return None
    report.examples += 1
    report.trained_turns += trained
    if tools:
        report.with_tools += 1
    row: dict[str, Any] = {"messages": out}
    if tools:
        row["tools"] = tools
    report.chars += len(json.dumps(row, ensure_ascii=False))
    return row


def convert_file(path: Path, report: Report) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            report.lines += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                report.skip("invalid JSON")
                continue
            row = convert_line(obj, n, report)
            if row is not None:
                rows.append(row)
    return rows


def split_holdout(rows: list[dict[str, Any]], fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic split by content hash, so re-running keeps the same validation set."""
    if fraction <= 0 or len(rows) < 50:
        return rows, []
    n_val = min(200, max(10, int(len(rows) * fraction)))
    keyed = sorted(rows, key=lambda r: hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest())
    val_ids = {id(r) for r in keyed[:n_val]}
    return [r for r in rows if id(r) not in val_ids], [r for r in rows if id(r) in val_ids]


def epochs_from_inventory(path: Path, job_id: str | None) -> tuple[int | None, str | None]:
    """n_epochs (and base model) of the original OpenAI job, if the inventory has it."""
    try:
        inv = json.loads(path.read_text())
    except (OSError, ValueError):
        return None, None
    jobs = [j for j in inv.get("jobs") or [] if j.get("status") == "succeeded"]
    if job_id:
        jobs = [j for j in jobs if j.get("id") == job_id]
    if not jobs:
        return None, None
    n = (jobs[0].get("hyperparameters") or {}).get("n_epochs")
    return (n if isinstance(n, int) else None), jobs[0].get("base")


def write_kit(out: Path, train: list[dict[str, Any]], val: list[dict[str, Any]], config: dict[str, Any]) -> None:
    ensure_private_dir(out)
    for name, rows in (("train.jsonl", train), ("validation.jsonl", val)):
        data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")
        write_private(out / "data" / name, data)
    write_private(out / "config.json", (json.dumps(config, indent=2) + "\n").encode())
    kit = resources.files("tunekeep") / "kit"
    for name in KIT_FILES:
        with resources.as_file(kit / name) as src:
            shutil.copyfile(src, out / name)
    (out / ".gitignore").write_text("data/\nruns/\n")


def build_config(base: str, revision: str | None, epochs: int, gpu: str, source_sha256: str, **overrides: Any) -> dict[str, Any]:
    cfg = {
        "base": base,
        "revision": revision if revision is not None else KNOWN_REVISIONS.get(base),
        "epochs": epochs,
        "lr": 2e-4,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
        "max_len": 4096,
        "batch_size": 4,
        "grad_accum": 4,
        "warmup_ratio": 0.03,
        "seed": 42,
        "gradient_checkpointing": True,
        "gpu": gpu,
        "timeout_hours": 2,
        "gguf": "q8_0",
        "source": {"training_file_sha256": source_sha256, "tool": f"tunekeep {__version__}"},
    }
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def render_report(report: Report, n_train: int, n_val: int) -> str:
    lines = [
        f"{report.examples} of {report.lines} lines usable: {report.trained_turns} assistant turns to train on"
        + (f", {report.with_tools} with tools" if report.with_tools else ""),
        f"About {report.chars // 4:,} tokens per epoch (rough: characters / 4).",
        f"Split: {n_train} train, {n_val} validation.",
    ]
    for reason, n in sorted(report.skipped.items(), key=lambda kv: -kv[1]):
        lines.append(f"Skipped {n}: {reason}")
    for w in report.warnings:
        lines.append(f"Warning: {w}")
    return "\n".join(lines)

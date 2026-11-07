"""Run API calls in parallel and append each result to a private JSONL file."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .client import APIError
from .files import open_private_append

# Wrong key, no access or the model is gone: no point sending the rest.
FATAL_STATUSES = (401, 403, 404)


@dataclass
class RunResult:
    total: int
    skipped: int
    recorded: int
    failed: int
    first_error: str | None


def done_ids(out: Path, key: str, value: str) -> set[str]:
    """Ids already in `out` for rows where row[key] == value (for resuming)."""
    done: set[str] = set()
    if not out.exists():
        return done
    with out.open("r", encoding="utf-8") as fh:
        for raw in fh:
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue  # a partial last line from an interrupted run
            if rec.get(key) == value and rec.get("id"):
                done.add(rec["id"])
    return done


def run_parallel(
    todo: list[Any],
    one: Callable[[Any], dict[str, Any]],
    out: Path,
    *,
    total: int,
    workers: int = 4,
    line_of: Callable[[Any], Any] = lambda item: getattr(item, "line", "?"),
    progress=None,
) -> RunResult:
    lock = threading.Lock()
    recorded = failed = 0
    first_error: str | None = None
    with open_private_append(out) as fh, ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, item): item for item in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            try:
                rec = fut.result()
            except APIError as e:
                failed += 1
                if first_error is None:
                    first_error = f"line {line_of(futures[fut])}: {e}"
                if e.status in FATAL_STATUSES:
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
    return RunResult(total, total - len(todo), recorded, failed, first_error)


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as fh:
        for n, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield n, obj

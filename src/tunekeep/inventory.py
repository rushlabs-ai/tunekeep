"""`tunekeep inventory`: everything an account has fine-tuned, and when it dies."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from . import __version__
from .client import APIError, Client
from .files import safe_name, write_private
from .schedule import DEPRECATIONS_URL, REPLACEMENT, TRAINING_ENDS, base_of, days_left, shutdown_for


def collect(client: Client, out_dir: Path, download: bool = True, today: dt.date | None = None) -> dict[str, Any]:
    notes: list[str] = []
    models = client.list_models()
    listed = {m.get("id"): m for m in models if m.get("id")}

    try:
        jobs = client.list_jobs()
    except APIError as e:
        jobs = []
        notes.append(f"Could not list fine-tuning jobs ({e}). Only models listed by /v1/models are shown.")

    by_model: dict[str, dict[str, Any]] = {}

    def entry(model_id: str) -> dict[str, Any]:
        if model_id not in by_model:
            api_value = (listed.get(model_id) or {}).get("shutdown_date")
            date, source = shutdown_for(model_id, api_value)
            by_model[model_id] = {
                "id": model_id,
                "base": base_of(model_id),
                "listed": model_id in listed,
                "shutdown_date": date.isoformat() if date else None,
                "shutdown_source": source,
                "days_left": days_left(date, today),
                "job": None,
                # Checkpoint ids end in ":ckpt-step-N"; this holds even when jobs can't be listed.
                "checkpoint_of": model_id.split(":ckpt-step-")[0] if ":ckpt-step-" in model_id else None,
            }
        return by_model[model_id]

    for model_id in listed:
        if model_id.startswith("ft:"):
            entry(model_id)

    files: dict[str, dict[str, Any]] = {}
    job_rows: list[dict[str, Any]] = []
    for job in jobs:
        job_id = job.get("id")
        ft_model = job.get("fine_tuned_model")
        row = {
            "id": job_id,
            "status": job.get("status"),
            "base": job.get("model"),
            "fine_tuned_model": ft_model,
            "method": (job.get("method") or {}).get("type") or "supervised",
            "created_at": job.get("created_at"),
            "training_file": job.get("training_file"),
            "validation_file": job.get("validation_file"),
            "result_files": job.get("result_files") or [],
            "hyperparameters": job.get("hyperparameters"),
            "trained_tokens": job.get("trained_tokens"),
            "seed": job.get("seed"),
            "suffix": job.get("suffix"),
            "checkpoints": [],
        }
        if ft_model:
            entry(ft_model)["job"] = job_id
            try:
                for ck in client.list_checkpoints(job_id):
                    name = ck.get("fine_tuned_model_checkpoint")
                    row["checkpoints"].append(
                        {"id": ck.get("id"), "model": name, "step": ck.get("step_number"), "metrics": ck.get("metrics")}
                    )
                    if name:
                        e = entry(name)
                        e["job"] = job_id
                        e["checkpoint_of"] = ft_model
            except APIError as e:
                notes.append(f"Could not list checkpoints for {job_id} ({e}).")
        job_rows.append(row)

        roles = [("training", row["training_file"]), ("validation", row["validation_file"])]
        roles += [("result", f) for f in row["result_files"]]
        for role, file_id in roles:
            if file_id and file_id not in files:
                files[file_id] = _file_record(client, file_id, role, job_id, out_dir, download)

    inventory = {
        "tool": f"tunekeep {__version__}",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "base_url": client.base_url,
        "training_ends": TRAINING_ENDS.isoformat(),
        "replacement": REPLACEMENT,
        "sources": [DEPRECATIONS_URL],
        "models": sorted(by_model.values(), key=lambda m: (m["checkpoint_of"] is not None, m["id"])),
        "jobs": job_rows,
        "files": list(files.values()),
        "notes": notes,
    }
    write_private(out_dir / "inventory.json", json.dumps(inventory, indent=2).encode("utf-8"))
    return inventory


def _file_record(client: Client, file_id: str, role: str, job_id: str, out_dir: Path, download: bool) -> dict[str, Any]:
    rec: dict[str, Any] = {"id": file_id, "role": role, "job": job_id, "filename": None, "bytes": None,
                           "purpose": None, "saved_to": None, "download": "skipped"}
    try:
        meta = client.get_file(file_id)
        rec.update(filename=meta.get("filename"), bytes=meta.get("bytes"), purpose=meta.get("purpose"))
    except APIError as e:
        rec["download"] = f"metadata unavailable: {e.message}"
        return rec
    if not download:
        return rec
    try:
        data = client.file_content(file_id)
    except APIError as e:
        # OpenAI does not return files uploaded with purpose=fine-tune.
        rec["download"] = f"blocked by the API ({e.status}): {e.message}"
        return rec
    target = out_dir / "files" / f"{file_id}-{safe_name(rec['filename'] or 'file')}"
    write_private(target, data)
    rec["saved_to"] = str(target)
    rec["download"] = "saved"
    return rec


def render(inv: dict[str, Any]) -> str:
    """Plain-text summary for the terminal."""
    lines: list[str] = []
    today = dt.datetime.now(dt.timezone.utc).date()
    left_training = (dt.date.fromisoformat(inv["training_ends"]) - today).days
    models = [m for m in inv["models"] if not m["checkpoint_of"]]
    ckpts = [m for m in inv["models"] if m["checkpoint_of"]]

    ck = f" (plus {len(ckpts)} checkpoint{'s' if len(ckpts) != 1 else ''})" if ckpts else ""
    lines.append(f"Fine-tuned models: {len(models)}{ck}")
    if left_training >= 0:
        lines.append(f"New fine-tuning jobs stop on {inv['training_ends']} ({left_training} days).")
    lines.append("")
    if models:
        w = max(len(m["id"]) for m in models)
        lines.append(f"{'MODEL'.ljust(w)}  SHUTS DOWN   DAYS LEFT  SOURCE")
        for m in models:
            d = m["days_left"]
            left = "unknown" if d is None else ("gone" if d < 0 else str(d))
            lines.append(f"{m['id'].ljust(w)}  {(m['shutdown_date'] or '-').ljust(11)}  {left.ljust(9)}  {m['shutdown_source']}")
    else:
        lines.append("No fine-tuned models found for this key.")

    blocked = [f for f in inv["files"] if f["role"] in ("training", "validation") and f["download"].startswith("blocked")]
    saved = [f for f in inv["files"] if f["download"] == "saved"]
    lines.append("")
    noun = "file" if len(blocked) == 1 else "files"
    lines.append(f"Files: {len(saved)} saved, {len(blocked)} training/validation {noun} the API would not return.")
    if blocked:
        lines.append("  Keep your own copy of these; tunekeep retrains only from your original data:")
        for f in blocked:
            lines.append(f"  - {f['id']}  {f['filename'] or ''}  (job {f['job']})")
    for note in inv["notes"]:
        lines.append(f"Note: {note}")

    alive = [m for m in models if m["days_left"] is not None and m["days_left"] >= 0]
    if alive:
        lines.append("")
        lines.append("Next: record how each model behaves while it still answers, e.g.")
        lines.append(f"  tunekeep record --model {alive[0]['id']} --prompts prompts.jsonl")
    return "\n".join(lines)

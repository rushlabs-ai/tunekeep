"""Shutdown dates for fine-tunable base models.

The API's `shutdown_date` field (on /v1/models) is the source of truth. This
table is only a fallback for accounts or proxies that do not return the field
yet. Fine-tuned models shut down with their base model.

Sources (checked 2026-11-06):
  https://developers.openai.com/api/docs/deprecations
  https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-retirement-schedule
"""

from __future__ import annotations

import datetime as dt

DEPRECATIONS_URL = "https://developers.openai.com/api/docs/deprecations"

# Last day OpenAI accepts new fine-tuning jobs for remaining customers.
TRAINING_ENDS = dt.date(2027, 1, 6)

# Base model id -> shutdown date (OpenAI). Fine-tunes follow their base model.
OPENAI_SHUTDOWNS: dict[str, dt.date] = {
    "gpt-4o-2024-08-06": dt.date(2027, 1, 14),
    "gpt-4o-2024-11-20": dt.date(2027, 1, 14),
    "gpt-4o-mini-2024-07-18": dt.date(2027, 1, 14),
    "gpt-4.1-2025-04-14": dt.date(2027, 1, 14),
    "gpt-4.1-mini-2025-04-14": dt.date(2027, 1, 14),
    # Already shut down on 2026-10-23.
    "gpt-4.1-nano-2025-04-14": dt.date(2026, 10, 23),
    "gpt-4o-2024-05-13": dt.date(2026, 10, 23),
    "o4-mini-2025-04-16": dt.date(2026, 10, 23),
    "gpt-3.5-turbo-0125": dt.date(2026, 10, 23),
    "gpt-3.5-turbo-1106": dt.date(2026, 10, 23),
    "gpt-3.5-turbo-0613": dt.date(2026, 10, 23),
    "gpt-4-0613": dt.date(2026, 10, 23),
    "babbage-002": dt.date(2026, 10, 23),
    "davinci-002": dt.date(2026, 10, 23),
}

REPLACEMENT = "gpt-6-luna"  # Named replacement; it cannot be fine-tuned.


def base_of(model_id: str) -> str | None:
    """Base model of a fine-tuned id: ft:<base>:<org>:<suffix>:<id>[:ckpt-step-N]."""
    if not model_id.startswith("ft:"):
        return None
    parts = model_id.split(":")
    return parts[1] if len(parts) > 1 and parts[1] else None


def parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def shutdown_for(model_id: str, api_value: str | None) -> tuple[dt.date | None, str]:
    """Return (date, source) where source is 'api', 'table' or 'unknown'."""
    from_api = parse_date(api_value)
    if from_api:
        return from_api, "api"
    base = base_of(model_id) or model_id
    if base in OPENAI_SHUTDOWNS:
        return OPENAI_SHUTDOWNS[base], "table"
    return None, "unknown"


def days_left(date: dt.date | None, today: dt.date | None = None) -> int | None:
    if date is None:
        return None
    return (date - (today or dt.datetime.now(dt.timezone.utc).date())).days

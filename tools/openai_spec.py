"""Extract the parts of OpenAI's official API spec that tunekeep relies on (development only).

    python3 tools/openai_spec.py            # writes tests/openai_spec.json

Source: https://github.com/openai/openai-openapi (openapi.json), pinned to a commit and checked
by SHA-256 before use. Only the request and response schemas of the endpoints tunekeep calls are
kept, with everything they reference, so the tests can check the fake server and tunekeep's
requests against OpenAI's own definitions without a network or an API key.
To update: pick a newer commit that is at least 7 days old, run once to see the new checksum,
review the diff of tests/openai_spec.json, then pin both values here.
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

COMMIT = "c2aecc2fbe1ddff3e7500a0eaa5c2bc4c19c5318"  # 2026-10-31
SHA256 = "d80bb28992f6c2a86876d866dcae0b1f15e130de94238592bc57a4a579448b5f"
URL = f"https://raw.githubusercontent.com/openai/openai-openapi/{COMMIT}/openapi.json"
OUT = Path(__file__).resolve().parents[1] / "tests" / "openai_spec.json"

# (method, path) -> what tunekeep does with it
ENDPOINTS = {
    ("get", "/models"): "inventory: models and shutdown dates",
    ("get", "/fine_tuning/jobs"): "inventory: fine-tuning jobs",
    ("get", "/fine_tuning/jobs/{fine_tuning_job_id}/checkpoints"): "inventory: checkpoints",
    ("get", "/files/{file_id}"): "inventory: file metadata",
    ("post", "/chat/completions"): "record and compare",
}


def refs(node, found: set[str]) -> None:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/components/schemas/"):
            found.add(ref.rsplit("/", 1)[-1])
        for v in node.values():
            refs(v, found)
    elif isinstance(node, list):
        for v in node:
            refs(v, found)


def main() -> int:
    raw = urllib.request.urlopen(URL, timeout=120).read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        print(f"checksum mismatch for {URL}: got {digest}", file=sys.stderr)
        return 1
    spec = json.loads(raw)
    schemas = spec["components"]["schemas"]
    endpoints, wanted = {}, set()
    for (method, path), purpose in ENDPOINTS.items():
        op = spec["paths"][path][method]
        entry = {"purpose": purpose}
        body = op.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema")
        if body:
            entry["request"] = body
        entry["response"] = op["responses"]["200"]["content"]["application/json"]["schema"]
        endpoints[f"{method.upper()} {path}"] = entry
        refs(entry, wanted)
    done: set[str] = set()
    while wanted - done:
        name = sorted(wanted - done)[0]
        done.add(name)
        refs(schemas[name], wanted)
    out = {
        "source": URL,
        "sha256": SHA256,
        "endpoints": endpoints,
        "schemas": {n: schemas[n] for n in sorted(done)},
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{OUT}: {len(endpoints)} endpoints, {len(done)} schemas, {OUT.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

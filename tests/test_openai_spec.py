"""Check the fake OpenAI server and tunekeep's own requests against OpenAI's official API spec.

tests/openai_spec.json is an extract of github.com/openai/openai-openapi, pinned by commit and
SHA-256 (see tools/openai_spec.py). This catches the fake server drifting from the real API
(missing or mistyped fields, values outside an enum) and tunekeep sending a request the API
would not accept, with no network and no API key.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fake_openai import FT, KEY, FakeOpenAI  # noqa: E402
from test_v02 import client_for, write_training  # noqa: E402

from tunekeep import compare, inventory  # noqa: E402
from tunekeep.record import load_prompts  # noqa: E402
from tunekeep.record import run as record_run  # noqa: E402

SPEC = json.loads((Path(__file__).parent / "openai_spec.json").read_text())


def validate(value, schema, path: str = "$") -> list[str]:
    """The JSON Schema subset the OpenAI spec uses. Returns problems as 'path: message'."""
    if "$ref" in schema:
        return validate(value, SPEC["schemas"][schema["$ref"].rsplit("/", 1)[-1]], path)
    errors: list[str] = []
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [validate(value, s, path) for s in schema[key]]
            if all(options):
                best = min(options, key=len)
                errors.append(f"{path}: matches none of {len(options)} {key} options (closest: {best[0]})")
    for sub in schema.get("allOf", []):
        errors += validate(value, sub, path)
    types = schema.get("type")
    if types is not None:
        types = types if isinstance(types, list) else [types]
        if schema.get("nullable"):
            types = types + ["null"]
        checks = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "null": lambda v: v is None,
        }
        if not any(checks[t](value) for t in types if t in checks):
            return errors + [f"{path}: expected {'/'.join(types)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing required field {name!r}")
        for name, v in value.items():
            if name in props:
                errors += validate(v, props[name], f"{path}.{name}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected field {name!r}")
            elif isinstance(schema.get("additionalProperties"), dict):
                errors += validate(v, schema["additionalProperties"], f"{path}.{name}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors


def endpoint(name: str) -> dict:
    return SPEC["endpoints"][name]


class ValidatorTests(unittest.TestCase):
    def test_catches_drift(self):
        model = {"$ref": "#/components/schemas/Model"}
        good = {"id": "m", "object": "model", "created": 1, "owned_by": "openai"}
        self.assertEqual(validate(good, model), [])
        self.assertTrue(validate({**good, "created": "yesterday"}, model))
        self.assertTrue(validate({"id": "m", "object": "model"}, model))
        self.assertTrue(validate({**good, "object": "models"}, model))


class SpecFactsTests(unittest.TestCase):
    """What tunekeep's design relies on, as OpenAI's own spec states it."""

    def test_models_have_a_shutdown_date(self):
        self.assertIn("shutdown_date", SPEC["schemas"]["Model"]["properties"])

    def test_reasoning_effort_none_exists(self):
        options = SPEC["schemas"]["ReasoningEffort"]["anyOf"][0]["enum"]
        self.assertIn("none", options)

    def test_file_purposes(self):
        purposes = json.dumps(SPEC["schemas"]["OpenAIFile"]["properties"]["purpose"])
        for p in ("fine-tune", "fine-tune-results"):
            self.assertIn(f'"{p}"', purposes)


class FakeServerMatchesSpec(unittest.TestCase):
    def get(self, fake: FakeOpenAI, path: str):
        req = urllib.request.Request(fake.base_url + path, headers={"Authorization": f"Bearer {KEY}"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def post(self, fake: FakeOpenAI, body: dict):
        req = urllib.request.Request(fake.base_url + "/chat/completions", json.dumps(body).encode(),
                                     {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def assertValid(self, value, schema):
        problems = validate(value, schema)
        self.assertEqual(problems, [], "\n".join(problems[:10]))

    def test_inventory_endpoints(self):
        with FakeOpenAI() as fake:
            self.assertValid(self.get(fake, "/models"), endpoint("GET /models")["response"])
            self.assertValid(self.get(fake, "/fine_tuning/jobs?limit=2"), endpoint("GET /fine_tuning/jobs")["response"])
            self.assertValid(self.get(fake, "/fine_tuning/jobs/ftjob-000/checkpoints"),
                             endpoint("GET /fine_tuning/jobs/{fine_tuning_job_id}/checkpoints")["response"])
            for f in ("file-train-0", "file-result-0"):
                self.assertValid(self.get(fake, f"/files/{f}"), endpoint("GET /files/{file_id}")["response"])

    def test_chat_completions(self):
        schema = endpoint("POST /chat/completions")["response"]
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}
        with FakeOpenAI() as fake:
            for body in (
                {"model": FT, "messages": [{"role": "user", "content": "hi"}]},
                {"model": FT, "messages": [{"role": "user", "content": "json: x"}]},
                {"model": FT, "messages": [{"role": "user", "content": "q"}], "tools": [tool]},
                {"model": "gpt-6-luna", "messages": [{"role": "user", "content": "q"}], "reasoning_effort": "low"},
            ):
                self.assertValid(self.post(fake, body), schema)


class TunekeepRequestsMatchSpec(unittest.TestCase):
    """Every request tunekeep sends in record and compare is one the API defines."""

    def test_record_and_compare_requests(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            inventory.collect(client_for(fake), tmp / "inv", download=False)
            for tools in (False, True):
                golden_path = tmp / f"golden-{tools}.jsonl"
                record_run(client_for(fake), FT, load_prompts(write_training(tmp / f"train-{tools}.jsonl", tools=tools)),
                           golden_path, progress=None)
                golden = compare.load_golden(golden_path)
                shots = compare.load_shots(tmp / f"train-{tools}.jsonl")
                compare.run(client_for(fake), golden[:2], tmp / f"c-{tools}.jsonl", candidate="gpt-6-luna", label="luna",
                            shots=shots, n_shots=2, progress=None)
                compare.run(client_for(fake), golden[:2], tmp / f"c4-{tools}.jsonl", candidate="gpt-4o-mini-2024-07-18",
                            label="mini", progress=None)
                if not tools:
                    compare.run(client_for(fake), golden[:2], tmp / "c-low.jsonl", candidate="gpt-6-luna", label="low",
                                effort="low", progress=None)
            sent = list(fake.state.chat_calls)
        self.assertGreater(len(sent), 10)
        schema = endpoint("POST /chat/completions")["request"]
        for body in sent:
            problems = validate(body, schema)
            self.assertEqual(problems, [], f"{json.dumps(body)[:300]}\n" + "\n".join(problems[:10]))


if __name__ == "__main__":
    unittest.main()

"""End-to-end tests against the fake OpenAI API. Run: python -m unittest -v"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fake_openai import CKPT, FT, KEY, FakeOpenAI  # noqa: E402

from tunekeep import cli, schedule  # noqa: E402
from tunekeep.client import APIError, Client  # noqa: E402
from tunekeep.files import safe_name  # noqa: E402
from tunekeep.inventory import collect, render  # noqa: E402
from tunekeep.record import PromptError, load_prompts, parse_line, run  # noqa: E402


def client_for(fake: FakeOpenAI, key: str = KEY) -> Client:
    return Client(api_key=key, base_url=fake.base_url, max_retries=3, _sleep=lambda s: None)


class ScheduleTests(unittest.TestCase):
    def test_base_of(self):
        self.assertEqual(schedule.base_of(FT), "gpt-4o-mini-2024-07-18")
        self.assertEqual(schedule.base_of(CKPT), "gpt-4o-mini-2024-07-18")
        self.assertIsNone(schedule.base_of("gpt-4o"))

    def test_api_value_wins_then_table(self):
        self.assertEqual(schedule.shutdown_for(FT, "2027-02-01"), (dt.date(2027, 2, 1), "api"))
        self.assertEqual(schedule.shutdown_for(FT, None), (dt.date(2027, 1, 14), "table"))
        self.assertEqual(schedule.shutdown_for("ft:unknown-base:o::x", None), (None, "unknown"))

    def test_days_left(self):
        self.assertEqual(schedule.days_left(dt.date(2027, 1, 14), today=dt.date(2026, 11, 6)), 69)


class InventoryTests(unittest.TestCase):
    def test_inventory_end_to_end(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            inv = collect(client_for(fake), out, today=dt.date(2026, 11, 6))
            ids = {m["id"]: m for m in inv["models"]}
            self.assertIn(FT, ids)
            self.assertIn(CKPT, ids)
            self.assertEqual(ids[FT]["shutdown_source"], "api")
            self.assertEqual(ids[FT]["days_left"], 69)
            self.assertEqual(ids[FT]["job"], "ftjob-000")
            self.assertEqual(ids[CKPT]["checkpoint_of"], FT)
            # Pagination: 5 jobs over pages of 2.
            self.assertEqual(len(inv["jobs"]), 5)
            files = {f["id"]: f for f in inv["files"]}
            self.assertTrue(files["file-train-0"]["download"].startswith("blocked by the API (400)"))
            self.assertEqual(files["file-result-0"]["download"], "saved")
            saved = Path(files["file-result-0"]["saved_to"])
            self.assertTrue(saved.read_bytes().startswith(b"step,"))
            self.assertEqual(stat.S_IMODE(os.stat(saved).st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(os.stat(out / "inventory.json").st_mode), 0o600)
            text = render(inv)
            self.assertIn("2027-01-14", text)
            self.assertIn("file-train-0", text)
            self.assertIn("tunekeep record --model", text)

    def test_falls_back_to_table_without_shutdown_field(self):
        with FakeOpenAI(shutdown_field=False) as fake, tempfile.TemporaryDirectory() as tmp:
            inv = collect(client_for(fake), Path(tmp), download=False)
            ft = next(m for m in inv["models"] if m["id"] == FT)
            self.assertEqual((ft["shutdown_date"], ft["shutdown_source"]), ("2027-01-14", "table"))

    def test_jobs_forbidden_is_a_note_not_a_crash(self):
        with FakeOpenAI(jobs_forbidden=True) as fake, tempfile.TemporaryDirectory() as tmp:
            inv = collect(client_for(fake), Path(tmp))
            self.assertTrue(any("Could not list fine-tuning jobs" in n for n in inv["notes"]))
            ids = {m["id"]: m for m in inv["models"]}
            self.assertIn(FT, ids)
            self.assertEqual(ids[CKPT]["checkpoint_of"], FT)

    def test_bad_key(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(APIError) as ctx:
                collect(client_for(fake, key="sk-wrong"), Path(tmp))
            self.assertEqual(ctx.exception.status, 401)
            self.assertNotIn("sk-wrong", str(ctx.exception))


class PromptParsingTests(unittest.TestCase):
    def test_training_line_strips_answer_and_weight(self):
        p = parse_line({"messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "weight": 1}]}, 1)
        self.assertEqual([m["role"] for m in p.messages], ["system", "user"])
        self.assertEqual(p.reference, {"role": "assistant", "content": "hello"})

    def test_weight_removed_from_earlier_assistant_turns(self):
        p = parse_line({"messages": [
            {"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "weight": 0},
            {"role": "user", "content": "c"}]}, 1)
        self.assertNotIn("weight", p.messages[1])
        self.assertIsNone(p.reference)

    def test_dpo_line(self):
        p = parse_line({"input": {"messages": [{"role": "user", "content": "q"}], "tools": []},
                        "preferred_output": [{"role": "assistant", "content": "good"}],
                        "non_preferred_output": [{"role": "assistant", "content": "bad"}]}, 3)
        self.assertEqual(p.messages, [{"role": "user", "content": "q"}])
        self.assertEqual(p.reference, [{"role": "assistant", "content": "good"}])

    def test_errors(self):
        for bad in ([1], {"foo": 1}, {"messages": []}, {"messages": [{"role": "assistant", "content": "x"}]}):
            with self.assertRaises(PromptError):
                parse_line(bad, 7)

    def test_dedupes(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "p.jsonl"
            line = json.dumps({"messages": [{"role": "user", "content": "same"}]})
            f.write_text(f"{line}\n\n{line}\n")
            self.assertEqual(len(load_prompts(f)), 1)


class RecordTests(unittest.TestCase):
    def _prompts(self, tmp: Path, n: int = 6, tools: bool = False) -> Path:
        f = tmp / "train.jsonl"
        with f.open("w") as fh:
            for i in range(n):
                row = {"messages": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}", "weight": 1}]}
                if tools:
                    row["tools"] = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
                fh.write(json.dumps(row) + "\n")
        return f

    def test_record_resume_and_retry(self):
        with FakeOpenAI(fail_429=2) as fake, tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            prompts = load_prompts(self._prompts(tmp))
            out = tmp / "golden.jsonl"
            r1 = run(client_for(fake), FT, prompts[:4], out, workers=2, progress=None)
            self.assertEqual((r1.recorded, r1.failed), (4, 0))
            r2 = run(client_for(fake), FT, prompts, out, workers=2, progress=None)
            self.assertEqual((r2.skipped, r2.recorded), (4, 2))
            rows = [json.loads(x) for x in out.read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual(len({r["id"] for r in rows}), 6)
            first = rows[0]
            self.assertTrue(first["response"]["message"]["content"].startswith("echo: q"))
            self.assertEqual(first["reference"]["content"][0], "a")
            self.assertEqual(first["request"]["temperature"], 0.0)
            self.assertEqual(stat.S_IMODE(os.stat(out).st_mode), 0o600)
            # The fake rejects 'weight'; no call may carry it.
            self.assertFalse(any("weight" in m for c in fake.state.chat_calls for m in c["messages"]))

    def test_tool_calls_are_recorded(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            out = tmp / "g.jsonl"
            run(client_for(fake), FT, load_prompts(self._prompts(tmp, 1, tools=True)), out, progress=None)
            row = json.loads(out.read_text())
            self.assertEqual(row["response"]["finish_reason"], "tool_calls")
            self.assertEqual(row["response"]["message"]["tool_calls"][0]["function"]["name"], "lookup")

    def test_missing_model_stops_early(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            res = run(client_for(fake), "ft:gone:o::x", load_prompts(self._prompts(tmp, 20)), tmp / "g.jsonl",
                      workers=1, progress=None)
            self.assertEqual(res.recorded, 0)
            self.assertIn("404", res.first_error)
            self.assertLess(len(fake.state.chat_calls), 20)


class CliTests(unittest.TestCase):
    def test_cli_inventory_and_record(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            env = {"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": fake.base_url}
            old = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    code = cli.main(["inventory", "--out", str(Path(tmp) / "o")])
                self.assertEqual(code, 0)
                self.assertIn(FT, buf.getvalue())
                self.assertNotIn(KEY, buf.getvalue())

                prompts = Path(tmp) / "p.jsonl"
                prompts.write_text(json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n")
                buf = io.StringIO()
                with redirect_stdout(buf), redirect_stderr(io.StringIO()):
                    code = cli.main(["record", "--model", FT, "--prompts", str(prompts), "--out", str(Path(tmp) / "g.jsonl")])
                self.assertEqual(code, 0)
                self.assertIn("Recorded 1", buf.getvalue())

                buf = io.StringIO()
                with redirect_stdout(buf):
                    code = cli.main(["record", "--model", FT, "--prompts", str(prompts), "--dry-run"])
                self.assertEqual(code, 0)
                self.assertIn("Dry run", buf.getvalue())
            finally:
                for k, v in old.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_missing_key_message(self):
        old = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaises(SystemExit) as ctx:
                Client.from_env()
            self.assertIn("OPENAI_API_KEY", str(ctx.exception))
        finally:
            if old is not None:
                os.environ["OPENAI_API_KEY"] = old


class FilesTests(unittest.TestCase):
    def test_safe_name(self):
        self.assertEqual(safe_name("../../etc/passwd"), "passwd")
        self.assertEqual(safe_name("a b/c?.jsonl"), "c_.jsonl")
        self.assertEqual(safe_name(""), "file")


if __name__ == "__main__":
    unittest.main()

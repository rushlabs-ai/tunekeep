"""Tests for compare, scoring, retrain and report. Run: python -m unittest discover -s tests -v"""

from __future__ import annotations

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

from fake_openai import FT, KEY, FakeOpenAI  # noqa: E402

from tunekeep import cli, compare, retrain, score  # noqa: E402
from tunekeep.client import Client  # noqa: E402
from tunekeep.record import load_prompts  # noqa: E402
from tunekeep.record import run as record_run  # noqa: E402


def client_for(fake: FakeOpenAI) -> Client:
    return Client(api_key=KEY, base_url=fake.base_url, max_retries=2, _sleep=lambda s: None)


class EnvMixin:
    def set_env(self, **env):
        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)

        def restore():
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        self.addCleanup(restore)


def write_training(path: Path, n: int = 6, tools: bool = False, extra: list[dict] | None = None) -> Path:
    with path.open("w") as fh:
        for i in range(n):
            row = {"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": f"q{i}"},
                                {"role": "assistant", "content": f"echo: q{i}"}]}
            if tools:
                row["tools"] = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
                row["messages"][-1] = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": json.dumps({"q": f"q{i}"})}}]}
            fh.write(json.dumps(row) + "\n")
        for row in extra or []:
            fh.write(json.dumps(row) + "\n")
    return path


class ScoreTests(unittest.TestCase):
    def test_text(self):
        a = {"role": "assistant", "content": "The answer is 42."}
        self.assertEqual(score.score(a, {"content": "The answer is 42."})["exact"], True)
        s = score.score(a, {"content": "the answer   is 42"})
        self.assertTrue(s["normalized"])
        self.assertEqual(s["score"], 1.0)
        s = score.score(a, {"content": "It is 42"})
        self.assertGreater(s["score"], 0)
        self.assertLess(s["score"], 1)

    def test_think_block_is_ignored(self):
        s = score.score({"content": "yes"}, {"content": "<think>\nhmm\n</think>\n\nyes"})
        self.assertTrue(s["exact"])

    def test_json(self):
        exp = {"content": '{"label": "spam", "score": 1}'}
        self.assertEqual(score.score(exp, {"content": '```json\n{"score": 1, "label": "Spam"}\n```'})["score"], 1.0)
        self.assertEqual(score.score(exp, {"content": '{"label": "ham", "score": 1}'})["score"], 0.5)
        s = score.score(exp, {"content": "not json"})
        self.assertEqual((s["score"], s["json_valid"]), (0.0, False))

    def test_tool_calls(self):
        def call(name, args):
            return {"type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        exp = {"content": None, "tool_calls": [call("a", {"x": 1}), call("b", {"y": 2})]}
        same = {"tool_calls": [call("b", {"y": 2}), call("a", {"x": 1})]}  # order does not matter
        self.assertTrue(score.score(exp, same)["exact"])
        half = {"tool_calls": [call("a", {"x": 1})]}
        self.assertEqual(score.score(exp, half)["score"], 0.5)
        self.assertEqual(score.score(exp, {"content": "I would call a"})["score"], 0.0)

    def test_tool_call_instead_of_text(self):
        s = score.score({"content": "hi"}, {"content": None, "tool_calls": [
            {"function": {"name": "a", "arguments": "{}"}}]})
        self.assertEqual(s["score"], 0.0)


class PlanParamsTests(unittest.TestCase):
    def test_luna_auto_uses_none_and_keeps_temperature(self):
        params, notes = compare.plan_params("gpt-6-luna", "auto", 0.0, has_tools=True)
        self.assertEqual(params, {"reasoning_effort": "none", "temperature": 0.0})
        self.assertEqual(notes, [])

    def test_luna_with_reasoning_drops_temperature(self):
        params, notes = compare.plan_params("gpt-6-luna", "low", 0.0, has_tools=False)
        self.assertEqual(params, {"reasoning_effort": "low"})
        self.assertTrue(notes)

    def test_luna_with_reasoning_and_tools_is_refused(self):
        with self.assertRaises(compare.CompareError):
            compare.plan_params("gpt-6-luna", "medium", None, has_tools=True)
        with self.assertRaises(compare.CompareError):
            compare.plan_params("gpt-6-luna", "default", None, has_tools=True)

    def test_other_families(self):
        with self.assertRaises(compare.CompareError):
            compare.plan_params("gpt-6-astra", "auto", None, has_tools=True)
        self.assertEqual(compare.plan_params("o4-mini", "auto", 0.0, False)[0], {"reasoning_effort": "low"})
        self.assertEqual(compare.plan_params("gpt-4o-mini", "auto", 0.0, True)[0], {"temperature": 0.0})
        self.assertEqual(compare.plan_params("my-local-model", "auto", 0.2, True)[0], {"temperature": 0.2})


class CompareTests(unittest.TestCase):
    def _golden(self, fake, tmp: Path, **kw) -> Path:
        prompts = load_prompts(write_training(tmp / "train.jsonl", **kw))
        out = tmp / "golden.jsonl"
        record_run(client_for(fake), FT, prompts, out, progress=None)
        return out

    def test_luna_end_to_end_and_resume(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            golden = compare.load_golden(self._golden(fake, tmp))
            out = tmp / "cmp.jsonl"
            res, notes = compare.run(client_for(fake), golden[:3], out, candidate="gpt-6-luna", label="luna", progress=None)
            self.assertEqual((res.recorded, res.failed), (3, 0))
            res, _ = compare.run(client_for(fake), golden, out, candidate="gpt-6-luna", label="luna", progress=None)
            self.assertEqual((res.skipped, res.recorded), (3, 3))
            rows = compare.load_rows(out)["luna"]
            self.assertEqual(len(rows), 6)
            s = compare.summarize(rows)
            self.assertEqual(s["vs_golden"]["n"], 6)
            self.assertAlmostEqual(s["vs_golden"]["match"], 4 / 6, places=3)  # q1 and q3 differ
            self.assertEqual(s["vs_reference"]["n"], 6)
            sent = [c for c in fake.state.chat_calls if c["model"] == "gpt-6-luna"]
            self.assertTrue(all(c["reasoning_effort"] == "none" and c["temperature"] == 0.0 for c in sent))

    def test_tools_with_reasoning_never_sends(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            golden = compare.load_golden(self._golden(fake, tmp, tools=True))
            before = len(fake.state.chat_calls)
            with self.assertRaises(compare.CompareError):
                compare.run(client_for(fake), golden, tmp / "c.jsonl", candidate="gpt-6-luna", label="x",
                            effort="medium", progress=None)
            self.assertEqual(len(fake.state.chat_calls), before)

    def test_tool_answers_score(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            golden = compare.load_golden(self._golden(fake, tmp, n=2, tools=True))
            compare.run(client_for(fake), golden, tmp / "c.jsonl", candidate="gpt-6-luna", label="x", progress=None)
            rows = compare.load_rows(tmp / "c.jsonl")["x"]
            self.assertTrue(all(r["scores"]["golden"]["kind"] == "tool" for r in rows))

    def test_few_shot_uses_other_examples_only(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            golden = compare.load_golden(self._golden(fake, tmp))
            shots = compare.load_shots(tmp / "train.jsonl")
            self.assertEqual(len(shots), 6)
            compare.run(client_for(fake), golden[:1], tmp / "c.jsonl", candidate="gpt-6-luna", label="s",
                        shots=shots, n_shots=2, progress=None)
            msgs = [c for c in fake.state.chat_calls if c["model"] == "gpt-6-luna"][-1]["messages"]
            self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user", "assistant", "user"])
            self.assertEqual(msgs[-1]["content"], "q0")
            self.assertNotIn("q0", [m["content"] for m in msgs[1:-1]])

    def test_golden_baseline(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            golden = compare.load_golden(self._golden(fake, Path(tmp)))
            b = compare.golden_baseline(golden)
            self.assertEqual(b["vs_reference"]["match"], 1.0)  # the fake fine-tune echoes its training answers


class ClientKeyTests(unittest.TestCase):
    def test_local_endpoint_needs_no_key(self):
        c = Client.for_endpoint("http://127.0.0.1:8080/v1", None)
        self.assertNotIn("Authorization", c._headers())

    def test_remote_endpoint_needs_a_key(self):
        with self.assertRaises(SystemExit):
            Client.for_endpoint("https://example.com/v1", None)


class RetrainConvertTests(unittest.TestCase):
    def conv(self, obj):
        rep = retrain.Report()
        return retrain.convert_line(obj, 1, rep), rep

    def test_tools_and_arguments(self):
        row, rep = self.conv({"tools": [{"type": "function", "function": {"name": "f"}}], "messages": [
            {"role": "developer", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "assistant", "content": "done", "weight": 0}]})
        m = row["messages"]
        self.assertEqual(m[0], {"role": "system", "content": "sys"})
        self.assertEqual(m[2]["content"], "")
        self.assertEqual(m[2]["tool_calls"][0]["function"]["arguments"], {"a": 1})
        self.assertEqual((m[2]["weight"], m[4]["weight"]), (1, 0))
        self.assertEqual(rep.trained_turns, 1)
        self.assertEqual(rep.with_tools, 1)

    def test_legacy_functions(self):
        row, _ = self.conv({"functions": [{"name": "g", "parameters": {}}], "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": None, "function_call": {"name": "g", "arguments": "{}"}},
            {"role": "function", "name": "g", "content": "42"},
            {"role": "assistant", "content": "42"}]})
        self.assertEqual(row["tools"], [{"type": "function", "function": {"name": "g", "parameters": {}}}])
        self.assertEqual(row["messages"][1]["tool_calls"][0]["function"], {"name": "g", "arguments": {}})
        self.assertEqual(row["messages"][2]["role"], "tool")
        self.assertEqual(row["messages"][2]["tool_call_id"], row["messages"][1]["tool_calls"][0]["id"])

    def test_skips(self):
        cases = {
            "preference (DPO) line: not supported in v0": {"input": {"messages": []}, "preferred_output": []},
            "images or audio in content: not supported in v0": {"messages": [
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]},
                {"role": "assistant", "content": "a"}]},
            "no assistant message to train on": {"messages": [
                {"role": "user", "content": "a"}, {"role": "assistant", "content": "b", "weight": 0}]},
            "no messages": {"messages": []},
        }
        for reason, obj in cases.items():
            row, rep = self.conv(obj)
            self.assertIsNone(row)
            self.assertEqual(rep.skipped, {reason: 1})

    def test_bad_arguments_warn(self):
        row, rep = self.conv({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "tool_calls": [
            {"function": {"name": "f", "arguments": "{oops"}}]}]})
        self.assertEqual(row["messages"][1]["tool_calls"][0]["function"]["arguments"], "{oops")
        self.assertTrue(rep.warnings)

    def test_holdout_is_deterministic(self):
        rows = [{"messages": [{"role": "user", "content": str(i)}, {"role": "assistant", "content": "a"}]} for i in range(100)]
        t1, v1 = retrain.split_holdout(rows, 0.05)
        t2, v2 = retrain.split_holdout(list(reversed(rows)), 0.05)
        self.assertEqual((len(t1), len(v1)), (90, 10))
        self.assertEqual(sorted(json.dumps(r) for r in v1), sorted(json.dumps(r) for r in v2))
        self.assertEqual(retrain.split_holdout(rows[:20], 0.05), (rows[:20], []))


class CliV02Tests(EnvMixin, unittest.TestCase):
    def run_cli(self, *argv) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf), redirect_stderr(io.StringIO()):
            code = cli.main(list(argv))
        return code, buf.getvalue()

    def test_record_compare_report(self):
        with FakeOpenAI() as fake, tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self.set_env(OPENAI_API_KEY=KEY, OPENAI_BASE_URL=fake.base_url)
            out = tmp / "out"
            evil = {"messages": [{"role": "user", "content": "<script>alert(1)</script>"},
                                 {"role": "assistant", "content": "<b>x</b>"}]}
            jsonp = {"messages": [{"role": "user", "content": "json: classify"},
                                  {"role": "assistant", "content": "{\"label\": \"spam\", \"score\": 1}"}]}
            train = write_training(tmp / "train.jsonl", extra=[evil, jsonp])
            self.assertEqual(self.run_cli("inventory", "--out", str(out))[0], 0)
            golden = out / "golden.jsonl"
            self.assertEqual(self.run_cli("record", "--model", FT, "--prompts", str(train), "--out", str(golden))[0], 0)

            code, text = self.run_cli("compare", "--golden", str(golden), "--candidate", "gpt-6-luna", "--dry-run")
            self.assertEqual(code, 0)
            self.assertIn('"reasoning_effort": "none"', text)

            code, text = self.run_cli("compare", "--golden", str(golden), "--candidate", "gpt-6-luna",
                                      "--examples", str(train), "--shots", "2")
            self.assertEqual(code, 0, text)
            self.assertIn("vs recorded fine-tune", text)
            self.assertIn("valid JSON 100.0%", text)
            cmp_file = out / "compare-gpt-6-luna_2-shot.jsonl"
            self.assertTrue(cmp_file.exists(), list(out.iterdir()))
            self.assertEqual(stat.S_IMODE(os.stat(cmp_file).st_mode), 0o600)

            code, text = self.run_cli("report", "--dir", str(out))
            self.assertEqual(code, 0)
            page = (out / "report.html").read_text()
            self.assertIn("Content-Security-Policy", page)
            self.assertNotIn("<script>alert(1)</script>", page)
            self.assertNotIn("<b>x</b>", page)
            self.assertIn("gpt-6-luna +2-shot", page)
            self.assertIn("Biggest differences", page)
            self.assertIn("2027-01-14", page)
            self.assertNotIn("<script", page.lower())

    def test_retrain_kit(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train = write_training(tmp / "train.jsonl", n=60)
            inv = tmp / "inventory.json"
            inv.write_text(json.dumps({"jobs": [{"id": "ftjob-1", "status": "succeeded", "base": "gpt-4o-mini-2024-07-18",
                                                 "hyperparameters": {"n_epochs": 4}}]}))
            code, text = self.run_cli("retrain", "--data", str(train), "--check", "--inventory", str(inv))
            self.assertEqual(code, 0)
            self.assertIn("Epochs: 4 (from the original OpenAI job)", text)
            self.assertFalse((tmp / "kit").exists())

            kit = tmp / "kit"
            code, text = self.run_cli("retrain", "--data", str(train), "--out", str(kit), "--inventory", str(inv),
                                      "--batch-size", "16", "--grad-accum", "1")
            self.assertEqual(code, 0, text)
            for name in ("train_lora.py", "modal_app.py", "requirements-train.txt", "requirements-gguf.txt", "config.json"):
                self.assertTrue((kit / name).exists(), name)
            cfg = json.loads((kit / "config.json").read_text())
            self.assertEqual((cfg["base"], cfg["epochs"], cfg["gpu"]), ("Qwen/Qwen3-1.7B", 4, "L40S"))
            self.assertEqual((cfg["batch_size"], cfg["grad_accum"]), (16, 1))
            self.assertEqual(cfg["revision"], retrain.KNOWN_REVISIONS["Qwen/Qwen3-1.7B"])
            self.assertEqual(len((kit / "data" / "train.jsonl").read_text().splitlines()), 50)
            self.assertEqual(len((kit / "data" / "validation.jsonl").read_text().splitlines()), 10)
            self.assertEqual(stat.S_IMODE(os.stat(kit / "data" / "train.jsonl").st_mode), 0o600)
            req = (kit / "requirements-train.txt").read_text()
            self.assertIn("--hash=sha256:", req)
            self.assertIn("torch==2.14.1", req)


class ReportColumnTests(unittest.TestCase):
    def test_training_column_only_when_there_are_training_answers(self):
        from tunekeep.report import _candidates
        base = {"candidate": "m", "rows": 1, "vs_golden": {"n": 1, "mean": 1.0, "match": 1.0, "exact": 1.0, "json_valid": None},
                "vs_reference": {"n": 0}, "latency_p50_ms": 1, "latency_p95_ms": 2, "prompt_tokens": 1,
                "completion_tokens": 1, "reasoning_tokens": 0}
        html = _candidates([base], [])
        self.assertNotIn("Score vs training answers", html)
        self.assertNotIn("Valid JSON", html)
        with_ref = dict(base, vs_reference={"n": 1, "mean": 0.5, "match": 0.5, "exact": 0.5})
        self.assertIn("Score vs training answers", _candidates([with_ref], []))


if __name__ == "__main__":
    unittest.main()

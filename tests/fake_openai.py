"""A fake OpenAI API for tests, shaped after the public API reference.

It covers what tunekeep calls: /models (with shutdown_date), paginated
/fine_tuning/jobs and checkpoints, /files metadata and content (training files
are blocked like the real API), and /chat/completions (deterministic answers,
optional 429s to exercise retries).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

KEY = "sk-test-123"
FT = "ft:gpt-4o-mini-2024-07-18:acme::AbC123"
FT_OLD = "ft:gpt-4.1-nano-2025-04-14:acme::Zz9"
CKPT = FT + ":ckpt-step-100"


class State:
    def __init__(self, shutdown_field: bool = True, fail_429: int = 0, jobs_forbidden: bool = False):
        self.shutdown_field = shutdown_field
        self.fail_429 = fail_429
        self.jobs_forbidden = jobs_forbidden
        self.chat_calls: list[dict] = []
        self.lock = threading.Lock()
        self.jobs = [
            {
                "object": "fine_tuning.job", "id": f"ftjob-{i:03d}", "model": "gpt-4o-mini-2024-07-18",
                "created_at": 1720000000 + i, "finished_at": 1720000500 + i,
                "fine_tuned_model": FT if i == 0 else None, "organization_id": "org-acme",
                "result_files": ["file-result-0"] if i == 0 else [], "status": "succeeded" if i == 0 else "failed",
                "validation_file": None, "training_file": "file-train-0" if i == 0 else f"file-train-{i}",
                "hyperparameters": {"n_epochs": 3, "batch_size": 1, "learning_rate_multiplier": 1.8},
                "trained_tokens": 12345 if i == 0 else None, "seed": 42, "error": None,
                "method": {"type": "supervised"},
            }
            for i in range(5)
        ]
        self.files = {
            "file-train-0": {"purpose": "fine-tune", "filename": "train.jsonl", "bytes": 2048},
            "file-result-0": {"purpose": "fine-tune-results", "filename": "step_metrics.csv", "bytes": 64},
        }
        for i in range(1, 5):
            self.files[f"file-train-{i}"] = {"purpose": "fine-tune", "filename": f"t{i}.jsonl", "bytes": 10}


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output quiet
            pass

        def _send(self, status, payload=None, raw: bytes | None = None, ctype="application/json", headers=None):
            body = raw if raw is not None else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _err(self, status, message, code=None, headers=None):
            self._send(status, {"error": {"message": message, "type": "invalid_request_error", "param": None, "code": code}}, headers=headers)

        def _authed(self) -> bool:
            if self.headers.get("Authorization") != f"Bearer {KEY}":
                self._err(401, "Incorrect API key provided.", "invalid_api_key")
                return False
            return True

        def do_GET(self):
            if not self._authed():
                return
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            parts = url.path.strip("/").split("/")[1:]  # drop "v1"
            if parts == ["models"]:
                ids = [("gpt-4o-mini-2024-07-18", "2027-01-14"), (FT, "2027-01-14"), (CKPT, "2027-01-14"), ("gpt-6-luna", None)]
                data = []
                for mid, sd in ids:
                    m = {"id": mid, "object": "model", "created": 1720000000, "owned_by": "org-acme" if mid.startswith("ft:") else "openai"}
                    if state.shutdown_field:
                        m["shutdown_date"] = sd
                    data.append(m)
                return self._send(200, {"object": "list", "data": data})
            if parts == ["fine_tuning", "jobs"]:
                if state.jobs_forbidden:
                    return self._err(403, "You do not have access to fine-tuning.")
                return self._send(200, self._page(state.jobs, q))
            if len(parts) == 4 and parts[:2] == ["fine_tuning", "jobs"] and parts[3] == "checkpoints":
                if parts[2] != "ftjob-000":
                    return self._send(200, {"object": "list", "data": [], "has_more": False})
                ck = {"object": "fine_tuning.job.checkpoint", "id": "ftckpt_1", "created_at": 1720000400,
                      "fine_tuned_model_checkpoint": CKPT, "fine_tuning_job_id": "ftjob-000", "step_number": 100,
                      "metrics": {"train_loss": 0.4, "full_valid_loss": 0.5}}
                return self._send(200, {"object": "list", "data": [ck], "has_more": False, "first_id": "ftckpt_1", "last_id": "ftckpt_1"})
            if len(parts) == 2 and parts[0] == "files":
                f = state.files.get(parts[1])
                if not f:
                    return self._err(404, f"No such File object: {parts[1]}")
                return self._send(200, {"object": "file", "id": parts[1], "created_at": 1720000000, "status": "processed", **f})
            if len(parts) == 3 and parts[0] == "files" and parts[2] == "content":
                f = state.files.get(parts[1])
                if not f:
                    return self._err(404, f"No such File object: {parts[1]}")
                if f["purpose"] == "fine-tune":
                    return self._err(400, "Not allowed to download files of purpose: fine-tune")
                return self._send(200, raw=b"step,train_loss\n1,0.9\n", ctype="application/octet-stream")
            return self._err(404, f"Unknown path {url.path}")

        def do_POST(self):
            if not self._authed():
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            if urlparse(self.path).path != "/v1/chat/completions":
                return self._err(404, "Unknown path")
            with state.lock:
                state.chat_calls.append(body)
                if state.fail_429 > 0:
                    state.fail_429 -= 1
                    return self._err(429, "Rate limit reached.", "rate_limit_exceeded", headers={"Retry-After": "0"})
            if body.get("model") == "gpt-6-luna":
                # Documented Chat Completions rules for GPT-6 Luna (reasoning effort defaults to medium).
                effort = body.get("reasoning_effort", "medium")
                if effort != "none" and ("temperature" in body or "top_p" in body):
                    return self._err(400, "Unsupported parameter: 'temperature' is not supported with this model.")
                if effort != "none" and body.get("tools"):
                    return self._err(400, "Function tools with reasoning_effort are not supported for gpt-6-luna "
                                          "in /v1/chat/completions.")
            if body.get("model") not in (FT, CKPT, "gpt-6-luna", "gpt-4o-mini-2024-07-18"):
                return self._err(404, f"The model `{body.get('model')}` does not exist or you do not have access to it.", "model_not_found")
            for m in body.get("messages", []):
                if "weight" in m:
                    return self._err(400, "Unrecognized request argument supplied: weight")
            last = body["messages"][-1].get("content") or ""
            message = {"role": "assistant", "content": f"echo: {last}", "refusal": None}
            if body.get("model") == "gpt-6-luna" and last.endswith(("1", "3")):
                message["content"] = f"Echo {last}, of course."
            if last.startswith("json:"):
                message["content"] = json.dumps({"label": "spam" if body.get("model") != "gpt-6-luna" else "ham", "score": 1})
            if body.get("tools"):
                name = body["tools"][0]["function"]["name"]
                message = {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps({"q": last})}}]}
            return self._send(200, {
                "id": "chatcmpl-1", "object": "chat.completion", "created": 1720000000, "model": body["model"],
                "choices": [{"index": 0, "message": message, "logprobs": None,
                             "finish_reason": "tool_calls" if body.get("tools") else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                          "completion_tokens_details": {"reasoning_tokens": 3 if body.get("reasoning_effort") not in (None, "none") else 0}},
                "system_fingerprint": "fp_test",
            })

        @staticmethod
        def _page(items, q):
            limit = int(q.get("limit", 20))
            start = 0
            if q.get("after"):
                ids = [i["id"] for i in items]
                start = ids.index(q["after"]) + 1
            chunk = items[start:start + min(limit, 2)]  # tiny pages to exercise pagination
            return {"object": "list", "data": chunk, "has_more": start + len(chunk) < len(items)}

    return Handler


class FakeOpenAI:
    def __init__(self, **kw):
        self.state = State(**kw)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

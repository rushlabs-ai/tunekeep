"""A small OpenAI API client built on the standard library.

Only the endpoints tunekeep needs. No third-party packages, so there is no
dependency tree to audit. The API key is sent in a header and never logged.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterator

from . import __version__

DEFAULT_BASE_URL = "https://api.openai.com/v1"
RETRY_STATUSES = {408, 409, 429, 500, 502, 503, 504}


class APIError(Exception):
    """An error response from the API, with the status and OpenAI's message."""

    def __init__(self, status: int, message: str, code: str | None = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.code = code


@dataclass
class Client:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    organization: str | None = None
    project: str | None = None
    azure: bool = False
    timeout: float = 120.0
    max_retries: int = 5
    _sleep: Any = field(default=time.sleep, repr=False)

    @classmethod
    def from_env(cls, base_url: str | None = None, azure: bool = False) -> "Client":
        if azure:
            key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
            name = "AZURE_OPENAI_API_KEY"
        else:
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            name = "OPENAI_API_KEY"
        if not key:
            raise SystemExit(f"tunekeep: set {name} in your environment first.")
        url = base_url or os.environ.get("OPENAI_BASE_URL", "").strip() or DEFAULT_BASE_URL
        if azure and url == DEFAULT_BASE_URL:
            raise SystemExit(
                "tunekeep: with --azure, pass --base-url https://<resource>.openai.azure.com/openai/v1"
            )
        return cls(
            api_key=key,
            base_url=url.rstrip("/"),
            organization=os.environ.get("OPENAI_ORG_ID") or None,
            project=os.environ.get("OPENAI_PROJECT_ID") or None,
            azure=azure,
        )

    # -- low level -----------------------------------------------------------------

    def _headers(self, accept_json: bool = True) -> dict[str, str]:
        h = {"User-Agent": f"tunekeep/{__version__}"}
        if self.azure:
            h["api-key"] = self.api_key
        else:
            h["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            h["OpenAI-Organization"] = self.organization
        if self.project:
            h["OpenAI-Project"] = self.project
        if accept_json:
            h["Accept"] = "application/json"
        return h

    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url += "?" + urllib.parse.urlencode(clean)
        return url

    def _request(self, method: str, path: str, *, params=None, body=None, raw=False):
        data = None
        headers = self._headers(accept_json=not raw)
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        url = self._url(path, params)
        attempt = 0
        while True:
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = resp.read()
                    return payload if raw else json.loads(payload or b"null")
            except urllib.error.HTTPError as e:
                status = e.code
                text = e.read().decode("utf-8", "replace")
                if status in RETRY_STATUSES and attempt < self.max_retries:
                    self._sleep(self._backoff(attempt, e.headers.get("Retry-After")))
                    attempt += 1
                    continue
                raise _api_error(status, text) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt < self.max_retries:
                    self._sleep(self._backoff(attempt, None))
                    attempt += 1
                    continue
                reason = getattr(e, "reason", e)
                raise APIError(0, f"could not reach {self.base_url}: {reason}") from None

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(60.0, max(0.0, float(retry_after)))
            except ValueError:
                pass
        return min(30.0, 0.5 * (2**attempt)) + random.uniform(0, 0.25)

    def get(self, path: str, **params) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, body: dict[str, Any]) -> Any:
        return self._request("POST", path, body=body)

    def paginate(self, path: str, limit: int = 100, **params) -> Iterator[dict[str, Any]]:
        """Yield every item of a cursor-paginated list (`after` = last id)."""
        after = None
        while True:
            page = self.get(path, limit=limit, after=after, **params)
            items = page.get("data") or []
            yield from items
            if not page.get("has_more") or not items:
                return
            after = page.get("last_id") or items[-1].get("id")

    # -- endpoints -----------------------------------------------------------------

    def list_models(self) -> list[dict[str, Any]]:
        return self.get("models").get("data") or []

    def list_jobs(self) -> list[dict[str, Any]]:
        return list(self.paginate("fine_tuning/jobs"))

    def list_checkpoints(self, job_id: str) -> list[dict[str, Any]]:
        return list(self.paginate(f"fine_tuning/jobs/{urllib.parse.quote(job_id)}/checkpoints"))

    def get_file(self, file_id: str) -> dict[str, Any]:
        return self.get(f"files/{urllib.parse.quote(file_id)}")

    def file_content(self, file_id: str) -> bytes:
        return self._request("GET", f"files/{urllib.parse.quote(file_id)}/content", raw=True)

    def chat(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.post("chat/completions", body)


def _api_error(status: int, text: str) -> APIError:
    message, code = text.strip()[:500] or "no response body", None
    try:
        err = json.loads(text).get("error") or {}
        message = err.get("message") or message
        code = err.get("code") or err.get("type")
    except (ValueError, AttributeError):
        pass
    return APIError(status, message, code)

"""Score the retrained model and the untouched base model on banking77, on a Modal L4.

    modal run evaluate.py --run <name of the retrain run>

Each model is served by llama-server (official llama.cpp release, checksum-pinned) and scored
with the released `tunekeep compare` from PyPI (pinned by hash), then `tunekeep report`
writes the HTML report. Results land in results/.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

import modal

HERE = Path(__file__).parent
LLAMA = "b11980"
LLAMA_FILES = {
    f"llama-{LLAMA}-bin-ubuntu-cuda-12.8-x64.tar.gz": "b8a992b3f63fcb47fe33f92aa4f587b2103ee85545fba06f1c94423e29afdabc",
    f"cudart-llama-{LLAMA}-bin-ubuntu-cuda-12.8-x64.tar.gz": "e560a6efd043f7edac962c431423317f0e5a9cdb04bc44c2800ebb1565453f46",
}
TUNEKEEP = ("tunekeep==0.2.1", "44d59f7ccfdda6814efc0039c39ac15c8a75894d06a1847e78e06e8249edbb2d")
BASE_GGUF = {
    "url": "https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/90862c4b9d2787eaed51d12237eafdfe7c5f6077/Qwen3-1.7B-Q8_0.gguf",
    "sha256": "061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a",
}

downloads = " && ".join(
    f"curl -fsSL -o /tmp/{name} https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA}/{name} && "
    f"echo '{sha}  /tmp/{name}' | sha256sum -c - && tar -xzf /tmp/{name} -C /opt/llama && rm /tmp/{name}"
    for name, sha in LLAMA_FILES.items()
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "libgomp1")
    .run_commands("mkdir -p /opt/llama && " + downloads)
    .run_commands(f"echo '{TUNEKEEP[0]} --hash=sha256:{TUNEKEEP[1]}' > /tmp/req.txt && "
                  "pip install --no-deps --require-hashes -r /tmp/req.txt")
    .add_local_file(HERE / "data" / "golden.jsonl", "/root/golden.jsonl")
)
volume = modal.Volume.from_name("tunekeep", create_if_missing=True)
app = modal.App("tunekeep-banking77-eval")


def _wait(url: str, seconds: float) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"{url} not ready")


def _base_model() -> str:
    path = Path("/vol/models/Qwen3-1.7B-Q8_0.gguf")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["curl", "-fsSL", "-o", f"{path}.part", BASE_GGUF["url"]], check=True)
        subprocess.run(["sh", "-c", f"echo '{BASE_GGUF['sha256']}  {path}.part' | sha256sum -c -"], check=True)
        os.replace(f"{path}.part", path)
        volume.commit()
    return str(path)


@app.function(image=image, gpu="L4", volumes={"/vol": volume}, timeout=60 * 60)
def evaluate(run: str) -> dict:
    volume.reload()
    server = next(Path("/opt/llama").rglob("llama-server"))
    env = dict(os.environ, LD_LIBRARY_PATH=":".join(sorted({str(p.parent) for p in Path("/opt/llama").rglob("*.so*")})))
    out = Path("/tmp/out")
    out.mkdir()
    candidates = [("qwen3-1.7b base", _base_model()), ("qwen3-1.7b + LoRA", f"/vol/runs/{run}/model-q8_0.gguf")]
    timings = {}
    for label, gguf in candidates:
        proc = subprocess.Popen([str(server), "-m", gguf, "--port", "8080", "--host", "127.0.0.1", "--jinja",
                                 "-ngl", "99", "-c", "16384", "-np", "8"], env=env,
                                stdout=open(f"/tmp/{label[:12].replace(' ', '_')}.log", "wb"), stderr=subprocess.STDOUT)
        _wait("http://127.0.0.1:8080/health", 300)
        t0 = time.time()
        subprocess.run(["tunekeep", "compare", "--golden", "/root/golden.jsonl", "--candidate", label,
                        "--candidate-base-url", "http://127.0.0.1:8080/v1", "--label", label, "--workers", "8",
                        "--max-tokens", "64", "--extra-body", json.dumps({"chat_template_kwargs": {"enable_thinking": False}}),
                        "--out", str(out / f"compare-{label.replace(' ', '_').replace('+', 'plus')}.jsonl")],
                       check=True, env=env)
        timings[label] = round(time.time() - t0, 1)
        proc.terminate()
        proc.wait(30)
    subprocess.run(["tunekeep", "report", "--dir", str(out), "--golden", "/root/golden.jsonl",
                    "--retrain-summary", f"/vol/runs/{run}/summary.json", "--out", str(out / "report.html")], check=True)
    files = {p.name: p.read_text() for p in out.iterdir()}
    files["summary.json"] = Path(f"/vol/runs/{run}/summary.json").read_text()
    return {"files": files, "seconds": timings, "gpu": "L4", "llama_cpp": LLAMA}


@app.local_entrypoint()
def main(run: str) -> None:
    res = evaluate.remote(run)
    dest = HERE / "results"
    dest.mkdir(exist_ok=True)
    for name, text in res["files"].items():
        (dest / name).write_text(text)
    print(json.dumps({"seconds": res["seconds"], "gpu": res["gpu"]}, indent=2))
    print(f"Results in {dest}")

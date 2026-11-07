"""Run the LoRA retrain on Modal (https://modal.com). Written by tunekeep.

    modal run modal_app.py                  # train on a GPU, then convert to GGUF on CPU
    modal run modal_app.py --no-gguf        # skip the GGUF step
    modal run modal_app.py --max-steps 5    # a cheap smoke test first

Your training data (data/) is uploaded to your own Modal account for the run.
Outputs (LoRA adapter, merged weights, GGUF, metrics) go to the Modal Volume
"tunekeep" under runs/<name>/. The job stops after `timeout_hours`, which caps
the cost of a run.

Packages are pinned with hashes (requirements-*.txt) and llama.cpp is pinned
to an exact commit.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).parent
KIT = "/root/kit"
CONFIG = json.loads((HERE / "config.json" if (HERE / "config.json").exists() else Path(KIT) / "config.json").read_text())

LLAMA_CPP_TAG = "b11980"
LLAMA_CPP_COMMIT = "f7c01c5226cac1de1052c17920ff38cdefea0ae6"

volume = modal.Volume.from_name("tunekeep", create_if_missing=True)
app = modal.App("tunekeep-retrain")


# Image definitions are evaluated lazily, so the local paths below are only read on your machine.
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(HERE / "requirements-train.txt"))
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_file(HERE / "config.json", f"{KIT}/config.json")
    .add_local_file(HERE / "train_lora.py", f"{KIT}/train_lora.py")
    .add_local_dir(HERE / "data", f"{KIT}/data")
)

gguf_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        f"git clone --depth 1 --branch {LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        f'test "$(git -C /opt/llama.cpp rev-parse HEAD)" = "{LLAMA_CPP_COMMIT}"',
    )
    .pip_install_from_requirements(str(HERE / "requirements-gguf.txt"))
)


@app.function(image=train_image, gpu=CONFIG.get("gpu", "L40S"), volumes={"/vol": volume},
              timeout=int(float(CONFIG.get("timeout_hours", 2)) * 3600))
def train(run: str, max_steps: int = 0) -> dict:
    import subprocess

    out = f"/vol/runs/{run}"
    cmd = ["python", f"{KIT}/train_lora.py", "--config", f"{KIT}/config.json", "--data", f"{KIT}/data", "--out", out]
    if max_steps:
        cmd += ["--max-steps", str(max_steps)]
    subprocess.run(cmd, check=True)
    volume.commit()
    return json.loads(Path(out, "summary.json").read_text())


@app.function(image=gguf_image, cpu=4, memory=32768, volumes={"/vol": volume}, timeout=2 * 3600)
def to_gguf(run: str, outtype: str) -> str:
    import subprocess

    volume.reload()
    out = Path(f"/vol/runs/{run}")
    target = out / f"model-{outtype}.gguf"
    subprocess.run(["python", "/opt/llama.cpp/convert_hf_to_gguf.py", str(out / "merged"),
                    "--outfile", str(target), "--outtype", outtype], check=True)
    volume.commit()
    return str(target.relative_to("/vol"))


@app.local_entrypoint()
def main(run: str = "", max_steps: int = 0, no_gguf: bool = False) -> None:
    run = run or time.strftime("%Y%m%d-%H%M%S")
    print(f"run {run}: training {CONFIG['base']} on {CONFIG.get('gpu', 'L40S')}")
    summary = train.remote(run, max_steps)
    local = HERE / "runs" / run
    local.mkdir(parents=True, exist_ok=True)
    (local / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("train_samples", "steps", "train_seconds", "validation_loss_before",
                                               "validation_loss_by_epoch")}, indent=2))
    files = [f"runs/{run}/adapter/", f"runs/{run}/merged/"]
    if not no_gguf and CONFIG.get("gguf"):
        files.append(to_gguf.remote(run, CONFIG["gguf"]))
    print(f"\nSummary saved to {local / 'summary.json'}. Download with:")
    for f in files:
        print(f"  modal volume get tunekeep {f} {local}/")

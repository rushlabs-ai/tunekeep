"""Check that the patched GGUF environment produces the same GGUF files as llama.cpp's own pins.

    modal run tools/verify_gguf_env.py

llama.cpp pins transformers 4.57.6 / torch 2.11.0 / protobuf 4.x for convert_hf_to_gguf.py, which
carry known advisories. The kit uses patched versions instead (requirements-gguf.txt). This
converts the same models with both environments and compares every tensor (name, type, shape,
bytes) and every metadata field. Models: the banking77 LoRA merge (Qwen3 causal LM, tunekeep
Volume) and Rush Small (Qwen3 reranker, rush-small Volume).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import modal

HERE = Path(__file__).parent
LLAMA = ("b11980", "f7c01c5226cac1de1052c17920ff38cdefea0ae6")
MODELS = {
    "banking77 (Qwen3 causal LM)": ("/tk/runs/banking77-v1/merged", "banking77"),
    "rush-small (Qwen3 reranker)": ("/rs/models/rush-small-qwen3-reranker-0.6b-v0", "rush-small-qwen3-reranker"),
}

app = modal.App("tunekeep-verify-gguf-env")
tk = modal.Volume.from_name("tunekeep")
rs = modal.Volume.from_name("rush-small")
scratch = modal.Volume.from_name("gguf-env-check", create_if_missing=True)


def env_image(requirements: Path) -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .run_commands(
            f"git clone --depth 1 --branch {LLAMA[0]} https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
            f'test "$(git -C /opt/llama.cpp rev-parse HEAD)" = "{LLAMA[1]}"',
        )
        .pip_install_from_requirements(str(requirements))
    )


old_image = env_image(HERE / "locks" / "gguf" / "llama-cpp-pins.txt")
new_image = env_image(HERE.parent / "src" / "tunekeep" / "kit" / "requirements-gguf.txt")
VOLS = {"/tk": tk, "/rs": rs, "/out": scratch}


def _convert(env: str) -> dict:
    import transformers

    out = {}
    for label, (src, name) in MODELS.items():
        # The directory name keeps "qwen3-reranker" so llama.cpp detects the reranker format.
        work = Path("/tmp") / name
        if work.exists():
            subprocess.run(["rm", "-rf", str(work)], check=True)
        subprocess.run(["cp", "-r", src, str(work)], check=True)
        dest = Path("/out") / env / f"{name}.gguf"
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["python", "/opt/llama.cpp/convert_hf_to_gguf.py", str(work), "--outfile", str(dest),
                        "--outtype", "q8_0"], check=True, stdout=subprocess.DEVNULL)
        out[label] = str(dest)
    scratch.commit()
    return {"transformers": transformers.__version__, "files": out}


@app.function(image=old_image, cpu=4, memory=16384, volumes=VOLS, timeout=3600)
def convert_old() -> dict:
    return _convert("old")


@app.function(image=new_image, cpu=4, memory=16384, volumes=VOLS, timeout=3600)
def convert_new() -> dict:
    return _convert("new")


@app.function(image=new_image, cpu=2, memory=8192, volumes=VOLS, timeout=1800)
def compare(old: dict, new: dict) -> dict:
    import sys

    sys.path.insert(0, "/opt/llama.cpp/gguf-py")
    from gguf import GGUFReader

    scratch.reload()

    def describe(path):
        r = GGUFReader(path)
        meta = {}
        for k, f in r.fields.items():
            if k.startswith("GGUF.") or k == "general.file_type":
                continue
            meta[k] = hashlib.sha256(b"".join(bytes(p) for p in f.parts)).hexdigest()
        tensors = {t.name: (str(t.tensor_type), tuple(int(x) for x in t.shape), hashlib.sha256(t.data.tobytes()).hexdigest())
                   for t in r.tensors}
        return meta, tensors

    report = {"old_transformers": old["transformers"], "new_transformers": new["transformers"], "models": {}}
    for label in MODELS:
        m1, t1 = describe(old["files"][label])
        m2, t2 = describe(new["files"][label])
        report["models"][label] = {
            "tensors": len(t1),
            "tensors_identical": t1 == t2,
            "tensors_differing": sorted(k for k in set(t1) | set(t2) if t1.get(k) != t2.get(k))[:10],
            "metadata_identical": m1 == m2,
            "metadata_differing": sorted(k for k in set(m1) | set(m2) if m1.get(k) != m2.get(k)),
        }
    return report


@app.local_entrypoint()
def main() -> None:
    o, n = convert_old.spawn(), convert_new.spawn()
    report = compare.remote(o.get(), n.get())
    (HERE / "verify_gguf_env.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

"""Optional end-to-end smoke test of the retraining kit on CPU.

Skipped unless both are set:
  TUNEKEEP_KIT_PYTHON  a Python with requirements-train.txt installed (CPU torch is fine)
  TUNEKEEP_KIT_BASE    a local Qwen3 model folder (config + tokenizer; weights optional)

With no weights in TUNEKEEP_KIT_BASE, a tiny random model with the same tokenizer
and chat template is built, so the test runs in seconds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from test_v02 import write_training  # noqa: E402

from tunekeep import retrain  # noqa: E402

PY = os.environ.get("TUNEKEEP_KIT_PYTHON")
BASE = os.environ.get("TUNEKEEP_KIT_BASE")

TINY = """
import json, shutil, sys
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM
src, dst = Path(sys.argv[1]), Path(sys.argv[2])
cfg = AutoConfig.from_pretrained(src)
for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=16).items():
    setattr(cfg, k, v)
if getattr(cfg, "layer_types", None):
    cfg.layer_types = cfg.layer_types[:cfg.num_hidden_layers]
AutoModelForCausalLM.from_config(cfg).save_pretrained(dst)
for f in src.iterdir():
    if f.name.startswith(("tokenizer", "vocab", "merges", "generation_config")):
        shutil.copy(f, dst / f.name)
"""


@unittest.skipUnless(PY and BASE, "set TUNEKEEP_KIT_PYTHON and TUNEKEEP_KIT_BASE to run the kit smoke test")
class KitSmokeTest(unittest.TestCase):
    def test_kit_trains_and_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            base = Path(BASE)
            if not any(base.glob("*.safetensors")):
                subprocess.run([PY, "-c", TINY, str(base), str(tmp / "tiny")], check=True)
                base = tmp / "tiny"
            extra = [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "skip me", "weight": 0},
                                   {"role": "user", "content": "q"}, {"role": "assistant", "content": "train me"}]}]
            rows = retrain.convert_file(write_training(tmp / "t.jsonl", n=8, extra=extra), retrain.Report())
            tool_rows = retrain.convert_file(write_training(tmp / "tools.jsonl", n=4, tools=True), retrain.Report())
            cfg = retrain.build_config(str(base), None, 2, "L4", "0" * 64, batch_size=2, grad_accum=2)
            kit = tmp / "kit"
            retrain.write_kit(kit, rows + tool_rows, rows[:2], cfg)
            out = tmp / "out"
            proc = subprocess.run([PY, str(kit / "train_lora.py"), "--config", str(kit / "config.json"),
                                   "--data", str(kit / "data"), "--out", str(out)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
            summary = json.loads((out / "summary.json").read_text())
            # 8 plain + 1 multi-turn (one trained turn) + 4 tool examples = 13 samples, none skipped.
            self.assertEqual(summary["train_samples"], 13)
            self.assertEqual(summary["skipped"], {"skipped_template": 0, "skipped_too_long": 0})
            # 13 samples / batch 2 = 7 micro-batches / accum 2 = 4 steps per epoch, 2 epochs.
            self.assertEqual(summary["steps"], 8)
            for name in ("adapter/adapter_config.json", "merged/config.json", "merged/tokenizer.json", "metrics.jsonl"):
                self.assertTrue((out / name).exists(), name)


if __name__ == "__main__":
    unittest.main()

# Retraining kit (written by tunekeep)

This folder retrains an open model with LoRA on **your own training data** and
exports it for local serving.

| File | What it is |
|---|---|
| `data/train.jsonl`, `data/validation.jsonl` | Your data, converted to plain chat format. Private (0600); not committed (`.gitignore`). |
| `config.json` | Base model (pinned revision), epochs, LoRA settings, GPU, time limit. Edit freely. |
| `train_lora.py` | The training script. Loss only on assistant messages; `"weight": 0` turns are skipped. |
| `modal_app.py` | Runs the script on a Modal GPU, then converts to GGUF on CPU. |
| `requirements-train.txt`, `requirements-gguf.txt` | Exact versions, pinned by hash. |

## Run on Modal

```sh
pip install modal && modal setup        # once
modal run modal_app.py --max-steps 5    # smoke test: a few minutes, cents
modal run modal_app.py                  # the real run
```

- Your `data/` is uploaded to **your** Modal account for the run.
- The job stops after `timeout_hours` (config.json), which caps what a run can cost.
  Set a workspace budget in Modal's settings as well.
- Outputs land in the Modal Volume `tunekeep` under `runs/<name>/`: `adapter/` (LoRA),
  `merged/` (full weights), `model-q8_0.gguf`, `metrics.jsonl`, `summary.json`.
  The run prints the `modal volume get` commands to download them.

On any other CUDA machine: `pip install -r requirements-train.txt`, then
`python train_lora.py --config config.json --data data --out out`.

## Serve and compare

```sh
llama-server -m model-q8_0.gguf --jinja --port 8080
tunekeep compare --golden tunekeep-out/golden-<model>.jsonl \
  --candidate retrained --candidate-base-url http://127.0.0.1:8080/v1 \
  --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}'
tunekeep report --retrain-summary runs/<name>/summary.json
```

The model was trained without thinking blocks (`enable_thinking: false` for Qwen3), so serve it the same way.

## Licences

The base model keeps its own licence (Qwen3: Apache-2.0). Your data is yours. Check that
your training data is not itself the output of a provider whose terms forbid training on it.

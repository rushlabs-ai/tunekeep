# tunekeep

Keep a fine-tuned model working after its provider retires the base model.

OpenAI shuts down fine-tuned **gpt-4o, gpt-4o-mini, gpt-4.1 and gpt-4.1-mini** models on
**14 January 2027**, and stops new fine-tuning jobs on **6 January 2027**. The named replacement,
`gpt-6-luna`, cannot be fine-tuned. Once a model is gone, so is the only exact record of how it behaved.

tunekeep helps you get out in time:

| Step | Command |
|---|---|
| 1. See every fine-tune you own and its days left | `tunekeep inventory` |
| 2. Record how each model answers, as a golden eval set | `tunekeep record` |
| 3. Check whether gpt-6-luna (or any model) already matches it | `tunekeep compare` |
| 4. Retrain an open model with LoRA from your own data | `tunekeep retrain` |
| 5. One HTML report: quality, speed and tokens side by side | `tunekeep report` |

**Do the recording first.** Everything else can happen later; recording cannot happen after the shutdown.

## Install

```sh
uv tool install tunekeep      # or: pipx install tunekeep
```

Python 3.10+. **No dependencies**: tunekeep uses only the Python standard library. (The retraining
kit it writes has its own pinned requirements, installed only where you run it.)

## 1. Inventory

```sh
export OPENAI_API_KEY=sk-...          # a key for the org/project that owns the fine-tunes
tunekeep inventory
```

```text
Fine-tuned models: 1 (plus 1 checkpoint)
New fine-tuning jobs stop on 2027-01-06 (60 days).

MODEL                                   SHUTS DOWN   DAYS LEFT  SOURCE
ft:gpt-4o-mini-2024-07-18:acme::AbC123  2027-01-14   68         api

Files: 1 saved, 1 training/validation file the API would not return.
  Keep your own copy of these; tunekeep retrains only from your original data:
  - file-train-0  train.jsonl  (job ftjob-000)

Next: record how each model behaves while it still answers, e.g.
  tunekeep record --model ft:gpt-4o-mini-2024-07-18:acme::AbC123 --prompts prompts.jsonl
```

`tunekeep-out/inventory.json` holds the full detail: jobs, hyperparameters, checkpoints and their
metrics, and every file. Result files (training metrics) are downloaded. Files you uploaded for
training are not: OpenAI does not return them through the API, so keep your own copy.

## 2. Record

```sh
tunekeep record --model ft:gpt-4o-mini-2024-07-18:acme::AbC123 --prompts prompts.jsonl
```

- `prompts.jsonl` can be your **training file as-is**. The final assistant message on each line is
  removed before sending and kept as the reference answer. DPO lines (`input.messages`) work too.
- Real production prompts make the best golden set. Use a few hundred, covering your real traffic.
- Answers are recorded at `temperature 0` by default. Use `--seed` for more repeatable runs.
- Safe to stop and re-run: prompts already recorded are skipped, failed ones are retried.
- `--dry-run` counts prompts and estimated input tokens without sending anything.

## 3. Compare

```sh
tunekeep compare --golden tunekeep-out/golden-ft_gpt-4o-mini-2024-07-18_acme_AbC123.jsonl \
  --candidate gpt-6-luna --examples train.jsonl --shots 5
```

Each recorded prompt goes to the candidate, optionally with a few examples from **your own**
training file, and the answer is scored against the recorded one:

- **Tool calls**: matched by function name, arguments compared field by field.
- **JSON answers**: the share of leaf fields with the same value; invalid JSON scores 0.
- **Text**: 100% when equal after normalising case and spacing, otherwise word overlap (F1). Overlap
  is a rough signal for free text, so the report shows the biggest differences side by side.

GPT-6 Luna reasons by default, and in Chat Completions it accepts `temperature` and function tools
only with `reasoning_effort: "none"`. tunekeep sends `none` by default (`--reasoning-effort auto`),
which is also closest to how a fine-tuned gpt-4o-mini answered. Try `--reasoning-effort low` to see
whether reasoning helps; tunekeep then drops `temperature`, and refuses before sending anything if your
prompts use tools.

Any OpenAI-compatible endpoint works as a candidate: `--candidate-base-url http://127.0.0.1:8080/v1`
for llama-server, vLLM or Ollama (no key needed on localhost; `--candidate-key-env` for others).

## 4. Retrain

```sh
tunekeep retrain --data train.jsonl          # default base: Qwen/Qwen3-1.7B, pinned revision
cd tunekeep-out/retrain && modal run modal_app.py
```

`retrain` converts your OpenAI training file (chat format, tools, legacy `functions`, per-message
`weight`) and writes a small kit: your data, a config, a training script and a [Modal](https://modal.com)
app. It uploads nothing itself. The kit:

- trains a LoRA adapter with loss only on assistant messages (`"weight": 0` turns are skipped, as on OpenAI);
- reuses your original job's epoch count when `tunekeep-out/inventory.json` has it;
- pins every package by hash, the base model by revision, and llama.cpp by commit;
- stops after a time limit (default 2 hours), which caps the cost of a run;
- exports merged weights and a q8_0 GGUF for llama-server or Ollama.

Use `--check` to only validate and summarise the data. See the kit's own README for running it
elsewhere and serving the result. Not yet supported: DPO, images and audio in training data.

## 5. Report

```sh
tunekeep report                      # reads tunekeep-out/, writes tunekeep-out/report.html
```

One self-contained HTML file with deadlines, every candidate's scores, latency and token use, and
the lowest-scoring answers side by side. It loads nothing from the network and runs no scripts, and
every value from your data is escaped.

## What tunekeep will not do

**It never trains on recorded outputs.** OpenAI's Services Agreement (section 3.3(e)) restricts using
Output to develop competing models. Recorded answers are used only to *evaluate* a replacement.
Retraining uses only your original training data. If that data was itself generated by an OpenAI
model, it counts as Output too. Check with your own counsel; this is not legal advice.

## Privacy

- tunekeep talks only to the API base URLs you give it. No telemetry, no other network calls.
- Outputs can contain production prompts, so the output folder is created `0700` and files `0600`.
- API keys are read from the environment and are never written to disk or printed. A candidate
  endpoint never receives your OpenAI key unless you name that variable with `--candidate-key-env`.

## Other providers

- `--base-url` / `OPENAI_BASE_URL` for OpenAI-compatible gateways.
- `--azure --base-url https://<resource>.openai.azure.com/openai/v1` with `AZURE_OPENAI_API_KEY`
  (**experimental**). Azure retires fine-tuned gpt-4o and gpt-4o-mini deployments on 1 October 2027, and
  the gpt-4.1 family on 14 October 2027.

## Development

```sh
python -m unittest discover -s tests -v
```

The tests run against a fake OpenAI server built from the public API reference: pagination, the
`shutdown_date` field, blocked training-file downloads, rate limits, tool calls and GPT-6 Luna's
parameter rules. The retraining kit is tested end to end on CPU with Qwen3-0.6B.
`tools/lock-kit.sh` regenerates the kit's hash-locked requirements.
**Reports from real accounts are very welcome.** Please open an issue with anything that behaves differently.

## Sources

- OpenAI deprecations: https://developers.openai.com/api/docs/deprecations
- GPT-6 parameters: https://developers.openai.com/api/docs/guides/latest-model
- Azure retirement schedule: https://learn.microsoft.com/en-us/azure/foundry/openai/concepts/model-retirement-schedule

## Licence

Apache-2.0. Built by [Rush Labs](https://rushlabs.ai), an applied AI lab.

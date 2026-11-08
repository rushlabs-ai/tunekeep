# Changelog

## 0.2.2

- Security: the retrain kit's GGUF step no longer uses llama.cpp's own pins (transformers 4.57.6,
  torch 2.11.0, protobuf 4.25.9, setuptools 81.0.0), which have 17 known advisories including remote
  code execution (GHSA-29pf-2h5f-8g72). It now uses transformers 5.22.1, torch 2.14.1, protobuf 6.33.6
  and setuptools 84.2.1; `tools/verify_gguf_env.py` shows the GGUF files are byte-identical.
- `retrain --batch-size` and `--grad-accum`.
- The report hides the training-answers and JSON columns when no run has them.
- Retrain kit: the GGUF step's container now includes config.json (#1).
- examples/banking77: a public proof run (Qwen3-1.7B, 50.3% → 90.6% accuracy).

## 0.2.1

- First release on PyPI (same features as 0.2.0; the 0.2.0 release workflow could not check out the private repository).

## 0.2.0

- `tunekeep compare`: score any candidate (gpt-6-luna, an OpenAI-compatible endpoint, your served
  retrained model) against the recorded golden set and the training answers. Tool calls, JSON fields
  and text are scored separately. Optional few-shot examples from your own training file. Applies
  GPT-6 Luna's Chat Completions rules (`reasoning_effort: none` for tools and temperature).
- `tunekeep retrain`: converts OpenAI fine-tuning JSONL (tools, legacy functions, per-message weights)
  and writes a LoRA kit for an open model (default Qwen3-1.7B, pinned revision): training script,
  Modal app with a time limit, hash-locked requirements, GGUF export via pinned llama.cpp.
- `tunekeep report`: one self-contained HTML report (strict CSP, no scripts, all data escaped).
- Golden rows are read in prompt-file order. Local endpoints never receive your OpenAI key.

## 0.1.0

- `tunekeep inventory`: fine-tuned models, checkpoints, jobs and files, with shutdown dates from the
  API's `shutdown_date` field (falls back to OpenAI's published schedule) and days left.
- `tunekeep record`: replays prompts (OpenAI training JSONL works as-is) through a fine-tuned model
  and saves a resumable golden eval set.

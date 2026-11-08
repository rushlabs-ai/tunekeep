# Changelog

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

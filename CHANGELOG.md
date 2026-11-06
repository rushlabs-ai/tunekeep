# Changelog

## 0.1.0

- `tunekeep inventory`: fine-tuned models, checkpoints, jobs and files, with shutdown dates from the
  API's `shutdown_date` field (falls back to OpenAI's published schedule) and days left.
- `tunekeep record`: replays prompts (OpenAI training JSONL works as-is) through a fine-tuned model
  and saves a resumable golden eval set.

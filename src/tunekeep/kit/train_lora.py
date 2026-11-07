"""LoRA fine-tune of an open model on OpenAI-format chat data. Written by tunekeep.

Runs on any machine with the pinned packages in requirements-train.txt (a CUDA GPU
for real runs; CPU works for tiny smoke tests). modal_app.py runs it on Modal.

Loss is computed only on assistant messages, one sample per trained assistant
turn, matching OpenAI's rule that every assistant message is trained unless it
has "weight": 0. Long samples are skipped and counted, never silently cut.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def render(tok, messages, tools, add_generation_prompt: bool) -> str:
    kw = {"tokenize": False, "add_generation_prompt": add_generation_prompt, "enable_thinking": False}
    if tools:
        kw["tools"] = tools
    return tok.apply_chat_template(messages, **kw)


def build_samples(tok, examples: list[dict], max_len: int, stats: dict) -> list[dict]:
    samples = []
    for ex in examples:
        msgs, tools = ex["messages"], ex.get("tools")
        clean = [{k: v for k, v in m.items() if k != "weight"} for m in msgs]
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant" or m.get("weight", 1) == 0:
                continue
            prompt = render(tok, clean[:i], tools, True)
            full = render(tok, clean[: i + 1], tools, False)
            if not full.startswith(prompt):
                # The template renders this turn differently in context; skip rather than mis-mask.
                stats["skipped_template"] += 1
                continue
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(c_ids) > max_len:
                stats["skipped_too_long"] += 1
                continue
            samples.append({"input_ids": p_ids + c_ids, "labels": [-100] * len(p_ids) + c_ids})
            stats["target_tokens"] += len(c_ids)
            stats["tokens"] += len(p_ids) + len(c_ids)
    return samples


def batches(samples: list[dict], batch_size: int, rng: random.Random, shuffle: bool):
    order = list(range(len(samples)))
    if shuffle:
        rng.shuffle(order)
        # Group similar lengths inside windows to waste less compute on padding.
        window = batch_size * 32
        order = [i for w in range(0, len(order), window)
                 for i in sorted(order[w:w + window], key=lambda j: len(samples[j]["input_ids"]))]
        groups = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
        rng.shuffle(groups)
    else:
        groups = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    yield from groups


def collate(torch, samples: list[dict], idx: list[int], pad_id: int, device):
    width = max(len(samples[i]["input_ids"]) for i in idx)
    ids = torch.full((len(idx), width), pad_id, dtype=torch.long)
    labels = torch.full((len(idx), width), -100, dtype=torch.long)
    mask = torch.zeros((len(idx), width), dtype=torch.long)
    for row, i in enumerate(idx):
        s = samples[i]
        n = len(s["input_ids"])
        ids[row, :n] = torch.tensor(s["input_ids"])
        labels[row, :n] = torch.tensor(s["labels"])
        mask[row, :n] = 1
    return ids.to(device), mask.to(device), labels.to(device)


def evaluate(torch, model, samples, batch_size, pad_id, device) -> float | None:
    if not samples:
        return None
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for idx in batches(samples, batch_size, random.Random(0), shuffle=False):
            ids, mask, labels = collate(torch, samples, idx, pad_id, device)
            n = int((labels[:, 1:] != -100).sum())
            if n == 0:
                continue
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            total += float(loss) * n
            count += n
    model.train()
    return total / count if count else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--max-steps", type=int, default=0, help="Stop early (smoke tests)")
    args = ap.parse_args()

    import torch
    import transformers
    import peft
    from huggingface_hub import snapshot_download
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = cfg["base"]
    if Path(base).is_dir():
        local, revision = base, None
    else:
        local = snapshot_download(base, revision=cfg.get("revision") or None,
                                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])
        revision = Path(local).name
    print(f"base {base} @ {revision or 'local'} on {device}", flush=True)

    tok = AutoTokenizer.from_pretrained(local)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    stats = {"skipped_template": 0, "skipped_too_long": 0, "target_tokens": 0, "tokens": 0}
    train = build_samples(tok, load_jsonl(Path(args.data) / "train.jsonl"), int(cfg["max_len"]), stats)
    vstats = {"skipped_template": 0, "skipped_too_long": 0, "target_tokens": 0, "tokens": 0}
    val = build_samples(tok, load_jsonl(Path(args.data) / "validation.jsonl"), int(cfg["max_len"]), vstats)
    print(f"samples: train {len(train)}, validation {len(val)}; skipped {stats}", flush=True)
    if not train:
        raise SystemExit("no training samples left; see the skipped counts above")

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(local, dtype=dtype)
    model.to(device)
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]), lora_dropout=float(cfg["lora_dropout"]),
                      target_modules=cfg.get("target_modules", "all-linear"), task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable parameters: {trainable:,}", flush=True)

    bs, accum, epochs = int(cfg["batch_size"]), int(cfg["grad_accum"]), int(cfg["epochs"])
    steps_per_epoch = math.ceil(math.ceil(len(train) / bs) / accum)
    total_steps = steps_per_epoch * epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = max(1, int(total_steps * float(cfg.get("warmup_ratio", 0.03))))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["lr"]),
                            weight_decay=0.0, fused=device.type == "cuda")

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    val_before = evaluate(torch, model, val, bs, pad_id, device)
    print(f"validation loss before: {val_before}", flush=True)

    metrics = (out / "metrics.jsonl").open("w")
    rng = random.Random(seed)
    step, window_loss, window_n = 0, 0.0, 0
    started = time.time()
    tokens_seen = 0
    epoch_val = []
    model.train()

    def optimizer_step(epoch: int) -> None:
        nonlocal step, window_loss, window_n
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % 10 == 0 or step == total_steps:
            row = {"step": step, "epoch": epoch + 1, "loss": round(window_loss / max(1, window_n), 4),
                   "lr": sched.get_last_lr()[0], "tokens_per_s": round(tokens_seen / (time.time() - started))}
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            print(row, flush=True)
            window_loss, window_n = 0.0, 0

    for epoch in range(epochs):
        pending = 0  # micro-batches since the last optimizer step
        for idx in batches(train, bs, rng, shuffle=True):
            ids, mask, labels = collate(torch, train, idx, pad_id, device)
            tokens_seen += int(mask.sum())
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            (loss / accum).backward()
            window_loss += loss.item()
            window_n += 1
            pending += 1
            if pending == accum:
                optimizer_step(epoch)
                pending = 0
                if step >= total_steps:
                    break
        if pending and step < total_steps:
            optimizer_step(epoch)  # the epoch's last, partial accumulation
        epoch_val.append(evaluate(torch, model, val, bs, pad_id, device))
        print(f"epoch {epoch + 1}: validation loss {epoch_val[-1]}", flush=True)
        if step >= total_steps:
            break
    metrics.close()
    train_seconds = round(time.time() - started)

    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(out / "merged")
    tok.save_pretrained(out / "merged")
    # The tokenizer is unchanged: keep the base model's own files so every converter reads them.
    for f in Path(local).iterdir():
        if f.name.startswith(("tokenizer", "vocab", "merges", "special_tokens_map", "chat_template", "generation_config")):
            shutil.copy(f, out / "merged" / f.name)

    samples_out = []
    merged.eval()
    for ex in load_jsonl(Path(args.data) / "validation.jsonl")[:3]:
        msgs = [{k: v for k, v in m.items() if k != "weight"} for m in ex["messages"]]
        last = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"), default=None)
        if last is None:
            continue
        prompt = render(tok, msgs[:last], ex.get("tools"), True)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            gen = merged.generate(**ids, max_new_tokens=256, do_sample=False)
        answer = tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        samples_out.append({"expected": msgs[last].get("content"), "got": answer})

    summary = {
        "base": base,
        "revision": revision,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "train_samples": len(train),
        "validation_samples": len(val),
        "skipped": {k: stats[k] for k in ("skipped_template", "skipped_too_long")},
        "target_tokens": stats["target_tokens"],
        "epochs": epochs,
        "steps": step,
        "train_seconds": train_seconds,
        "tokens_per_second": round(tokens_seen / max(1, train_seconds)),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
        "validation_loss_before": val_before,
        "validation_loss_by_epoch": epoch_val,
        "trainable_parameters": trainable,
        "samples": samples_out,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "config": cfg,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("samples", "config")}, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

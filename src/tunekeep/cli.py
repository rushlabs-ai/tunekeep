"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from . import __version__
from .client import APIError, Client
from .files import ensure_private_dir, write_private

DEFAULT_OUT = "tunekeep-out"


def _client(args: argparse.Namespace) -> Client:
    return Client.from_env(base_url=args.base_url, azure=args.azure)


def _err(msg: str) -> int:
    print(f"tunekeep: {msg}", file=sys.stderr)
    return 2


def cmd_inventory(args: argparse.Namespace) -> int:
    from .inventory import collect, render

    out = Path(args.out)
    ensure_private_dir(out)
    inv = collect(_client(args), out, download=not args.no_download)
    print(render(inv))
    print(f"\nFull inventory: {out / 'inventory.json'}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    from .record import TERMS_NOTE, PromptError, estimate_tokens, load_prompts, run

    try:
        prompts = load_prompts(Path(args.prompts))
    except (OSError, PromptError) as e:
        return _err(str(e))
    if args.limit:
        prompts = prompts[: args.limit]
    out = Path(args.out) if args.out else Path(DEFAULT_OUT) / f"golden-{_slug(args.model)}.jsonl"
    print(f"{len(prompts)} unique prompts, about {estimate_tokens(prompts):,} input tokens.")
    if not args.model.startswith("ft:"):
        print("Note: this is not a fine-tuned model id (ft:...). Recording anyway.")
    if args.dry_run:
        print(f"Dry run: nothing sent. Output would go to {out}")
        return 0
    print(TERMS_NOTE)
    res = run(
        _client(args),
        args.model,
        prompts,
        out,
        workers=args.workers,
        temperature=None if args.temperature < 0 else args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
        progress=sys.stderr,
    )
    print(f"Recorded {res.recorded}, already had {res.skipped}, failed {res.failed}. Golden set: {out}")
    return _finish(res)


def _finish(res) -> int:
    if res.first_error:
        print(f"First error: {res.first_error}", file=sys.stderr)
        print("Run the same command again to retry only the missing prompts.", file=sys.stderr)
    return 1 if res.failed else 0


def cmd_compare(args: argparse.Namespace) -> int:
    from .compare import CompareError, load_golden, load_rows, load_shots, plan_params, render_summary, run, summarize
    from .record import PromptError

    golden_path = Path(args.golden)
    try:
        golden = load_golden(golden_path)
        shots = load_shots(Path(args.examples)) if args.examples else []
    except (OSError, CompareError, PromptError) as e:
        return _err(str(e))
    if args.limit:
        golden = golden[: args.limit]
    n_shots = args.shots if args.shots is not None else (5 if args.examples else 0)
    if n_shots and not shots:
        return _err("--shots needs --examples with single-turn, plain-text training lines")
    extra = None
    if args.extra_body:
        try:
            extra = json.loads(args.extra_body)
        except ValueError:
            extra = None
        if not isinstance(extra, dict):
            return _err("--extra-body must be a JSON object")
    effort = args.reasoning_effort
    label = args.label or args.candidate + (f" +{n_shots}-shot" if n_shots else "") + (
        f" effort={effort}" if effort != "auto" else "")
    out = Path(args.out) if args.out else golden_path.parent / f"compare-{_slug(label)}.jsonl"
    temperature = None if args.temperature < 0 else args.temperature
    has_tools = any(r["request"].get("tools") for r in golden)
    try:
        params, notes = plan_params(args.candidate, effort, temperature, has_tools)
    except CompareError as e:
        return _err(str(e))
    print(f"{len(golden)} recorded prompts -> {label}" + (f", {n_shots} examples each" if n_shots else ""))
    print(f"Parameters: {json.dumps(params) if params else 'model defaults'}")
    for n in notes:
        print(f"Note: {n}")
    if args.dry_run:
        print(f"Dry run: nothing sent. Output would go to {out}")
        return 0
    if args.candidate_base_url:
        client = Client.for_endpoint(args.candidate_base_url, args.candidate_key_env)
    else:
        client = _client(args)
    try:
        res, _ = run(client, golden, out, candidate=args.candidate, label=label, effort=effort,
                     temperature=temperature, max_tokens=args.max_tokens, shots=shots, n_shots=n_shots,
                     extra_body=extra, workers=args.workers, progress=sys.stderr)
    except CompareError as e:
        return _err(str(e))
    rows = load_rows(out).get(label) or []
    if rows:
        print(render_summary(summarize(rows)))
    print(f"Compared {res.recorded}, already had {res.skipped}, failed {res.failed}. Results: {out}")
    return _finish(res)


def cmd_retrain(args: argparse.Namespace) -> int:
    from . import retrain as rt

    data = Path(args.data)
    report = rt.Report()
    try:
        rows = rt.convert_file(data, report)
        val_rows = rt.convert_file(Path(args.validation), rt.Report()) if args.validation else None
    except OSError as e:
        return _err(str(e))
    if val_rows is not None:
        train_rows = rows
    else:
        train_rows, val_rows = rt.split_holdout(rows, args.holdout)
    print(rt.render_report(report, len(train_rows), len(val_rows)))
    if not train_rows:
        return _err("nothing to train on")

    epochs, job_base = (None, None)
    inv_path = Path(args.inventory) if args.inventory else Path(DEFAULT_OUT) / "inventory.json"
    if inv_path.exists():
        epochs, job_base = rt.epochs_from_inventory(inv_path, args.job)
    if args.epochs:
        epochs = args.epochs
    if epochs:
        print(f"Epochs: {epochs}" + (" (from the original OpenAI job)" if not args.epochs else ""))
    else:
        epochs = 3
        print("Epochs: 3 (default; pass --epochs, or --job with an inventory to reuse the original job's)")
    if job_base:
        print(f"Original job base: {job_base}")
    if args.check:
        print("Check only: nothing written.")
        return 0

    revision = args.revision
    if revision is None and args.base not in rt.KNOWN_REVISIONS and not Path(args.base).is_dir():
        print(f"Note: {args.base} has no pinned revision; the run records the one it used. Pin with --revision.")
    cfg = rt.build_config(args.base, revision, epochs, args.gpu, rt.sha256_file(data),
                          lr=args.lr, lora_r=args.lora_r, max_len=args.max_len, timeout_hours=args.timeout_hours,
                          batch_size=args.batch_size, grad_accum=args.grad_accum)
    cfg["gguf"] = None if args.gguf == "none" else args.gguf
    out = Path(args.out)
    rt.write_kit(out, train_rows, val_rows, cfg)
    print(f"\nKit written to {out}/ (data files are private: 0600).")
    print("Run it on Modal (needs `modal setup` once; uploads data/ to your Modal account):")
    print(f"  cd {out} && modal run modal_app.py --max-steps 5   # smoke test first")
    print(f"  cd {out} && modal run modal_app.py")
    print(f"The job stops after {cfg['timeout_hours']} h, which caps its cost. Or run train_lora.py on any CUDA machine "
          "with requirements-train.txt installed.")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .compare import CompareError
    from .report import build

    d = Path(args.dir)
    golden = [Path(p) for p in args.golden] if args.golden else sorted(d.glob("golden-*.jsonl"))
    compare = [Path(p) for p in args.compare] if args.compare else sorted(d.glob("compare-*.jsonl"))
    inventory = Path(args.inventory) if args.inventory else d / "inventory.json"
    trained = [Path(p) for p in args.retrain_summary or []]
    if not golden and not compare and not inventory.exists():
        return _err(f"nothing to report in {d}/: run inventory, record or compare first")
    try:
        page = build(golden, compare, inventory, trained)
    except (OSError, CompareError) as e:
        return _err(str(e))
    out = Path(args.out) if args.out else d / "report.html"
    write_private(out, page.encode("utf-8"))
    print(f"Report: {out}")
    return 0


def _slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model).strip("_")[:100]


def build_parser() -> argparse.ArgumentParser:
    from .compare import EFFORTS
    from .retrain import DEFAULT_BASE, GPUS

    p = argparse.ArgumentParser(
        prog="tunekeep",
        description="Keep a fine-tuned model working after its provider retires the base model.",
    )
    p.add_argument("--version", action="version", version=f"tunekeep {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--base-url", help="API base URL (default: OPENAI_BASE_URL or https://api.openai.com/v1)")
    common.add_argument("--azure", action="store_true", help="Azure OpenAI v1 API (uses AZURE_OPENAI_API_KEY); experimental")
    sub = p.add_subparsers(dest="command", required=True)

    inv = sub.add_parser("inventory", parents=[common], help="List fine-tuned models, jobs, files and shutdown dates")
    inv.add_argument("--out", default=DEFAULT_OUT, help=f"Output directory (default: {DEFAULT_OUT})")
    inv.add_argument("--no-download", action="store_true", help="Do not try to download job files")
    inv.set_defaults(func=cmd_inventory)

    rec = sub.add_parser("record", parents=[common], help="Record a fine-tuned model's answers as a golden eval set")
    rec.add_argument("--model", required=True, help="Fine-tuned model id, e.g. ft:gpt-4o-mini-2024-07-18:org::abc123")
    rec.add_argument("--prompts", required=True, help="JSONL of prompts (OpenAI training format works as-is)")
    rec.add_argument("--out", help=f"Output JSONL (default: {DEFAULT_OUT}/golden-<model>.jsonl)")
    rec.add_argument("--workers", type=int, default=4, help="Parallel requests (default: 4)")
    rec.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default 0; -1 to omit)")
    rec.add_argument("--max-tokens", type=int, help="max_completion_tokens per answer")
    rec.add_argument("--seed", type=int, help="Seed for more repeatable answers")
    rec.add_argument("--limit", type=int, help="Only the first N prompts")
    rec.add_argument("--dry-run", action="store_true", help="Count prompts and tokens; send nothing")
    rec.set_defaults(func=cmd_record)

    cmp = sub.add_parser("compare", parents=[common], help="Score another model against a recorded golden set")
    cmp.add_argument("--golden", required=True, help="Golden set from `tunekeep record`")
    cmp.add_argument("--candidate", required=True, help="Model to test, e.g. gpt-6-luna, or your served retrained model")
    cmp.add_argument("--candidate-base-url", help="OpenAI-compatible endpoint for the candidate, e.g. http://127.0.0.1:8080/v1")
    cmp.add_argument("--candidate-key-env", help="Env var holding the candidate endpoint's key (none needed for localhost)")
    cmp.add_argument("--examples", help="Your training JSONL, for few-shot examples (your own data only)")
    cmp.add_argument("--shots", type=int, help="Examples per prompt (default 5 with --examples)")
    cmp.add_argument("--reasoning-effort", choices=EFFORTS, default="auto",
                     help="auto: 'none' for GPT-6 Luna/Sol and GPT-5.x so tools and temperature work; "
                          "'default' sends nothing")
    cmp.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default 0; -1 to omit)")
    cmp.add_argument("--max-tokens", type=int, help="max_completion_tokens per answer")
    cmp.add_argument("--extra-body", help='JSON merged into each request, e.g. \'{"chat_template_kwargs":{"enable_thinking":false}}\'')
    cmp.add_argument("--label", help="Name for this run in results and the report")
    cmp.add_argument("--workers", type=int, default=4, help="Parallel requests (default: 4)")
    cmp.add_argument("--limit", type=int, help="Only the first N prompts")
    cmp.add_argument("--out", help="Output JSONL (default: next to the golden set)")
    cmp.add_argument("--dry-run", action="store_true", help="Show what would be sent; send nothing")
    cmp.set_defaults(func=cmd_compare)

    rt = sub.add_parser("retrain", help="Write a LoRA retraining kit for an open model from your training data")
    rt.add_argument("--data", required=True, help="Your original OpenAI training JSONL")
    rt.add_argument("--validation", help="Your validation JSONL (default: hold out a slice of --data)")
    rt.add_argument("--holdout", type=float, default=0.05, help="Validation share when no --validation (default 0.05)")
    rt.add_argument("--base", default=DEFAULT_BASE, help=f"Hugging Face model id or local path (default: {DEFAULT_BASE})")
    rt.add_argument("--revision", help="Hugging Face revision (commit) to pin")
    rt.add_argument("--epochs", type=int, help="Epochs (default: the original job's, else 3)")
    rt.add_argument("--inventory", help=f"inventory.json, to reuse the original job's epochs (default: {DEFAULT_OUT}/inventory.json)")
    rt.add_argument("--job", help="Fine-tuning job id in the inventory to copy settings from")
    rt.add_argument("--lr", type=float, help="Learning rate (default 2e-4)")
    rt.add_argument("--lora-r", type=int, help="LoRA rank (default 16)")
    rt.add_argument("--max-len", type=int, help="Max tokens per sample; longer ones are skipped (default 4096)")
    rt.add_argument("--batch-size", type=int, help="Samples per micro-batch (default 4)")
    rt.add_argument("--grad-accum", type=int, help="Micro-batches per optimizer step (default 4)")
    rt.add_argument("--gpu", choices=GPUS, default="L40S", help="Modal GPU (default L40S)")
    rt.add_argument("--timeout-hours", type=float, help="Hard stop for the Modal job (default 2)")
    rt.add_argument("--gguf", default="q8_0", choices=["q8_0", "bf16", "f16", "none"], help="GGUF export (default q8_0)")
    rt.add_argument("--out", default=f"{DEFAULT_OUT}/retrain", help=f"Kit directory (default: {DEFAULT_OUT}/retrain)")
    rt.add_argument("--check", action="store_true", help="Only check and summarise the data")
    rt.set_defaults(func=cmd_retrain)

    rp = sub.add_parser("report", help="Write one self-contained HTML report")
    rp.add_argument("--dir", default=DEFAULT_OUT, help=f"Where to look for results (default: {DEFAULT_OUT})")
    rp.add_argument("--golden", nargs="*", help="Golden sets (default: <dir>/golden-*.jsonl)")
    rp.add_argument("--compare", nargs="*", help="Compare results (default: <dir>/compare-*.jsonl)")
    rp.add_argument("--inventory", help="inventory.json (default: <dir>/inventory.json)")
    rp.add_argument("--retrain-summary", nargs="*", help="summary.json files from retrain runs")
    rp.add_argument("--out", help="Output HTML (default: <dir>/report.html)")
    rp.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except APIError as e:
        print(f"tunekeep: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ntunekeep: interrupted. Re-run to continue where you left off.", file=sys.stderr)
        return 130

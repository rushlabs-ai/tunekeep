"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .client import APIError, Client
from .files import ensure_private_dir


def _client(args: argparse.Namespace) -> Client:
    return Client.from_env(base_url=args.base_url, azure=args.azure)


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
        print(f"tunekeep: {e}", file=sys.stderr)
        return 2
    if args.limit:
        prompts = prompts[: args.limit]
    out = Path(args.out) if args.out else Path("tunekeep-out") / f"golden-{_slug(args.model)}.jsonl"
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
    )
    print(f"Recorded {res.recorded}, already had {res.skipped}, failed {res.failed}. Golden set: {out}")
    if res.first_error:
        print(f"First error: {res.first_error}", file=sys.stderr)
        print("Run the same command again to retry only the missing prompts.", file=sys.stderr)
    return 1 if res.failed else 0


def _slug(model: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "_" for c in model)[:100]


def build_parser() -> argparse.ArgumentParser:
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
    inv.add_argument("--out", default="tunekeep-out", help="Output directory (default: tunekeep-out)")
    inv.add_argument("--no-download", action="store_true", help="Do not try to download job files")
    inv.set_defaults(func=cmd_inventory)

    rec = sub.add_parser("record", parents=[common], help="Record a fine-tuned model's answers as a golden eval set")
    rec.add_argument("--model", required=True, help="Fine-tuned model id, e.g. ft:gpt-4o-mini-2024-07-18:org::abc123")
    rec.add_argument("--prompts", required=True, help="JSONL of prompts (OpenAI training format works as-is)")
    rec.add_argument("--out", help="Output JSONL (default: tunekeep-out/golden-<model>.jsonl)")
    rec.add_argument("--workers", type=int, default=4, help="Parallel requests (default: 4)")
    rec.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (default 0; -1 to omit)")
    rec.add_argument("--max-tokens", type=int, help="max_completion_tokens per answer")
    rec.add_argument("--seed", type=int, help="Seed for more repeatable answers")
    rec.add_argument("--limit", type=int, help="Only the first N prompts")
    rec.add_argument("--dry-run", action="store_true", help="Count prompts and tokens; send nothing")
    rec.set_defaults(func=cmd_record)
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

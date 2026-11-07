#!/usr/bin/env sh
# Regenerate the hash-locked requirements shipped in src/tunekeep/kit/.
# Policy: exact pins, nothing newer than CUTOFF (at least 7 days before the lock date).
set -eu
CUTOFF="${CUTOFF:-2026-10-31T00:00:00Z}"
cd "$(dirname "$0")/.."
KIT=src/tunekeep/kit
COMMON="--generate-hashes --python-version 3.12 --python-platform x86_64-manylinux_2_28 --exclude-newer $CUTOFF --no-header"

UV_NO_CONFIG=1 uv pip compile tools/locks/train.in $COMMON -o "$KIT/requirements-train.txt"

uv pip compile tools/locks/gguf/pyproject.toml $COMMON -o "$KIT/requirements-gguf.txt.tmp"
{
  echo "# torch comes from the PyTorch CPU index; every file is pinned by hash, so no index can substitute it."
  echo "--extra-index-url https://download.pytorch.org/whl/cpu"
  cat "$KIT/requirements-gguf.txt.tmp"
} > "$KIT/requirements-gguf.txt"
rm "$KIT/requirements-gguf.txt.tmp"
echo "locked with cutoff $CUTOFF"

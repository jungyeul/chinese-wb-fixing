#!/usr/bin/env bash
set -euo pipefail

if rg -n '/Users/|/home/|C:\\|fixing_wb|mds-2024-25' \
  --glob '!**/.git/**' \
  --glob '!**/.ipynb_checkpoints/**' \
  --glob '!**/__pycache__/**' \
  --glob '!scripts/check_paths.sh' \
  --glob '!docs/MIGRATION_REPORT.md' \
  --glob '!docs/PAPER_ARTIFACT_CHECKLIST.md' \
  .; then
  echo "Found local absolute or legacy source paths." >&2
  exit 1
fi

echo "No local absolute or legacy source paths found."

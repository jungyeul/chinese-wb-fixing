#!/usr/bin/env bash
set -euo pipefail

cd viewer/error-analysis
python -m http.server 8000

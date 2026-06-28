#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python projection_code/make_IBM2_projection.py
python projection_code/make_LTP_projection.py
python projection_code/make_Levenshtein_projection.py
python projection_code/make_step1_projection.py
python projection_code/make_step2_projection.py
python projection_code/evaluate_generated.py
python projection_code/compare_generated.py

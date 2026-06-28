#!/usr/bin/env bash
set -euo pipefail

if [[ -f data/local_resources/triplet_no_dup_threshold.csv ]]; then
  python src/analysis/run_step2_feature_ablation.py
else
  echo "Local glyph table is not present; showing stored ablation summary instead."
  python - <<'PY'
import pandas as pd
path = "results/ablation/step2_feature_ablation_outputs/eva_ALL.csv"
df = pd.read_csv(path)
print(df.to_string(index=False))
PY
fi

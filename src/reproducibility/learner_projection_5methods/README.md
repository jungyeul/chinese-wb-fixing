# Learner Projection Five-Method Summary

This folder collects the learner-corpus projection artifacts for five methods:

- IBM2
- LTP
- Levenshtein
- step1
- step2

## Quick Files

- `combined_eval_summary.csv`: one-row-per-method metric summary.
- `manifest.csv`: code, projection CSV, eva CSV, and provenance for each method.
- `csv/`: normalized projection result CSV files.
- `eva/`: one eva CSV per method.
- `code/`: code snapshots used for projection or evaluation.
- `projection_code/`: runnable scripts that generate method-specific projection CSVs.
- `inputs/`: local input files needed by the runnable scripts.
- `generated_csv/`: output location for regenerated projection CSVs.
- `generated_eva/`: output location for regenerated eva CSVs.
- `source_combined/`: original combined source CSV/eva files kept for provenance.
- `requirements.txt`: Python packages needed for the heavy projection scripts.

## Method Mapping

| Method | Projection CSV | Eva CSV | Code |
|---|---|---|---|
| IBM2 | `csv/IBM2_projection.csv` | `eva/eva_IBM2.csv` | `projection_code/make_IBM2_projection.py` |
| LTP | `csv/LTP_projection.csv` | `eva/eva_LTP.csv` | `projection_code/make_LTP_projection.py` |
| Levenshtein | `csv/Levenshtein_projection.csv` | `eva/eva_Levenshtein.csv` | `projection_code/make_Levenshtein_projection.py` |
| step1 | `csv/step1_projection.csv` | `eva/eva_step1.csv` | `projection_code/make_step1_projection.py` |
| step2 | `csv/step2_projection.csv` | `eva/eva_step2.csv` | `projection_code/make_step2_projection.py` |

## Re-run

From this folder:

```bash
conda activate fixwb
python projection_code/make_IBM2_projection.py
python projection_code/make_LTP_projection.py
python projection_code/make_Levenshtein_projection.py
python projection_code/make_step1_projection.py
python projection_code/make_step2_projection.py
python projection_code/evaluate_generated.py
python projection_code/compare_generated.py
```

Or run the whole sequence:

```bash
./run_all.sh
```

The three baseline scripts are lightweight and only need pandas/stdlib. The
step1/step2 scripts need `ltp`; step2 also needs the Transformer stack listed
in `requirements.txt`. The local `fixwb` conda environment has the dependencies
needed for the full run. Step2 may create a local shape-table cache file named
`inputs/triplet_no_dup_threshold.csv.pkl` when it runs; that cache is generated
from the CSV and is not required to be checked in.

## Notes

- IBM2 now comes from the real learner IBM2 output:
  `inputs/learners_three_outputs_add_ibm_only.csv`.
  That output corresponds to the pre-trained table model:
  `0211验证ibm train learners/model_learner_ibm2_tables_target_only.pkl`.
- LTP/Levenshtein came from the earlier combined learner output:
  `source_combined/three_outputs_learners_charpairs_textcols.csv`.
- The older baseline methods were split into one normalized CSV per method with:
  `Index`, `Target_ID`, `Gold_Source`, `Target_Text`, `Projected_Source`.
- step1 and step2 came from the current summary package:
  `transformer/data/step1_step2_summary_20260507`.
- All five projection CSVs have 1079 learner rows.
- Local runnable inputs are in `inputs/`:
  - `inputs/learners_three_outputs_add_ibm_only.csv` for IBM2.
  - `inputs/three_outputs_learners_charpairs_textcols.csv` for LTP/Levenshtein and step1.
  - `inputs/learner_corpus.csv` and `inputs/triplet_no_dup_threshold.csv` for step2.

## Verification

- Recomputed Token Precision/Recall/F1 from each local projection CSV and confirmed they exactly match the corresponding eva CSV.
- Confirmed `step1_projection.csv`, `step2_projection.csv`, `eva_step1.csv`, and `eva_step2.csv` are byte-identical copies from `step1_step2_summary_20260507`.
- Confirmed `IBM2_projection.csv` is an exact column split from `inputs/learners_three_outputs_add_ibm_only.csv`.
- Confirmed `LTP_projection.csv` and `Levenshtein_projection.csv` are exact column splits from `source_combined/three_outputs_learners_charpairs_textcols.csv`.
- Confirmed the updated IBM2 output differs from Levenshtein in 193 learner rows.
- Re-ran all five projection scripts in this folder under `conda activate fixwb`.
- Confirmed all generated projection CSVs and generated eva CSVs are byte-identical to the stored targets via `projection_code/compare_generated.py`.
- Direct code correspondence:
  - step1 and step2 code files are the direct learner projection scripts.
  - IBM2 uses the real learner IBM2 output associated with `model_learner_ibm2_tables_target_only.pkl`.
  - LTP/Levenshtein code files are provenance/evaluation snapshots for the older baseline pipeline; their projection CSVs come from the original combined learner output table.

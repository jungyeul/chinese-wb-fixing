# Reproducibility Migration Report

## Summary

This cleanup added paper-first reproducibility materials from the local working
folders into `chinese-wb-fixing` using a `src/`-first layout. Existing colleague
files in the target repository were left in place. No commits were created.

## Source Materials Used

- `fixing_wb/transformer/data/learner_projection_5methods_summary_20260514/`
- `fixing_wb/transformer/data/0423/`
- `fixing_wb/transformer/data/0430/`
- `fixing_wb/transformer/data/0415/`
- `fixing_wb/data/anotation_for_levenshtein/`
- `fixing_wb/data/annotation_for_cos_126/`
- `fixing_wb/data/lev_vs_cos/`
- `fixing_wb/data/segmentation_ltp_and_stanza/`
- `fixing_wb/source_ltp_segment_with_id_updated.csv`
- `error-analysis-viewer/`

## Repository Layout

- `src/pipeline/`: runnable projection, segmentation, dataset construction, and
  evaluation code.
- `src/reproducibility/`: preserved learner projection bundle used for
  paper-facing checks.
- `data/learner_benchmark/`: learner benchmark CSVs.
- `data/learner_benchmark/construction/`: target-selection and segmentation
  proposal files used during learner benchmark construction.
- `data/learner_annotation/`: annotation and adjudication materials.
- `data/local_resources/`: local runtime inputs used by rerun scripts.
- `results/`: learner, synthetic, ablation, figure, and analysis outputs.
- `viewer/error-analysis/`: static error-analysis viewer.
- `scripts/`: root-runnable convenience commands.

## Migration Map

| Source | Target |
|---|---|
| `transformer/data/learner_projection_5methods_summary_20260514/` | `src/reproducibility/learner_projection_5methods/` |
| `transformer/data/learner_projection_5methods_summary_20260514/inputs/learner_corpus.csv` | `data/learner_benchmark/learner_corpus.csv` |
| `data/FINAL_seg_merged_with_labeled.csv` | `data/learner_benchmark/FINAL_seg_merged_with_labeled.csv` |
| `data/levcos+human.csv` | `data/learner_benchmark/levcos_human_selected_pairs.csv` |
| `data/cosine_vs_levenshtein.csv` | `data/learner_benchmark/cosine_vs_levenshtein_reference_selection.csv` |
| `data/levenshtein_distances.csv` | `data/learner_benchmark/construction/reference_selection/levenshtein_distances.csv` |
| `data/levenshtein_target.csv` | `data/learner_benchmark/construction/reference_selection/levenshtein_target.csv` |
| `data/cosine_similarities.csv` | `data/learner_benchmark/construction/reference_selection/cosine_similarities.csv` |
| `data/cosine_target.csv` | `data/learner_benchmark/construction/reference_selection/cosine_target.csv` |
| `data/cosine_vs_levenshtein.csv` | `data/learner_benchmark/construction/reference_selection/cosine_vs_levenshtein.csv` |
| `data/final_data_without_annotation.csv` | `data/learner_benchmark/construction/segmentation_candidates/final_data_without_annotation.csv` |
| `data/final_data_with_annotation.csv` | `data/learner_benchmark/construction/segmentation_candidates/final_data_with_annotation.csv` |
| `data/levcos+projection.csv` | `data/learner_benchmark/construction/segmentation_candidates/levcos+projection.csv` |
| `data/levcos+ltp.csv` | `data/learner_benchmark/construction/segmentation_candidates/levcos+ltp.csv` |
| `data/segmentation_ltp_and_stanza/` | `data/learner_benchmark/construction/segmentation_candidates/segmentation_ltp_and_stanza/` |
| `data/anotation_for_levenshtein/` | `data/learner_annotation/anotation_for_levenshtein/` |
| `data/annotation_for_cos_126/` | `data/learner_annotation/annotation_for_cos_126/` |
| `data/lev_vs_cos/` | `data/learner_annotation/reference_choice_lev_vs_cos/` |
| `source_ltp_segment_with_id_updated.csv` | `data/learner_annotation/reference_choice_lev_vs_cos/source_ltp_segment_with_id_updated.csv` |
| `transformer/data/0423/syn_step1_outputs/` | `results/synthetic/syn_step1_outputs/` |
| `transformer/data/0423/syn_step2_outputs/` | `results/synthetic/syn_step2_outputs/` |
| `transformer/data/0423/synthetic_*` | `results/synthetic/` and `results/figures/` |
| `transformer/data/0430/step2_feature_ablation_outputs/` | `results/ablation/step2_feature_ablation_outputs/` |
| `transformer/data/0415/step2_weight_optimization_spanlen_cv/` | `results/ablation/step2_weight_optimization_spanlen_cv/` |
| `transformer/data/step1_step2_summary_20260507/eva_learner_ibm2_final/` | `results/analysis/ibm_diagnostic/eva_learner_ibm2_final/` |
| `error-analysis-viewer/` | `viewer/error-analysis/` |

## Commands

```bash
python -m pip install -r requirements.txt
scripts/check_paths.sh
scripts/run_learner_eval.sh
scripts/run_synthetic_eval.sh
scripts/run_ablation.sh
scripts/summarize_annotation_workflow.sh
scripts/serve_error_viewer.sh
```

## Verification Notes

- `scripts/check_paths.sh` checks for local absolute paths and legacy source
  paths in executable/documentation areas.
- `scripts/run_learner_eval.sh` recomputes the learner D/P1/P2 token F1 summary
  from stored projection CSVs.
- `scripts/run_synthetic_eval.sh` writes a compact F1 table from the synthetic
  result summary.
- `scripts/run_ablation.sh` uses local P2 resources when present and otherwise
  prints the stored ablation summary.
- `scripts/serve_error_viewer.sh` starts a local static server for the
  browser-based error viewer.

## Local Resources

For reruns from raw inputs, see `docs/DATA_PREPARATION.md`. That document
records the expected CTB 5.1 input format, glyph table schema, downloaded model
requirements, and placement conventions for local runtime resources.

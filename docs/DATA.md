# Data Documentation

## Learner Benchmark

Path: `data/learner_benchmark/learner_corpus.csv`

- Rows: 1,079.
- Columns: `Index`, `Target_ID`, `Target_Text`, `Source_Text`.
- Use: final paper-facing learner benchmark for D/P1/P2 evaluation.
- Source token count: 32,536.
- Target token count: 33,140.
- Format: whitespace-separated Chinese tokens.

Path: `data/learner_benchmark/FINAL_seg_merged_with_labeled.csv`

- Rows: 1,079.
- Columns: `Index`, `Target_ID`, `Target_Text`, `Source_Text`, `Source_Type`.
- Use: selected source-target pairs with source provenance labels.

Path: `data/learner_benchmark/cosine_vs_levenshtein_reference_selection.csv`

- Use: automatic reference-selection evidence.
- Paper values represented: 765 automatically agreed cases and 314 cases that
  required manual adjudication.

Path: `data/learner_benchmark/construction/`

- `reference_selection/`: Levenshtein and cosine target-selection scores and
  selected targets.
- `segmentation_candidates/`: LTP/Stanza/projection working files used before
  source-side manual review.

## Learner Annotation

Path: `data/learner_annotation/`

- `anotation_for_levenshtein/`: source-side segmentation review and final
  Levenshtein-selected rows.
- `annotation_for_cos_126/`: annotator files for cosine-selected/manual cases.
- `reference_choice_lev_vs_cos/`: human Levenshtein-vs-cosine target-choice
  files, merged 953/126 decision table, and the ID-based cosine source
  replacement artifact `source_ltp_segment_with_id_updated.csv`.
- Includes CSV, XLSX, and HTML review tools copied from the source materials.
- Annotator labels are anonymized as `annotator_a`, `annotator_b`, and
  `annotator_c`; see `docs/ANNOTATION_ANONYMIZATION.md`.

Workflow documentation: `docs/ANNOTATION_WORKFLOW.md`

## Synthetic Benchmark

Path: `pipeline/synthetic_data/syn_dat_outputs/`

- Files: 10 CSV files.
- Noise levels: 1% through 10%.
- Rows per file: 1,079.
- Columns: `Index`, `CTB_ID`, `Gold_Source`, `Target_Text`.
- Use: CTB-derived synthetic benchmark generated from CTB 5.1.

Generation script:

```bash
python src/pipeline/dataset_construction/synthetic_ctb_generation/generate_synthetic_ctb.py \
  --ctb-seg-dir /path/to/ctb5.1_507K/segmented
```

## System Outputs And Evaluation Results

Path: `src/reproducibility/learner_projection_5methods/`

- `csv/`: stored learner output CSVs for D, P1, P2, and diagnostic methods.
- `eva/`: stored evaluation summaries.
- `statistical_significance/`: paired significance test script and result notes.

Path: `results/synthetic/`

- Synthetic P1/P2 output CSVs and evaluation summaries.
- `synthetic_all_methods_report.csv` is the source table for the synthetic
  figure data.

Path: `results/ablation/`

- P2 feature-ablation outputs and 5-fold cross-validation search artifacts.

Path: `viewer/error-analysis/`

- Static browser viewer.
- `data.js` stores the 1,079-sentence viewer data.
- `data/all_errors.csv` stores local error rows used for Table 3 counts.

## Checksums

Representative SHA-256 checksums:

```text
6c1df24a16c81e688468e8a3179ac1bc3abffe89de2ae1018b5ba066e016251e  data/learner_benchmark/learner_corpus.csv
754fb5d36d8470e751f0244ec209625c79d7ec78dea420184bcbcbd3c1a71b4d  data/learner_benchmark/FINAL_seg_merged_with_labeled.csv
ab0700f345170a4cd7bb8b29e90b14f743b8f0f80989b490099fb5f141e242f9  data/learner_annotation/reference_choice_lev_vs_cos/source_ltp_segment_with_id_updated.csv
1ff8c4ced4a26f3ba79afb10fab3fe3a68c11cfb7d563573a6708d58d5f869ff  pipeline/synthetic_data/syn_dat_outputs/ctb5_synthetic_charnoise_1pct_ids400_550_1079.csv
64552a685cfadf455a329ac27d05bb33c315018675f7295a10a808c8b553dfc7  pipeline/synthetic_data/syn_dat_outputs/ctb5_synthetic_charnoise_10pct_ids400_550_1079.csv
daddd16b6da2dc8070f4951a6ce7d57547aaf44e35421f35299fbb6698c3fb65  results/ablation/step2_feature_ablation_outputs/eva_ALL.csv
```

# File Manifest

## Code

- `src/pipeline/alignment/step1_identical_alignment_projection.py`: P1 exact
  identical-character projection.
- `src/pipeline/alignment/step2_similarity_alignment_projection.py`: P2
  similarity-based residual alignment and projection.
- `src/pipeline/dataset_construction/mucgec_reference_selection/`: MuCGEC
  reference-selection scripts.
- `src/pipeline/dataset_construction/synthetic_ctb_generation/generate_synthetic_ctb.py`:
  CTB-derived synthetic benchmark generator.
- `src/pipeline/evaluation/jp_algorithm.py`: local token-span exact-match
  evaluation implementation.
- `src/pipeline/evaluation/evaluate_learner_methods.py`: root-runnable learner
  evaluation wrapper.
- `src/pipeline/evaluation/summarize_synthetic_results.py`: root-runnable
  synthetic result table wrapper.
- `src/reproducibility/learner_projection_5methods/`: preserved paper-facing
  learner projection bundle.
- `src/analysis/`: analysis, ablation, plotting, and viewer-generation helpers.

## Data

- `data/learner_benchmark/`: final learner source-target pairs and reference
  selection evidence.
- `data/learner_benchmark/construction/`: target-selection and segmentation
  proposal files used before final learner benchmark construction.
- `data/learner_annotation/`: annotator outputs and review artifacts.
- `data/learner_annotation/reference_choice_lev_vs_cos/source_ltp_segment_with_id_updated.csv`:
  1,079-row ID-keyed source segmentation table after applying the 126-row
  cosine source replacement rule.
- `docs/ANNOTATION_ANONYMIZATION.md`: anonymized annotator label conventions.
- `pipeline/synthetic_data/`: pre-existing synthetic generation notebooks and
  synthetic CSV outputs.
- `data/local_resources/`: local runtime resources used by rerun scripts, such
  as the glyph table.

## Results

- `results/learner_benchmark/`: generated learner evaluation summaries.
- `results/synthetic/`: synthetic system outputs and result summaries.
- `results/figures/`: paper-facing synthetic figures.
- `results/ablation/`: feature ablation and cross-validation outputs.
- `results/analysis/`: alignment coverage, error analysis, and IBM diagnostic
  artifacts.

## Viewer

- `viewer/error-analysis/`: approved import of the static paper error-analysis
  viewer, including `index.html`, `app.js`, `styles.css`, `data.js`, and
  `data/all_errors.csv`.

## Scripts

- `scripts/check_paths.sh`: check for local absolute or legacy source paths.
- `scripts/run_learner_eval.sh`: recompute learner D/P1/P2 metrics from stored
  projection CSVs.
- `scripts/run_synthetic_eval.sh`: write synthetic paper F1 table from stored
  summaries.
- `scripts/run_ablation.sh`: run ablation if the local glyph table is present;
  otherwise print stored ablation results.
- `scripts/summarize_annotation_workflow.sh`: verify learner annotation and
  reference-choice workflow counts.
- `scripts/serve_error_viewer.sh`: serve the static viewer locally.

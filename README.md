# Chinese Word Boundary Recovery

Reproducibility materials for the paper **Chinese Word Boundary Recovery through
Character Alignment Projection**.

This repository organizes the learner benchmark, synthetic benchmark, projection
pipeline, evaluation scripts, stored experiment outputs, ablation artifacts, and
error-analysis viewer used for the paper.

## Repository Layout

```text
data/
  learner_benchmark/        Final learner benchmark and construction files
  learner_annotation/       Annotation, adjudication, and Lev-vs-Cos records
  local_resources/          Local runtime resources for full reruns
docs/
  ANNOTATION_WORKFLOW.md    Learner annotation and target-choice data flow
  DATA.md                   Dataset schemas, counts, and checksums
  DATA_PREPARATION.md       Local resource placement and raw-data notes
  PAPER_ARTIFACT_CHECKLIST.md
src/
  pipeline/                 Alignment, projection, segmentation, evaluation
  analysis/                 Plotting, ablation, and audit helpers
  reproducibility/          Preserved paper-facing learner projection bundle
results/
  learner_benchmark/        Generated learner evaluation summaries
  synthetic/                Synthetic experiment outputs and summaries
  ablation/                 P2 grid-search and ablation outputs
  analysis/                 IBM diagnostic, annotation, and error artifacts
viewer/
  error-analysis/           Static browser-based error-analysis viewer
scripts/
  *.sh                      Root-runnable convenience commands
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Conda users can start from:

```bash
conda env create -f environment.yml
conda activate chinese-wb-fixing
```

## Quick Checks

Run from the repository root:

```bash
scripts/check_paths.sh
scripts/summarize_annotation_workflow.sh
scripts/run_learner_eval.sh
scripts/run_synthetic_eval.sh
scripts/run_ablation.sh
```

## Learner Benchmark

Final benchmark:

```text
data/learner_benchmark/learner_corpus.csv
data/learner_benchmark/FINAL_seg_merged_with_labeled.csv
```

Key counts:

- 1,079 learner source-target pairs
- Source tokens: 32,536
- Target tokens: 33,140
- Reference-selection split: 765 automatic cases and 314 human-choice cases
- Final source-side provenance labels: 953 `lev`, 126 `cos`

The annotation and reference-choice workflow is documented in:

```text
docs/ANNOTATION_WORKFLOW.md
```

The learner construction materials preserve the review trail for Levenshtein
source segmentation, the 126 cosine-selected source rows, and the ID-based
replacement artifact:

```text
data/learner_annotation/anotation_for_levenshtein/
data/learner_annotation/annotation_for_cos_126/
data/learner_annotation/reference_choice_lev_vs_cos/source_ltp_segment_with_id_updated.csv
```

To verify the construction counts:

```bash
scripts/summarize_annotation_workflow.sh
```

## Main Learner Evaluation

To recompute learner benchmark metrics from the stored D/P1/P2 outputs:

```bash
scripts/run_learner_eval.sh
```

Expected token-level F1:

| Method | Description | Token F1 |
|---|---|---:|
| D | Direct segmentation baseline | 0.9891 |
| P1 | Identical-character projection | 0.9899 |
| P2 | Similarity-based projection | 0.9916 |

Generated summary:

```text
results/learner_benchmark/learner_eval_summary.csv
```

## Projection Pipeline

Core code:

```text
src/pipeline/alignment/step1_identical_alignment_projection.py
src/pipeline/alignment/step2_similarity_alignment_projection.py
src/pipeline/evaluation/jp_algorithm.py
```

P1 uses exact identical-character alignment and deterministic target-to-source
boundary projection. P2 adds residual similarity alignment with glyph, pinyin,
position, and contextual embedding features.

The paper-facing P2 configuration is:

```text
embedding = 0.1
glyph     = 0.6
pinyin    = 0.0
position  = 0.3
tau       = 0.85
```

## Synthetic Benchmark

Synthetic CSVs are under:

```text
pipeline/synthetic_data/syn_dat_outputs/
```

The benchmark has 10 settings, one for each noise level from 1% to 10%, with
1,079 examples per setting.

To regenerate the compact synthetic F1 table from stored outputs:

```bash
scripts/run_synthetic_eval.sh
```

Generated summary:

```text
results/synthetic/synthetic_token_f1_table.csv
```

To regenerate synthetic data from CTB 5.1 segmented files, see:

```text
docs/DATA_PREPARATION.md
```

## Ablation

Stored P2 ablation outputs are under:

```text
results/ablation/step2_feature_ablation_outputs/
```

To print the stored ablation summary, or rerun when local P2 resources are
available:

```bash
scripts/run_ablation.sh
```

Expected token-level F1 values:

| Setting | Token F1 |
|---|---:|
| P2 | 0.9916 |
| without glyph similarity | 0.9899 |
| without phonological similarity | 0.9916 |
| without positional similarity | 0.9899 |
| without semantic similarity | 0.9902 |

## Error Analysis Viewer

Open the static viewer with:

```bash
scripts/serve_error_viewer.sh
```

Then visit the printed local URL. Viewer files are in:

```text
viewer/error-analysis/
```

The viewer data covers the learner benchmark and supports the paper's
over-segmentation, under-segmentation, and drift analysis.

## Local Runtime Resources

Some full reruns expect local resources supplied by the user, such as CTB 5.1
segmented files, downloaded LTP/HuggingFace models, and the glyph similarity
table used by P2. Placement and schemas are documented in:

```text
docs/DATA_PREPARATION.md
data/local_resources/README.md
```

## Documentation

Useful entry points:

- `docs/PAPER_ARTIFACT_CHECKLIST.md`: paper artifact index.
- `docs/DATA.md`: data schemas, counts, and representative checksums.
- `docs/ANNOTATION_WORKFLOW.md`: learner annotation and target-choice logic.
- `docs/ANNOTATION_ANONYMIZATION.md`: public anonymized annotator label
  conventions.
- `docs/MIGRATION_REPORT.md`: migration map from the working materials.
- `docs/FILE_MANIFEST.md`: important files and their purpose.

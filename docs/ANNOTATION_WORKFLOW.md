# Learner Benchmark Annotation Workflow

This document describes the learner benchmark construction files used before the
final 1,079-example benchmark. It focuses on file relationships that can be
checked from the repository.

## Overview

The learner benchmark construction has two related decisions:

1. Select a corrected target sentence for each learner source sentence.
2. Produce a gold source-side word segmentation for the learner sentence.

The final evaluation file is:

```text
data/learner_benchmark/FINAL_seg_merged_with_labeled.csv
```

It contains 1,079 rows with:

- `Target_Text`: tokenized target/reference text.
- `Source_Text`: manually checked learner source segmentation.
- `Source_Type`: provenance label for the source-side segmentation row.

`data/learner_benchmark/levcos_human_selected_pairs.csv` is the same final
benchmark without the `Source_Type` column.

## Stage 1: Target Candidate Selection

Path:

```text
data/learner_benchmark/construction/reference_selection/
```

Files:

- `levenshtein_distances.csv`: candidate references scored by edit distance.
- `levenshtein_target.csv`: selected Levenshtein target for 1,079 learner
  examples.
- `cosine_similarities.csv`: candidate references scored by cosine similarity.
- `cosine_target.csv`: selected cosine target for 1,079 learner examples.
- `cosine_vs_levenshtein.csv`: comparison of the two selected targets.

Checked counts:

- 765 rows have the same selected target under both methods.
- 314 rows have different Levenshtein and cosine selected targets.

The 765 same-target rows can pass the target-choice step automatically. The 314
different-target rows are the target-choice review set.

## Stage 2: Automatic Segmentation Proposals

Path:

```text
data/learner_benchmark/construction/segmentation_candidates/
```

Files:

- `segmentation_ltp_and_stanza/source_ltp_segment_with_id.csv`: LTP source-side
  segmentation.
- `segmentation_ltp_and_stanza/target_ltp_segment_with_id.csv`: LTP target-side
  segmentation.
- `segmentation_ltp_and_stanza/stanza_segmented_output.csv`: Stanza
  source-side segmentation comparison material.
- `final_data_without_annotation.csv`: 1,079-row table before adding annotator
  columns.
- `final_data_with_annotation.csv`: 1,079-row table with annotator columns.
- `levcos+ltp.csv`: LTP source segmentation paired with IDs.
- `levcos+projection.csv`: projection output with final source-type labels.

These files are working materials for constructing source-side segmentation
proposals before human review.

## Stage 3: Levenshtein-Based Source Segmentation Review

Path:

```text
data/learner_annotation/anotation_for_levenshtein/
```

The directory name preserves the original spelling from the working folder.

Main files:

- `FINAL_seg_lev.csv`: 1,079-row Levenshtein-side final segmented learner data.
- `segmentation_annotator_a_1-200_comparison_results.csv`
- `segmentation_annotator_a_201-493_787-1079_comparison_results.csv`
- `segmentation_annotator_b_1-200_comparison_results.csv`
- `segmentation_annotator_b_201-786_comparison_results.csv`
- `segmentation_annotator_c_1-200_comparison_results.csv`
- `segmentation_annotator_c_494-1079_comparison_results.csv`
- `results_1-200_annotators_comparison.xlsx`
- `results_201-1079_annotators_comparison.xlsx`

The file names show the review split:

- Rows 1-200 were reviewed by annotator A, annotator B, and annotator C.
- Rows 201-493 were reviewed by annotator A and annotator B.
- Rows 494-786 were reviewed by annotator B and annotator C.
- Rows 787-1079 were reviewed by annotator A and annotator C.

This gives each row at least two human reviews, while each annotator covers
about two thirds of the dataset.

The final review workbooks contain 40 rows for the 1-200 block and 220 rows for
the 201-1079 block.

## Stage 4: Levenshtein vs Cosine Target Choice

Path:

```text
data/learner_annotation/reference_choice_lev_vs_cos/
```

Files:

- `annotator_a_annotation.csv`
- `annotator_b_annotation.csv`
- `annotator_c_annotation.csv`
- `merged_with_final_target.csv`
- `final_source_target_cos.csv`
- `final_source_target_cos_with_ltp.csv`
- `final_source_target_cos_with_ltp_id_source.csv`
- `final_source_target_with_projection.csv`
- `source_ltp_segment_with_id_updated.csv`
- `sentence_selector.html`

The three anonymized annotator CSVs contain `selected_type` decisions. Rows where
Levenshtein and cosine targets are identical are marked `auto`. Rows where they
differ receive a human target-choice label.

Checked final target-choice counts from `merged_with_final_target.csv`:

- 953 rows select the Levenshtein target.
- 126 rows select the cosine target.
- 1,079 rows have `needs_review = False`.

`final_source_target_cos.csv` contains the 126 rows selected by the cosine
target-choice workflow before source-side segmentation review.

`final_source_target_cos_with_ltp_id_source.csv` contains the cosine-selected
source-side segmented text keyed by `id`. The corresponding construction
artifact `source_ltp_segment_with_id_updated.csv` is the 1,079-row source LTP
table after applying the ID-based replacement rule: when an `id` appears in the
126-row cosine source file, its `source_segmented` value is copied into the
`source_ltp_segmented` column. In the current local files,
`source_ltp_segment_with_id_updated.csv` and the construction copy
`segmentation_ltp_and_stanza/source_ltp_segment_with_id.csv` are identical, so
the construction copy appears to already contain this replacement.

## Stage 5: Cosine-Selected 126-Row Segmentation Review

Path:

```text
data/learner_annotation/annotation_for_cos_126/
```

Files:

- `FINAL_seg_cos.csv`: 126-row segmented file for the cosine-selected set.
- `annotator_a_annotation.csv`
- `annotator_b_annotation.csv`
- `annotator_c_annotation.csv`
- `annotation_combined.xlsx`
- `annotation_review_results.xlsx`

The review workbook contains 20 rows requiring comparison/adjudication in this
126-row set.

## Stage 6: Final Merge

Final files:

```text
data/learner_benchmark/FINAL_seg_merged_with_labeled.csv
data/learner_benchmark/levcos_human_selected_pairs.csv
data/learner_benchmark/learner_corpus.csv
```

Checked final counts:

- 1,079 total rows.
- `Source_Type = lev`: 953 rows.
- `Source_Type = cos`: 126 rows.

Relationship checks:

- `levcos_human_selected_pairs.csv` equals
  `FINAL_seg_merged_with_labeled.csv` after removing `Source_Type`.
- The 126 `Source_Type = cos` IDs match `annotation_for_cos_126/FINAL_seg_cos.csv`.
- The 126 IDs in `final_source_target_cos_with_ltp_id_source.csv` all appear in
  `source_ltp_segment_with_id_updated.csv`.
- For those 126 IDs, `final_source_target_cos_with_ltp_id_source.csv`
  `source_segmented` equals `source_ltp_segment_with_id_updated.csv`
  `source_ltp_segmented`.
- For these 126 rows, final `Source_Text` matches
  `annotation_for_cos_126/FINAL_seg_cos.csv`.
- For these 126 rows, final `Target_Text` matches
  `anotation_for_levenshtein/FINAL_seg_lev.csv`.

Interpretation: in the final merged file, `Source_Type` should be read as the
provenance of the source-side segmentation row. It should not be read as a
standalone guarantee that `Target_Text` was replaced by the cosine-selected
target in the final benchmark file.

## Verification

Run:

```bash
scripts/summarize_annotation_workflow.sh
```

The command writes:

```text
results/analysis/annotation_workflow/annotation_workflow_summary.json
results/analysis/annotation_workflow/annotation_workflow_key_counts.csv
```

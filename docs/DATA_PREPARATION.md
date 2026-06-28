# Data Preparation Notes

This file documents local resources used when rerunning generation scripts from
raw inputs. The processed learner benchmark, synthetic CSV files, system outputs,
evaluation summaries, and viewer artifacts used for the paper-facing checks are
organized elsewhere in the repository.

## CTB 5.1

- Paper use: clean target source for the synthetic benchmark.
- Local input: CTB 5.1 segmented files such as `chtb_0400.seg` through
  `chtb_0550.seg`.
- Format: CTB segmented text, GB18030-compatible encoding, one
  segmented sentence per content line with CTB markup ignored.
- Generation command:

```bash
python src/pipeline/dataset_construction/synthetic_ctb_generation/generate_synthetic_ctb.py \
  --ctb-seg-dir /path/to/ctb5.1_507K/segmented \
  --output-dir pipeline/synthetic_data/syn_dat_outputs
```

## Compound-Based Segmentation Guideline

- Paper use: source-side learner benchmark annotation.
- Working materials: final segmented learner benchmark and annotator output files
  are under `data/learner_benchmark/` and `data/learner_annotation/`.
- If an author-approved standalone guideline is added later, place it under
  `docs/` and link it from `docs/PAPER_ARTIFACT_CHECKLIST.md`.

## Annotation Agreement Calculations

- Paper values: boundary-level pairwise F1 = 0.987 and Krippendorff alpha =
  0.981.
- Working materials: `results/analysis/alignment_coverage/JP_IAA_Summary_after_annotator_c_punctuation_fix.csv`
  and annotator comparison files.
- If a full recomputation script is added later, put it under
  `src/analysis/annotation_agreement/` and write outputs to
  `results/analysis/alignment_coverage/`.

## Glyph Similarity Table

- Paper use: P2 glyph similarity feature.
- Local path expected by scripts:
  `data/local_resources/triplet_no_dup_threshold.csv`
- Columns:
  - `CharA`
  - `CharB`
  - `similarity`
- Values: similarity scores in `[0, 1]`; scripts load both directions as
  equivalent character-pair scores.

## Downloaded Models

The following are runtime dependencies, not repository data:

- `bert-base-chinese` from HuggingFace, used for contextual embedding similarity.
- LTP model, used for direct segmentation and source base segmentation.

Install Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

## MuCGEC Raw Development File

The repository uses processed selected learner benchmark files for evaluation.
Users who want to rerun reference selection from the MuCGEC development set can
place the raw development file under `data/local_resources/` and adapt the
scripts in `src/pipeline/dataset_construction/mucgec_reference_selection/`.

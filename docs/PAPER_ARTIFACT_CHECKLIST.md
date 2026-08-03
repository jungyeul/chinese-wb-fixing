# Paper Artifact Index

Source of truth: `2605.28128v1.pdf`, "Chinese Word Boundary Recovery through
Character Alignment Projection".

This index maps paper-facing artifacts to repository paths and verification
commands. It is intended as a reproducibility guide rather than a development
audit log.

| Paper item | Paper location | Artifact / command | Repository path |
|---|---|---|---|
| Learner Chinese benchmark | Section 3.1, Table 4, Table 6 | 1,079 source-target pairs, tokenized source and target text | `data/learner_benchmark/learner_corpus.csv` |
| Learner token counts | Table 6 | Source tokens 32,536; target tokens 33,140 | `scripts/run_learner_eval.sh`; `docs/DATA.md` |
| MuCGEC reference selection | Section 3.1, Table 5 | 765 automatic cases and 314 manual-adjudication cases | `data/learner_benchmark/cosine_vs_levenshtein_reference_selection.csv`; `data/learner_benchmark/construction/reference_selection/`; `src/pipeline/dataset_construction/mucgec_reference_selection/` |
| Learner annotation materials | Section 3.1, Table 5 | Annotator outputs, review tools, adjudication files, and ID-based 126-row cosine source replacement artifact | `data/learner_annotation/`; `docs/ANNOTATION_WORKFLOW.md` |
| Manual adjudication counts | Table 5 | 115 full-agreement cases, 199 two-of-three majority cases, 0 disputed cases | `data/learner_annotation/`; `data/learner_benchmark/FINAL_seg_merged_with_labeled.csv` |
| Source-side segmentation guideline materials | Section 3.1 | Final tokenized benchmark and author annotation materials | `data/learner_benchmark/`; `data/learner_annotation/`; `docs/DATA_PREPARATION.md` |
| Annotation agreement values | Section 3.1, Appendix D | Boundary pairwise F1 0.987; Krippendorff alpha 0.981 | `results/analysis/alignment_coverage/JP_IAA_Summary_after_annotator_c_punctuation_fix.csv`; `docs/DATA_PREPARATION.md` |
| CTB-derived synthetic benchmark | Section 3.2 | CTB 5.1 IDs 400-550, 1,079 examples per setting | `pipeline/synthetic_data/syn_dat_outputs/`; `src/pipeline/dataset_construction/synthetic_ctb_generation/generate_synthetic_ctb.py` |
| Synthetic noise settings | Section 3.2, Figure 1 | Noise levels 1% through 10%; substitution 0.483, deletion 0.318, insertion 0.199 | `pipeline/synthetic_data/syn_data_creation.ipynb`; `src/pipeline/dataset_construction/synthetic_ctb_generation/generate_synthetic_ctb.py` |
| Direct segmentation baseline D | Section 4.1, Table 1 | Stored output and evaluation wrapper | `src/reproducibility/learner_projection_5methods/csv/LTP_projection.csv`; `src/pipeline/segmentation/`; `scripts/run_learner_eval.sh` |
| P1 identical-character projection | Section 2.2, Table 1 | Exact-character alignment and deterministic projection | `src/pipeline/alignment/step1_identical_alignment_projection.py`; `src/reproducibility/learner_projection_5methods/csv/step1_projection.csv` |
| P2 similarity projection | Section 2.2, Appendix B, Table 1 | Residual alignment with glyph, pinyin, position, and embedding features | `src/pipeline/alignment/step2_similarity_alignment_projection.py`; `src/reproducibility/learner_projection_5methods/csv/step2_projection.csv` |
| P2 feature weights and threshold | Appendix B | embedding 0.1, glyph 0.6, pinyin 0.0, position 0.3, tau 0.85 | `src/analysis/run_step2_feature_ablation.py`; `docs/DATA_PREPARATION.md` |
| Grid search and 5-fold CV | Appendix B, Figure 2 | 0.1 weight increments; thresholds 0.30 to 0.90 by 0.05 | `results/ablation/step2_weight_optimization_spanlen_cv/`; `src/analysis/optimize_step2_weights_spanlen_cv.py` |
| Ablation table | Section 5.2, Table 2 | P2 and feature-removal token F1 values | `results/ablation/step2_feature_ablation_outputs/eva_ALL.csv`; `scripts/run_ablation.sh` |
| Learner main table | Section 4.2, Table 1 | D 0.9891, P1 0.9899, P2 0.9916 | `scripts/run_learner_eval.sh`; `results/learner_benchmark/learner_eval_summary.csv` |
| Paired significance tests | Section 4.2 | P2 vs P1 p=0.0006; P2 vs D p=0.0026; D vs P1 p=0.2690 | `src/pipeline/evaluation/paired_significance_test.py`; `src/reproducibility/learner_projection_5methods/statistical_significance/` |
| Synthetic results and figure | Section 4.3, Figure 1 | Token-level F1 across 1%-10% noise | `results/synthetic/synthetic_all_methods_report.csv`; `results/figures/`; `scripts/run_synthetic_eval.sh` |
| Error type table | Section 5.4, Table 3 | Over-segmentation, under-segmentation, and drift counts | `viewer/error-analysis/data/all_errors.csv`; `results/analysis/error_analysis/` |
| Static error-analysis viewer | Appendix G | Browser-based viewer over all learner sentences | `viewer/error-analysis/`; `scripts/serve_error_viewer.sh` |
| IBM-style alignment diagnostic | Section 5.3, Appendix E | Stored IBM-style projection diagnostic and notebooks | `results/analysis/ibm_diagnostic/`; `src/pipeline/alignment/ibm_style_diagnostic/` |

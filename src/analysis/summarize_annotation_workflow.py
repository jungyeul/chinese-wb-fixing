#!/usr/bin/env python3
"""Summarize learner benchmark annotation and reference-choice files."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "results/analysis/annotation_workflow"


def compact(text: object) -> str:
    return "".join(str(text).split())


def csv_rows(path: Path) -> int:
    return len(pd.read_csv(path))


def main() -> None:
    benchmark = REPO_ROOT / "data/learner_benchmark"
    construction = benchmark / "construction"
    annotation = REPO_ROOT / "data/learner_annotation"

    final = pd.read_csv(benchmark / "FINAL_seg_merged_with_labeled.csv")
    final_plain = pd.read_csv(benchmark / "levcos_human_selected_pairs.csv")
    lev_final = pd.read_csv(annotation / "anotation_for_levenshtein/FINAL_seg_lev.csv")
    cos_final = pd.read_csv(annotation / "annotation_for_cos_126/FINAL_seg_cos.csv")
    ref_compare = pd.read_csv(
        construction / "reference_selection/cosine_vs_levenshtein.csv"
    )
    ref_choice = pd.read_csv(
        annotation / "reference_choice_lev_vs_cos/merged_with_final_target.csv"
    )
    source_ltp = pd.read_csv(
        construction / "segmentation_candidates/segmentation_ltp_and_stanza/source_ltp_segment_with_id.csv"
    )
    source_ltp_updated = pd.read_csv(
        annotation / "reference_choice_lev_vs_cos/source_ltp_segment_with_id_updated.csv"
    )
    cos_source = pd.read_csv(
        annotation / "reference_choice_lev_vs_cos/final_source_target_cos_with_ltp_id_source.csv"
    )

    source_type_counts = final["Source_Type"].value_counts().to_dict()
    is_same_counts = ref_compare["is_same"].value_counts().to_dict()
    final_choice_counts = ref_choice["final_selected_type"].value_counts().to_dict()

    lev_join = final[final["Source_Type"].eq("cos")].merge(
        lev_final, on="Target_ID", suffixes=("_final", "_lev")
    )
    cos_join = final[final["Source_Type"].eq("cos")].merge(
        cos_final, on="Target_ID", suffixes=("_final", "_cos")
    )
    ref_join = final[final["Source_Type"].eq("cos")].merge(
        ref_choice, left_on="Target_ID", right_on="id"
    )
    source_ltp_compare = source_ltp_updated.merge(
        source_ltp, on="id", suffixes=("_updated", "_construction")
    )
    cos_source_join = cos_source.merge(source_ltp_updated, on="id")

    summary = {
        "row_counts": {
            "final_benchmark": len(final),
            "final_plain_benchmark": len(final_plain),
            "levenshtein_segmentation_final": len(lev_final),
            "cosine_126_segmentation_final": len(cos_final),
            "reference_choice_merged": len(ref_choice),
            "levenshtein_target": csv_rows(
                construction / "reference_selection/levenshtein_target.csv"
            ),
            "cosine_target": csv_rows(
                construction / "reference_selection/cosine_target.csv"
            ),
        },
        "reference_selection": {
            "cosine_and_levenshtein_same": int(is_same_counts.get(True, 0)),
            "cosine_and_levenshtein_different": int(is_same_counts.get(False, 0)),
        },
        "reference_choice": {
            "final_selected_type_counts": {
                str(k): int(v) for k, v in final_choice_counts.items()
            },
            "needs_review_counts": {
                str(k): int(v) for k, v in ref_choice["needs_review"].value_counts().items()
            },
            "annotator_selected_type_counts": {
                col: {str(k): int(v) for k, v in ref_choice[col].value_counts().items()}
                for col in [
                    "selected_type_annotator_c",
                    "selected_type_annotator_b",
                    "selected_type_annotator_a",
                ]
            },
        },
        "final_benchmark": {
            "source_type_counts": {str(k): int(v) for k, v in source_type_counts.items()},
            "plain_equals_labeled_without_source_type": bool(
                final_plain.equals(final[["Index", "Target_ID", "Target_Text", "Source_Text"]])
            ),
        },
        "cosine_source_type_relationships": {
            "cos_source_ids_match_final_seg_cos": bool(
                set(final.loc[final["Source_Type"].eq("cos"), "Target_ID"].astype(int))
                == set(cos_final["Target_ID"].astype(int))
            ),
            "final_cos_source_text_equals_final_seg_cos": bool(
                (cos_join["Source_Text_final"] == cos_join["Source_Text_cos"]).all()
            ),
            "final_cos_target_text_equals_final_seg_lev": bool(
                (lev_join["Target_Text_final"] == lev_join["Target_Text_lev"]).all()
            ),
            "final_cos_target_compact_equals_levenshtein_target": int(
                sum(
                    compact(a) == compact(b)
                    for a, b in zip(ref_join["Target_Text"], ref_join["levenshtein_target"])
                )
            ),
            "final_cos_target_compact_equals_cosine_target": int(
                sum(
                    compact(a) == compact(b)
                    for a, b in zip(ref_join["Target_Text"], ref_join["cosine_target"])
                )
            ),
            "final_cos_source_compact_equals_raw_source": int(
                sum(
                    compact(a) == compact(b)
                    for a, b in zip(ref_join["Source_Text"], ref_join["source"])
                )
            ),
        },
        "cosine_source_ltp_replacement_artifact": {
            "source_ltp_updated_rows": len(source_ltp_updated),
            "cosine_source_rows": len(cos_source),
            "cosine_source_ids_present_in_updated": int(
                source_ltp_updated["id"].isin(cos_source["id"]).sum()
            ),
            "cosine_source_segmented_equals_updated_source_ltp_segmented": int(
                (
                    cos_source_join["source_segmented"].astype(str)
                    == cos_source_join["source_ltp_segmented"].astype(str)
                ).sum()
            ),
            "updated_equals_construction_source_ltp_segment_with_id": bool(
                (
                    source_ltp_compare["source_ltp_segmented_updated"].astype(str)
                    == source_ltp_compare["source_ltp_segmented_construction"].astype(str)
                ).all()
            ),
        },
        "review_workbooks": {
            "levenshtein_review_1_200_rows": len(
                pd.read_excel(
                    annotation
                    / "anotation_for_levenshtein/results_1-200_annotators_comparison.xlsx"
                )
            ),
            "levenshtein_review_201_1079_rows": len(
                pd.read_excel(
                    annotation
                    / "anotation_for_levenshtein/results_201-1079_annotators_comparison.xlsx"
                )
            ),
            "cosine_126_review_rows": len(
                pd.read_excel(
                    annotation
                    / "annotation_for_cos_126/annotation_review_results.xlsx"
                )
            ),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "annotation_workflow_summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    counts_rows = [
        ("final_benchmark_rows", summary["row_counts"]["final_benchmark"]),
        ("reference_same_automatic", summary["reference_selection"]["cosine_and_levenshtein_same"]),
        (
            "reference_different_human_choice",
            summary["reference_selection"]["cosine_and_levenshtein_different"],
        ),
        ("final_source_type_lev", source_type_counts.get("lev", 0)),
        ("final_source_type_cos", source_type_counts.get("cos", 0)),
        ("levenshtein_review_rows", summary["review_workbooks"]["levenshtein_review_1_200_rows"] + summary["review_workbooks"]["levenshtein_review_201_1079_rows"]),
        ("cosine_126_review_rows", summary["review_workbooks"]["cosine_126_review_rows"]),
        (
            "cosine_source_ids_present_in_updated_ltp_file",
            summary["cosine_source_ltp_replacement_artifact"][
                "cosine_source_ids_present_in_updated"
            ],
        ),
    ]
    pd.DataFrame(counts_rows, columns=["metric", "value"]).to_csv(
        OUT_DIR / "annotation_workflow_key_counts.csv", index=False
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {(OUT_DIR / 'annotation_workflow_key_counts.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

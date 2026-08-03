#!/usr/bin/env python3
"""
Step 1 v2: conservative boundary projection plus exact-match merge constraint.

- Use LTP segmentation on source as the base segmentation
- Use exact-match alignment with insertion/deletion only
- Conservatively project target boundaries onto source
- If a target word forms a contiguous exact-match span in source, remove internal
  source boundaries inside that span

Usage:
    python step1_exact_boundary_projection_v2_merge.py \
        --input learner_corpus.csv \
        --output learner_step1_exact_boundary_projection_v2_merge.csv
"""

import argparse
import pandas as pd
from tqdm import tqdm
from ltp import LTP


ltp_model = None


def get_ltp_model():
    global ltp_model
    if ltp_model is None:
        print("Loading LTP model...")
        ltp_model = LTP()
    return ltp_model


def ltp_segment(text):
    """Segment text using LTP."""
    text_clean = text.replace(" ", "")
    if len(text_clean) == 0:
        return text
    model = get_ltp_model()
    result = model.pipeline([text_clean], tasks=["cws"])
    return " ".join(result.cws[0])


def exact_alignment_ins_del_only(source, target):
    """Align identical characters using insertion/deletion only."""
    m, n = len(source), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                )

    aligned = {}
    unaligned_src = []
    unaligned_tgt = []
    i, j = m, n

    while i > 0 or j > 0:
        if i > 0 and j > 0 and source[i - 1] == target[j - 1]:
            aligned[j - 1] = i - 1
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or dp[i - 1][j] <= dp[i][j - 1]):
            unaligned_src.append(i - 1)
            i -= 1
        else:
            unaligned_tgt.append(j - 1)
            j -= 1

    unaligned_src.reverse()
    unaligned_tgt.reverse()
    return aligned, unaligned_src, unaligned_tgt


def get_word_final_positions(segmented_text):
    """Get word-final character positions from segmented text."""
    word_final = set()
    pos = 0
    for token in segmented_text.split():
        pos += len(token)
        word_final.add(pos - 1)
    return word_final


def positions_to_segmented(text_clean, word_final_positions):
    """Convert word-final positions to segmented string."""
    result = []
    for i, char in enumerate(text_clean):
        result.append(char)
        if i in word_final_positions and i < len(text_clean) - 1:
            result.append(" ")
    return "".join(result)


def get_token_spans(segmented_text):
    """Return token spans as (token, start_idx, end_idx)."""
    spans = []
    pos = 0
    for token in segmented_text.split():
        start = pos
        end = pos + len(token) - 1
        spans.append((token, start, end))
        pos += len(token)
    return spans


def project_boundaries_conservatively(target_text, aligned, base_boundaries):
    """
    Project target boundaries only when both sides of the boundary are aligned
    to adjacent source positions.
    """
    final_boundaries = set(base_boundaries)
    projected_boundaries = set()
    skipped_boundaries = []

    target_chars = target_text.replace(" ", "")
    target_boundaries = get_word_final_positions(target_text)

    for tgt_left_idx in sorted(target_boundaries):
        tgt_right_idx = tgt_left_idx + 1

        if tgt_right_idx >= len(target_chars):
            continue

        src_left_idx = aligned.get(tgt_left_idx)
        src_right_idx = aligned.get(tgt_right_idx)

        if src_left_idx is None or src_right_idx is None:
            skipped_boundaries.append(f"{tgt_left_idx}:unaligned")
            continue

        if src_right_idx != src_left_idx + 1:
            skipped_boundaries.append(
                f"{tgt_left_idx}:non_adjacent({src_left_idx},{src_right_idx})"
            )
            continue

        final_boundaries.add(src_left_idx)
        projected_boundaries.add(src_left_idx)

    return final_boundaries, projected_boundaries, skipped_boundaries


def apply_exact_match_merge_constraint(target_text, aligned, word_final_positions):
    """
    If a target token maps to a contiguous exact-match span in source, remove
    internal source boundaries inside that span.

    The merge is only allowed when the target token length and the aligned
    source span length are the same. This prevents merging across extra source
    characters that happen to sit between the aligned endpoints.
    """
    merged_spans = []
    skipped_tokens = []

    for token, tgt_start, tgt_end in get_token_spans(target_text):
        if len(token) <= 1:
            continue

        src_positions = []
        missing = False
        for tgt_idx in range(tgt_start, tgt_end + 1):
            src_idx = aligned.get(tgt_idx)
            if src_idx is None:
                missing = True
                break
            src_positions.append(src_idx)

        if missing:
            skipped_tokens.append(f"{token}:unaligned")
            continue

        src_span_len = src_positions[-1] - src_positions[0] + 1
        if src_span_len != len(token):
            skipped_tokens.append(
                f"{token}:span_length_mismatch(target={len(token)},source={src_span_len})"
            )
            continue

        expected = list(range(src_positions[0], src_positions[0] + len(src_positions)))
        if src_positions != expected:
            skipped_tokens.append(
                f"{token}:non_contiguous({src_positions[0]}-{src_positions[-1]})"
            )
            continue

        src_start = src_positions[0]
        src_end = src_positions[-1]
        removed = []
        for src_idx in range(src_start, src_end):
            if src_idx in word_final_positions:
                word_final_positions.discard(src_idx)
                removed.append(src_idx)

        if removed:
            merged_spans.append(f"{token}:{src_start}-{src_end}")

    return word_final_positions, merged_spans, skipped_tokens


def run_step1(source_text, target_text):
    source_chars = source_text.replace(" ", "")
    target_chars = target_text.replace(" ", "")

    if len(source_chars) == 0:
        return {
            "projected": source_text,
            "aligned": {},
            "unaligned_src": [],
            "unaligned_tgt": [],
            "ltp_seg": source_text,
            "projected_boundaries": set(),
            "skipped_boundaries": [],
            "merged_spans": [],
            "merge_skips": [],
        }

    ltp_seg = ltp_segment(source_text)
    ltp_word_final = get_word_final_positions(ltp_seg)
    aligned, unaligned_src, unaligned_tgt = exact_alignment_ins_del_only(
        source_chars, target_chars
    )

    final_word_final, projected_boundaries, skipped_boundaries = (
        project_boundaries_conservatively(
            target_text=target_text,
            aligned=aligned,
            base_boundaries=ltp_word_final,
        )
    )

    final_word_final, merged_spans, merge_skips = apply_exact_match_merge_constraint(
        target_text=target_text,
        aligned=aligned,
        word_final_positions=final_word_final,
    )

    if source_chars:
        final_word_final.add(len(source_chars) - 1)

    projected = positions_to_segmented(source_chars, final_word_final)

    return {
        "projected": projected,
        "aligned": aligned,
        "unaligned_src": unaligned_src,
        "unaligned_tgt": unaligned_tgt,
        "ltp_seg": ltp_seg,
        "projected_boundaries": projected_boundaries,
        "skipped_boundaries": skipped_boundaries,
        "merged_spans": merged_spans,
        "merge_skips": merge_skips,
    }


def alignments_to_string(source, aligned):
    parts = []
    for tgt_idx in sorted(aligned.keys()):
        src_idx = aligned[tgt_idx]
        parts.append(f"{tgt_idx}->{src_idx}:{source[src_idx]}")
    return " ".join(parts)


def detect_separator(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()
    tab_count = first_line.count("\t")
    comma_count = first_line.count(",")
    return "\t" if tab_count > comma_count else ","


def main():
    parser = argparse.ArgumentParser(
        description="Step 1 v2: conservative projection plus exact-match merge constraint"
    )
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    args = parser.parse_args()

    sep = detect_separator(args.input)
    df = pd.read_csv(args.input, sep=sep)
    print(f"Loaded {len(df)} rows")

    if "Gold_Source" in df.columns:
        source_col = "Gold_Source"
    elif "Source_Text" in df.columns:
        source_col = "Source_Text"
    else:
        raise ValueError("Cannot find source column")

    if "Target_Text" not in df.columns:
        raise ValueError("Cannot find target column: Target_Text")

    id_col = (
        "CTB_ID"
        if "CTB_ID" in df.columns
        else ("Target_ID" if "Target_ID" in df.columns else None)
    )

    print(f"Using columns: source='{source_col}', target='Target_Text'")

    results = []
    total_chars = 0
    total_aligned = 0
    total_unaligned = 0
    total_projected_boundaries = 0
    total_skipped_boundaries = 0
    total_merged_spans = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        source = row[source_col]
        target = row["Target_Text"]

        try:
            result = run_step1(source, target)
            align_str = alignments_to_string(source.replace(" ", ""), result["aligned"])

            total_chars += len(source.replace(" ", ""))
            total_aligned += len(result["aligned"])
            total_unaligned += len(result["unaligned_src"])
            total_projected_boundaries += len(result["projected_boundaries"])
            total_skipped_boundaries += len(result["skipped_boundaries"])
            total_merged_spans += len(result["merged_spans"])
        except Exception as e:
            print(f"Error at row {idx}: {e}")
            result = {
                "projected": source,
                "aligned": {},
                "unaligned_src": [],
                "unaligned_tgt": [],
                "ltp_seg": source,
                "projected_boundaries": set(),
                "skipped_boundaries": [f"ERROR:{e}"],
                "merged_spans": [],
                "merge_skips": [],
            }
            align_str = "ERROR"

        results.append(
            {
                "Index": row.get("Index", idx),
                "ID": row.get(id_col, "") if id_col else "",
                "Source": source,
                "Target_Text": target,
                "LTP_Source": result["ltp_seg"],
                "Alignments": align_str,
                "Projected_Boundary_Count": len(result["projected_boundaries"]),
                "Skipped_Boundaries": " ".join(result["skipped_boundaries"]),
                "Merged_Spans": " ".join(result["merged_spans"]),
                "Merge_Skips": " ".join(result["merge_skips"]),
                "Projected_Source": result["projected"],
            }
        )

    print("\n=== Statistics ===")
    print(f"Total chars: {total_chars}")
    if total_chars > 0:
        print(f"Aligned (exact match): {total_aligned} ({100 * total_aligned / total_chars:.2f}%)")
        print(f"Unaligned source chars: {total_unaligned} ({100 * total_unaligned / total_chars:.2f}%)")
    else:
        print("Aligned (exact match): 0 (0.00%)")
        print("Unaligned source chars: 0 (0.00%)")
    print(f"Projected target boundaries: {total_projected_boundaries}")
    print(f"Skipped target boundaries: {total_skipped_boundaries}")
    print(f"Merged target-token spans: {total_merged_spans}")

    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output, index=False)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

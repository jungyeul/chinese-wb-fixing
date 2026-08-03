#!/usr/bin/env python3
"""
Shared target-to-source boundary projection pipeline.

This file extracts the projection component used in Step 1 and Step 2.
It assumes that character alignments are already provided in target-to-source
format:

    aligned[target_char_index] = source_char_index

The projection pipeline does not create alignments. It only uses alignments,
target word boundaries, and a base source segmentation to produce a projected
source segmentation.

Main behavior:
1. Start from base source boundaries, usually from LTP segmentation.
2. Project target word boundaries only when the two target-side boundary
   characters align to adjacent source characters.
3. Merge internal source boundaries when a full target token maps to a
   contiguous same-length source span.
4. Convert final source boundary positions back into a segmented source string.
"""

from typing import Dict, List, Set, Tuple, Optional, Any


AlignmentT2S = Dict[int, int]  # target_char_index -> source_char_index


def get_word_final_positions(segmented_text: str) -> Set[int]:
    """
    Return word-final character positions from a segmented sentence.

    Example:
        segmented_text = "我 喜欢 苹果"
        raw text       = "我喜欢苹果"
        output         = {0, 2, 4}

    These positions are character indices in the unsegmented sentence.
    """
    word_final: Set[int] = set()
    pos = 0
    for token in segmented_text.split():
        pos += len(token)
        word_final.add(pos - 1)
    return word_final


def positions_to_segmented(text_clean: str, word_final_positions: Set[int]) -> str:
    """
    Convert word-final positions into a segmented string.

    Args:
        text_clean: Source sentence without spaces.
        word_final_positions: Character indices where a word should end.

    Returns:
        Segmented sentence with spaces inserted after word-final positions.
    """
    result: List[str] = []
    for i, char in enumerate(text_clean):
        result.append(char)
        if i in word_final_positions and i < len(text_clean) - 1:
            result.append(" ")
    return "".join(result)


def get_token_spans(segmented_text: str) -> List[Tuple[str, int, int]]:
    """
    Return token spans as inclusive character spans.

    Returns:
        A list of (token, start_idx, end_idx), where start_idx and end_idx
        refer to positions in the unsegmented sentence.
    """
    spans: List[Tuple[str, int, int]] = []
    pos = 0
    for token in segmented_text.split():
        start = pos
        end = pos + len(token) - 1
        spans.append((token, start, end))
        pos += len(token)
    return spans


def project_boundaries_conservatively(
    target_text: str,
    aligned: AlignmentT2S,
    base_boundaries: Set[int],
) -> Tuple[Set[int], Set[int], List[str]]:
    """
    Project target word boundaries onto the source conservatively.

    A target boundary after target character j is projected only when:
        1. target[j] is aligned to source[i]
        2. target[j + 1] is aligned to source[i + 1]
        3. the two source positions are adjacent

    This avoids projecting boundaries across non-adjacent or uncertain matches.

    Args:
        target_text: Segmented target sentence.
        aligned: Target-to-source char alignment: target_idx -> source_idx.
        base_boundaries: Existing source word-final positions, usually from LTP.

    Returns:
        final_boundaries: Base boundaries plus safely projected boundaries.
        projected_boundaries: Boundaries added by projection.
        skipped_boundaries: Diagnostic strings explaining skipped boundaries.
    """
    final_boundaries = set(base_boundaries)
    projected_boundaries: Set[int] = set()
    skipped_boundaries: List[str] = []

    target_chars = target_text.replace(" ", "")
    target_boundaries = get_word_final_positions(target_text)

    for tgt_left_idx in sorted(target_boundaries):
        tgt_right_idx = tgt_left_idx + 1

        # No boundary after the final character needs projection.
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


def apply_exact_match_merge_constraint(
    target_text: str,
    aligned: AlignmentT2S,
    word_final_positions: Set[int],
) -> Tuple[Set[int], List[str], List[str]]:
    """
    Merge internal source boundaries when a target token maps cleanly to source.

    For each multi-character target token, the merge is allowed only when:
        1. every target character in the token is aligned to source;
        2. the aligned source positions form a contiguous span;
        3. the source span length equals the target token length.

    The span-length check prevents merging across extra source characters that
    happen to lie between aligned endpoints.

    Args:
        target_text: Segmented target sentence.
        aligned: Target-to-source char alignment: target_idx -> source_idx.
        word_final_positions: Current source word-final positions.

    Returns:
        word_final_positions: Updated word-final positions after merging.
        merged_spans: Diagnostic strings for successful merges.
        skipped_tokens: Diagnostic strings for skipped merge attempts.
    """
    merged_spans: List[str] = []
    skipped_tokens: List[str] = []

    # Copy so caller's set is not modified unexpectedly.
    word_final_positions = set(word_final_positions)

    for token, tgt_start, tgt_end in get_token_spans(target_text):
        if len(token) <= 1:
            continue

        src_positions: List[int] = []
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
        removed: List[int] = []

        # Remove boundaries inside this clean source span.
        for src_idx in range(src_start, src_end):
            if src_idx in word_final_positions:
                word_final_positions.discard(src_idx)
                removed.append(src_idx)

        if removed:
            merged_spans.append(f"{token}:{src_start}-{src_end}")

    return word_final_positions, merged_spans, skipped_tokens


def project_source_segmentation(
    source_text: str,
    target_text: str,
    base_source_segmentation: str,
    aligned: AlignmentT2S,
) -> Dict[str, Any]:
    """
    Run the full projection pipeline.

    Args:
        source_text: Source sentence, segmented or unsegmented. Spaces are removed.
        target_text: Segmented target sentence.
        base_source_segmentation: Base source segmentation, usually LTP output.
        aligned: Target-to-source character alignments.

    Returns:
        Dictionary with projected segmentation and diagnostics.
    """
    source_chars = source_text.replace(" ", "")

    if len(source_chars) == 0:
        return {
            "projected": source_text,
            "base_boundaries": set(),
            "final_boundaries": set(),
            "projected_boundaries": set(),
            "skipped_boundaries": [],
            "merged_spans": [],
            "merge_skips": [],
        }

    base_boundaries = get_word_final_positions(base_source_segmentation)

    final_boundaries, projected_boundaries, skipped_boundaries = (
        project_boundaries_conservatively(
            target_text=target_text,
            aligned=aligned,
            base_boundaries=base_boundaries,
        )
    )

    final_boundaries, merged_spans, merge_skips = apply_exact_match_merge_constraint(
        target_text=target_text,
        aligned=aligned,
        word_final_positions=final_boundaries,
    )

    # Always mark the final source character as a word boundary.
    final_boundaries.add(len(source_chars) - 1)

    projected = positions_to_segmented(source_chars, final_boundaries)

    return {
        "projected": projected,
        "base_boundaries": base_boundaries,
        "final_boundaries": final_boundaries,
        "projected_boundaries": projected_boundaries,
        "skipped_boundaries": skipped_boundaries,
        "merged_spans": merged_spans,
        "merge_skips": merge_skips,
    }


if __name__ == "__main__":
    # Minimal example for quick sanity check.
    source = "我喜欢吃苹果"
    target = "我 喜欢 吃 苹果"
    base_source_seg = "我 喜 欢 吃 苹果"

    # target index -> source index
    aligned = {
        0: 0,  # 我
        1: 1,  # 喜
        2: 2,  # 欢
        3: 3,  # 吃
        4: 4,  # 苹
        5: 5,  # 果
    }

    result = project_source_segmentation(
        source_text=source,
        target_text=target,
        base_source_segmentation=base_source_seg,
        aligned=aligned,
    )

    print(result["projected"])
    print("Projected boundaries:", sorted(result["projected_boundaries"]))
    print("Merged spans:", result["merged_spans"])

#!/usr/bin/env python3
"""
Step 2: LTP base + target-to-source exact match + 4-feature similarity.

- Keep Step 1 unchanged as the baseline
- Step 1: target->source exact-match alignment with insertion/deletion only
- Step 2: align remaining unaligned chars with 4 features (embedding, position,
  pinyin, shape), also target->source
- Reuse the same projection pipeline as updated Step 1, including the span-length
  check in the merge rule

Usage:
    python step2_ltp_4features_t2s_projection_spanlen.py \
        --input learner_corpus.csv \
        --shape triplet_no_dup_threshold.csv \
        --output learner_step2_t2s_spanlen_cvbest_t0.85.csv \
        --threshold 0.85
"""

import argparse
import os
import pickle
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from pypinyin import pinyin, Style
import Levenshtein
from tqdm import tqdm
from ltp import LTP


# ============== Global Models ==============
tokenizer = None
model = None
device = None
ltp_model = None


def load_bert_model(model_name='bert-base-chinese'):
    global tokenizer, model, device
    if model is None:
        print(f"Loading BERT model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        print(f"Using device: {device}")


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


# ============== Load Shape Table ==============
def load_shape_table(filepath):
    cache_path = filepath + '.pkl'

    if os.path.exists(cache_path):
        if os.path.getmtime(cache_path) > os.path.getmtime(filepath):
            print(f"Loading cached shape table: {cache_path}")
            with open(cache_path, 'rb') as f:
                shape_table = pickle.load(f)
            print(f"Loaded {len(shape_table)} pairs from cache")
            return shape_table

    print(f"Reading {filepath}...")
    df = pd.read_csv(filepath)

    shape_table = {(a, b): s for a, b, s in zip(df['CharA'], df['CharB'], df['similarity'])}
    shape_table.update({(b, a): s for a, b, s in zip(df['CharA'], df['CharB'], df['similarity'])})

    print(f"Built table with {len(shape_table)} pairs")

    with open(cache_path, 'wb') as f:
        pickle.dump(shape_table, f)

    return shape_table


# ============== Feature 1: Embedding Similarity ==============
def get_char_embeddings(text):
    chars = list(text)
    inputs = tokenizer(chars, return_tensors='pt', is_split_into_words=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[0, 1:len(chars) + 1, :]
    return embeddings


def compute_embedding_similarity(source, target):
    src_emb = get_char_embeddings(source)
    tgt_emb = get_char_embeddings(target)

    similarity = torch.cosine_similarity(
        src_emb.unsqueeze(1),
        tgt_emb.unsqueeze(0),
        dim=2
    )
    return similarity.cpu().numpy()


# ============== Feature 2: Pinyin Similarity ==============
def get_pinyin(char):
    py = pinyin(char, style=Style.TONE3, heteronym=False)
    return py[0][0] if py else char


def pinyin_similarity(char1, char2):
    py1 = get_pinyin(char1)
    py2 = get_pinyin(char2)
    dist = Levenshtein.distance(py1, py2)
    max_len = max(len(py1), len(py2))
    return 1 - dist / max_len if max_len > 0 else 1.0


# ============== Step 1: Exact Match (target -> source) ==============
def exact_alignment_ins_del_only(source, target):
    """Align identical characters using insertion/deletion only, target->source."""
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


# ============== Step 2: 4-Feature Similarity (target -> source) ==============
def align_similar_chars_4feat_t2s(source, target, unaligned_src, unaligned_tgt,
                                  emb_sim_matrix, shape_table, threshold=0.5):
    weights = {
        'embedding': 0.1,
        'position': 0.3,
        'pinyin': 0.0,
        'shape': 0.6,
    }

    new_alignments = {}
    used_src = set()

    for tgt_idx in unaligned_tgt:
        tgt_char = target[tgt_idx]
        best_src_idx = -1
        best_score = threshold

        for src_idx in unaligned_src:
            if src_idx in used_src:
                continue

            src_char = source[src_idx]

            emb_sim = emb_sim_matrix[src_idx, tgt_idx]
            shape_sim = shape_table.get((src_char, tgt_char), 0.0)
            py_sim = pinyin_similarity(src_char, tgt_char)
            pos_diff = abs(src_idx - tgt_idx) / max(len(source), len(target))
            pos_sim = 1 - pos_diff

            score = (
                weights['embedding'] * emb_sim +
                weights['shape'] * shape_sim +
                weights['pinyin'] * py_sim +
                weights['position'] * pos_sim
            )

            if score > best_score:
                best_score = score
                best_src_idx = src_idx

        if best_src_idx != -1:
            new_alignments[tgt_idx] = best_src_idx
            used_src.add(best_src_idx)

    return new_alignments


# ============== Projection Helpers (unchanged from updated Step 1) ==============
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


# ============== Main Processing ==============
def process_pair(source_text, target_text, shape_table, threshold=0.5):
    source_chars = source_text.replace(" ", "")
    target_chars = target_text.replace(" ", "")

    if len(source_chars) == 0:
        return {
            'align_str': '',
            'projected': source_text,
            'ltp_seg': source_text,
            'projected_boundaries': set(),
            'skipped_boundaries': [],
            'merged_spans': [],
            'merge_skips': [],
            'stats': {
                'step1_aligned': 0,
                'step2_aligned': 0,
                'still_unaligned_src': 0,
                'still_unaligned_tgt': 0,
            },
        }

    # Base segmentation from LTP
    ltp_seg = ltp_segment(source_text)
    ltp_word_final = get_word_final_positions(ltp_seg)

    # Step 1 exact match, target -> source
    step1_aligned, unaligned_src, unaligned_tgt = exact_alignment_ins_del_only(
        source_chars, target_chars
    )

    # Step 2 fuzzy similarity on remaining unaligned chars, target -> source
    emb_sim_matrix = compute_embedding_similarity(source_chars, target_chars)
    step2_aligned = align_similar_chars_4feat_t2s(
        source_chars,
        target_chars,
        unaligned_src,
        unaligned_tgt,
        emb_sim_matrix,
        shape_table,
        threshold,
    )

    # Combine alignments and reuse updated Step 1 projection unchanged
    all_aligned = {**step1_aligned, **step2_aligned}

    final_word_final, projected_boundaries, skipped_boundaries = (
        project_boundaries_conservatively(
            target_text=target_text,
            aligned=all_aligned,
            base_boundaries=ltp_word_final,
        )
    )

    final_word_final, merged_spans, merge_skips = apply_exact_match_merge_constraint(
        target_text=target_text,
        aligned=all_aligned,
        word_final_positions=final_word_final,
    )

    if source_chars:
        final_word_final.add(len(source_chars) - 1)

    projected = positions_to_segmented(source_chars, final_word_final)

    align_parts = []
    for tgt_idx in sorted(step1_aligned.keys()):
        src_idx = step1_aligned[tgt_idx]
        align_parts.append(f"[E]{tgt_idx}->{src_idx}:{source_chars[src_idx]}")
    for tgt_idx in sorted(step2_aligned.keys()):
        src_idx = step2_aligned[tgt_idx]
        align_parts.append(f"[S]{tgt_idx}->{src_idx}:{source_chars[src_idx]}")
    align_str = " ".join(align_parts)

    matched_src_step2 = set(step2_aligned.values())
    still_unaligned_src = len([x for x in unaligned_src if x not in matched_src_step2])
    still_unaligned_tgt = len([x for x in unaligned_tgt if x not in step2_aligned])

    return {
        'align_str': align_str,
        'projected': projected,
        'ltp_seg': ltp_seg,
        'projected_boundaries': projected_boundaries,
        'skipped_boundaries': skipped_boundaries,
        'merged_spans': merged_spans,
        'merge_skips': merge_skips,
        'stats': {
            'step1_aligned': len(step1_aligned),
            'step2_aligned': len(step2_aligned),
            'still_unaligned_src': still_unaligned_src,
            'still_unaligned_tgt': still_unaligned_tgt,
        },
    }


def detect_separator(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        first_line = f.readline()
    return '\t' if first_line.count('\t') > first_line.count(',') else ','


def main():
    parser = argparse.ArgumentParser(
        description='Step 2: target->source exact+4-feature similarity with updated Step 1 projection'
    )
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--shape', required=True, help='Shape similarity table CSV')
    parser.add_argument('--output', required=True, help='Output CSV file')
    parser.add_argument('--threshold', type=float, default=0.85, help='Similarity threshold for Step 2')
    parser.add_argument('--model', default='bert-base-chinese', help='BERT model name')
    args = parser.parse_args()

    load_bert_model(args.model)
    shape_table = load_shape_table(args.shape)

    sep = detect_separator(args.input)
    df = pd.read_csv(args.input, sep=sep)
    print(f"Loaded {len(df)} rows")

    if 'Gold_Source' in df.columns:
        source_col = 'Gold_Source'
    elif 'Source_Text' in df.columns:
        source_col = 'Source_Text'
    else:
        raise ValueError('Cannot find source column')

    if 'Target_Text' not in df.columns:
        raise ValueError('Cannot find target column: Target_Text')

    id_col = 'CTB_ID' if 'CTB_ID' in df.columns else ('Target_ID' if 'Target_ID' in df.columns else None)

    print(f"Using columns: source='{source_col}', target='Target_Text'")
    print(f"Similarity threshold: {args.threshold}")

    results = []
    total_stats = {
        'step1_aligned': 0,
        'step2_aligned': 0,
        'still_unaligned_src': 0,
        'still_unaligned_tgt': 0,
    }
    total_chars = 0
    total_projected_boundaries = 0
    total_skipped_boundaries = 0
    total_merged_spans = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Processing'):
        source = row[source_col]
        target = row['Target_Text']

        try:
            result = process_pair(source, target, shape_table, args.threshold)
            for k in total_stats:
                total_stats[k] += result['stats'].get(k, 0)
            total_chars += len(source.replace(' ', ''))
            total_projected_boundaries += len(result['projected_boundaries'])
            total_skipped_boundaries += len(result['skipped_boundaries'])
            total_merged_spans += len(result['merged_spans'])
        except Exception as e:
            print(f"Error at row {idx}: {e}")
            result = {
                'align_str': 'ERROR',
                'projected': source,
                'ltp_seg': source,
                'projected_boundaries': set(),
                'skipped_boundaries': [f'ERROR:{e}'],
                'merged_spans': [],
                'merge_skips': [],
                'stats': {
                    'step1_aligned': 0,
                    'step2_aligned': 0,
                    'still_unaligned_src': 0,
                    'still_unaligned_tgt': 0,
                },
            }

        results.append({
            'Index': row.get('Index', idx),
            'ID': row.get(id_col, '') if id_col else '',
            'Source': source,
            'Target_Text': target,
            'LTP_Source': result['ltp_seg'],
            'Alignments': result['align_str'],
            'Projected_Boundary_Count': len(result['projected_boundaries']),
            'Skipped_Boundaries': ' '.join(result['skipped_boundaries']),
            'Merged_Spans': ' '.join(result['merged_spans']),
            'Merge_Skips': ' '.join(result['merge_skips']),
            'Projected_Source': result['projected'],
        })

    print("\n=== Statistics ===")
    print(f"Total source chars: {total_chars}")
    if total_chars > 0:
        print(f"Step 1 aligned (exact): {total_stats['step1_aligned']} ({100 * total_stats['step1_aligned'] / total_chars:.2f}%)")
        print(f"Step 2 aligned (4-feat): {total_stats['step2_aligned']} ({100 * total_stats['step2_aligned'] / total_chars:.2f}%)")
        total_aligned = total_stats['step1_aligned'] + total_stats['step2_aligned']
        print(f"Total aligned: {total_aligned} ({100 * total_aligned / total_chars:.2f}%)")
        print(f"Still unaligned source chars: {total_stats['still_unaligned_src']} ({100 * total_stats['still_unaligned_src'] / total_chars:.2f}%)")
        print(f"Still unaligned target chars: {total_stats['still_unaligned_tgt']}")
    else:
        print("Step 1 aligned (exact): 0 (0.00%)")
        print("Step 2 aligned (4-feat): 0 (0.00%)")
        print("Total aligned: 0 (0.00%)")
        print("Still unaligned source chars: 0 (0.00%)")
        print("Still unaligned target chars: 0")
    print(f"Projected target boundaries: {total_projected_boundaries}")
    print(f"Skipped target boundaries: {total_skipped_boundaries}")
    print(f"Merged target-token spans: {total_merged_spans}")

    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output, index=False)
    print(f"Saved to: {args.output}")


if __name__ == '__main__':
    main()

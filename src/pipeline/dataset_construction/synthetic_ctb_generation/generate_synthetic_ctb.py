#!/usr/bin/env python3
"""Generate the CTB-derived synthetic benchmark described in the paper.

This script expects a local CTB 5.1 segmented directory containing files such
as chtb_0400.seg.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd


ERROR_DIST = {
    "delete": 0.318,
    "insert": 0.199,
    "replace": 0.483,
}


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def load_ctb_segmented(seg_dir: Path, id_min: int, id_max: int) -> List[Tuple[str, List[str]]]:
    sent_list: List[Tuple[str, List[str]]] = []
    pattern = str(seg_dir / "chtb_*.seg")
    seg_files = []
    for seg_path in sorted(glob.glob(pattern)):
        match = re.search(r"(\d+)", os.path.basename(seg_path))
        if not match:
            continue
        file_id = int(match.group(1))
        if id_min <= file_id <= id_max:
            seg_files.append(seg_path)

    if not seg_files:
        raise FileNotFoundError(f"No CTB .seg files found in {seg_dir} for IDs {id_min}-{id_max}")

    meta_lines = {"完", "（ 完 ）", "( 完 )", "（ 完)", "(完 ）"}
    for seg_path in seg_files:
        fname = os.path.basename(seg_path)
        with open(seg_path, "r", encoding="gb18030", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line or not has_cjk(line):
                    continue
                if line.startswith("<") and line.endswith(">"):
                    continue
                if line in meta_lines or line.startswith("新华社 "):
                    continue
                tokens = line.split()
                if tokens and not all(tok.startswith("<") and tok.endswith(">") for tok in tokens):
                    sent_list.append((f"{fname}:{line_no}", tokens))
    return sent_list


def sample_op() -> str:
    draw = random.random()
    if draw < ERROR_DIST["delete"]:
        return "delete"
    if draw < ERROR_DIST["delete"] + ERROR_DIST["insert"]:
        return "insert"
    return "replace"


def apply_global_char_noise(all_tokens: List[List[str]], noise_ratio: float) -> List[List[str]]:
    token_chars = [[list(tok) for tok in sent] for sent in all_tokens]
    total_chars = sum(len(tok) for sent in all_tokens for tok in sent)
    num_ops = max(1, int(round(noise_ratio * total_chars)))
    char_pool = [ch for sent in all_tokens for tok in sent for ch in tok if ch.strip()]
    if not char_pool:
        return all_tokens

    for _ in range(num_ops):
        sent = token_chars[random.randrange(len(token_chars))]
        primary = sample_op()
        fallback = {
            "delete": ["delete", "replace", "insert"],
            "insert": ["insert", "replace", "delete"],
            "replace": ["replace", "delete", "insert"],
        }[primary]
        for op_type in fallback:
            candidates = [(idx, len(chars)) for idx, chars in enumerate(sent) if chars or op_type == "insert"]
            if not candidates:
                continue
            tok_idx, length = random.choice(candidates)
            if op_type == "delete" and length:
                del sent[tok_idx][random.randrange(length)]
                break
            if op_type == "replace" and length:
                char_idx = random.randrange(length)
                old = sent[tok_idx][char_idx]
                new = random.choice(char_pool)
                for _ in range(3):
                    if new != old:
                        break
                    new = random.choice(char_pool)
                sent[tok_idx][char_idx] = new
                break
            if op_type == "insert":
                sent[tok_idx].insert(random.randrange(length + 1), random.choice(char_pool))
                break
    return [["".join(chars) for chars in sent] for sent in token_chars]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctb-seg-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("pipeline/synthetic_data/syn_dat_outputs"), type=Path)
    parser.add_argument("--size", default=1079, type=int)
    parser.add_argument("--id-min", default=400, type=int)
    parser.add_argument("--id-max", default=550, type=int)
    parser.add_argument("--base-sample-seed", default=42, type=int)
    args = parser.parse_args()

    all_sents = load_ctb_segmented(args.ctb_seg_dir, args.id_min, args.id_max)
    if args.size > len(all_sents):
        raise ValueError(f"Requested {args.size} sentences but only found {len(all_sents)}")

    random.seed(args.base_sample_seed)
    sampled_indices = random.sample(range(len(all_sents)), args.size)
    sampled_ids = [all_sents[i][0] for i in sampled_indices]
    sampled_tokens = [all_sents[i][1] for i in sampled_indices]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for noise_pct in range(1, 11):
        random.seed(noise_pct)
        noisy_tokens = apply_global_char_noise(sampled_tokens, noise_pct / 100.0)
        rows = []
        for idx, (ctb_id, clean_tok, noisy_tok) in enumerate(zip(sampled_ids, sampled_tokens, noisy_tokens)):
            rows.append(
                {
                    "Index": idx,
                    "CTB_ID": ctb_id,
                    "Gold_Source": " ".join(noisy_tok),
                    "Target_Text": " ".join(clean_tok),
                }
            )
        output = args.output_dir / f"ctb5_synthetic_charnoise_{noise_pct}pct_ids{args.id_min}_{args.id_max}_{args.size}.csv"
        pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()

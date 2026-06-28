import pandas as pd
import Levenshtein
from ltp import LTP

# =====================
# 对齐函数
# =====================
def create_combined_list(S, T):
    a = list(S)
    b = list(T)
    edits = list(Levenshtein.editops(S, T))
    idx = 0
    S_out, T_out = [], []
    i, j = 0, 0
    
    while i < len(a) or j < len(b):
        if idx < len(edits):
            op, i1, j1 = edits[idx]
        else:
            op, i1, j1 = None, None, None
        
        if op == "delete" and i == i1:
            S_out.append(a[i])
            T_out.append("")
            i += 1
            idx += 1
        elif op == "insert" and j == j1:
            S_out.append("")
            T_out.append(b[j])
            j += 1
            idx += 1
        elif op == "replace" and i == i1 and j == j1:
            S_out.append(a[i])
            T_out.append(b[j])
            i += 1
            j += 1
            idx += 1
        else:
            if i < len(a) and j < len(b):
                S_out.append(a[i])
                T_out.append(b[j])
                i += 1
                j += 1
            elif i < len(a):
                S_out.append(a[i])
                i += 1
            elif j < len(b):
                T_out.append(b[j])
                j += 1
    return S_out, T_out

# =====================
# 投射函数
# =====================
def project_wb(S_out, T_out, target_words, keep_empty=True, preserve_source_order=True):
    assert len(S_out) == len(T_out), "S_out and T_out must be equal length"
    N = len(T_out)

    def emit_pair(src_words, tgt_words_aligned, s, t):
        if s or keep_empty:
            src_words.append(s)
            tgt_words_aligned.append(t)

    tgt_words_aligned = []
    src_words = []
    wi = 0
    k = 0
    pending_del = []
    cur_src_buf = []
    cur_tgt_buf = []
    started_word = False
    chars_needed = len(target_words[wi]) if target_words else 0

    def flush_pending_del():
        nonlocal pending_del
        if pending_del:
            emit_pair(src_words, tgt_words_aligned, "".join(pending_del), "")
            pending_del = []

    def flush_current_piece():
        nonlocal cur_src_buf, cur_tgt_buf
        emit_pair(src_words, tgt_words_aligned, "".join(cur_src_buf), "".join(cur_tgt_buf))
        cur_src_buf = []
        cur_tgt_buf = []

    while k < N and wi < len(target_words):
        if T_out[k] == "":
            if S_out[k] != "":
                if started_word:
                    flush_current_piece()
                emit_pair(src_words, tgt_words_aligned, S_out[k], "")
            k += 1
            continue

        if not started_word:
            flush_pending_del()
            started_word = True

        if S_out[k] != "":
            cur_src_buf.append(S_out[k])
        cur_tgt_buf.append(T_out[k])
        chars_needed -= 1
        k += 1

        if chars_needed == 0:
            flush_current_piece()
            wi += 1
            started_word = False
            if wi < len(target_words):
                chars_needed = len(target_words[wi])

    while wi < len(target_words):
        flush_pending_del()
        emit_pair(src_words, tgt_words_aligned, "", target_words[wi])
        wi += 1

    flush_pending_del()

    while k < N:
        if T_out[k] == "":
            del_buf = []
            while k < N and T_out[k] == "":
                if S_out[k] != "":
                    del_buf.append(S_out[k])
                k += 1
            if del_buf:
                emit_pair(src_words, tgt_words_aligned, "".join(del_buf), "")
            continue

        tail_src_buf, tail_tgt_buf = [], []
        while k < N and T_out[k] != "":
            if S_out[k] != "":
                tail_src_buf.append(S_out[k])
            tail_tgt_buf.append(T_out[k])
            k += 1
        emit_pair(src_words, tgt_words_aligned, "".join(tail_src_buf), "".join(tail_tgt_buf))

    return src_words, tgt_words_aligned

# =====================
# 主流程
# =====================
def main():
    import argparse
    from pathlib import Path

    base = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Apply the legacy LTP/Levenshtein projection helper.")
    parser.add_argument(
        "--input",
        default=str(base / "inputs/three_outputs_learners_charpairs_textcols.csv"),
        help="Input CSV with Source_Text and Target_Text columns.",
    )
    parser.add_argument(
        "--output",
        default=str(base / "generated_csv/ltp_levenshtein_projection_helper.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    # 初始化 LTP
    ltp = LTP("LTP/small")

    new_sources = []
    for _, row in df.iterrows():
        src = row["Source_Text"].replace(" ", "")
        tgt = row["Target_Text"].replace(" ", "")
        try:
            # 1. 对齐
            S_out, T_out = create_combined_list(src, tgt)
            # 2. target 分词
            output = ltp.pipeline([tgt], tasks=["cws"])
            target_words = output["cws"][0]
            # 3. 投射
            src_proj, tgt_proj = project_wb(S_out, T_out, target_words,
                                            keep_empty=True, preserve_source_order=True)
            new_sources.append(" ".join([tok for tok in src_proj if tok]))
        except Exception as e:
            new_sources.append(src)

    df["Source_Text"] = new_sources
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved to {output}")

if __name__ == "__main__":
    main()

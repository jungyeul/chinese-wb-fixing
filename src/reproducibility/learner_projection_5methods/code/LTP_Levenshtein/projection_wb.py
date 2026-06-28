from typing import List, Tuple

########################################################
# Step 1. Character-level alignment (Levenshtein DP)
########################################################
def levenshtein_align(source: str, target: str):
    m, n = len(source), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
        back[i][0] = "D"
    for j in range(1, n + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if source[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
                back[i][j] = "M"
            else:
                choices = [
                    (dp[i-1][j] + 1, "D"),
                    (dp[i][j-1] + 1, "I"),
                    (dp[i-1][j-1] + 1, "R")
                ]
                dp[i][j], back[i][j] = min(choices)

    eS, eT, log = [], [], []
    i, j = m, n
    while i > 0 or j > 0:
        if back[i][j] == "M":
            eS.append(source[i-1]); eT.append(target[j-1])
            log.append(f"{i-1} {i} {source[i-1]} --> {target[j-1]}")
            i, j = i-1, j-1
        elif back[i][j] == "R":
            # ⚠️ 改动点：当成对齐，而不是替换
            eS.append(source[i-1]); eT.append(target[j-1])
            log.append(f"{i-1} {i} {source[i-1]} ~ {target[j-1]} substitute (ignored)")
            i, j = i-1, j-1
        elif back[i][j] == "D":
            eS.append(source[i-1]); eT.append("ϵ")
            log.append(f"{i-1} {i} {source[i-1]} --> ϵ delete")
            i -= 1
        elif back[i][j] == "I":
            eS.append("ϵ"); eT.append(target[j-1])
            log.append(f"insert {target[j-1]} at {j-1} in target")
            j -= 1

    return eS[::-1], eT[::-1], log[::-1]


########################################################
# Step 2. Projection algorithm (Algorithm 1)
########################################################
def project_wb(eS: List[str], eT: List[str], WT: List[str],
               mode: str = "buffer", keep_empty: bool = True) -> Tuple[List[str], List[str]]:
    """
    Project target word boundaries (WT) onto source using aligned sequences (eS, eT).
    Args:
        eS, eT: aligned character arrays (same length N)
        WT: tokenized target words (list of strings)
        mode: "buffer" or "preserve"
        keep_empty: whether to emit (ϵ, word) for insertions
    Returns:
        cWS: projected source segmentation
        cWT: aligned target segmentation
    """
    N = len(eS)
    cWS, cWT = [], []
    k, i = 0, 0
    pending, curS, curT = [], [], []
    need = len(WT[0]) if WT else 0

    while k < N and i < len(WT):
        if eT[k] == "ϵ":  # deletion
            if eS[k] != "ϵ":
                if mode == "preserve":
                    cWS.append(eS[k]); cWT.append("ϵ")
                else:
                    pending.append(eS[k])
            k += 1
        else:  # real target char
            if mode == "buffer" and pending:
                cWS.append("".join(pending)); cWT.append("ϵ")
                pending = []
            if eS[k] != "ϵ":
                curS.append(eS[k])
            curT.append(eT[k])
            need -= 1
            k += 1
            if need == 0:
                cWS.append("".join(curS) if curS else "ϵ")
                cWT.append(WT[i])
                curS, curT = [], []
                i += 1
                if i < len(WT):
                    need = len(WT[i])

    # trailing target words
    while i < len(WT):
        if keep_empty:
            cWS.append("ϵ"); cWT.append(WT[i])
        i += 1

    # flush pending deletions
    if mode == "buffer" and pending:
        cWS.append("".join(pending)); cWT.append("ϵ")

    return cWS, cWT

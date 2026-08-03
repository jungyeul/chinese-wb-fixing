import csv
import os
from time import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Set file paths
data_path = os.path.join(os.path.dirname(__file__), '../data/MuCGEC_dev.txt')
out_path = os.path.join(os.path.dirname(__file__), '../data/cosine_target.csv')
dist_path = os.path.join(os.path.dirname(__file__), '../data/cosine_similarities.csv')

results = []     # To store best-matched targets
dist_rows = []   # To store full candidate similarity info

start = time()  # Start timing

# Read and process input data line by line
with open(data_path, encoding='utf-8') as f:
    lines = list(f)
    for idx, line in enumerate(tqdm(lines, desc="Processing sentences"), 1):
        line = line.strip()
        if not line:
            continue
        cols = line.split('\t')
        if len(cols) < 2:
            continue
        original = cols[1]
        candidates = cols[2:]
        if not candidates:
            continue

        # Encode original and candidate sentences
        original_emb = model.encode(original, convert_to_tensor=True)
        candidate_embs = model.encode(candidates, convert_to_tensor=True)

        # Compute cosine similarity scores
        cosine_scores = util.cos_sim(original_emb, candidate_embs)[0].tolist()
        max_idx = cosine_scores.index(max(cosine_scores))
        best_target = candidates[max_idx]

        # Save the best match
        results.append([idx, original, best_target])

        # Save the full similarity row
        dist_row = [idx, original] + candidates + [f"{score:.4f}" for score in cosine_scores]
        dist_rows.append(dist_row)

# Write best-matching results (excluding certain labels)
with open(out_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'source', 'target'])
    for row in results:
        if row[2] not in ['没有错误', '无法标注']:  # Exclude "no error" or "unable to annotate"
            writer.writerow(row)

# Write full similarity data for all candidates
with open(dist_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    max_cand = max((len(row) - 2) // 2 for row in dist_rows) if dist_rows else 0

    # Generate header: cand_1, cand_2, ..., sim_1, sim_2, ...
    header = ['id', 'source']
    for i in range(max_cand):
        header.append(f'cand_{i+1}')
    for i in range(max_cand):
        header.append(f'sim_{i+1}')
    writer.writerow(header)

    # Pad candidates and scores if necessary and write each row
    for row in dist_rows:
        n = (len(row) - 2) // 2
        cands = row[2:2+n]
        sims = row[2+n:]
        cands += [''] * (max_cand - len(cands))
        sims += [''] * (max_cand - len(sims))
        writer.writerow([row[0], row[1]] + cands + sims)

print(f"✅ Done in {time() - start:.2f} seconds.")

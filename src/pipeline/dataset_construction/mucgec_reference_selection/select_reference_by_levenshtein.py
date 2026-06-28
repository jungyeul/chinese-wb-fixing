import csv
import os
import Levenshtein

# Main data paths
data_path = os.path.join(os.path.dirname(__file__), '../data/MuCGEC_dev.txt')
out_path = os.path.join(os.path.dirname(__file__), '../data/levenshtein_target.csv')
dist_path = os.path.join(os.path.dirname(__file__), '../data/levenshtein_distances.csv')

results = []
dist_rows = []
with open(data_path, encoding='utf-8') as f:
    for idx, line in enumerate(f, 1):
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
        min_dist = None
        min_target = None
        dist_list = []
        for cand in candidates:
            dist = Levenshtein.distance(original, cand)
            dist_list.append(dist)
            if (min_dist is None) or (dist < min_dist):
                min_dist = dist
                min_target = cand
        # delete '没有错误' or '无法标注'
        results.append([idx, original, min_target])
        dist_row = [idx, original] + candidates + [str(d) for d in dist_list]
        dist_rows.append(dist_row)

# Output the minimum distance pair, filter here
with open(out_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'source', 'target'])
    for row in results:
        if row[2] not in ['没有错误', '无法标注']:
            writer.writerow(row)

# Output all distances
with open(dist_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    max_cand = max((len(row)-2)//2 for row in dist_rows) if dist_rows else 0
    header = ['id', 'source']
    for i in range(max_cand):
        header.append(f'cand_{i+1}')
    for i in range(max_cand):
        header.append(f'dist_{i+1}')
    writer.writerow(header)
    for row in dist_rows:
        n = (len(row)-2)//2
        cands = row[2:2+n]
        dists = row[2+n:]
        cands += [''] * (max_cand - len(cands))
        dists += [''] * (max_cand - len(dists))
        writer.writerow([row[0], row[1]] + cands + dists)

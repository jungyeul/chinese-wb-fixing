import pandas as pd
import os
from jp_algorithm import evaluate, load_txt_file

# ===== 1️⃣ 定义输入文件 =====
csv_files = [
    "data/anhui_three_outputs.csv",
    "data/canto_three_outputs.csv",
    "data/learners_three_outputs_add_ibm_only.csv",
    "data/nw_three_outputs.csv",
    "data/wu_three_outputs.csv"
    
]

methods = {
    'gold': 'Gold_Source',
    'LTP': 'LTP', 
    'Levenshtein': 'Levenshtein',
    'IBM2': 'IBM2'
}

# ===== 2️⃣ 创建输出目录 =====
base_out_dir = "data"
os.makedirs(base_out_dir, exist_ok=True)

all_results = []

# ===== 3️⃣ 遍历所有 CSV 文件 =====
for csv_path in csv_files:
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\n=== Evaluating {dataset_name} ===")

    # 读取 CSV
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 为四种方法生成 txt 文件
#    for name, col in methods.items():
#        dir_path = f"data/txt-{name}"
#        os.makedirs(dir_path, exist_ok=True)
#        filepath = os.path.join(dir_path, f"{dataset_name}.txt")
#        df[col].to_csv(filepath, index=False, header=False)
#        print(f"✅ Created: {filepath}")
    for name, col in methods.items():
        dir_path = f"data/txt-{name}"
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"{dataset_name}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            for sent in df[col]:
                f.write(str(sent).strip() + "\n")
        print(f"✅ Created: {filepath}")


    # # 加载 gold
    # gold_file = f"data/txt-gold/{dataset_name}.txt"
    # gold_ud = load_txt_file(gold_file)

    # # 三个系统
    # systems = {
    #     'LTP': f"data/txt-LTP/{dataset_name}.txt",
    #     'Levenshtein': f"data/txt-Levenshtein/{dataset_name}.txt",
    #     'IBM2': f"data/txt-IBM2/{dataset_name}.txt"
    # }

    # # ===== 运行评估 =====
    # results = {}
    # print("=" * 60)
    # print(f"SEGMENTATION EVALUATION RESULTS for {dataset_name}")
    # print("=" * 60)

    # for method_name, pred_file in systems.items():
    #     print(f"\n{method_name}:")
    #     print("-" * 40)
    #     pred_ud = load_txt_file(pred_file)
    #     scores = evaluate(gold_ud, pred_ud)
# 加载 gold 文件路径（注意这里只定义路径，不立刻加载）
    gold_file = f"data/txt-gold/{dataset_name}.txt"

    # 三个系统
    systems = {
        'LTP': f"data/txt-LTP/{dataset_name}.txt",
        'Levenshtein': f"data/txt-Levenshtein/{dataset_name}.txt",
        'IBM2': f"data/txt-IBM2/{dataset_name}.txt"
    }

    for method_name, pred_file in systems.items():
        print(f"\n{method_name}:")
        print("-" * 40)

        # ✅ 每次循环都重新加载 gold_ud，避免被 evaluate() 改写
        gold_ud = load_txt_file(gold_file)
        pred_ud = load_txt_file(pred_file)

        scores = evaluate(gold_ud, pred_ud)


        token_score = scores['Tokens']
        sent_score = scores['Sentences']

        print("\n  Token-level:")
        print(f"    Precision: {token_score.precision:.4f}")
        print(f"    Recall:    {token_score.recall:.4f}")
        print(f"    F1:        {token_score.f1:.4f}")

        print("\n  Sentence-level:")
        print(f"    Precision: {sent_score.precision:.4f}")
        print(f"    Recall:    {sent_score.recall:.4f}")
        print(f"    F1:        {sent_score.f1:.4f}")

        all_results.append({
            "Dataset": dataset_name,
            "Method": method_name,
            "Token_Precision": token_score.precision,
            "Token_Recall": token_score.recall,
            "Token_F1": token_score.f1,
            "Sentence_Precision": sent_score.precision,
            "Sentence_Recall": sent_score.recall,
            "Sentence_F1": sent_score.f1
        })

# ===== 4️⃣ 汇总输出 =====
summary_df = pd.DataFrame(all_results)
summary_df = summary_df.sort_values(["Dataset", "Method"]).reset_index(drop=True)

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(summary_df.to_string(index=False))

out_path = os.path.join(base_out_dir, "evaluation_results_all.csv")
summary_df.to_csv(out_path, index=False)
print(f"\n✅ All results saved to: {out_path}")

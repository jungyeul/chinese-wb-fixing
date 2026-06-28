import pandas as pd
from pathlib import Path
from jp_algorithm import evaluate, load_txt_file

# ===== 1️⃣ 定义输入文件 =====
script_dir = Path(__file__).parent
csv_files = [
    script_dir.parent / "data" / "0430" / "learner_step3_wordorder_safe_v2_t0.85.csv",
]

methods = {
    "gold": "Source",
    "Projected_Source": "Projected_Source",
}

all_results = []

# ===== 2️⃣ 遍历所有 CSV 文件（目前就 1 个）=====
for csv_file in csv_files:
    csv_path = Path(csv_file).resolve()
    dataset_name = csv_path.stem  # learner_roberta_output
    print(f"\n=== Evaluating {dataset_name} ===")

    # 输出目录：csv 的上一级 / eva_{dataset}
    base_out_dir = (csv_path.parent / f"eva_{dataset_name}")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    # 读取 CSV
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # ===== 3️⃣ 生成 txt 文件（写到 base_out_dir 下）=====
    for name, col in methods.items():
        dir_path = base_out_dir / f"txt-{name}"
        dir_path.mkdir(parents=True, exist_ok=True)

        txt_path = dir_path / f"{dataset_name}.txt"
        with txt_path.open("w", encoding="utf-8") as f:
            for sent in df[col]:
                f.write(str(sent).strip() + "\n")
        print(f"✅ Created: {txt_path}")

    # ===== 4️⃣ 跑评估（Projected_Source vs Source）=====
    gold_file = base_out_dir / "txt-gold" / f"{dataset_name}.txt"
    pred_file = base_out_dir / "txt-Projected_Source" / f"{dataset_name}.txt"

    print(f"\nProjected_Source:")
    print("-" * 40)

    gold_ud = load_txt_file(str(gold_file))
    pred_ud = load_txt_file(str(pred_file))
    scores = evaluate(gold_ud, pred_ud)

    token_score = scores["Tokens"]
    sent_score = scores["Sentences"]

    print("\n  Token-level:")
    print(f"    Precision: {token_score.precision:.4f}")
    print(f"    Recall:    {token_score.recall:.4f}")
    print(f"    F1:        {token_score.f1:.4f}")

    print("\n  Sentence-level:")
    print(f"    Precision: {sent_score.precision:.4f}")
    print(f"    Recall:    {sent_score.recall:.4f}")
    print(f"    F1:        {sent_score.f1:.4f}")

    result_row = {
        "Dataset": dataset_name,
        "Method": "Projected_Source",
        "Token_Precision": token_score.precision,
        "Token_Recall": token_score.recall,
        "Token_F1": token_score.f1,
        "Sentence_Precision": sent_score.precision,
        "Sentence_Recall": sent_score.recall,
        "Sentence_F1": sent_score.f1,
    }
    all_results.append(result_row)

    # ===== 5️⃣ 单文件输出到 eva_{dataset}/ =====
    summary_df = pd.DataFrame([result_row]).sort_values(["Dataset", "Method"]).reset_index(drop=True)

    out_path = base_out_dir / f"eva_{dataset_name}.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\n✅ Results saved to: {out_path}")

# （可选）如果你未来 csv_files 放多个文件，这里也可以再额外写一个总表到 learner/ 下

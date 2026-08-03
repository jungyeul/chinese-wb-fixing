# ltp_segment.py  (Python 3.9 兼容 & 更稳健)
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from ltp import LTP

def _load_ltp_or_raise(model_path: Union[str, Path]) -> LTP:
    model_path = Path(model_path)
    if not model_path.exists() or not model_path.is_dir():
        raise FileNotFoundError(
            f"模型目录不存在：{model_path.resolve()}\n"
            f"请确保这里是你克隆/解压后的模型文件夹（应包含 config.json）。"
        )
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"未找到 config.json：{(model_path / 'config.json').resolve()}\n"
            f"请确认模型是否下载完整（例如使用 git lfs clone 或解压官方 base.tgz）。"
        )
    return LTP(str(model_path))

def ltp_segment_id_src_tgt(
    input_csv: Union[str, Path],
    output_csv: Union[str, Path],
    model_name: Union[str, Path] = "./base",  # 指向本地模型目录
    target_col: str = "final_target",         # 你的目标列，默认为 final_target
    batch_size: int = 64,
    keep_na: bool = False,
    ltp_model: Optional[LTP] = None,
) -> None:
    """
    读取包含 id, source, <target_col> 的 CSV，用 LTP 分词，
    输出仅包含：id, source_segmented, target_segmented 三列的 CSV。
    """
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    df = pd.read_csv(input_csv)
    for col in ["id", "source", target_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {input_csv}")

    # 准备文本
    src = df["source"]
    tgt = df[target_col]
    if not keep_na:
        src = src.fillna("")
        tgt = tgt.fillna("")
    # 避免出现数值/None类型
    src = src.astype(str)
    tgt = tgt.astype(str)

    # 加载模型（优先用传入的实例，其次校验本地目录）
    ltp = ltp_model or _load_ltp_or_raise(model_name)

    def segment_series(series: pd.Series) -> list:
        out = []
        n = len(series)
        for i in range(0, n, batch_size):
            batch = series.iloc[i:i+batch_size].tolist()
            res = ltp.pipeline(batch, tasks=["cws"])
            out.extend([" ".join(tokens) if tokens else "" for tokens in res.cws])
        return out

    source_segmented = segment_series(src)
    target_segmented = segment_series(tgt)

    out_df = pd.DataFrame({
        "id": df["id"],
        "source_segmented": source_segmented,
        "target_segmented": target_segmented
    })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

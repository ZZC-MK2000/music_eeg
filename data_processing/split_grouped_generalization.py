import argparse
import json
import re
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _parse_nmed_recording(recording_name: str) -> Tuple[str, str]:
    # Examples: 02_1_raw.mat, 14_2_raw
    m = re.search(r"(\d{2})_(\d)", str(recording_name))
    if not m:
        return "unknown", "unknown"
    subject_id = f"sub_{m.group(1)}"
    session_id = f"sess_{m.group(2)}"
    return subject_id, session_id


def _ensure_group_columns(df: pd.DataFrame, derive_from: str) -> pd.DataFrame:
    if {"subject_id", "session_id"}.issubset(df.columns):
        return df

    if derive_from not in df.columns:
        raise ValueError(
            "缺少 subject_id/session_id 列，且未找到可推断列: "
            f"{derive_from}. 请在输入特征表中加入分组列。"
        )

    subject_ids = []
    session_ids = []
    for v in df[derive_from].astype(str).tolist():
        sid, sess = _parse_nmed_recording(v)
        subject_ids.append(sid)
        session_ids.append(sess)

    out = df.copy()
    if "subject_id" not in out.columns:
        out["subject_id"] = subject_ids
    if "session_id" not in out.columns:
        out["session_id"] = session_ids
    return out


def _pick_group_column(df: pd.DataFrame, split_mode: str, custom_group_col: str) -> str:
    if split_mode == "subject":
        if "subject_id" not in df.columns:
            raise ValueError("split_mode=subject 需要 subject_id 列。")
        return "subject_id"
    if split_mode == "session":
        if "session_id" not in df.columns:
            raise ValueError("split_mode=session 需要 session_id 列。")
        return "session_id"
    if not custom_group_col:
        raise ValueError("split_mode=custom 时必须传入 --group-col。")
    if custom_group_col not in df.columns:
        raise ValueError(f"未找到 group 列: {custom_group_col}")
    return custom_group_col


def _split_by_group(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].astype(str).values

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    val_size_rel = val_size / (1.0 - test_size)
    groups_tv = train_val_df[group_col].astype(str).values
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_rel, random_state=random_state + 1)
    train_idx_rel, val_idx_rel = next(gss_val.split(train_val_df, groups=groups_tv))

    train_df = train_val_df.iloc[train_idx_rel].copy()
    val_df = train_val_df.iloc[val_idx_rel].copy()

    return train_df, val_df, test_df


def _summarize_split(df: pd.DataFrame, group_col: str) -> Dict:
    label_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
    return {
        "n_rows": int(len(df)),
        "n_groups": int(df[group_col].nunique()),
        "groups": sorted(df[group_col].astype(str).unique().tolist()),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
    }


def split_grouped_features(
    input_csv: Path,
    output_dir: Path,
    split_mode: str = "subject",
    group_col: str = "",
    derive_from: str = "recording_id",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
    drop_unknown: bool = False,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")

    df = pd.read_csv(input_csv)
    if "label" not in df.columns:
        raise ValueError("输入特征表缺少 label 列。")

    if drop_unknown:
        df = df[df["label"] != "unknown"].copy()

    df = _ensure_group_columns(df, derive_from=derive_from)
    chosen_group_col = _pick_group_column(df, split_mode, group_col)

    train_df, val_df, test_df = _split_by_group(
        df=df,
        group_col=chosen_group_col,
        test_size=test_size,
        val_size=val_size,
        random_state=seed,
    )

    train_groups = set(train_df[chosen_group_col].astype(str))
    val_groups = set(val_df[chosen_group_col].astype(str))
    test_groups = set(test_df[chosen_group_col].astype(str))
    if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
        raise RuntimeError("分组泄漏：train/val/test 之间存在重叠组。")

    train_path = output_dir / "train_features.csv"
    val_path = output_dir / "val_features.csv"
    test_path = output_dir / "test_features.csv"
    summary_path = output_dir / "split_summary.json"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "input_csv": str(input_csv),
        "split_mode": split_mode,
        "group_col": chosen_group_col,
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "train": _summarize_split(train_df, chosen_group_col),
        "val": _summarize_split(val_df, chosen_group_col),
        "test": _summarize_split(test_df, chosen_group_col),
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("分组划分完成:")
    print(f"  mode={split_mode}, group_col={chosen_group_col}")
    print(f"  train/val/test = {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"  输出: {train_path}, {val_path}, {test_path}")
    print(f"  摘要: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="按被试/会话分组划分 EEG 特征，避免数据泄漏。")
    parser.add_argument("--input-csv", type=str, required=True, help="输入特征CSV，需包含 label 列。")
    parser.add_argument("--output-dir", type=str, default=".", help="输出目录。")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="subject",
        choices=["subject", "session", "custom"],
        help="分组划分模式：subject/session/custom。",
    )
    parser.add_argument("--group-col", type=str, default="", help="split_mode=custom 时使用的分组列名。")
    parser.add_argument("--derive-from", type=str, default="recording_id", help="当缺少 subject/session 列时用于推断的列名。")
    parser.add_argument("--test-size", type=float, default=0.2, help="测试集比例。")
    parser.add_argument("--val-size", type=float, default=0.2, help="验证集比例（相对于全量）。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--drop-unknown", action="store_true", help="是否剔除 label=unknown。")
    args = parser.parse_args()

    split_grouped_features(
        input_csv=Path(args.input_csv),
        output_dir=Path(args.output_dir),
        split_mode=args.split_mode,
        group_col=args.group_col,
        derive_from=args.derive_from,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        drop_unknown=args.drop_unknown,
    )


if __name__ == "__main__":
    main()

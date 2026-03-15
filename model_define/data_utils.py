from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_split(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    non_feature_cols = {
        "label",
        "subject_id",
        "session_id",
        "recording_id",
        "song_id",
        "trial_id",
        "window_id",
        "group_id",
    }
    candidate_cols = [c for c in df.columns if c not in non_feature_cols]
    feature_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        raise ValueError(f"{csv_path.name} 中未找到可用数值特征列。")

    x = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    return x, y

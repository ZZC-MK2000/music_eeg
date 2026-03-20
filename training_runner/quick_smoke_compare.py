from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from music_eeg.model_define.trainers import train_eeg_chanattn, train_eeg_msresnet, train_eeg_resmlp


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    processed = root / "processed_data"
    results = root / "training_runner" / "results"
    reports_csv_dir = results / "reports" / "csv"
    results.mkdir(parents=True, exist_ok=True)
    reports_csv_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(processed / "train_features.csv").sample(n=384, random_state=42)
    val_df = pd.read_csv(processed / "val_features.csv").sample(n=192, random_state=42)
    test_df = pd.read_csv(processed / "test_features.csv").sample(n=192, random_state=42)

    feature_cols = [c for c in train_df.columns if c.startswith("f_")]

    le = LabelEncoder()
    y_train = le.fit_transform(train_df["label"].astype(str).values)
    y_val = le.transform(val_df["label"].astype(str).values)
    y_test = le.transform(test_df["label"].astype(str).values)

    x_train = train_df[feature_cols].to_numpy(np.float32)
    x_val = val_df[feature_cols].to_numpy(np.float32)
    x_test = test_df[feature_cols].to_numpy(np.float32)

    device = torch.device("cpu")
    specs = [
        ("resmlp", train_eeg_resmlp),
        ("msresnet", train_eeg_msresnet),
        ("chanattn", train_eeg_chanattn),
    ]

    rows = []
    for name, fn in specs:
        print(f"\\n=== training {name} ===")
        _, _, test_prob = fn(
            x_train=x_train,
            y_train_enc=y_train,
            x_val=x_val,
            y_val_enc=y_val,
            x_test=x_test,
            n_classes=len(le.classes_),
            seed=42,
            epochs=1,
            batch_size=64,
            learning_rate=1e-3,
            patience=1,
            device=device,
            loss_name="focal",
            use_onecycle=True,
        )
        pred = test_prob.argmax(axis=1)
        rows.append(
            {
                "model_variant": name,
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_macro_f1": float(f1_score(y_test, pred, average="macro")),
            }
        )

    summary = pd.DataFrame(rows).sort_values(["test_macro_f1", "test_accuracy"], ascending=False)
    out_csv = reports_csv_dir / "smoke_compare_summary.csv"
    summary.to_csv(out_csv, index=False)

    print("\\n=== smoke comparison summary ===")
    print(summary.to_string(index=False))
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()

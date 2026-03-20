from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from music_eeg.model_define.pipeline import run_training


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "processed_data"
    results_dir = project_root / "training_runner" / "results"
    reports_json_dir = results_dir / "reports" / "json"
    reports_csv_dir = results_dir / "reports" / "csv"
    models_sklearn_dir = results_dir / "models" / "sklearn"
    models_encoder_dir = results_dir / "models" / "encoders"
    figures_dir = results_dir / "figures"

    results_dir.mkdir(parents=True, exist_ok=True)
    reports_json_dir.mkdir(parents=True, exist_ok=True)
    reports_csv_dir.mkdir(parents=True, exist_ok=True)
    models_sklearn_dir.mkdir(parents=True, exist_ok=True)
    models_encoder_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "formal_resmlp_focal_onecycle",
            "model_variant": "resmlp",
            "deep_loss": "focal",
            "use_onecycle": True,
        },
        {
            "name": "formal_msresnet_focal_onecycle",
            "model_variant": "msresnet",
            "deep_loss": "focal",
            "use_onecycle": True,
        },
        {
            "name": "formal_chanattn_focal_onecycle",
            "model_variant": "chanattn",
            "deep_loss": "focal",
            "use_onecycle": True,
        },
    ]

    metrics: list[dict] = []
    for cfg in configs:
        run_name = cfg["name"]
        report_name = f"report_{run_name}.json"
        encoder_name = f"encoder_{run_name}.pkl"
        model_name = f"model_{run_name}.joblib"

        print(f"\n=== Running {run_name} ===")
        result = run_training(
            data_dir=str(data_dir),
            report_path=str(reports_json_dir / report_name),
            model_path=str(models_sklearn_dir / model_name),
            encoder_path=str(models_encoder_dir / encoder_name),
            seed=42,
            epochs=120,
            batch_size=128,
            learning_rate=1e-3,
            patience=30,
            robust_seed_offsets=(0, 7, 19),
            use_smote=False,
            use_calibration=False,
            cv_folds=4,
            cv_weight=0.35,
            selection_mode="val_priority",
            calibration_min_gain=0.0,
            run_classical_models=False,
            classical_profile="disabled",
            model_variant=cfg["model_variant"],
            deep_loss=cfg["deep_loss"],
            use_onecycle=cfg["use_onecycle"],
            device="cuda",
        )

        metrics.append(
            {
                "run": run_name,
                "best_model": result.get("best_model"),
                "model_variant": cfg["model_variant"],
                "deep_loss": cfg["deep_loss"],
                "use_onecycle": cfg["use_onecycle"],
                "seed": 42,
                "robust_seed_offsets": "0,7,19",
                "epochs": 120,
                "batch_size": 128,
                "patience": 30,
                "test_accuracy": result.get("test_accuracy"),
                "test_balanced_accuracy": result.get("test_balanced_accuracy"),
                "test_macro_f1": result.get("test_macro_f1"),
                "best_val_accuracy": result.get("best_val_accuracy"),
                "best_val_accuracy_std": result.get("best_val_accuracy_std"),
            }
        )

    summary_df = pd.DataFrame(metrics)
    summary_df = summary_df.sort_values(["test_macro_f1", "test_accuracy"], ascending=False)
    summary_csv = reports_csv_dir / "formal_compare_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    settings = {
        "seed": 42,
        "robust_seed_offsets": [0, 7, 19],
        "epochs": 120,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "patience": 30,
        "device": "cuda",
        "use_smote": False,
        "use_calibration": False,
        "cv_folds": 4,
        "cv_weight": 0.35,
        "selection_mode": "val_priority",
        "calibration_min_gain": 0.0,
        "run_classical_models": False,
    }
    settings_path = reports_json_dir / "formal_compare_settings.json"
    settings_path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Formal comparison summary ===")
    print(summary_df.to_string(index=False))
    print("Saved summary:", summary_csv)
    print("Saved settings:", settings_path)


if __name__ == "__main__":
    main()

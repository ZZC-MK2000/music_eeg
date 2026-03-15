from music_eeg.model_define.pipeline import run_training
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))




def main() -> None:
    data_dir = PROJECT_ROOT / "processed_data"
    results_dir = PROJECT_ROOT / "training_runner" / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        {
            "name": "resmlp_focal_onecycle",
            "model_variant": "resmlp",
            "deep_loss": "focal",
            "use_onecycle": True,
        },
        {
            "name": "msresnet_focal_onecycle",
            "model_variant": "msresnet",
            "deep_loss": "focal",
            "use_onecycle": True,
        },
    ]

    metrics = []
    for cfg in configs:
        report_name = f"run_{cfg['name']}.json"
        encoder_name = f"encoder_{cfg['name']}.pkl"
        model_name = f"model_{cfg['name']}.joblib"
        print(f"\\n=== Running {cfg['name']} ===")
        result = run_training(
            data_dir=str(data_dir),
            report_path=str(results_dir / report_name),
            model_path=str(results_dir / model_name),
            encoder_path=str(results_dir / encoder_name),
            seed=42,
            epochs=100,
            batch_size=128,
            learning_rate=1e-3,
            patience=25,
            robust_seed_offsets=(0,),
            use_smote=False,
            use_calibration=False,
            cv_folds=2,
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
                "run": cfg["name"],
                "best_model": result["best_model"],
                "test_accuracy": result["test_accuracy"],
                "test_macro_f1": result["test_macro_f1"],
            }
        )

    out_csv = results_dir / "one_click_summary.csv"
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("run,best_model,test_accuracy,test_macro_f1\\n")
        for row in metrics:
            f.write(f"{row['run']},{row['best_model']},{row['test_accuracy']:.6f},{row['test_macro_f1']:.6f}\\n")

    labels = [m["run"] for m in metrics]
    accs = [m["test_accuracy"] for m in metrics]
    f1s = [m["test_macro_f1"] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(labels, accs)
    axes[0].set_title("Test Accuracy")
    axes[0].set_ylim(0, 1)
    axes[1].bar(labels, f1s)
    axes[1].set_title("Test Macro-F1")
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig_path = figures_dir / "one_click_metrics.png"
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)

    print("\\nSaved summary:", out_csv)
    print("Saved figure:", fig_path)


if __name__ == "__main__":
    main()

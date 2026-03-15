import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "processed_data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "training_runner" / "results"

if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))


def _add_train_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--run-name", type=str, default="eeg_run")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--robust-seed-offsets", type=int, nargs="+", default=[0, 7, 19])
    parser.add_argument("--use-smote", action="store_true")
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--cv-weight", type=float, default=0.35)
    parser.add_argument("--selection-mode", type=str, default="val_priority", choices=["val_priority", "robust"])
    parser.add_argument("--calibration-min-gain", type=float, default=0.0)
    parser.add_argument("--model-variant", type=str, default="all", choices=["all", "resmlp", "msresnet"])
    parser.add_argument("--deep-loss", type=str, default="auto", choices=["auto", "ce", "focal"])
    parser.add_argument("--use-onecycle", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])


def _resolve_toggle(enable_flag: bool, disable_flag: bool, default_value: bool) -> bool:
    if disable_flag:
        return False
    if enable_flag:
        return True
    return default_value


def _run_train(args: argparse.Namespace, *, epochs: int | None = None, patience: int | None = None) -> None:
    from music_eeg.model_define.pipeline import run_training

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name
    model_path = output_dir / f"model_{run_name}.joblib"
    encoder_path = output_dir / f"encoder_{run_name}.pkl"
    report_path = output_dir / f"report_{run_name}.json"

    use_smote = _resolve_toggle(args.use_smote, args.no_smote, default_value=True)
    use_calibration = _resolve_toggle(args.use_calibration, args.no_calibration, default_value=True)

    run_training(
        data_dir=args.data_dir,
        model_path=str(model_path),
        encoder_path=str(encoder_path),
        report_path=str(report_path),
        seed=args.seed,
        epochs=args.epochs if epochs is None else epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience if patience is None else patience,
        robust_seed_offsets=tuple(args.robust_seed_offsets),
        use_smote=use_smote,
        use_calibration=use_calibration,
        cv_folds=args.cv_folds,
        cv_weight=args.cv_weight,
        selection_mode=args.selection_mode,
        calibration_min_gain=args.calibration_min_gain,
        run_classical_models=False,
        classical_profile="disabled",
        model_variant=args.model_variant,
        deep_loss=args.deep_loss,
        use_onecycle=args.use_onecycle,
        device=args.device,
    )

    print("Training outputs:")
    print("-", model_path)
    print("-", encoder_path)
    print("-", report_path)


def _run_workflow(args: argparse.Namespace) -> None:
    if args.profile == "quick":
        _run_train(args, epochs=min(args.epochs, 40), patience=min(args.patience, 12))
        return

    if args.profile == "nmedt-dl":
        args.model_variant = "resmlp"
        args.deep_loss = "focal"
        args.use_onecycle = True
        args.no_smote = True
        args.no_calibration = True
        _run_train(args)
        return

    raise ValueError(f"Unsupported workflow profile: {args.profile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EEG 训练与评估命令行（training_runner）")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_train = subparsers.add_parser("train", help="训练并评估模型")
    _add_train_options(p_train)

    p_workflow = subparsers.add_parser("workflow", help="训练预设流程")
    p_workflow.add_argument("--profile", type=str, required=True, choices=["quick", "nmedt-dl"])
    _add_train_options(p_workflow)

    args = parser.parse_args()

    if args.command == "train":
        _run_train(args)
        return

    _run_workflow(args)


if __name__ == "__main__":
    main()

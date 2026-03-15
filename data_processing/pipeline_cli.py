import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_INPUT_DIR = PROJECT_ROOT / "NMED-T"
DEFAULT_PROCESSED_DIR = PACKAGE_ROOT / "processed_data"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _add_extract_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-dir", type=str, required=True, help="包含 .mat 文件的目录")
    parser.add_argument("--output-csv", type=str, required=True, help="输出特征 CSV 路径")
    parser.add_argument("--sfreq-default", type=float, default=125.0)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=1.0)
    parser.add_argument("--eeg-key", type=str, default="")
    parser.add_argument("--min-samples", type=int, default=1024)
    parser.add_argument("--max-channels", type=int, default=256)


def _add_split_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_PROCESSED_DIR))
    parser.add_argument("--split-mode", type=str, default="subject", choices=["subject", "session", "custom"])
    parser.add_argument("--group-col", type=str, default="")
    parser.add_argument("--derive-from", type=str, default="recording_id")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-unknown", action="store_true")

def _add_extract_parser(subparsers):
    p = subparsers.add_parser("extract-nmedt", help="将 NMED-T .mat 转换为窗口特征 CSV")
    _add_extract_options(p)


def _add_split_parser(subparsers):
    p = subparsers.add_parser("split", help="按被试/会话分组划分 train/val/test")
    _add_split_options(p)


def _add_workflow_parser(subparsers):
    p = subparsers.add_parser("workflow", help="按预设流程执行数据处理：NMED-T 主流程")
    p.add_argument("--profile", type=str, required=True, choices=["nmedt", "nmedt-dl"])

    # NMED-T extract/split options.
    p.add_argument("--input-dir", type=str, default=str(DEFAULT_RAW_INPUT_DIR), help="NMED-T .mat 目录（profile=nmedt 时使用）")
    p.add_argument("--feature-csv", type=str, default=str(DEFAULT_PROCESSED_DIR / "nmedt_features.csv"))
    p.add_argument("--sfreq-default", type=float, default=125.0)
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--step-sec", type=float, default=1.0)
    p.add_argument("--eeg-key", type=str, default="")
    p.add_argument("--min-samples", type=int, default=1024)
    p.add_argument("--max-channels", type=int, default=256)
    p.add_argument("--split-mode", type=str, default="subject", choices=["subject", "session", "custom"])
    p.add_argument("--group-col", type=str, default="")
    p.add_argument("--derive-from", type=str, default="recording_id")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--drop-unknown", action="store_true")
    p.add_argument("--skip-extract", action="store_true")
    p.add_argument("--skip-split", action="store_true")



def _run_extract(args: argparse.Namespace) -> None:
    from nmedt_mat_to_features import convert_mat_to_feature_csv

    convert_mat_to_feature_csv(
        input_dir=Path(args.input_dir),
        output_csv=Path(args.output_csv),
        sfreq_default=args.sfreq_default,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        eeg_key=args.eeg_key,
        min_samples=args.min_samples,
        max_channels=args.max_channels,
    )


def _run_split(args: argparse.Namespace) -> None:
    from split_grouped_generalization import split_grouped_features

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


def _run_workflow(args: argparse.Namespace) -> None:
    from nmedt_mat_to_features import convert_mat_to_feature_csv
    from split_grouped_generalization import split_grouped_features

    if args.profile in {"nmedt", "nmedt-dl"}:
        feature_csv = Path(args.feature_csv)
        output_dir = Path(DEFAULT_PROCESSED_DIR)
        input_dir = Path(args.input_dir)

        window_sec = args.window_sec
        step_sec = args.step_sec
        if args.profile == "nmedt-dl":
            # Input-side defaults for stronger temporal context.
            if args.window_sec == 2.0:
                window_sec = 4.0
            if args.step_sec == 1.0:
                step_sec = 0.5

        if not input_dir.exists():
            raise FileNotFoundError(f"未找到原始数据目录: {input_dir}")

        if not args.skip_extract:
            convert_mat_to_feature_csv(
                input_dir=input_dir,
                output_csv=feature_csv,
                sfreq_default=args.sfreq_default,
                window_sec=window_sec,
                step_sec=step_sec,
                eeg_key=args.eeg_key,
                min_samples=args.min_samples,
                max_channels=args.max_channels,
            )

        if not args.skip_split:
            split_grouped_features(
                input_csv=feature_csv,
                output_dir=output_dir,
                split_mode=args.split_mode,
                group_col=args.group_col,
                derive_from=args.derive_from,
                test_size=args.test_size,
                val_size=args.val_size,
                seed=args.seed,
                drop_unknown=args.drop_unknown,
            )
        return


def main():
    parser = argparse.ArgumentParser(description="EEG 原始数据处理命令行（仅提取与划分）")
    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_extract_parser(subparsers)
    _add_split_parser(subparsers)
    _add_workflow_parser(subparsers)

    args = parser.parse_args()

    if args.command == "extract-nmedt":
        _run_extract(args)
        return

    if args.command == "split":
        _run_split(args)
        return

    if args.command == "workflow":
        _run_workflow(args)
        return


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _moved_message() -> str:
    return (
        "训练入口已迁移到 training_runner。\n"
        "请使用：python ../training_runner/train_cli.py train --data-dir ../processed_data ..."
    )


if __name__ == "__main__":
    print(_moved_message())

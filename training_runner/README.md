# music_eeg: Training Runner Guide

This document explains the current project layout and how to run the full workflow after the refactor.

Project intro: this repository now focuses on EEG deep-learning pipelines only (ResMLP, MSResNet, ChanAttn). Legacy classical-ML tuning switches were removed from the training CLI.

## Project structure (high level)

```text
music_eeg/
├── data_processing/          # raw-data processing only (extract/split)
├── processed_data/           # feature-engineering outputs only
├── model_define/             # model definitions + training pipeline
└── training_runner/          # training/eval entrypoints, notebooks, results
	├── train_cli.py
	├── run_train_eval_report.py
	├── notebooks/
	└── results/
	    ├── reports/json/       # report_*.json, run_*.json, settings json
	    ├── reports/csv/        # comparison/ablation/smoke/formal summary csv
	    ├── models/sklearn/     # model_*.joblib
	    ├── models/encoders/    # encoder_*.pkl, label encoders
	    ├── models/checkpoints/ # best_*.pt
	    └── figures/            # exported png charts
```

## Folder responsibilities

- `data_processing`: convert NMED-T `.mat` to features, then split into `train/val/test`.
- `processed_data`: keep only feature csv/summary files.
- `model_define`: central model/pipeline code (`models.py`, `trainers.py`, `pipeline.py`).
- `training_runner`: where training is started, evaluated, and where outputs are written.

## Quick start

Run all commands from project root `music_eeg/`.

1) Prepare features from raw data:

```bash
python data_processing/pipeline_cli.py workflow --profile nmedt --input-dir ../NMED-T --feature-csv processed_data/nmedt_features.csv
```

2) Train one experiment:

```bash
python training_runner/train_cli.py train --data-dir processed_data --output-dir training_runner/results --run-name msresnet_focal --device cuda --model-variant msresnet --deep-loss focal --use-onecycle --epochs 100 --patience 25 --batch-size 128 --robust-seed-offsets 0
```

3) Run one-click dual-model comparison:

```bash
python training_runner/run_train_eval_report.py
```

4) Open analysis notebook:

- `training_runner/notebooks/comparison_report.ipynb`

## Important notes

- Training commands are no longer in `data_processing/pipeline_cli.py`.
- Use `training_runner/train_cli.py` for all training/evaluation CLI runs.
- New outputs are grouped by type under `training_runner/results/reports`, `training_runner/results/models`, and `training_runner/results/figures`.

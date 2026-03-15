# data_processing

This folder is dedicated to raw data processing only.

## Responsibilities

- Convert raw NMED-T `.mat` files to feature tables.
- Split feature tables into train/val/test.
- Data-only workflow orchestration.

## CLI

```bash
python data_processing/pipeline_cli.py extract-nmedt --input-dir ../NMED-T --output-csv ../processed_data/nmedt_features.csv
```

```bash
python data_processing/pipeline_cli.py split --input-csv ../processed_data/nmedt_features.csv --output-dir ../processed_data --split-mode subject
```

```bash
python data_processing/pipeline_cli.py workflow --profile nmedt --input-dir ../NMED-T --feature-csv ../processed_data/nmedt_features.csv
```

Training commands were moved to `training_runner/train_cli.py`.

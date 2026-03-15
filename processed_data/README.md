# processed_data

This folder stores generated artifacts from the EEG processing/training pipeline.

Typical contents:
- extracted features (`nmedt_features*.csv`)
- dataset split files (`train_features.csv`, `val_features.csv`, `test_features.csv`)
- training reports and models (`*.json`, `*.joblib`, `*.pt`, `*.pkl`)

Raw dataset files should remain in `../NMED-T` (project root).

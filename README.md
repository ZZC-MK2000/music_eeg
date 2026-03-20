# music_eeg

音乐 EEG 实验与深度学习分析项目。

该仓库包含从实验数据采集、原始数据处理到模型训练与评估的完整流程，适合做音乐情绪相关 EEG 研究与复现实验。

## 项目功能

- 实验采集（`data_collector`）
  - 播放音乐刺激（classical/jazz/rock/ambient）
  - 记录实验序列、评分与事件时间戳
  - 支持 Trigger 设备联动
- 数据处理（`data_processing`）
  - 将 NMED-T 的 `.mat` 原始 EEG 数据转为窗口特征 CSV
  - 支持按 subject/session 分组划分 train/val/test，降低泄漏风险
- 模型定义（`model_define`）
  - 提供 ResMLP、MSResNet、ChanAttn 等模型与训练/评估 pipeline
- 训练与报告（`training_runner`）
  - 一键训练、对比评估、消融分析
  - 输出 JSON/CSV 报告、模型权重、图表

## 项目结构

```text
music_eeg/
├── README.md
├── requirements.txt
├── requirements.collector.txt
├── requirements.processing.txt
├── data_collector/              # 实验采集程序
│   ├── main.py
│   ├── materials/music/
│   ├── save/                    # 本地采集结果（已在 .gitignore 中忽略）
│   └── utils/
├── data_processing/             # 原始数据处理（提取/划分）
│   ├── pipeline_cli.py
│   ├── nmedt_mat_to_features.py
│   └── split_grouped_generalization.py
├── model_define/                # 模型与训练核心实现
│   ├── models.py
│   ├── trainers.py
│   ├── pipeline.py
│   └── evaluation.py
├── training_runner/             # 训练入口与结果管理
│   ├── train_cli.py
│   ├── run_train_eval_report.py
│   ├── notebooks/
│   └── results/                 # 训练输出（已在 .gitignore 中忽略）
└── processed_data/              # 特征产物目录（已在 .gitignore 中忽略）
```

## 环境要求

- Python 3.10+（建议）
- Windows（采集模块包含串口 Trigger、PsychoPy 依赖）

## 安装依赖

在仓库根目录执行：

```bash
pip install -r requirements.txt
```

说明：

- `requirements.collector.txt`：实验采集依赖（PsychoPy、串口等）
- `requirements.processing.txt`：处理/训练依赖（pandas、torch、mne 等）

## 使用说明

### 1) 实验数据采集

```bash
python data_collector/main.py
```

默认会在 `data_collector/save/` 生成：

- `experiment_sequence.csv`
- `music_ratings.csv`
- `music_timestamps.csv`

### 2) 原始数据处理（NMED-T）

先提取特征：

```bash
python data_processing/pipeline_cli.py extract-nmedt \
  --input-dir D:/datasets/NMEDT \
  --output-csv processed_data/nmedt_features.csv \
  --window-sec 2 --step-sec 1
```

再做分组划分：

```bash
python data_processing/pipeline_cli.py split \
  --input-csv processed_data/nmedt_features.csv \
  --output-dir processed_data \
  --split-mode subject \
  --drop-unknown
```

或直接跑处理工作流：

```bash
python data_processing/pipeline_cli.py workflow \
  --profile nmedt \
  --input-dir D:/datasets/NMEDT \
  --feature-csv processed_data/nmedt_features.csv
```

### 3) 模型训练与评估

训练单次实验：

```bash
python training_runner/train_cli.py train \
  --data-dir processed_data \
  --output-dir training_runner/results \
  --run-name msresnet_focal \
  --model-variant msresnet \
  --deep-loss focal \
  --use-onecycle \
  --epochs 100
```

快速预设工作流：

```bash
python training_runner/train_cli.py workflow --profile quick
```

NMED-T 深度学习预设：

```bash
python training_runner/train_cli.py workflow --profile nmedt-dl
```

一键训练+报告：

```bash
python training_runner/run_train_eval_report.py
```

## 输出说明

- 训练报告：`training_runner/results/reports/json/`、`training_runner/results/reports/csv/`
- 模型文件：`training_runner/results/models/`
- 图表：`training_runner/results/figures/`

## Git 与大文件说明

为避免仓库体积膨胀，以下目录默认忽略：

- `processed_data/`
- `training_runner/results/`
- `data_collector/save/`

建议将大规模数据与模型产物保存在本地或对象存储中，仓库仅保留代码与必要文档。

## 相关文档

- `data_collector/README.md`：采集模块说明
- `data_processing/README.md`：数据处理说明
- `training_runner/README.md`：训练与评估说明

# Music EEG Experiment System

## 项目概述
该项目用于构建音乐EEG实验范式，通过播放不同类型的音乐刺激并记录被试的脑电信号和主观评分，研究音乐感知的神经机制。系统包含实验程序运行模块和数据收集模块。

## 项目结构
```
data_collector/            # 实验程序主目录
├── main.py                # 实验主程序入口
├── README.md              # 项目文档
├── requirements.txt       # Python依赖列表
├── materials/             # 实验材料
│   └── music/             # 音乐文件目录
├── save/                  # 实验数据保存目录
│   ├── experiment_sequence.csv  # 实验序列记录
│   ├── music_ratings.csv        # 音乐评分数据
│   └── music_timestamps.csv     # 事件时间戳记录
└── utils/                 # 工具模块
    ├── MusicPlayer.py     # 音乐播放控制
    ├── ScoreMusic.py      # 音乐评分组件
    ├── Trigger.py         # 触发器控制
    └── ...                # 其他工具模块

data_processing/           # 原始数据处理目录（仅提取/划分）
├── pipeline_cli.py        # 数据处理命令行（extract/split/workflow）
├── nmedt_mat_to_features.py         # NMED-T .mat 转窗口特征 CSV
├── split_grouped_generalization.py  # 跨被试/跨会话分组划分
└── README.md              # 数据处理说明

model_define/              # 模型定义与训练 pipeline
├── models.py              # 深度模型结构
├── data_utils.py          # 数据加载与特征列选择
├── trainers.py            # 训练逻辑
├── evaluation.py          # 消融与显著性检验
└── pipeline.py            # 训练主流程

training_runner/           # 一键训练、评估与图表产出
├── train_cli.py           # 训练命令行入口
├── run_train_eval_report.py  # 一键训练+汇总
├── notebooks/             # 训练/评估 notebook
└── results/               # 报告、模型与图表

processed_data/            # 仅存放特征工程产物（csv/summary）
```

## 配置指南
1. **音乐文件配置**：
   - 将音乐文件按类型放入`materials/music/`目录
   - 支持的格式：MP3, WAV

2. **实验参数配置**：
   修改`main.py`中的params字典：
   ```python
   params = {
       'music_files': {
           'classical': ['materials/music/classical.mp3'],
           # 添加其他音乐类型...
       },
       'trigger_index': {
           'classical': 1,
           # 添加其他音乐类型的trigger索引...
       },
       'music_duration': 30,  # 音乐播放时长(秒)
       'rest_duration': 10,   # 试次间休息时长(秒)
       'total_blocks': 1,     # 实验块数
       'trials_per_genre': 1  # 每块中每种音乐类型的试次数
   }
   ```

3. **Trigger设备配置**：
   - 默认使用COM5端口
   - 如需修改，编辑`utils/MusicPlayer.py`中的ActiviewTrigger初始化

## 运行说明
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 启动实验：
   ```bash
   python main.py
   ```

3. 实验流程：
   - 显示开始界面
   - 按顺序播放配置的音乐
   - 每段音乐结束后进行三维评分（唤醒度、效价、喜好度）
   - 实验结束后自动保存数据

## 数据处理统一命令行
在项目根目录运行以下命令：

1. 查看帮助：
   ```bash
   python data_processing/pipeline_cli.py --help
   ```

2. NMED-T .mat 转窗口特征 CSV：
   ```bash
   python data_processing/pipeline_cli.py extract-nmedt \
     --input-dir D:/datasets/NMEDT \
     --output-csv data_processing/nmedt_features.csv \
     --window-sec 2 --step-sec 1
   ```

3. 跨被试划分（无泄漏）：
   ```bash
   python data_processing/pipeline_cli.py split \
     --input-csv data_processing/nmedt_features.csv \
     --output-dir data_processing \
     --split-mode subject \
     --drop-unknown
   ```

4. 训练并输出报告：
   ```bash
   python data_processing/pipeline_cli.py train \
     --data-dir data_processing \
     --epochs 160 \
     --batch-size 16
   ```

说明：
1. 如果特征 CSV 中没有 subject_id/session_id，可在 split 命令里使用 --derive-from recording_id 自动推断。
2. 训练报告会输出主结果、消融实验、显著性检验，便于论文写作。

## 快速使用说明
下面给出两条最常用流程，可直接复制执行。

### 数据策略
1. 主流程：使用 NMED-T 数据集，作为论文主实验与正式结果来源。
2. 快速验证：保留当前自采小数据集，用于功能联调和快速回归检查。

推荐做法：开发阶段先跑 quick；准备论文结果时跑 nmedt。

### 流程A：使用 NMED-T 数据集
1. 提取窗口特征：
   ```bash
   python data_processing/pipeline_cli.py extract-nmedt \
     --input-dir D:/datasets/NMEDT \
     --output-csv data_processing/nmedt_features.csv \
     --window-sec 2 --step-sec 1
   ```

2. 按被试划分（推荐论文评测）：
   ```bash
   python data_processing/pipeline_cli.py split \
     --input-csv data_processing/nmedt_features.csv \
     --output-dir data_processing \
     --split-mode subject \
     --drop-unknown
   ```

3. 训练并生成报告：
   ```bash
   python data_processing/pipeline_cli.py train \
     --data-dir data_processing \
     --epochs 160 --batch-size 16
   ```

### 流程B：使用你自己的特征 CSV
如果你已经有完整特征文件（含 label 列），可从划分步骤开始：

1. 分组划分：
   ```bash
   python data_processing/pipeline_cli.py split \
     --input-csv data_processing/your_features.csv \
     --output-dir data_processing \
     --split-mode session \
     --drop-unknown
   ```

2. 模型训练：
   ```bash
   python data_processing/pipeline_cli.py train --data-dir data_processing
   ```

运行完成后，`data_processing/` 下会生成：
1. `eeg_1dcnn_report.json`：主结果、消融、显著性检验。
2. `best_model.joblib` 或 `best_*.pt`：最佳模型文件。
3. `label_encoder.pkl`：标签编码器。

## 一键工作流（推荐）
统一入口支持两种 profile：

1. `nmedt`：从 `.mat` 提取特征 -> 分组划分 -> 训练。
2. `quick`：直接使用现成 `train/val/test_features.csv` 快速训练验证。

### 1) NMED-T 主流程（论文主实验）
```bash
python data_processing/pipeline_cli.py workflow \
   --profile nmedt \
   --input-dir D:/datasets/NMEDT \
   --feature-csv data_processing/nmedt_features.csv \
   --data-dir data_processing \
   --split-mode subject \
   --drop-unknown \
   --epochs 160
```

### 2) 小数据快速验证（开发调试）
```bash
python data_processing/pipeline_cli.py workflow \
   --profile quick \
   --data-dir data_processing \
   --epochs 40
```

提示：
1. `quick` 模式会自动把训练轮数和早停阈值收敛到更快的默认范围，便于快速验证。
2. `nmedt` 模式可用 `--skip-extract` 或 `--skip-split` 复用已有中间结果。

## 数据格式
1. **experiment_sequence.csv**：
   - 列：Block, Genre, Music_File, Trigger_Index
   - 记录实验试次顺序和参数

2. **music_ratings.csv**：
   - 列：Trial_Num, Block, Genre, Music_File, Arousal, Valence, Liking
   - 记录被试对每段音乐的主观评分（1-9分）

3. **music_timestamps.csv**：
   - 列：Event, Timestamp, Unix_Timestamp, Trigger_Index, Trigger_Sent, Music_Type, Music_File
   - 记录实验事件精确时间戳

## 常见问题
1. **音乐文件无法播放**：
   - 检查文件路径是否正确
   - 确认文件格式为MP3或WAV

2. **Trigger连接失败**：
   - 检查COM端口是否正确
   - 确认Trigger设备已连接

3. **数据未保存**：
   - 确保save/目录存在且有写入权限
   - 检查实验是否正常完成（未提前退出）
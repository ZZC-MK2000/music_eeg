# 实验结果表模板（可直接贴正文）

## 表1 主结果（跨被试/跨会话泛化）

表1展示不同模型在 NMED-T 数据集上的主结果。所有结果均基于分组无泄漏划分，报告多随机种子平均值与标准差。

| 模型 | 评测协议 | Val Acc (mean±std) | Test Acc | Test Balanced Acc | Test Macro-F1 | 备注 |
|---|---|---:|---:|---:|---:|---|
| EEG-CNN | LOSO/LOSO-Session | xx.xx±x.xx | xx.xx | xx.xx | xx.xx | 本文实现 |
| EEG-ResMLP | LOSO/LOSO-Session | xx.xx±x.xx | xx.xx | xx.xx | xx.xx | 本文实现 |
| SVM (RBF) | LOSO/LOSO-Session | xx.xx±x.xx | xx.xx | xx.xx | xx.xx | PCA+标准化 |
| RandomForest | LOSO/LOSO-Session | xx.xx±x.xx | xx.xx | xx.xx | xx.xx | class-weight |
| ExtraTrees | LOSO/LOSO-Session | xx.xx±x.xx | xx.xx | xx.xx | xx.xx | class-weight |
| MLP (sklearn) | LOSO/LOSO-Session | xx.xx±x.xx | xx.xx | xx.xx | xx.xx | PCA+标准化 |
| LogisticRegression | LOSO/LOSO-Session | xx.xx±x.xx | xx.xx | xx.xx | xx.xx | 基线模型 |
| Top-3 Weighted Ensemble | LOSO/LOSO-Session | xx.xx±x.xx | xx.xx | xx.xx | xx.xx | 集成模型 |

注：
1. LOSO 表示 Leave-One-Subject-Out，LOSO-Session 表示 Leave-One-Session-Out。
2. 主要比较指标建议以 Test Balanced Acc 和 Test Macro-F1 为主，避免类别不平衡导致偏差。

---

## 表2 消融实验（特征工程贡献）

表2用于验证各类特征对性能的贡献，建议固定分类器为 Logistic Regression 或你最终最佳传统模型。

| 特征配置 | 特征维度 | Val Acc (mean±std) | Val Balanced Acc | Val Macro-F1 | 相对全特征变化 |
|---|---:|---:|---:|---:|---:|
| All Features | xxx | xx.xx±x.xx | xx.xx | xx.xx | 0.00 |
| Only Time Domain | xxx | xx.xx±x.xx | xx.xx | xx.xx | -x.xx |
| Time + Hjorth | xxx | xx.xx±x.xx | xx.xx | xx.xx | -x.xx |
| Time + Frequency | xxx | xx.xx±x.xx | xx.xx | xx.xx | -x.xx |
| Without Hjorth | xxx | xx.xx±x.xx | xx.xx | xx.xx | -x.xx |
| Without Connectivity (corr_stats) | xxx | xx.xx±x.xx | xx.xx | xx.xx | -x.xx |
| Without Covariance Eigen Spectrum | xxx | xx.xx±x.xx | xx.xx | xx.xx | -x.xx |

注：
1. 相对全特征变化 = 当前配置 Val Acc mean - All Features Val Acc mean。
2. 可在文中强调“删除后下降最大”的特征组，作为方法有效性依据。

---

## 表3 显著性检验（最佳模型 vs 其他模型）

表3用于说明最佳模型提升不是随机波动。建议基于同一组随机种子结果做配对检验。

| 最佳模型 | 对比模型 | 检验方法 | p-value | 是否显著 (alpha=0.05) | 最佳模型均值 | 对比模型均值 |
|---|---|---|---:|---|---:|---:|
| BestModelName | BaselineA | Wilcoxon / paired t-test | 0.xxxx | Yes/No | xx.xx | xx.xx |
| BestModelName | BaselineB | Wilcoxon / paired t-test | 0.xxxx | Yes/No | xx.xx | xx.xx |
| BestModelName | BaselineC | Wilcoxon / paired t-test | 0.xxxx | Yes/No | xx.xx | xx.xx |

注：
1. 若差值分布不满足正态，优先使用 Wilcoxon signed-rank test。
2. 建议正文中同时报告效应方向（最佳模型均值 - 对比模型均值）。

---

## 结果文字模板（可直接改数字）

在 NMED-T 数据集上，我们采用分组无泄漏评测协议（跨被试/跨会话）对比了深度模型与传统机器学习模型。主结果见表1。最佳模型在 Test Balanced Acc 与 Test Macro-F1 上均优于其他方法，说明模型在类别不平衡条件下仍具备较好的泛化能力。消融实验（表2）显示，去除 Hjorth 与跨通道连接性相关特征会带来明显性能下降，表明这些特征对音乐 EEG 识别任务具有关键贡献。显著性检验（表3）进一步表明最佳模型相对主要基线的提升在统计上显著（p<0.05）。

---

## 图表编号建议

1. 图1：整体方法流程图（预处理-特征-分组划分-训练评估）。
2. 图2：最佳模型混淆矩阵。
3. 图3：消融实验柱状图（Val Acc 或 Balanced Acc）。
4. 表1：主结果。
5. 表2：消融实验。
6. 表3：显著性检验。

import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from .data_utils import load_split
from .evaluation import pairwise_significance, run_feature_ablation
from .trainers import (
    resolve_torch_device,
    train_eeg_msresnet,
    train_eeg_resmlp,
)


def run_training(
    data_dir: str = ".",
    model_path: str = "best_model.joblib",
    encoder_path: str = "label_encoder.pkl",
    report_path: str = "eeg_1dcnn_report.json",
    seed: int = 42,
    epochs: int = 160,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    patience: int = 25,
    robust_seed_offsets: Tuple[int, ...] = (0, 7, 19),
    use_smote: bool = True,
    use_calibration: bool = True,
    cv_folds: int = 4,
    cv_weight: float = 0.35,
    selection_mode: str = "val_priority",
    calibration_min_gain: float = 0.0,
    device: str = "auto",
    run_classical_models: bool = True,
    classical_profile: str = "full",
    model_variant: str = "all",
    deep_loss: str = "auto",
    use_onecycle: bool = False,
):
    data_root = Path(data_dir)
    train_csv = data_root / "train_features.csv"
    val_csv = data_root / "val_features.csv"
    test_csv = data_root / "test_features.csv"

    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("未找到 train/val/test_features.csv，请先运行数据划分单元。")

    np.random.seed(seed)
    robust_seeds = [seed + x for x in robust_seed_offsets]
    print(f"鲁棒评估随机种子: {robust_seeds}")
    torch_device = resolve_torch_device(device)

    x_train, y_train = load_split(train_csv)
    x_val, y_val = load_split(val_csv)
    x_test, y_test = load_split(test_csv)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    candidates: List[Dict] = []

    print(f"\n===== 训练候选模型：EEG 深度网络（variant={model_variant}, loss={deep_loss}, onecycle={use_onecycle}） =====")

    def _collect_deep_candidate(name: str, trainer, default_loss: str = "ce", default_onecycle: bool = False):
        selected_loss = default_loss if deep_loss == "auto" else deep_loss
        selected_onecycle = use_onecycle or default_onecycle
        states = []
        val_probs = []
        test_probs = []
        val_accs = []
        for sd in robust_seeds:
            st, val_prob, test_prob = trainer(
                x_train=x_train,
                y_train_enc=y_train_enc,
                x_val=x_val,
                y_val_enc=y_val_enc,
                x_test=x_test,
                n_classes=len(label_encoder.classes_),
                seed=sd,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                patience=patience,
                device=torch_device,
                loss_name=selected_loss,
                use_onecycle=selected_onecycle,
            )
            states.append(st)
            val_probs.append(val_prob)
            test_probs.append(test_prob)
            val_accs.append(accuracy_score(y_val_enc, val_prob.argmax(axis=1)))

        val_prob = np.mean(val_probs, axis=0)
        test_prob = np.mean(test_probs, axis=0)
        val_acc = float(np.mean(val_accs))
        val_acc_std = float(np.std(val_accs))
        candidates.append(
            {
                "name": name,
                "val_acc": val_acc,
                "robust_score_mean": val_acc,
                "val_acc_std": val_acc_std,
                "val_prob": val_prob,
                "test_prob": test_prob,
                "seed_val_accs": [float(x) for x in val_accs],
                "artifact": {"type": "torch", "state": states[0]},
            }
        )
        print(f"[{name}] val_acc_mean={val_acc:.4f} | val_acc_std={val_acc_std:.4f}")

    if model_variant in {"all", "resmlp"}:
        _collect_deep_candidate("eeg_resmlp", train_eeg_resmlp)

    if model_variant in {"all", "msresnet"}:
        _collect_deep_candidate("eeg_msresnet", train_eeg_msresnet, default_loss="focal", default_onecycle=True)

    print("\n===== 传统机器学习已禁用 =====")

    ranked = sorted(candidates, key=lambda x: x["val_acc"], reverse=True)
    print("\n===== 验证集排名 =====")
    for idx, cand in enumerate(ranked, start=1):
        print(f"{idx:>2}. {cand['name']} | val_acc={cand['val_acc']:.4f} | std={cand.get('val_acc_std', 0.0):.4f}")

    top_k = min(3, len(ranked))
    top_candidates = ranked[:top_k]
    weights = np.array([max(c["val_acc"], 1e-6) for c in top_candidates], dtype=np.float32)
    weights = weights / weights.sum()
    ens_val_prob = np.zeros_like(top_candidates[0]["val_prob"])
    ens_test_prob = np.zeros_like(top_candidates[0]["test_prob"])
    for w, cand in zip(weights, top_candidates):
        ens_val_prob += w * cand["val_prob"]
        ens_test_prob += w * cand["test_prob"]

    ens_val_acc = accuracy_score(y_val_enc, ens_val_prob.argmax(axis=1))
    ensemble_candidate = {
        "name": "ensemble_top3_weighted",
        "val_acc": ens_val_acc,
        "val_acc_std": float(np.std([c["val_acc"] for c in top_candidates])),
        "val_prob": ens_val_prob,
        "test_prob": ens_test_prob,
        "artifact": {"type": "ensemble", "members": [c["name"] for c in top_candidates], "weights": weights.tolist()},
    }
    candidates.append(ensemble_candidate)
    print(f"[ensemble_top3_weighted] val_acc={ens_val_acc:.4f}")

    ablation_results = run_feature_ablation(x_train=x_train, y_train=y_train_enc, x_val=x_val, y_val=y_val_enc, seeds=robust_seeds)
    if ablation_results:
        print("\n===== 特征消融（LogReg 基线） =====")
        for row in ablation_results:
            print(
                f"{row['name']:<22} acc={row['val_acc_mean']:.4f} +/- {row['val_acc_std']:.4f} | "
                f"bacc={row['val_balanced_acc_mean']:.4f} | macro_f1={row['val_macro_f1_mean']:.4f}"
            )

    significance_results = pairwise_significance(candidates)

    best_candidate = sorted(candidates, key=lambda x: x["val_acc"], reverse=True)[0]
    print(f"\n最终选中模型: {best_candidate['name']} (val_acc={best_candidate['val_acc']:.4f})")

    if best_candidate["artifact"]["type"] == "sklearn":
        joblib.dump(
            {
                "model_type": best_candidate["name"],
                "model": best_candidate["artifact"]["model"],
                "label_classes": label_encoder.classes_.tolist(),
            },
            data_root / model_path,
        )
    elif best_candidate["artifact"]["type"] == "torch":
        torch.save(best_candidate["artifact"]["state"], data_root / f"best_{best_candidate['name']}.pt")
    else:
        with open(data_root / "best_ensemble_meta.json", "w", encoding="utf-8") as f:
            json.dump(best_candidate["artifact"], f, ensure_ascii=False, indent=2)

    joblib.dump(label_encoder, data_root / encoder_path)

    test_pred_enc = best_candidate["test_prob"].argmax(axis=1)
    test_acc = accuracy_score(y_test_enc, test_pred_enc)
    test_balanced_acc = balanced_accuracy_score(y_test_enc, test_pred_enc)
    test_macro_f1 = f1_score(y_test_enc, test_pred_enc, average="macro", zero_division=0)
    report = classification_report(y_test_enc, test_pred_enc, target_names=label_encoder.classes_, output_dict=True, zero_division=0)
    conf_mat = confusion_matrix(y_test_enc, test_pred_enc)

    result = {
        "best_model": best_candidate["name"],
        "best_val_accuracy": float(best_candidate["val_acc"]),
        "best_robust_score": float(best_candidate.get("robust_score_mean", best_candidate["val_acc"])),
        "best_val_accuracy_std": float(best_candidate.get("val_acc_std", 0.0)),
        "test_accuracy": float(test_acc),
        "test_balanced_accuracy": float(test_balanced_acc),
        "test_macro_f1": float(test_macro_f1),
        "robust_seeds": robust_seeds,
        "use_smote": use_smote,
        "use_calibration": use_calibration,
        "cv_folds": cv_folds,
        "cv_weight": cv_weight,
        "selection_mode": selection_mode,
        "calibration_min_gain": calibration_min_gain,
        "run_classical_models": run_classical_models,
        "classical_profile": classical_profile,
        "model_variant": model_variant,
        "deep_loss": deep_loss,
        "use_onecycle": use_onecycle,
        "classes": label_encoder.classes_.tolist(),
        "confusion_matrix": conf_mat.tolist(),
        "classification_report": report,
        "significance_vs_best": significance_results,
        "feature_ablation": ablation_results,
        "evaluation_notes": {
            "split_warning": "当前评估依赖现有 train/val/test csv。若窗口来自同一连续段，可能高估泛化性能。建议补充被试级/会话级划分。",
            "paper_tip": "建议在论文正文报告 acc、balanced acc、macro F1（mean±std）并附显著性检验。",
        },
        "all_candidates": [
            {
                "name": c["name"],
                "val_acc": float(c["val_acc"]),
                "robust_score_mean": float(c.get("robust_score_mean", c["val_acc"])),
                "val_acc_std": float(c.get("val_acc_std", 0.0)),
                "cv_acc_mean": float(c.get("cv_acc_mean", 0.0)),
            }
            for c in sorted(candidates, key=lambda x: x["val_acc"], reverse=True)
        ],
    }

    with open(data_root / report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== 测试结果 ===")
    print(f"最佳模型: {best_candidate['name']}")
    print(f"测试集准确率: {test_acc:.4f}")
    print(f"测试集平衡准确率: {test_balanced_acc:.4f}")
    print(f"测试集Macro-F1: {test_macro_f1:.4f}")
    print("分类报告:")
    print(classification_report(y_test_enc, test_pred_enc, target_names=label_encoder.classes_, zero_division=0))
    print("混淆矩阵:")
    print(conf_mat)

    return result

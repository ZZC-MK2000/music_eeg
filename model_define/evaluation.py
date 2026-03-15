from typing import Dict, List, Optional

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def infer_channel_count(feature_dim: int) -> Optional[int]:
    if feature_dim <= 15:
        return None
    remainder = feature_dim - 15
    if remainder % 15 != 0:
        return None
    channels = remainder // 15
    return channels if channels > 0 else None


def build_feature_groups(feature_dim: int) -> Optional[Dict[str, np.ndarray]]:
    channels = infer_channel_count(feature_dim)
    if channels is None:
        return None

    groups: Dict[str, np.ndarray] = {}
    idx = 0

    def make_mask(start: int, end: int) -> np.ndarray:
        mask = np.zeros(feature_dim, dtype=bool)
        mask[start:end] = True
        return mask

    groups["time_domain"] = make_mask(idx, idx + 6 * channels)
    idx += 6 * channels
    groups["hjorth"] = make_mask(idx, idx + 3 * channels)
    idx += 3 * channels
    groups["spectral_entropy"] = make_mask(idx, idx + channels)
    idx += channels
    groups["band_power"] = make_mask(idx, idx + 5 * channels)
    idx += 5 * channels
    groups["ratio_feats"] = make_mask(idx, idx + 4)
    idx += 4
    groups["corr_stats"] = make_mask(idx, idx + 3)
    idx += 3
    groups["eig_spectrum"] = make_mask(idx, idx + 8)

    return groups


def run_feature_ablation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seeds: List[int],
) -> List[Dict]:
    groups = build_feature_groups(x_train.shape[1])
    if groups is None:
        return []

    all_mask = np.ones(x_train.shape[1], dtype=bool)
    time_mask = groups["time_domain"]
    hjorth_mask = groups["hjorth"]
    freq_mask = groups["spectral_entropy"] | groups["band_power"] | groups["ratio_feats"]

    configs = [
        ("all_features", all_mask),
        ("only_time", time_mask),
        ("time_plus_hjorth", time_mask | hjorth_mask),
        ("time_plus_freq", time_mask | freq_mask),
        ("without_hjorth", all_mask & (~hjorth_mask)),
        ("without_connectivity", all_mask & (~groups["corr_stats"])),
        ("without_cov_eigs", all_mask & (~groups["eig_spectrum"])),
    ]

    results = []
    for name, mask in configs:
        xtr = x_train[:, mask]
        xva = x_val[:, mask]

        per_seed_acc = []
        per_seed_bacc = []
        per_seed_f1 = []
        for sd in seeds:
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            C=1.5,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=sd,
                        ),
                    ),
                ]
            )
            model.fit(xtr, y_train)
            pred = model.predict(xva)
            per_seed_acc.append(float(accuracy_score(y_val, pred)))
            per_seed_bacc.append(float(balanced_accuracy_score(y_val, pred)))
            per_seed_f1.append(float(f1_score(y_val, pred, average="macro", zero_division=0)))

        results.append(
            {
                "name": name,
                "n_features": int(mask.sum()),
                "val_acc_mean": float(np.mean(per_seed_acc)),
                "val_acc_std": float(np.std(per_seed_acc)),
                "val_balanced_acc_mean": float(np.mean(per_seed_bacc)),
                "val_macro_f1_mean": float(np.mean(per_seed_f1)),
                "seed_val_accs": [float(x) for x in per_seed_acc],
            }
        )

    return sorted(results, key=lambda x: x["val_acc_mean"], reverse=True)


def pairwise_significance(candidates: List[Dict], alpha: float = 0.05) -> List[Dict]:
    ranked = sorted(candidates, key=lambda x: x["val_acc"], reverse=True)
    if not ranked:
        return []

    best = ranked[0]
    best_scores = best.get("seed_val_accs")
    if not best_scores or len(best_scores) < 2:
        return []

    best_scores_arr = np.array(best_scores, dtype=np.float64)
    outputs = []

    for cand in ranked[1:]:
        cand_scores = cand.get("seed_val_accs")
        if not cand_scores or len(cand_scores) != len(best_scores):
            continue

        cand_scores_arr = np.array(cand_scores, dtype=np.float64)
        diff = best_scores_arr - cand_scores_arr
        if np.allclose(diff, 0.0):
            p_val = 1.0
            test_name = "wilcoxon"
        else:
            try:
                p_val = float(wilcoxon(best_scores_arr, cand_scores_arr, zero_method="wilcox").pvalue)
                test_name = "wilcoxon"
            except ValueError:
                p_val = float(ttest_rel(best_scores_arr, cand_scores_arr).pvalue)
                test_name = "paired_ttest"

        outputs.append(
            {
                "best_model": best["name"],
                "compared_model": cand["name"],
                "test": test_name,
                "p_value": p_val,
                "significant_at_0_05": bool(p_val < alpha),
                "best_mean": float(np.mean(best_scores_arr)),
                "compared_mean": float(np.mean(cand_scores_arr)),
            }
        )

    return outputs

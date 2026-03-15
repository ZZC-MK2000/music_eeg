import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import detrend, welch
from scipy.stats import kurtosis, skew

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


BANDS = [
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("gamma", 30.0, 40.0),
]


def _song_id_from_trigger(song_trigger: int) -> str:
    # NMED-T README: triggers 21-30 map to song index 1-10.
    if 21 <= song_trigger <= 30:
        return f"song_{song_trigger - 20}"
    return f"song_{song_trigger}"


def _safe_numeric_array(x: Any) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(x)
    except Exception:
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return arr


def _parse_ids_from_filename(path: Path) -> Dict[str, str]:
    name = path.name
    stem = path.stem

    out = {
        "recording_id": stem,
        "subject_id": "unknown",
        "session_id": "unknown",
        "song_id": "unknown",
    }

    # Raw file style: 02_1_raw.mat
    m_raw = re.search(r"(\d{2})_(\d)_raw", name)
    if m_raw:
        out["subject_id"] = f"sub_{m_raw.group(1)}"
        out["session_id"] = f"sess_{m_raw.group(2)}"

    # Aggregated per-song style: song21_Imputed.mat
    m_song = re.search(r"song(\d+)", name, flags=re.IGNORECASE)
    if m_song:
        out["song_id"] = _song_id_from_trigger(int(m_song.group(1)))

    return out


def _extract_scalar_by_name(container: Any, names: Tuple[str, ...]) -> Optional[float]:
    names_lower = {n.lower() for n in names}

    def walk(obj: Any, key_name: str = "") -> Optional[float]:
        k = key_name.lower()
        if k in names_lower:
            arr = _safe_numeric_array(obj)
            if arr is not None and arr.size == 1:
                return float(arr.reshape(-1)[0])

        if isinstance(obj, dict):
            for kk, vv in obj.items():
                got = walk(vv, str(kk))
                if got is not None:
                    return got
            return None

        if hasattr(obj, "_fieldnames"):
            for fld in obj._fieldnames:
                got = walk(getattr(obj, fld), fld)
                if got is not None:
                    return got
            return None

        if isinstance(obj, np.ndarray) and obj.dtype == object:
            for item in obj.flat:
                got = walk(item, key_name)
                if got is not None:
                    return got

        return None

    return walk(container)


def _collect_numeric_2d_arrays(container: Any, prefix: str = "") -> List[Tuple[str, np.ndarray]]:
    found: List[Tuple[str, np.ndarray]] = []

    if isinstance(container, dict):
        for k, v in container.items():
            if str(k).startswith("__"):
                continue
            sub = f"{prefix}.{k}" if prefix else str(k)
            found.extend(_collect_numeric_2d_arrays(v, sub))
        return found

    if hasattr(container, "_fieldnames"):
        for fld in container._fieldnames:
            v = getattr(container, fld)
            sub = f"{prefix}.{fld}" if prefix else fld
            found.extend(_collect_numeric_2d_arrays(v, sub))
        return found

    arr = _safe_numeric_array(container)
    if arr is not None and arr.ndim == 2:
        found.append((prefix or "array", arr))
        return found

    if isinstance(container, np.ndarray) and container.dtype == object:
        for i, item in enumerate(container.flat):
            sub = f"{prefix}[{i}]" if prefix else f"obj[{i}]"
            found.extend(_collect_numeric_2d_arrays(item, sub))

    return found


def _load_mat_any(mat_path: Path) -> Dict[str, Any]:
    try:
        return loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        if not HAS_H5PY:
            raise RuntimeError(
                "该 .mat 可能是 v7.3(HDF5) 格式，且未安装 h5py。请先安装 h5py，或先将文件转为旧版 mat。"
            )

        out: Dict[str, Any] = {}
        with h5py.File(mat_path, "r") as f:
            def visitor(name: str, obj: Any):
                if isinstance(obj, h5py.Dataset):
                    try:
                        out[name] = np.array(obj)
                    except Exception:
                        pass

            f.visititems(visitor)
        if not out:
            raise RuntimeError(f"未从 {mat_path.name} 读取到可用 dataset。")
        return out


def _select_eeg_array(
    data: Dict[str, Any],
    eeg_key: str,
    min_samples: int,
    max_channels: int,
) -> Tuple[str, np.ndarray]:
    if eeg_key:
        if eeg_key not in data:
            raise KeyError(f"指定的 --eeg-key 不存在: {eeg_key}")
        arr = _safe_numeric_array(data[eeg_key])
        if arr is None or arr.ndim != 2:
            raise ValueError(f"--eeg-key={eeg_key} 不是二维数值数组")
        return eeg_key, arr

    candidates = _collect_numeric_2d_arrays(data)
    scored: List[Tuple[float, str, np.ndarray]] = []

    for name, arr in candidates:
        if arr.size == 0:
            continue
        d0, d1 = arr.shape
        c = min(d0, d1)
        t = max(d0, d1)
        if c < 4 or c > max_channels or t < min_samples:
            continue

        # Prefer EEG-like array: channels not too small, many time points.
        score = float(t) - 10.0 * abs(c - 64)
        scored.append((score, name, arr))

    if not scored:
        raise RuntimeError("未自动找到 EEG 主数组，请使用 --eeg-key 指定键名。")

    scored.sort(key=lambda x: x[0], reverse=True)
    _, best_name, best_arr = scored[0]
    return best_name, best_arr


def _orient_channels_first(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("EEG 数组必须是二维")
    if x.shape[0] <= x.shape[1]:
        return x.astype(np.float32)
    return x.T.astype(np.float32)


def _hjorth_parameters(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = detrend(x, axis=1, type="linear")
    dx = np.diff(x, axis=1)
    ddx = np.diff(dx, axis=1)
    var_x = np.var(x, axis=1) + 1e-12
    var_dx = np.var(dx, axis=1) + 1e-12
    var_ddx = np.var(ddx, axis=1) + 1e-12
    activity = var_x
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / (mobility + 1e-12)
    return activity, mobility, complexity


def _extract_features_window(window_data: np.ndarray, sfreq: float) -> np.ndarray:
    mean_feat = window_data.mean(axis=1)
    std_feat = window_data.std(axis=1)
    rms_feat = np.sqrt((window_data ** 2).mean(axis=1))
    skew_feat = skew(window_data, axis=1, bias=False, nan_policy="omit")
    kurt_feat = kurtosis(window_data, axis=1, fisher=True, bias=False, nan_policy="omit")
    zcr_feat = np.mean(np.abs(np.diff(np.signbit(window_data), axis=1)), axis=1).astype(np.float32)

    hj_activity, hj_mobility, hj_complexity = _hjorth_parameters(window_data)

    freqs, psd = welch(window_data, fs=sfreq, nperseg=min(window_data.shape[1], 256), axis=1)
    total_power = psd.sum(axis=1, keepdims=True) + 1e-12
    psd_norm = psd / total_power
    spectral_entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=1)

    band_feats = []
    band_power_list = []
    for _, f_low, f_high in BANDS:
        idx = (freqs >= f_low) & (freqs < f_high)
        band_power = psd[:, idx].sum(axis=1) + 1e-12
        rel_power = band_power / total_power.ravel()
        band_feats.append(rel_power)
        band_power_list.append(band_power)

    theta_p = band_power_list[1].mean()
    alpha_p = band_power_list[2].mean()
    beta_p = band_power_list[3].mean()
    gamma_p = band_power_list[4].mean()

    ratio_feats = np.array(
        [
            theta_p / (alpha_p + 1e-12),
            (theta_p + alpha_p) / (beta_p + 1e-12),
            beta_p / (alpha_p + 1e-12),
            gamma_p / (beta_p + 1e-12),
        ],
        dtype=np.float32,
    )

    corr_mat = np.corrcoef(window_data)
    corr_mat = np.nan_to_num(corr_mat, nan=0.0, posinf=0.0, neginf=0.0)
    tri_idx = np.triu_indices_from(corr_mat, k=1)
    corr_vals = corr_mat[tri_idx]
    if corr_vals.size == 0:
        corr_stats = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        corr_stats = np.array([corr_vals.mean(), corr_vals.std(), np.median(corr_vals)], dtype=np.float32)

    cov_mat = np.cov(window_data)
    eigvals = np.linalg.eigvalsh(cov_mat)
    eigvals = np.sort(np.clip(eigvals, 1e-12, None))[::-1]
    top_k = 8
    if eigvals.size < top_k:
        eigvals = np.pad(eigvals, (0, top_k - eigvals.size), mode="constant", constant_values=1e-12)
    eig_feat = np.log(eigvals[:top_k] + 1e-12).astype(np.float32)

    features = np.concatenate(
        [
            mean_feat,
            std_feat,
            rms_feat,
            skew_feat,
            kurt_feat,
            zcr_feat,
            hj_activity,
            hj_mobility,
            hj_complexity,
            spectral_entropy,
            *band_feats,
            ratio_feats,
            corr_stats,
            eig_feat,
        ]
    )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _normalize_subject_id(raw_value: Any) -> str:
    text = str(raw_value).strip()
    m = re.search(r"(\d+)", text)
    if not m:
        return "unknown"
    return f"sub_{int(m.group(1)):02d}"


def _iter_song_imputed_rows(
    *,
    mat_path: Path,
    obj: Dict[str, Any],
    window_sec: float,
    step_sec: float,
    sfreq_default: float,
) -> Tuple[List[Dict[str, Any]], str]:
    name = mat_path.name
    m_song = re.search(r"song(\d+)", name, flags=re.IGNORECASE)
    if not m_song:
        return [], ""

    song_trigger = int(m_song.group(1))
    data_key = f"data{song_trigger}"
    if data_key not in obj:
        return [], ""

    raw = _safe_numeric_array(obj[data_key])
    if raw is None or raw.ndim != 3:
        return [], ""

    fs = _extract_scalar_by_name(obj, ("fs", "srate", "sampling_rate", "sfreq", "Fs"))
    sfreq = float(fs) if fs is not None and fs > 1 else float(sfreq_default)

    n0, n1, n2 = raw.shape
    if n0 <= n1 and n0 <= 256:
        arr = raw.astype(np.float32)
    elif n1 <= n0 and n1 <= 256:
        arr = np.transpose(raw, (1, 0, 2)).astype(np.float32)
    else:
        raise ValueError(f"{name} 的三维数组形状异常: {raw.shape}")

    subs_key = f"subs{song_trigger}"
    subs = obj.get(subs_key, None)
    if isinstance(subs, np.ndarray):
        subs_list = [str(x) for x in subs.reshape(-1).tolist()]
    else:
        subs_list = []

    rows: List[Dict[str, Any]] = []
    window_size = int(round(window_sec * sfreq))
    step_size = int(round(step_sec * sfreq))
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_sec 与 step_sec 必须为正数")

    song_id = _song_id_from_trigger(song_trigger)
    default_label = song_id

    for sub_idx in range(arr.shape[2]):
        eeg = arr[:, :, sub_idx]
        if eeg.shape[1] < window_size:
            continue

        subject_text = subs_list[sub_idx] if sub_idx < len(subs_list) else f"S{sub_idx + 1:02d}"
        subject_id = _normalize_subject_id(subject_text)
        row_meta = {
            "recording_id": f"{mat_path.stem}_{subject_id}",
            "subject_id": subject_id,
            "session_id": "unknown",
            "song_id": song_id,
        }
        label = default_label

        n_windows = (eeg.shape[1] - window_size) // step_size + 1
        for i in range(n_windows):
            start = i * step_size
            end = start + window_size
            wd = eeg[:, start:end]
            feat = _extract_features_window(wd, sfreq=sfreq)
            rows.append(
                {
                    **{f"f_{j:04d}": float(v) for j, v in enumerate(feat)},
                    "label": label,
                    "subject_id": subject_id,
                    "session_id": "unknown",
                    "recording_id": f"{mat_path.stem}_{subject_id}",
                    "song_id": song_id,
                    "window_id": i,
                    "window_start_sec": float(start / sfreq),
                    "window_end_sec": float(end / sfreq),
                    "sfreq": float(sfreq),
                    "n_channels": int(eeg.shape[0]),
                    "source_mat": mat_path.name,
                    "used_eeg_key": data_key,
                    "stim_trigger": int(song_trigger),
                }
            )

    return rows, f"[OK] {mat_path.name}: key={data_key}, sfreq={sfreq:.2f}, channels={arr.shape[0]}, participants={arr.shape[2]}"


def convert_mat_to_feature_csv(
    input_dir: Path,
    output_csv: Path,
    sfreq_default: float,
    window_sec: float,
    step_sec: float,
    eeg_key: str,
    min_samples: int,
    max_channels: int,
) -> Dict[str, Any]:
    mat_files = sorted(list(input_dir.glob("*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"{input_dir} 下未找到 .mat 文件")

    song_imputed_files = sorted(input_dir.glob("song*_Imputed.mat"))
    if song_imputed_files:
        mat_files = song_imputed_files
        print(f"检测到 song*_Imputed.mat，自动切换为10首音乐分类模式，共 {len(mat_files)} 个文件。")

    rows = []
    logs = []

    for mat_path in mat_files:
        meta = _parse_ids_from_filename(mat_path)

        try:
            obj = _load_mat_any(mat_path)

            song_rows, song_log = _iter_song_imputed_rows(
                mat_path=mat_path,
                obj=obj,
                window_sec=window_sec,
                step_sec=step_sec,
                sfreq_default=sfreq_default,
            )
            if song_rows:
                rows.extend(song_rows)
                logs.append(song_log)
                continue

            fs = _extract_scalar_by_name(obj, ("fs", "srate", "sampling_rate", "sfreq", "Fs"))
            sfreq = float(fs) if fs is not None and fs > 1 else float(sfreq_default)

            used_key, raw_eeg = _select_eeg_array(obj, eeg_key=eeg_key, min_samples=min_samples, max_channels=max_channels)
            eeg = _orient_channels_first(raw_eeg)

            window_size = int(round(window_sec * sfreq))
            step_size = int(round(step_sec * sfreq))
            if window_size <= 0 or step_size <= 0:
                raise ValueError("window_sec 与 step_sec 必须为正数")
            if eeg.shape[1] < window_size:
                logs.append(f"[SKIP] {mat_path.name}: 样本长度不足一个窗口")
                continue

            n_windows = (eeg.shape[1] - window_size) // step_size + 1
            label = meta["song_id"]

            for i in range(n_windows):
                start = i * step_size
                end = start + window_size
                wd = eeg[:, start:end]
                feat = _extract_features_window(wd, sfreq=sfreq)

                row = {
                    **{f"f_{j:04d}": float(v) for j, v in enumerate(feat)},
                    "label": label,
                    "subject_id": meta["subject_id"],
                    "session_id": meta["session_id"],
                    "recording_id": meta["recording_id"],
                    "song_id": meta["song_id"],
                    "window_id": i,
                    "window_start_sec": float(start / sfreq),
                    "window_end_sec": float(end / sfreq),
                    "sfreq": float(sfreq),
                    "n_channels": int(eeg.shape[0]),
                    "source_mat": mat_path.name,
                    "used_eeg_key": used_key,
                }
                rows.append(row)

            logs.append(
                f"[OK] {mat_path.name}: key={used_key}, sfreq={sfreq:.2f}, channels={eeg.shape[0]}, windows={n_windows}"
            )
        except Exception as exc:
            logs.append(f"[ERR] {mat_path.name}: {exc}")

    if not rows:
        raise RuntimeError("未提取到任何窗口特征，请检查输入文件结构或 --eeg-key 参数。")

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    summary = {
        "n_files": len(mat_files),
        "n_rows": int(len(df)),
        "n_features": int(len([c for c in df.columns if c.startswith("f_")])),
        "label_counts": df["label"].value_counts().to_dict(),
        "subjects": sorted(df["subject_id"].astype(str).unique().tolist()),
        "sessions": sorted(df["session_id"].astype(str).unique().tolist()),
        "songs": sorted(df["song_id"].astype(str).unique().tolist()),
        "logs": logs,
    }

    summary_path = output_csv.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"特征CSV已保存: {output_csv}")
    print(f"摘要已保存: {summary_path}")
    print(f"总窗口数: {len(df)}")
    print("标签分布:")
    print(df["label"].value_counts())

    return summary


def main():
    parser = argparse.ArgumentParser(description="将 NMED-T .mat 转换为窗口级 EEG 特征 CSV。")
    parser.add_argument("--input-dir", type=str, required=True, help="包含 .mat 文件的目录")
    parser.add_argument("--output-csv", type=str, required=True, help="输出特征 CSV 路径")
    parser.add_argument("--sfreq-default", type=float, default=125.0, help="当 .mat 内未找到采样率时使用")
    parser.add_argument("--window-sec", type=float, default=2.0, help="窗口长度(秒)")
    parser.add_argument("--step-sec", type=float, default=1.0, help="步长(秒)")
    parser.add_argument(
        "--eeg-key",
        type=str,
        default="",
        help="可选，手动指定 EEG 数组 key（如 data.eeg）。为空时自动推断。",
    )
    parser.add_argument("--min-samples", type=int, default=1024, help="自动识别 EEG 数组时的最小样本点")
    parser.add_argument("--max-channels", type=int, default=256, help="自动识别 EEG 数组时的最大通道数")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()

import importlib.util
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from torch.utils.data import DataLoader, TensorDataset

from .models import EEG1DCNN, EEGChannelAttnNet, EEGMSResNet1D, EEGResMLPNet

if importlib.util.find_spec("imblearn.over_sampling") is not None:
    from imblearn.over_sampling import SMOTE

    HAS_IMBLEARN = True
else:
    SMOTE = None
    HAS_IMBLEARN = False


def _build_loader(dataset: TensorDataset, batch_size: int, shuffle: bool, use_cuda: bool) -> DataLoader:
    # Windows frequently hits shared-memory mapping limits with multi-worker CUDA loaders.
    if sys.platform.startswith("win"):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=use_cuda,
        )

    env_workers = os.environ.get("MUSIC_EEG_NUM_WORKERS")
    num_workers = int(env_workers) if env_workers is not None else 2
    if use_cuda:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=max(0, num_workers),
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _maybe_print_cuda_runtime(use_cuda: bool, tag: str) -> None:
    if not use_cuda:
        return
    alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
    print(f"[{tag}] CUDA内存: allocated={alloc_mb:.1f}MB, reserved={reserved_mb:.1f}MB")


def resolve_torch_device(device: str) -> torch.device:
    requested = (device or "auto").lower()

    if requested == "auto":
        target = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求使用GPU训练，但当前环境不可用CUDA。")
        target = torch.device("cuda:0")
    elif requested == "cpu":
        target = torch.device("cpu")
    else:
        raise ValueError(f"不支持的设备参数: {device}")

    if target.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(target)
        gpu_name = torch.cuda.get_device_name(target)
        print(f"训练设备: {target} ({gpu_name})")
    else:
        print("训练设备: cpu")

    return target


def maybe_apply_smote(x_train: np.ndarray, y_train_enc: np.ndarray, seed: int, use_smote: bool) -> tuple[np.ndarray, np.ndarray]:
    if not use_smote:
        return x_train, y_train_enc

    if not HAS_IMBLEARN:
        print("SMOTE跳过：未安装 imbalanced-learn")
        return x_train, y_train_enc

    class_counts = np.bincount(y_train_enc)
    min_class = int(class_counts.min())
    if min_class < 2:
        print("SMOTE跳过：最小类别样本数不足2")
        return x_train, y_train_enc

    k_neighbors = max(1, min(3, min_class - 1))
    smote = SMOTE(random_state=seed, k_neighbors=k_neighbors)
    x_out, y_out = smote.fit_resample(x_train, y_train_enc)
    print(f"SMOTE已启用: {x_train.shape[0]} -> {x_out.shape[0]} 样本")
    return x_out, y_out


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        tgt_logp = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        tgt_p = p.gather(1, target.unsqueeze(1)).squeeze(1)
        at = self.alpha[target]
        loss = -at * (1.0 - tgt_p).pow(self.gamma) * tgt_logp
        return loss.mean()


def _build_criterion(class_weights: np.ndarray, device: torch.device, loss_name: str, label_smoothing: float = 0.0) -> nn.Module:
    alpha = torch.tensor(class_weights, device=device)
    if loss_name == "focal":
        return FocalLoss(alpha=alpha, gamma=2.0)
    return nn.CrossEntropyLoss(weight=alpha, label_smoothing=label_smoothing)


def train_eeg_cnn(
    x_train: np.ndarray,
    y_train_enc: np.ndarray,
    x_val: np.ndarray,
    y_val_enc: np.ndarray,
    x_test: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    loss_name: str = "ce",
    use_onecycle: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_n = ((x_train - x_mean) / x_std).astype(np.float32, copy=False)
    x_val_n = ((x_val - x_mean) / x_std).astype(np.float32, copy=False)
    x_test_n = ((x_test - x_mean) / x_std).astype(np.float32, copy=False)

    x_train_t = torch.from_numpy(x_train_n).unsqueeze(1).contiguous()
    x_val_t = torch.from_numpy(x_val_n).unsqueeze(1).contiguous()
    x_test_t = torch.from_numpy(x_test_n).unsqueeze(1).contiguous()
    y_train_t = torch.from_numpy(y_train_enc).long()
    y_val_t = torch.from_numpy(y_val_enc).long()

    use_cuda = device.type == "cuda"
    train_loader = _build_loader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True, use_cuda=use_cuda)
    val_loader = _build_loader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size, shuffle=False, use_cuda=use_cuda)

    model = EEG1DCNN(_n_channels=x_train.shape[1], n_classes=n_classes).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    class_counts = np.bincount(y_train_enc)
    class_weights = (len(y_train_enc) / (len(class_counts) * class_counts)).astype(np.float32)
    criterion = _build_criterion(class_weights, device, loss_name=loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    if use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=0.2,
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-5)

    best_state = None
    best_val_acc = -1.0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)

            noise = 0.01 * torch.randn_like(xb)
            xb_aug = xb + noise

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(xb_aug)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
            if use_onecycle:
                scheduler.step()

            pred = logits.argmax(dim=1)
            train_correct += (pred == yb).sum().item()
            train_total += yb.size(0)

        model.eval()
        val_logits = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=use_cuda)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                    logits = model(xb)
                val_logits.append(logits.cpu())

        val_logits = torch.cat(val_logits, dim=0)
        val_prob = F.softmax(val_logits, dim=1).numpy()
        val_pred = val_prob.argmax(axis=1)
        val_acc = accuracy_score(y_val_enc, val_pred)
        train_acc = train_correct / max(train_total, 1)

        if not use_onecycle:
            scheduler.step(val_acc)
        print(f"[EEG-CNN] Epoch {epoch:03d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        if epoch == 1:
            _maybe_print_cuda_runtime(use_cuda=use_cuda, tag="EEG-CNN")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            best_state = {
                "model": model.state_dict(),
                "x_mean": x_mean.astype(np.float32),
                "x_std": x_std.astype(np.float32),
                "input_dim": int(x_train.shape[1]),
            }
        else:
            wait += 1
            if wait >= patience:
                print(f"[EEG-CNN] 早停触发：{patience} 个 epoch 验证准确率未提升。")
                break

    if best_state is None:
        raise RuntimeError("EEG-CNN 训练失败，未得到有效状态。")

    model.load_state_dict(best_state["model"])
    model.eval()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            val_logits = model(x_val_t.to(device, non_blocking=use_cuda)).cpu()
            test_logits = model(x_test_t.to(device, non_blocking=use_cuda)).cpu()

    val_prob = F.softmax(val_logits, dim=1).numpy()
    test_prob = F.softmax(test_logits, dim=1).numpy()
    return best_state, val_prob, test_prob


def train_eeg_resmlp(
    x_train: np.ndarray,
    y_train_enc: np.ndarray,
    x_val: np.ndarray,
    y_val_enc: np.ndarray,
    x_test: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    loss_name: str = "ce",
    use_onecycle: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_n = ((x_train - x_mean) / x_std).astype(np.float32)
    x_val_n = ((x_val - x_mean) / x_std).astype(np.float32)
    x_test_n = ((x_test - x_mean) / x_std).astype(np.float32)

    x_train_t = torch.from_numpy(x_train_n)
    x_val_t = torch.from_numpy(x_val_n)
    x_test_t = torch.from_numpy(x_test_n)
    y_train_t = torch.from_numpy(y_train_enc).long()
    y_val_t = torch.from_numpy(y_val_enc).long()

    use_cuda = device.type == "cuda"
    train_loader = _build_loader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True, use_cuda=use_cuda)
    val_loader = _build_loader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size, shuffle=False, use_cuda=use_cuda)

    model = EEGResMLPNet(input_dim=x_train.shape[1], n_classes=n_classes).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    class_counts = np.bincount(y_train_enc)
    class_weights = (len(y_train_enc) / (len(class_counts) * class_counts)).astype(np.float32)
    criterion = _build_criterion(class_weights, device, loss_name=loss_name, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-3)
    if use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=0.2,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 10), eta_min=1e-5)

    best_state = None
    best_val_acc = -1.0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            noise = 0.005 * torch.randn_like(xb)
            xb_aug = xb + noise

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(xb_aug)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            if use_onecycle:
                scheduler.step()

            pred = logits.argmax(dim=1)
            train_correct += (pred == yb).sum().item()
            train_total += yb.size(0)

        if not use_onecycle:
            scheduler.step()

        model.eval()
        val_logits = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=use_cuda)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                    logits = model(xb)
                val_logits.append(logits.cpu())

        val_logits = torch.cat(val_logits, dim=0)
        val_prob = F.softmax(val_logits, dim=1).numpy()
        val_pred = val_prob.argmax(axis=1)
        val_acc = accuracy_score(y_val_enc, val_pred)
        train_acc = train_correct / max(train_total, 1)

        print(f"[EEG-ResMLP] Epoch {epoch:03d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        if epoch == 1:
            _maybe_print_cuda_runtime(use_cuda=use_cuda, tag="EEG-ResMLP")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            best_state = {
                "model": model.state_dict(),
                "x_mean": x_mean.astype(np.float32),
                "x_std": x_std.astype(np.float32),
                "input_dim": int(x_train.shape[1]),
            }
        else:
            wait += 1
            if wait >= patience:
                print(f"[EEG-ResMLP] 早停触发：{patience} 个 epoch 验证准确率未提升。")
                break

    if best_state is None:
        raise RuntimeError("EEG-ResMLP 训练失败，未得到有效状态。")

    model.load_state_dict(best_state["model"])
    model.eval()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            val_logits = model(x_val_t.to(device, non_blocking=use_cuda)).cpu()
            test_logits = model(x_test_t.to(device, non_blocking=use_cuda)).cpu()

    val_prob = F.softmax(val_logits, dim=1).numpy()
    test_prob = F.softmax(test_logits, dim=1).numpy()
    return best_state, val_prob, test_prob


def set_random_states(estimator, seed: int):
    params = estimator.get_params(deep=True)
    update = {}
    for key in params:
        if key.endswith("random_state"):
            update[key] = seed
    if update:
        estimator.set_params(**update)
    return estimator


def train_eeg_msresnet(
    x_train: np.ndarray,
    y_train_enc: np.ndarray,
    x_val: np.ndarray,
    y_val_enc: np.ndarray,
    x_test: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    loss_name: str = "focal",
    use_onecycle: bool = True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_n = ((x_train - x_mean) / x_std).astype(np.float32, copy=False)
    x_val_n = ((x_val - x_mean) / x_std).astype(np.float32, copy=False)
    x_test_n = ((x_test - x_mean) / x_std).astype(np.float32, copy=False)

    x_train_t = torch.from_numpy(x_train_n).unsqueeze(1).contiguous()
    x_val_t = torch.from_numpy(x_val_n).unsqueeze(1).contiguous()
    x_test_t = torch.from_numpy(x_test_n).unsqueeze(1).contiguous()
    y_train_t = torch.from_numpy(y_train_enc).long()
    y_val_t = torch.from_numpy(y_val_enc).long()

    use_cuda = device.type == "cuda"
    train_loader = _build_loader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True, use_cuda=use_cuda)
    val_loader = _build_loader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size, shuffle=False, use_cuda=use_cuda)

    model = EEGMSResNet1D(x_train.shape[1], n_classes=n_classes).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    class_counts = np.bincount(y_train_enc)
    class_weights = (len(y_train_enc) / (len(class_counts) * class_counts)).astype(np.float32)
    criterion = _build_criterion(class_weights, device, loss_name=loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-3)
    if use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=0.2,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 10), eta_min=1e-5)

    best_state = None
    best_val_acc = -1.0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            xb_aug = xb + 0.008 * torch.randn_like(xb)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(xb_aug)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            scaler.step(optimizer)
            scaler.update()
            if use_onecycle:
                scheduler.step()

            pred = logits.argmax(dim=1)
            train_correct += (pred == yb).sum().item()
            train_total += yb.size(0)

        if not use_onecycle:
            scheduler.step()

        model.eval()
        val_logits = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=use_cuda)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                    logits = model(xb)
                val_logits.append(logits.cpu())

        val_logits = torch.cat(val_logits, dim=0)
        val_prob = F.softmax(val_logits, dim=1).numpy()
        val_pred = val_prob.argmax(axis=1)
        val_acc = accuracy_score(y_val_enc, val_pred)
        train_acc = train_correct / max(train_total, 1)

        print(f"[EEG-MSResNet] Epoch {epoch:03d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        if epoch == 1:
            _maybe_print_cuda_runtime(use_cuda=use_cuda, tag="EEG-MSResNet")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            best_state = {
                "model": model.state_dict(),
                "x_mean": x_mean.astype(np.float32),
                "x_std": x_std.astype(np.float32),
                "input_dim": int(x_train.shape[1]),
            }
        else:
            wait += 1
            if wait >= patience:
                print(f"[EEG-MSResNet] 早停触发：{patience} 个 epoch 验证准确率未提升。")
                break

    if best_state is None:
        raise RuntimeError("EEG-MSResNet 训练失败，未得到有效状态。")

    model.load_state_dict(best_state["model"])
    model.eval()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            val_logits = model(x_val_t.to(device, non_blocking=use_cuda)).cpu()
            test_logits = model(x_test_t.to(device, non_blocking=use_cuda)).cpu()

    val_prob = F.softmax(val_logits, dim=1).numpy()
    test_prob = F.softmax(test_logits, dim=1).numpy()
    return best_state, val_prob, test_prob


def train_eeg_chanattn(
    x_train: np.ndarray,
    y_train_enc: np.ndarray,
    x_val: np.ndarray,
    y_val_enc: np.ndarray,
    x_test: np.ndarray,
    n_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    device: torch.device,
    loss_name: str = "focal",
    use_onecycle: bool = True,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_n = ((x_train - x_mean) / x_std).astype(np.float32, copy=False)
    x_val_n = ((x_val - x_mean) / x_std).astype(np.float32, copy=False)
    x_test_n = ((x_test - x_mean) / x_std).astype(np.float32, copy=False)

    x_train_t = torch.from_numpy(x_train_n).unsqueeze(1).contiguous()
    x_val_t = torch.from_numpy(x_val_n).unsqueeze(1).contiguous()
    x_test_t = torch.from_numpy(x_test_n).unsqueeze(1).contiguous()
    y_train_t = torch.from_numpy(y_train_enc).long()
    y_val_t = torch.from_numpy(y_val_enc).long()

    use_cuda = device.type == "cuda"
    train_loader = _build_loader(TensorDataset(x_train_t, y_train_t), batch_size=batch_size, shuffle=True, use_cuda=use_cuda)
    val_loader = _build_loader(TensorDataset(x_val_t, y_val_t), batch_size=batch_size, shuffle=False, use_cuda=use_cuda)

    model = EEGChannelAttnNet(x_train.shape[1], n_classes=n_classes).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    class_counts = np.bincount(y_train_enc)
    class_weights = (len(y_train_enc) / (len(class_counts) * class_counts)).astype(np.float32)
    criterion = _build_criterion(class_weights, device, loss_name=loss_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=2e-3)
    if use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=max(1, len(train_loader)),
            pct_start=0.2,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 10), eta_min=1e-5)

    best_state = None
    best_val_acc = -1.0
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            xb_aug = xb + 0.006 * torch.randn_like(xb)

            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                logits = model(xb_aug)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.5)
            scaler.step(optimizer)
            scaler.update()
            if use_onecycle:
                scheduler.step()

            pred = logits.argmax(dim=1)
            train_correct += (pred == yb).sum().item()
            train_total += yb.size(0)

        if not use_onecycle:
            scheduler.step()

        model.eval()
        val_logits = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device, non_blocking=use_cuda)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                    logits = model(xb)
                val_logits.append(logits.cpu())

        val_logits = torch.cat(val_logits, dim=0)
        val_prob = F.softmax(val_logits, dim=1).numpy()
        val_pred = val_prob.argmax(axis=1)
        val_acc = accuracy_score(y_val_enc, val_pred)
        train_acc = train_correct / max(train_total, 1)

        print(f"[EEG-ChanAttn] Epoch {epoch:03d} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")
        if epoch == 1:
            _maybe_print_cuda_runtime(use_cuda=use_cuda, tag="EEG-ChanAttn")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            wait = 0
            best_state = {
                "model": model.state_dict(),
                "x_mean": x_mean.astype(np.float32),
                "x_std": x_std.astype(np.float32),
                "input_dim": int(x_train.shape[1]),
            }
        else:
            wait += 1
            if wait >= patience:
                print(f"[EEG-ChanAttn] 早停触发：{patience} 个 epoch 验证准确率未提升。")
                break

    if best_state is None:
        raise RuntimeError("EEG-ChanAttn 训练失败，未得到有效状态。")

    model.load_state_dict(best_state["model"])
    model.eval()

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            val_logits = model(x_val_t.to(device, non_blocking=use_cuda)).cpu()
            test_logits = model(x_test_t.to(device, non_blocking=use_cuda)).cpu()

    val_prob = F.softmax(val_logits, dim=1).numpy()
    test_prob = F.softmax(test_logits, dim=1).numpy()
    return best_state, val_prob, test_prob


def fit_and_select_multiseed(
    name: str,
    estimators: list[tuple[str, object]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    seeds: list[int],
    cv_folds: int,
    cv_weight: float,
    use_calibration: bool,
    selection_mode: str,
    calibration_min_gain: float,
):
    best = None
    best_primary = -np.inf
    best_robust = -np.inf
    best_val_acc_std = np.inf

    for tag, est in estimators:
        print(f"[{name}:{tag}] 开始评估，seeds={seeds}, cv_folds={cv_folds}", flush=True)
        val_probs = []
        test_probs = []
        val_accs = []
        cv_scores = []
        robust_scores = []
        calibration_improved_count = 0

        for sd in seeds:
            t0 = time.perf_counter()
            print(f"[{name}:{tag}] seed={sd} -> CV中...", flush=True)
            model = clone(est)
            model = set_random_states(model, sd)

            min_class_count = int(np.bincount(y_train).min())
            n_splits = max(2, min(cv_folds, min_class_count))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=sd)
            cv_acc = float(cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy").mean())

            print(f"[{name}:{tag}] seed={sd} -> 训练与预测中...", flush=True)
            model.fit(x_train, y_train)
            base_val_prob = model.predict_proba(x_val)
            base_test_prob = model.predict_proba(x_test)
            base_val_acc = accuracy_score(y_val, base_val_prob.argmax(axis=1))

            chosen_val_prob = base_val_prob
            chosen_test_prob = base_test_prob
            chosen_val_acc = base_val_acc

            if use_calibration:
                calibrator = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=3)
                calibrator.fit(x_train, y_train)
                cal_val_prob = calibrator.predict_proba(x_val)
                cal_test_prob = calibrator.predict_proba(x_test)
                cal_val_acc = accuracy_score(y_val, cal_val_prob.argmax(axis=1))

                if cal_val_acc >= base_val_acc + calibration_min_gain:
                    chosen_val_prob = cal_val_prob
                    chosen_test_prob = cal_test_prob
                    chosen_val_acc = cal_val_acc
                    calibration_improved_count += 1

            val_probs.append(chosen_val_prob)
            test_probs.append(chosen_test_prob)
            val_accs.append(chosen_val_acc)
            cv_scores.append(cv_acc)
            robust_scores.append((1.0 - cv_weight) * chosen_val_acc + cv_weight * cv_acc)
            t1 = time.perf_counter()
            print(f"[{name}:{tag}] seed={sd} 完成，用时 {t1 - t0:.1f}s", flush=True)

        val_prob_mean = np.mean(val_probs, axis=0)
        test_prob_mean = np.mean(test_probs, axis=0)
        val_acc_mean = float(np.mean(val_accs))
        robust_score_mean = float(np.mean(robust_scores))
        val_acc_std = float(np.std(val_accs))
        cv_acc_mean = float(np.mean(cv_scores))
        print(
            f"[{name}:{tag}] val_acc_mean={val_acc_mean:.4f} | robust={robust_score_mean:.4f} | "
            f"cv_acc_mean={cv_acc_mean:.4f} | val_acc_std={val_acc_std:.4f}"
        )

        final_model = clone(est)
        final_model = set_random_states(final_model, seeds[0])
        final_model.fit(x_train, y_train)
        if use_calibration and calibration_improved_count >= (len(seeds) // 2 + 1):
            calibrated_final = CalibratedClassifierCV(estimator=final_model, method="sigmoid", cv=3)
            calibrated_final.fit(x_train, y_train)
            final_model = calibrated_final

        cur_primary = robust_score_mean if selection_mode == "robust" else val_acc_mean
        should_update = (
            best is None
            or cur_primary > best_primary
            or (
                np.isclose(cur_primary, best_primary)
                and (
                    robust_score_mean > best_robust
                    or (np.isclose(robust_score_mean, best_robust) and val_acc_std < best_val_acc_std)
                )
            )
        )

        if should_update:
            best = {
                "name": f"{name}:{tag}",
                "model": final_model,
                "val_acc": val_acc_mean,
                "robust_score_mean": robust_score_mean,
                "val_acc_std": val_acc_std,
                "cv_acc_mean": cv_acc_mean,
                "val_prob": val_prob_mean,
                "test_prob": test_prob_mean,
                "seed_val_accs": [float(x) for x in val_accs],
            }
            best_primary = cur_primary
            best_robust = robust_score_mean
            best_val_acc_std = val_acc_std

    return best

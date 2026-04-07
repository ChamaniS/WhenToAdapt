import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import copy
import random
import json
import csv
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score
)
import matplotlib.pyplot as plt
from PIL import Image


# =========================
# Config
# =========================
SEED = 42
DATA_ROOT = r"/lustre06/project/6008975/csj5/Breasttumor_classi_renamed/"
OUTPUT_DIR = "breast_classi_feature_mixstyle"
MODEL_NAME = "efficientnet_b0_breast_tumor_feature_mixstyle.pth"

# Optional external pretrained weights
WEIGHTS_PATH = r"/lustre06/project/6008975/csj5/narvalenv/pretrained/efficientnet_b0_rwightman-7f5810bc.pth"

output_file = r"/lustre06/project/6008975/csj5/narvalenv/featurelvl_breast_classi.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sys.stdout = open(output_file, "w")

BATCH_SIZE = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
LR = 1e-3
WEIGHT_DECAY =0
NUM_WORKERS = 0
IMG_SIZE = 224

CLIENT_NAMES =["BUSBRA", "BUS", "BUSI", "UDIAT"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
USE_AMP = DEVICE.type == "cuda"

# Feature-level MixStyle hyperparameters
USE_MIXSTYLE = True
MIXSTYLE_P = 0.6
MIXSTYLE_ALPHA = 0.2
MIXSTYLE_EPS = 1e-6

# Apply MixStyle after these feature blocks in EfficientNet
# These indices refer to backbone.features[i]
MIXSTYLE_LAYERS = (2, 4, 6)

# Reference images to visualize
REFERENCE_IMAGES = {
    "BUSBRA": "0001-r.png",
    "BUS": "00104.png",
    "BUSI": "101.png",
    "UDIAT": "000007.png",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "reference_grids"), exist_ok=True)

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)


# =========================
# Reproducibility
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)


# =========================
# Transforms
# =========================
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

vis_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


# =========================
# Dataset utilities
# =========================
def set_dataset_paths(root: str, client_names: List[str]) -> Dict[str, Dict[str, str]]:
    paths = {}
    for client in client_names:
        paths[client] = {
            "train": os.path.join(root, client, "train"),
            "val": os.path.join(root, client, "val"),
            "test": os.path.join(root, client, "test"),
        }
    return paths


def check_class_alignment(datasets_list):
    base_classes = datasets_list[0].classes
    base_class_to_idx = datasets_list[0].class_to_idx

    for i, ds in enumerate(datasets_list[1:], start=2):
        if ds.classes != base_classes:
            raise ValueError(
                f"Class mismatch detected in dataset {i}.\n"
                f"Expected classes: {base_classes}\n"
                f"Found classes   : {ds.classes}\n"
                "All clients must have the same class folder names."
            )
        if ds.class_to_idx != base_class_to_idx:
            raise ValueError(
                f"Class-to-index mismatch detected in dataset {i}.\n"
                "All clients must use identical class folder naming."
            )

    return base_classes, base_class_to_idx


def build_client_datasets(paths_dict, split, transform):
    ds_list = []
    for client in CLIENT_NAMES:
        split_dir = paths_dict[client][split]
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing directory: {split_dir}")
        ds = datasets.ImageFolder(split_dir, transform=transform)
        ds_list.append(ds)

    classes, class_to_idx = check_class_alignment(ds_list)
    return ds_list, classes, class_to_idx


def build_combined_dataset(ds_list):
    if len(ds_list) == 1:
        return ds_list[0]
    return ConcatDataset(ds_list)


def count_samples(ds):
    if isinstance(ds, ConcatDataset):
        return sum(len(d) for d in ds.datasets)
    return len(ds)


def find_image_path_for_client(client_root: str, filename: str) -> Optional[str]:
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(client_root, split)
        if not os.path.isdir(split_dir):
            continue
        for root, _, files in os.walk(split_dir):
            for fn in files:
                if fn.lower() == filename.lower():
                    return os.path.join(root, fn)
    return None


def sample_context_paths_same_client(client_root: str, exclude_path: str, k: int = 3) -> List[str]:
    train_dir = os.path.join(client_root, "train")
    candidates = []
    for root, _, files in os.walk(train_dir):
        for fn in files:
            if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                p = os.path.join(root, fn)
                if os.path.normpath(p) != os.path.normpath(exclude_path):
                    candidates.append(p)

    if len(candidates) == 0:
        return [exclude_path] * k

    if len(candidates) >= k:
        return random.sample(candidates, k)
    return [random.choice(candidates) for _ in range(k)]


class PathListDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None, loader=default_loader):
        self.samples = list(samples)
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def load_image_as_tensor(path: str, transform) -> torch.Tensor:
    img = default_loader(path).convert("RGB")
    return transform(img)


def tensor_to_display_image(t: torch.Tensor, mean_vals=mean, std_vals=std):
    if t.ndim == 4:
        t = t[0]
    t = t.detach().cpu().float().clone()
    mean_t = torch.tensor(mean_vals).view(3, 1, 1)
    std_t = torch.tensor(std_vals).view(3, 1, 1)
    t = t * std_t + mean_t
    t = torch.clamp(t, 0.0, 1.0)
    return t.permute(1, 2, 0).numpy()


# =========================
# Feature-level MixStyle
# =========================
class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (torch.rand(1).item() > self.p) or x.size(0) <= 1:
            return x

        if x.ndim != 4:
            return x

        B, C, H, W = x.size()
        x_view = x.view(B, C, -1)
        mu = x_view.mean(dim=2).view(B, C, 1, 1)
        var = x_view.var(dim=2, unbiased=False).view(B, C, 1, 1)
        sigma = (var + self.eps).sqrt()

        x_norm = (x - mu) / sigma

        lm = np.random.beta(self.alpha, self.alpha, size=B).astype(np.float32)
        lm = torch.from_numpy(lm).to(x.device).view(B, 1, 1, 1)

        perm = torch.randperm(B, device=x.device)
        mu2 = mu[perm]
        sigma2 = sigma[perm]

        mu_mix = mu * lm + mu2 * (1.0 - lm)
        sigma_mix = sigma * lm + sigma2 * (1.0 - lm)

        return x_norm * sigma_mix + mu_mix


# =========================
# Pretrained weights loading
# =========================
def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state", "model", "net", "model_state_dict"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format.")


def load_pretrained_weights(model, weights_path: str, device=DEVICE):
    if not weights_path:
        print("[WEIGHTS] No WEIGHTS_PATH provided, using torchvision initialization.")
        return model

    if not os.path.isfile(weights_path):
        print(f"[WEIGHTS] WARNING: WEIGHTS_PATH not found: {weights_path}")
        print("[WEIGHTS] Falling back to random / torchvision initialization.")
        return model

    ckpt = torch.load(weights_path, map_location=device)
    state_dict = extract_state_dict(ckpt)
    state_dict = strip_module_prefix(state_dict)

    model_state = model.state_dict()
    filtered_state = {}

    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered_state[k] = v

    missing, unexpected = model.load_state_dict(filtered_state, strict=False)

    print(f"[WEIGHTS] Loaded from: {weights_path}")
    print(f"[WEIGHTS] Matched keys: {len(filtered_state)}")
    print(f"[WEIGHTS] Missing keys: {len(missing)}")
    print(f"[WEIGHTS] Unexpected keys: {len(unexpected)}")
    return model


# =========================
# Model
# =========================
class EfficientNetFeatureMixStyle(nn.Module):
    def __init__(
        self,
        num_classes: int,
        weights_path: Optional[str] = None,
        mixstyle_layers: Tuple[int, ...] = MIXSTYLE_LAYERS,
        mix_p: float = MIXSTYLE_P,
        mix_alpha: float = MIXSTYLE_ALPHA,
        mix_eps: float = MIXSTYLE_EPS,
    ):
        super().__init__()

        try:
            backbone = efficientnet_b0(weights=None)
        except Exception:
            backbone = efficientnet_b0(pretrained=False)

        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = nn.Linear(in_features, num_classes)

        self.backbone = backbone
        self.mixstyle = MixStyle(p=mix_p, alpha=mix_alpha, eps=mix_eps)
        self.mixstyle_layers = set(mixstyle_layers)

        if weights_path:
            self.backbone = load_pretrained_weights(self.backbone, weights_path, device=DEVICE)

    def forward(self, x):
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if self.training and USE_MIXSTYLE and i in self.mixstyle_layers:
                x = self.mixstyle(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.classifier(x)
        return x


def build_model(num_classes: int):
    model = EfficientNetFeatureMixStyle(
        num_classes=num_classes,
        weights_path=WEIGHTS_PATH,
        mixstyle_layers=MIXSTYLE_LAYERS,
        mix_p=MIXSTYLE_P,
        mix_alpha=MIXSTYLE_ALPHA,
        mix_eps=MIXSTYLE_EPS
    )
    return model


def compute_specificity_from_cm(cm):
    num_classes = cm.shape[0]
    per_class_specificity = []
    total = cm.sum()

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = total - tp - fp - fn
        denom = tn + fp
        specificity = tn / denom if denom > 0 else 0.0
        per_class_specificity.append(specificity)

    macro_specificity = float(np.mean(per_class_specificity))
    return per_class_specificity, macro_specificity


def average_state_dicts_weighted(models, weights):
    avg_sd = copy.deepcopy(models[0].state_dict())
    for k in avg_sd.keys():
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k] for i in range(len(models)))
    return avg_sd


def compute_class_weights_from_dataset(dataset):
    targets = [s[1] for s in dataset.samples]
    counts = np.bincount(targets, minlength=len(dataset.classes)).astype(np.float32)
    total = counts.sum()
    weights = total / (counts + 1e-8)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


# =========================
# Reference image comparison grids
# =========================
def input_mixstyle(images: torch.Tensor, p: float = MIXSTYLE_P, alpha: float = MIXSTYLE_ALPHA, eps: float = MIXSTYLE_EPS):
    if images.ndim != 4:
        raise ValueError("input_mixstyle expects a 4D tensor [B,C,H,W].")

    if not torch.is_floating_point(images):
        images = images.float()

    B, C, H, W = images.shape
    if B <= 1:
        return images

    if np.random.rand() > p:
        return images

    x = images.view(B, C, -1)
    mu = x.mean(dim=2).view(B, C, 1, 1)
    var = x.var(dim=2, unbiased=False).view(B, C, 1, 1)
    sigma = torch.sqrt(var + eps)

    x_norm = (images - mu) / sigma

    lam_np = np.random.beta(alpha, alpha, size=B).astype(np.float32)
    lam = torch.from_numpy(lam_np).to(images.device).view(B, 1, 1, 1)

    perm = torch.randperm(B, device=images.device)
    mu2 = mu[perm]
    sigma2 = sigma[perm]

    mu_mix = mu * lam + mu2 * (1.0 - lam)
    sigma_mix = sigma * lam + sigma2 * (1.0 - lam)

    out = x_norm * sigma_mix + mu_mix
    return out


@torch.no_grad()
def save_reference_comparison_grids(paths_dict, class_names):
    """
    Saves per-client and combined comparison grids:
      1) original image
      2) harmonized image with input-level MixStyle (visualization proxy)
      3) absolute difference between original and harmonized
    """
    client_rows = []

    for client in CLIENT_NAMES:
        client_root = os.path.join(DATA_ROOT, client)
        filename = REFERENCE_IMAGES.get(client, None)
        if filename is None:
            continue

        ref_path = find_image_path_for_client(client_root, filename)
        if ref_path is None:
            raise FileNotFoundError(
                f"Could not find reference image '{filename}' inside {client_root}/train|val|test"
            )

        context_paths = sample_context_paths_same_client(client_root, ref_path, k=3)

        raw_ref = default_loader(ref_path).convert("RGB")
        raw_ref_disp = np.asarray(raw_ref.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0

        batch_paths = [ref_path] + context_paths
        batch_tensors = torch.stack([load_image_as_tensor(p, train_tfms) for p in batch_paths], dim=0).to(DEVICE)

        mixed_batch = input_mixstyle(batch_tensors, p=1.0, alpha=MIXSTYLE_ALPHA, eps=MIXSTYLE_EPS)
        mixed_ref_disp = tensor_to_display_image(mixed_batch[0])

        abs_diff = np.abs(raw_ref_disp - mixed_ref_disp)

        client_rows.append((client, raw_ref_disp, mixed_ref_disp, abs_diff))

        fig, axes = plt.subplots(3, 1, figsize=(6, 14))

        axes[0].imshow(raw_ref_disp)
        axes[0].set_title(f"{client}\nOriginal Image Before Harmonization")
        axes[0].axis("off")

        axes[1].imshow(mixed_ref_disp)
        axes[1].set_title("Harmonized Image with Input-Level MixStyle")
        axes[1].axis("off")

        axes[2].imshow(abs_diff)
        axes[2].set_title("Absolute Difference |Original - Harmonized|")
        axes[2].axis("off")

        plt.tight_layout()
        out_path = os.path.join(OUTPUT_DIR, "reference_grids", f"{client}_reference_grid.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved reference grid: {out_path}")

    if len(client_rows) == 4:
        fig, axes = plt.subplots(4, 3, figsize=(14, 18))
        for r, (client, raw_disp, mix_disp, diff_disp) in enumerate(client_rows):
            axes[r, 0].imshow(raw_disp)
            axes[r, 0].set_title(f"{client} - Original")
            axes[r, 0].axis("off")

            axes[r, 1].imshow(mix_disp)
            axes[r, 1].set_title(f"{client} - Harmonized")
            axes[r, 1].axis("off")

            axes[r, 2].imshow(diff_disp)
            axes[r, 2].set_title(f"{client} - |Difference|")
            axes[r, 2].axis("off")

        plt.tight_layout()
        combined_path = os.path.join(OUTPUT_DIR, "reference_grids", "all_clients_reference_grid.png")
        plt.savefig(combined_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved combined reference grid: {combined_path}")


# =========================
# Training / evaluation
# =========================
def run_epoch(model, loader, criterion, optimizer=None, train=True):
    model.train() if train else model.eval()

    running_loss = 0.0
    all_preds = []
    all_targets = []

    loop = tqdm(loader, desc="Train" if train else "Eval", leave=False)

    for images, labels in loop:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(labels.detach().cpu().numpy())
        loop.set_postfix(loss=float(loss.item()))

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    return epoch_loss, epoch_acc, all_targets, all_preds


def evaluate_loader(model, loader, criterion, class_names, title_prefix="test", save_dir=OUTPUT_DIR, save_cm=True):
    loss, acc, targets, preds = run_epoch(model, loader, criterion, optimizer=None, train=False)

    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    precision_macro = precision_score(targets, preds, average="macro", zero_division=0)
    recall_macro = recall_score(targets, preds, average="macro", zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(targets, preds)
    kappa = cohen_kappa_score(targets, preds)
    per_class_specificity, macro_specificity = compute_specificity_from_cm(cm)

    print(f"\n=== {title_prefix.upper()} ===")
    print(f"Loss        : {loss:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision_macro:.4f}")
    print(f"Recall      : {recall_macro:.4f}")
    print(f"F1-score    : {f1_macro:.4f}")
    print(f"Balanced Acc: {bal_acc:.4f}")
    print(f"Kappa       : {kappa:.4f}")
    print(f"Specificity : {macro_specificity:.4f}")

    print("\nPer-class Specificity:")
    for idx, cls_name in enumerate(class_names):
        print(f"{cls_name:15s}: {per_class_specificity[idx]:.4f}")

    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=class_names, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print(cm)

    if save_cm:
        plt.figure(figsize=(7, 6))
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix - {title_prefix}")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")
        plt.yticks(tick_marks, class_names)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"confusion_matrix_{title_prefix}.png"), dpi=300)
        plt.close()

    return {
        "split": title_prefix,
        "loss": float(loss),
        "accuracy": float(acc),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "kappa": float(kappa),
        "specificity_macro": float(macro_specificity),
        "balanced_acc": float(bal_acc),
        "per_class_specificity": [float(x) for x in per_class_specificity],
        "cm": cm.tolist(),
    }


def plot_round_curves(history, out_dir):
    rounds = np.arange(1, len(history["round"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history["train_loss"], label="Train Loss")
    plt.plot(rounds, history["val_loss"], label="Val Loss")
    plt.xlabel("Communication Round")
    plt.ylabel("Loss")
    plt.title("Federated Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fed_train_val_loss.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history["train_acc"], label="Train Acc")
    plt.plot(rounds, history["val_acc"], label="Val Acc")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Federated Training Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fed_train_val_acc.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(rounds, history["global_test_acc"], label="Global Test Acc")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Global Test Accuracy Across Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fed_global_test_acc.png"), dpi=300)
    plt.close()


def save_metrics_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "split",
            "loss",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "kappa",
            "specificity_macro",
            "balanced_acc"
        ])
        for r in results:
            writer.writerow([
                r["split"],
                f'{r["loss"]:.6f}',
                f'{r["accuracy"]:.6f}',
                f'{r["precision_macro"]:.6f}',
                f'{r["recall_macro"]:.6f}',
                f'{r["f1_macro"]:.6f}',
                f'{r["kappa"]:.6f}',
                f'{r["specificity_macro"]:.6f}',
                f'{r["balanced_acc"]:.6f}',
            ])


# =========================
# Main
# =========================
def main():
    print("DEVICE:", DEVICE)
    print("WEIGHTS_PATH:", WEIGHTS_PATH)

    paths = set_dataset_paths(DATA_ROOT, CLIENT_NAMES)

    # Build per-client datasets
    train_datasets, class_names, class_to_idx = build_client_datasets(paths, "train", train_tfms)
    val_datasets, _, _ = build_client_datasets(paths, "val", eval_tfms)
    test_datasets, _, _ = build_client_datasets(paths, "test", eval_tfms)

    train_ds_all = build_combined_dataset(train_datasets)
    val_ds_all = build_combined_dataset(val_datasets)
    test_ds_all = build_combined_dataset(test_datasets)

    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Train samples (all clients):", count_samples(train_ds_all))
    print("Val samples   (all clients):", count_samples(val_ds_all))
    print("Test samples  (all clients):", count_samples(test_ds_all))

    for client, ds_tr, ds_va, ds_te in zip(CLIENT_NAMES, train_datasets, val_datasets, test_datasets):
        print(f"{client:15s} | train={len(ds_tr):5d}  val={len(ds_va):5d}  test={len(ds_te):5d}")

    # Save reference grids before training
    save_reference_comparison_grids(paths, class_names)

    # DataLoaders
    train_loaders = {}
    val_loaders = {}
    test_loaders = {}

    for client, ds_tr, ds_va, ds_te in zip(CLIENT_NAMES, train_datasets, val_datasets, test_datasets):
        train_loaders[client] = DataLoader(
            ds_tr,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        val_loaders[client] = DataLoader(
            ds_va,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        test_loaders[client] = DataLoader(
            ds_te,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )

    global_val_loader = DataLoader(
        val_ds_all,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    global_test_loader = DataLoader(
        test_ds_all,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Model
    global_model = build_model(num_classes).to(DEVICE)

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(global_model.state_dict())

    history = defaultdict(list)
    round_metrics = []
    start_time = time.time()

    for r in range(COMM_ROUNDS):
        print(f"\n==================== Communication Round {r + 1}/{COMM_ROUNDS} ====================")

        local_models = []
        local_weights = []

        round_train_losses = []
        round_train_accs = []

        # -------- Local training on each client --------
        for client_name in CLIENT_NAMES:
            print(f"\n[Client {client_name}]")

            local_model = copy.deepcopy(global_model).to(DEVICE)
            optimizer = optim.AdamW(local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            train_loader = train_loaders[client_name]
            val_loader = val_loaders[client_name]

            client_train_ds = train_loader.dataset
            client_class_weights = compute_class_weights_from_dataset(client_train_ds).to(DEVICE)
            client_criterion = nn.CrossEntropyLoss(weight=client_class_weights)

            client_epoch_losses = []
            client_epoch_accs = []

            for ep in range(LOCAL_EPOCHS):
                tr_loss, tr_acc, _, _ = run_epoch(
                    local_model,
                    train_loader,
                    client_criterion,
                    optimizer=optimizer,
                    train=True
                )
                client_epoch_losses.append(tr_loss)
                client_epoch_accs.append(tr_acc)
                print(f"  Local Epoch {ep + 1}/{LOCAL_EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

            val_loss, val_acc, _, _ = run_epoch(
                local_model,
                val_loader,
                client_criterion,
                optimizer=None,
                train=False
            )
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            local_models.append(local_model.cpu())
            local_weights.append(len(train_loader.dataset))

            round_train_losses.append(float(np.mean(client_epoch_losses)))
            round_train_accs.append(float(np.mean(client_epoch_accs)))

        total_train_size = sum(local_weights)
        if total_train_size == 0:
            raise RuntimeError("Total training size across clients is 0. Check your dataset splits.")

        # -------- FedAvg aggregation --------
        norm_weights = [w / total_train_size for w in local_weights]
        avg_state = average_state_dicts_weighted(local_models, norm_weights)
        global_model.load_state_dict(avg_state)
        global_model.to(DEVICE)

        # -------- Global validation --------
        combined_train_targets = []
        for d in train_datasets:
            combined_train_targets.extend([s[1] for s in d.samples])

        counts = np.bincount(combined_train_targets, minlength=num_classes).astype(np.float32)
        counts[counts == 0] = 1.0
        cw = counts.sum() / counts
        cw = cw / np.mean(cw)
        combined_class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
        combined_criterion = nn.CrossEntropyLoss(weight=combined_class_weights)

        global_val_loss, global_val_acc, _, _ = run_epoch(
            global_model,
            global_val_loader,
            combined_criterion,
            optimizer=None,
            train=False
        )

        # Save best model
        if global_val_loss < best_val_loss:
            best_val_loss = global_val_loss
            best_model_wts = copy.deepcopy(global_model.state_dict())
            torch.save(best_model_wts, os.path.join(OUTPUT_DIR, MODEL_NAME))
            print("\nSaved best global model.")

        rm = {
            "round": r + 1,
            "train_loss": float(np.mean(round_train_losses)),
            "train_acc": float(np.mean(round_train_accs)),
            "val_loss": global_val_loss,
            "val_acc": global_val_acc,
        }

        # =========================
        # TEST AFTER THIS ROUND
        # =========================
        print("\n" + "=" * 30)
        print(f"GLOBAL TEST AFTER ROUND {r + 1} (ALL CLIENTS TOGETHER)")
        print("=" * 30)

        global_test_result = evaluate_loader(
            global_model,
            global_test_loader,
            combined_criterion,
            class_names,
            title_prefix=f"global_round_{r + 1}",
            save_dir=OUTPUT_DIR,
            save_cm=True
        )

        rm["global_test_loss"] = global_test_result["loss"]
        rm["global_test_acc"] = global_test_result["accuracy"]
        rm["global_test_precision"] = global_test_result["precision_macro"]
        rm["global_test_recall"] = global_test_result["recall_macro"]
        rm["global_test_f1"] = global_test_result["f1_macro"]
        rm["global_test_kappa"] = global_test_result["kappa"]
        rm["global_test_specificity"] = global_test_result["specificity_macro"]
        rm["global_test_balanced_acc"] = global_test_result["balanced_acc"]

        print("\n" + "=" * 30)
        print(f"INDIVIDUAL CLIENT TESTS AFTER ROUND {r + 1}")
        print("=" * 30)

        for client_name in CLIENT_NAMES:
            client_train_ds = train_loaders[client_name].dataset
            client_cw = compute_class_weights_from_dataset(client_train_ds).to(DEVICE)
            client_criterion = nn.CrossEntropyLoss(weight=client_cw)

            client_result = evaluate_loader(
                global_model,
                test_loaders[client_name],
                client_criterion,
                class_names,
                title_prefix=f"{client_name}_round_{r + 1}",
                save_dir=OUTPUT_DIR,
                save_cm=True
            )

            rm[f"{client_name}_test_loss"] = client_result["loss"]
            rm[f"{client_name}_test_acc"] = client_result["accuracy"]
            rm[f"{client_name}_test_precision"] = client_result["precision_macro"]
            rm[f"{client_name}_test_recall"] = client_result["recall_macro"]
            rm[f"{client_name}_test_f1"] = client_result["f1_macro"]
            rm[f"{client_name}_test_kappa"] = client_result["kappa"]
            rm[f"{client_name}_test_specificity"] = client_result["specificity_macro"]
            rm[f"{client_name}_test_balanced_acc"] = client_result["balanced_acc"]

        round_metrics.append(rm)

        history["round"].append(r + 1)
        history["train_loss"].append(rm["train_loss"])
        history["train_acc"].append(rm["train_acc"])
        history["val_loss"].append(rm["val_loss"])
        history["val_acc"].append(rm["val_acc"])
        history["global_test_acc"].append(rm["global_test_acc"])

        print(
            f"\n[ROUND {r + 1}] "
            f"Train Loss: {rm['train_loss']:.4f} | Train Acc: {rm['train_acc']:.4f} | "
            f"Val Loss: {rm['val_loss']:.4f} | Val Acc: {rm['val_acc']:.4f} | "
            f"Global Test Acc: {rm['global_test_acc']:.4f}"
        )

        plot_round_curves(history, OUTPUT_DIR)

    elapsed = time.time() - start_time
    print(f"\nFederated training finished in {elapsed / 60:.2f} minutes")

    # Load best weights
    global_model.load_state_dict(best_model_wts)

    # =========================
    # Final evaluation
    # =========================
    final_results = []

    print("\n==============================")
    print("FINAL TEST RESULTS: ALL CLIENTS TOGETHER")
    print("==============================")
    result_all = evaluate_loader(
        global_model,
        global_test_loader,
        combined_criterion,
        class_names,
        title_prefix="all_clients_final",
        save_dir=OUTPUT_DIR,
        save_cm=True
    )
    final_results.append(result_all)

    print("\n==============================")
    print("FINAL TEST RESULTS: EACH CLIENT SEPARATELY")
    print("==============================")
    for client_name in CLIENT_NAMES:
        client_train_ds = train_loaders[client_name].dataset
        client_cw = compute_class_weights_from_dataset(client_train_ds).to(DEVICE)
        client_criterion = nn.CrossEntropyLoss(weight=client_cw)

        result_client = evaluate_loader(
            global_model,
            test_loaders[client_name],
            client_criterion,
            class_names,
            title_prefix=f"{client_name}_final",
            save_dir=OUTPUT_DIR,
            save_cm=True
        )
        final_results.append(result_client)

    # Save round metrics
    with open(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"), "w") as f:
        json.dump(round_metrics, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(round_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(round_metrics)

    # Save final metrics
    save_metrics_csv(final_results, os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    with open(os.path.join(OUTPUT_DIR, "final_test_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    # Save final model
    final_model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    torch.save({
        "model_state": global_model.state_dict(),
        "class_names": class_names,
        "client_names": CLIENT_NAMES,
        "mixstyle_layers": list(MIXSTYLE_LAYERS),
        "use_mixstyle": USE_MIXSTYLE
    }, final_model_path)

    print("\nSaved outputs to:")
    print(final_model_path)
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.json"))


if __name__ == "__main__":
    main()
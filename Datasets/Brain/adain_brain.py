import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import copy
import csv
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

# =========================================================
# Config
# =========================================================
SEED = 42
DATA_ROOT = r"/lustre06/project/6008975/csj5/braintumor"
OUTPUT_DIR = "brain_tumor_federated_comogan"
MODEL_NAME = "efficientnet_b0_brain_tumor_fedavg_comogan.pth"
WEIGHTS_PATH = r"/lustre06/project/6008975/csj5/narvalenv/pretrained/efficientnet_b0_rwightman-7f5810bc.pth"

BATCH_SIZE = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
LR = 1e-4
NUM_WORKERS = 0
IMG_SIZE = 224

CLIENT_NAMES = ["Sartajbhuvaji", "rm1000", "thomasdubail", "figshare"]

# AdaIN settings
HARMONIZE_TRAIN = True
HARMONIZE_VAL_TEST = False  # keep validation/test untouched by default
REFERENCE_IMAGES = {
    "Sartajbhuvaji": "gg (1).jpg",
    "rm1000": "Te-glTr_0000.jpg",
    "thomasdubail": "G_1.jpg",
    "figshare": "Te-gl_1.jpg",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# =========================================================
# Reproducibility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# Transforms
# =========================================================
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

# For AdaIN in pixel space we do NOT normalize.
to_tensor_no_norm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])
to_pil_from_tensor = transforms.ToPILImage()

# =========================================================
# Helpers
# =========================================================
def set_dataset_paths(root: str, client_names: List[str]):
    """
    Expected structure:
    DATA_ROOT/
        Sartajbhuvaji/train, val, test
        rm1000/train, val, test
        thomasdubail/train, val, test
        figshare/train, val, test
    Each split folder must contain class subfolders.
    """
    paths = {}
    for client in client_names:
        paths[client] = {
            "train": os.path.join(root, client, "train"),
            "val": os.path.join(root, client, "val"),
            "test": os.path.join(root, client, "test"),
        }
    return paths


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def is_image_file(fn: str) -> bool:
    return fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))


def list_images_recursive(d: str) -> List[str]:
    if not os.path.isdir(d):
        return []
    out = []
    for root, _, files in os.walk(d):
        for fn in files:
            if is_image_file(fn):
                out.append(os.path.join(root, fn))
    return out


def find_file_recursive(root_dir: str, target_filename: str) -> Optional[str]:
    if not os.path.isdir(root_dir):
        return None
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if fn == target_filename:
                return os.path.join(root, fn)
    return None


def relpath_from_root(root_dir: str, abs_path: str) -> str:
    return os.path.relpath(abs_path, root_dir)


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


def build_model(num_classes):
    """
    Build EfficientNet-B0 and initialize it from a local checkpoint.

    Expected checkpoint: ImageNet-pretrained EfficientNet-B0 weights.
    We load the backbone weights first, then replace the classifier head for
    the target brain-tumor classes.
    """
    model = efficientnet_b0(weights=None)

    if WEIGHTS_PATH is not None:
        if not os.path.isfile(WEIGHTS_PATH):
            raise FileNotFoundError(f"WEIGHTS_PATH not found: {WEIGHTS_PATH}")

        ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")

        # Support a few common checkpoint formats.
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                state_dict = ckpt["model"]
            elif "net" in ckpt:
                state_dict = ckpt["net"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        # Strip common wrappers like "module." if present.
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("module.", "")
            cleaned_state_dict[new_k] = v

        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"Loaded EfficientNet-B0 weights from: {WEIGHTS_PATH}")
        if missing:
            print(f"[Weights] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[Weights] Unexpected keys: {len(unexpected)}")
    else:
        print("WEIGHTS_PATH is None; initializing EfficientNet-B0 from scratch.")

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def compute_specificity_from_cm(cm):
    """
    Multiclass specificity for each class using one-vs-rest:
    specificity_i = TN_i / (TN_i + FP_i)
    """
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
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
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

    cm = confusion_matrix(targets, preds)
    precision_macro = precision_score(targets, preds, average="macro", zero_division=0)
    recall_macro = recall_score(targets, preds, average="macro", zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(targets, preds)
    per_class_specificity, macro_specificity = compute_specificity_from_cm(cm)

    print(f"\n=== {title_prefix.upper()} ===")
    print(f"Loss        : {loss:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {precision_macro:.4f}")
    print(f"Recall      : {recall_macro:.4f}")
    print(f"F1-score    : {f1_macro:.4f}")
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
        "loss": loss,
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "kappa": kappa,
        "specificity_macro": macro_specificity,
        "per_class_specificity": per_class_specificity,
        "cm": cm.tolist(),
    }


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
            ])


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

# =========================================================
# AdaIN helpers
# =========================================================
def image_load_as_tensor(path: str):
    img = default_loader(path)  # PIL image
    t = to_tensor_no_norm(img)  # CxHxW, float in [0,1]
    return t


def compute_channel_mean_std(tensor: torch.Tensor):
    # tensor: CxHxW
    c = tensor.view(tensor.shape[0], -1)
    mu = c.mean(dim=1)
    std = c.std(dim=1, unbiased=False)
    std = torch.clamp(std, min=1e-6)
    return mu, std


def adain_transfer(content_t: torch.Tensor, style_t: torch.Tensor):
    mu_c, std_c = compute_channel_mean_std(content_t)
    mu_s, std_s = compute_channel_mean_std(style_t)

    mu_c = mu_c[:, None, None]
    std_c = std_c[:, None, None]
    mu_s = mu_s[:, None, None]
    std_s = std_s[:, None, None]

    normalized = (content_t - mu_c) / std_c
    out = normalized * std_s + mu_s
    out = torch.clamp(out, 0.0, 1.0)
    return out


def get_reference_style_path(client_name: str, split_dir: str) -> Optional[str]:
    """
    Finds the reference image inside the given client's split tree.
    The file name is taken from REFERENCE_IMAGES.
    """
    ref_fn = REFERENCE_IMAGES.get(client_name)
    if ref_fn is None:
        return None
    return find_file_recursive(split_dir, ref_fn)


def harmonize_split_tree(src_root: str, dst_root: str, style_path: Optional[str], harmonize: bool = True):
    """
    Copy a full ImageFolder tree while optionally applying AdaIN to each image.
    Preserves class subfolders exactly so ImageFolder still works.
    """
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    ensure_dir(dst_root)

    for class_name in sorted(os.listdir(src_root)):
        src_class_dir = os.path.join(src_root, class_name)
        if not os.path.isdir(src_class_dir):
            continue
        dst_class_dir = os.path.join(dst_root, class_name)
        ensure_dir(dst_class_dir)

        for root, _, files in os.walk(src_class_dir):
            rel = os.path.relpath(root, src_root)
            dst_subdir = os.path.join(dst_root, rel)
            ensure_dir(dst_subdir)

            for fn in files:
                if not is_image_file(fn):
                    continue
                src_fp = os.path.join(root, fn)
                dst_fp = os.path.join(dst_subdir, fn)

                if not harmonize or style_path is None or not os.path.exists(style_path):
                    shutil.copy2(src_fp, dst_fp)
                    continue

                try:
                    content_t = image_load_as_tensor(src_fp)
                    style_t = image_load_as_tensor(style_path)
                    hm_t = adain_transfer(content_t, style_t)
                    to_pil_from_tensor(hm_t).save(dst_fp)
                except Exception as e:
                    print(f"[AdaIN] Failed harmonizing {src_fp}: {e}. Copying original.")
                    shutil.copy2(src_fp, dst_fp)


def create_harmonized_round(round_idx: int, paths_dict: Dict[str, Dict[str, str]], out_base_dir: str):
    """
    Build harmonized train/val/test directories for one communication round.
    Returns per-client dicts of used directories and a mapping of style paths.
    """
    round_root = os.path.join(out_base_dir, f"harmonized_round_{round_idx}")
    if os.path.exists(round_root):
        shutil.rmtree(round_root)
    ensure_dir(round_root)

    used_paths = {"train": {}, "val": {}, "test": {}}
    style_map = {}

    for client in CLIENT_NAMES:
        client_round_root = os.path.join(round_root, client)
        ensure_dir(client_round_root)

        train_src = paths_dict[client]["train"]
        val_src = paths_dict[client]["val"]
        test_src = paths_dict[client]["test"]

        style_path = get_reference_style_path(client, train_src)
        style_map[client] = style_path
        if style_path is None:
            print(f"[AdaIN] Warning: reference image not found for {client}. Using copy-only for that client.")
        else:
            print(f"[AdaIN] {client} style reference: {style_path}")

        train_dst = os.path.join(client_round_root, "train")
        val_dst = os.path.join(client_round_root, "val")
        test_dst = os.path.join(client_round_root, "test")

        harmonize_split_tree(train_src, train_dst, style_path, harmonize=HARMONIZE_TRAIN)
        harmonize_split_tree(val_src, val_dst, style_path, harmonize=HARMONIZE_VAL_TEST)
        harmonize_split_tree(test_src, test_dst, style_path, harmonize=HARMONIZE_VAL_TEST)

        used_paths["train"][client] = train_dst
        used_paths["val"][client] = val_dst
        used_paths["test"][client] = test_dst

    return used_paths, style_map, round_root


def make_reference_comparison_grid(original_train_roots: Dict[str, str], harmonized_train_roots: Dict[str, str], save_dir: str, round_idx: int):
    """
    Creates a 3 x 4 grid:
      row 1 = original image
      row 2 = harmonized image using AdaIN
      row 3 = amplified absolute difference
    One column per dataset/client.
    """
    ensure_dir(save_dir)

    n = len(CLIENT_NAMES)
    fig, axs = plt.subplots(3, n, figsize=(4.2 * n, 11))
    if n == 1:
        axs = np.array([[axs[0]], [axs[1]], [axs[2]]])

    for col, client in enumerate(CLIENT_NAMES):
        ref_fn = REFERENCE_IMAGES.get(client)
        orig_root = original_train_roots[client]
        hm_root = harmonized_train_roots[client]

        orig_fp = find_file_recursive(orig_root, ref_fn) if ref_fn else None
        hm_fp = find_file_recursive(hm_root, ref_fn) if ref_fn else None

        axs[0, col].axis("off")
        axs[1, col].axis("off")
        axs[2, col].axis("off")

        if orig_fp is None:
            axs[0, col].text(0.5, 0.5, f"Missing\n{client}", ha="center", va="center")
            axs[1, col].text(0.5, 0.5, "Missing", ha="center", va="center")
            axs[2, col].text(0.5, 0.5, "Missing", ha="center", va="center")
            axs[0, col].set_title(client)
            continue

        try:
            orig = np.array(Image.open(orig_fp).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        except Exception as e:
            axs[0, col].text(0.5, 0.5, f"Read error\n{e}", ha="center", va="center")
            axs[0, col].set_title(client)
            continue

        axs[0, col].imshow(orig)
        if col == 0:
            axs[0, col].set_ylabel("Original", fontsize=11)
        axs[0, col].set_title(f"{client}\n{ref_fn}", fontsize=10)

        if hm_fp is None:
            axs[1, col].text(0.5, 0.5, "No harmonized copy", ha="center", va="center")
            axs[2, col].text(0.5, 0.5, "No diff", ha="center", va="center")
            continue

        try:
            hm = np.array(Image.open(hm_fp).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        except Exception as e:
            axs[1, col].text(0.5, 0.5, f"Read error\n{e}", ha="center", va="center")
            axs[2, col].text(0.5, 0.5, "No diff", ha="center", va="center")
            continue

        axs[1, col].imshow(hm)
        if col == 0:
            axs[1, col].set_ylabel("AdaIN", fontsize=11)

        diff = np.abs(hm.astype(np.float32) - orig.astype(np.float32))
        amplified = diff * 3.0
        amplified = np.clip(amplified, 0, 255).astype(np.uint8)
        if amplified.max() < 8:
            if diff.max() > 0:
                amplified = (diff / (diff.max() + 1e-8) * 255.0).astype(np.uint8)
            else:
                amplified = np.zeros_like(amplified, dtype=np.uint8)

        axs[2, col].imshow(amplified)
        if col == 0:
            axs[2, col].set_ylabel("Abs diff x3", fontsize=11)

    fig.suptitle(f"AdaIN Comparison Grids - Round {round_idx}", fontsize=15)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    out_png = os.path.join(save_dir, f"comparison_grid_round_{round_idx}.png")
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[VIS] Saved comparison grid: {out_png}")


def plot_per_class_summaries(round_metrics, out_dir):
    # Optional convenience plot: global test accuracy curve already handled elsewhere.
    pass

# =========================================================
# Main
# =========================================================
def main():
    paths = set_dataset_paths(DATA_ROOT, CLIENT_NAMES)

    # Build baseline datasets to validate class ordering.
    train_datasets, class_names, class_to_idx = build_client_datasets(paths, "train", train_tfms)
    val_datasets, _, _ = build_client_datasets(paths, "val", eval_tfms)
    test_datasets, _, _ = build_client_datasets(paths, "test", eval_tfms)

    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Train samples (all clients):", sum(len(ds) for ds in train_datasets))
    print("Val samples   (all clients):", sum(len(ds) for ds in val_datasets))
    print("Test samples  (all clients):", sum(len(ds) for ds in test_datasets))

    for client, ds_tr, ds_va, ds_te in zip(CLIENT_NAMES, train_datasets, val_datasets, test_datasets):
        print(f"{client:15s} | train={len(ds_tr):5d}  val={len(ds_va):5d}  test={len(ds_te):5d}")

    global_model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(global_model.state_dict())
    history = defaultdict(list)
    round_metrics = []
    start_time = time.time()

    for r in range(COMM_ROUNDS):
        print(f"\n==================== Communication Round {r + 1}/{COMM_ROUNDS} ====================")

        # Create AdaIN-harmonized copies for this round.
        if HARMONIZE_TRAIN or HARMONIZE_VAL_TEST:
            used_paths, style_map, round_root = create_harmonized_round(r + 1, paths, OUTPUT_DIR)

            # Save comparison grid for the four reference images.
            original_train_roots = {client: paths[client]["train"] for client in CLIENT_NAMES}
            harmonized_train_roots = {client: used_paths["train"][client] for client in CLIENT_NAMES}
            grid_dir = os.path.join(round_root, "ComparisonGrid")
            make_reference_comparison_grid(original_train_roots, harmonized_train_roots, grid_dir, r + 1)

            train_paths = used_paths["train"]
            val_paths = used_paths["val"]
            test_paths = used_paths["test"]
        else:
            train_paths = {client: paths[client]["train"] for client in CLIENT_NAMES}
            val_paths = {client: paths[client]["val"] for client in CLIENT_NAMES}
            test_paths = {client: paths[client]["test"] for client in CLIENT_NAMES}

        # Build client-specific loaders from the chosen paths.
        train_loaders = {}
        val_loaders = {}
        test_loaders = {}
        for client in CLIENT_NAMES:
            train_ds = datasets.ImageFolder(train_paths[client], transform=train_tfms)
            val_ds = datasets.ImageFolder(val_paths[client], transform=eval_tfms)
            test_ds = datasets.ImageFolder(test_paths[client], transform=eval_tfms)

            train_loaders[client] = DataLoader(
                train_ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )
            val_loaders[client] = DataLoader(
                val_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )
            test_loaders[client] = DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
            )

        global_val_loader = DataLoader(
            ConcatDataset([datasets.ImageFolder(val_paths[c], transform=eval_tfms) for c in CLIENT_NAMES]),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        global_test_loader = DataLoader(
            ConcatDataset([datasets.ImageFolder(test_paths[c], transform=eval_tfms) for c in CLIENT_NAMES]),
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
        )

        local_models = []
        local_weights = []
        round_train_losses = []
        round_train_accs = []

        # -------- Local training on each client --------
        for client_name in CLIENT_NAMES:
            print(f"\n[Client {client_name}]")

            local_model = copy.deepcopy(global_model).to(DEVICE)
            optimizer = optim.AdamW(local_model.parameters(), lr=LR, weight_decay=1e-4)

            train_loader = train_loaders[client_name]
            val_loader = val_loaders[client_name]

            client_epoch_losses = []
            client_epoch_accs = []

            for ep in range(LOCAL_EPOCHS):
                tr_loss, tr_acc, _, _ = run_epoch(
                    local_model,
                    train_loader,
                    criterion,
                    optimizer=optimizer,
                    train=True,
                )
                client_epoch_losses.append(tr_loss)
                client_epoch_accs.append(tr_acc)
                print(f"  Local Epoch {ep + 1}/{LOCAL_EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

            val_loss, val_acc, _, _ = run_epoch(
                local_model,
                val_loader,
                criterion,
                optimizer=None,
                train=False,
            )
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            local_models.append(local_model)
            local_weights.append(len(train_loader.dataset))

            round_train_losses.append(float(np.mean(client_epoch_losses)))
            round_train_accs.append(float(np.mean(client_epoch_accs)))

        total_train_size = sum(local_weights)
        if total_train_size == 0:
            raise RuntimeError("Total training size across clients is 0. Check your dataset splits.")

        # -------- FedAvg aggregation --------
        norm_weights = [w / total_train_size for w in local_weights]
        global_model.load_state_dict(average_state_dicts_weighted(local_models, norm_weights))

        # -------- Global validation --------
        global_val_loss, global_val_acc, _, _ = run_epoch(
            global_model,
            global_val_loader,
            criterion,
            optimizer=None,
            train=False,
        )

        # -------- Save best model by global validation loss --------
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
        # TEST AFTER THIS COMM ROUND
        # =========================
        print("\n" + "=" * 30)
        print(f"GLOBAL TEST AFTER ROUND {r + 1} (ALL CLIENTS TOGETHER)")
        print("=" * 30)

        global_test_result = evaluate_loader(
            global_model,
            global_test_loader,
            criterion,
            class_names,
            title_prefix=f"global_round_{r + 1}",
            save_dir=OUTPUT_DIR,
            save_cm=True,
        )

        rm["global_test_loss"] = global_test_result["loss"]
        rm["global_test_acc"] = global_test_result["accuracy"]
        rm["global_test_precision"] = global_test_result["precision_macro"]
        rm["global_test_recall"] = global_test_result["recall_macro"]
        rm["global_test_f1"] = global_test_result["f1_macro"]
        rm["global_test_kappa"] = global_test_result["kappa"]
        rm["global_test_specificity"] = global_test_result["specificity_macro"]

        print("\n" + "=" * 30)
        print(f"INDIVIDUAL CLIENT TESTS AFTER ROUND {r + 1}")
        print("=" * 30)

        for client_name in CLIENT_NAMES:
            client_result = evaluate_loader(
                global_model,
                test_loaders[client_name],
                criterion,
                class_names,
                title_prefix=f"{client_name}_round_{r + 1}",
                save_dir=OUTPUT_DIR,
                save_cm=True,
            )

            rm[f"{client_name}_test_loss"] = client_result["loss"]
            rm[f"{client_name}_test_acc"] = client_result["accuracy"]
            rm[f"{client_name}_test_precision"] = client_result["precision_macro"]
            rm[f"{client_name}_test_recall"] = client_result["recall_macro"]
            rm[f"{client_name}_test_f1"] = client_result["f1_macro"]
            rm[f"{client_name}_test_kappa"] = client_result["kappa"]
            rm[f"{client_name}_test_specificity"] = client_result["specificity_macro"]

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
    # Final Test evaluation
    # =========================
    final_results = []

    print("\n==============================")
    print("FINAL TEST RESULTS: ALL CLIENTS TOGETHER")
    print("==============================")
    result_all = evaluate_loader(
        global_model,
        global_test_loader,
        criterion,
        class_names,
        title_prefix="all_clients_final",
        save_dir=OUTPUT_DIR,
        save_cm=True,
    )
    final_results.append(result_all)

    print("\n==============================")
    print("FINAL TEST RESULTS: EACH CLIENT SEPARATELY")
    print("==============================")
    for client_name in CLIENT_NAMES:
        result_client = evaluate_loader(
            global_model,
            test_loaders[client_name],
            criterion,
            class_names,
            title_prefix=f"{client_name}_final",
            save_dir=OUTPUT_DIR,
            save_cm=True,
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

    print("\nSaved outputs to:")
    print(os.path.join(OUTPUT_DIR, MODEL_NAME))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "harmonized_round_1", "ComparisonGrid", "comparison_grid_round_1.png"))


if __name__ == "__main__":
    main()

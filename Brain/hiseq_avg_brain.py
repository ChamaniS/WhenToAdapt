import os
import time
import copy
import random
import json
import csv
import glob
import numpy as np
from collections import defaultdict
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage import exposure

# =========================================================
# Logging
# =========================================================
output_file = r"/lustre06/project/6008975/csj5/narvalenv/hismat_avg_brain.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
sys.stdout = open(output_file, "w")

# =========================
# Config
# =========================
SEED = 42
DATA_ROOT = r"/lustre06/project/6008975/csj5/braintumor/"
OUTPUT_DIR = r"/lustre06/project/6008975/csj5/narvalenv/brain_tumor_avg_histmatch/"
MODEL_NAME = "efficientnet_b0_brain_tumor_fedavg_histmatch.pth"

# Put the downloaded EfficientNet-B0 weights file here
WEIGHTS_PATH = r"/lustre06/project/6008975/csj5/narvalenv/pretrained/efficientnet_b0_rwightman-7f5810bc.pth"

BATCH_SIZE = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
LR = 1e-4
NUM_WORKERS = 0
IMG_SIZE = 224

CLIENT_NAMES = ["Sartajbhuvaji", "rm1000", "thomasdubail", "figshare"]

# Use None to average ALL training samples from the largest client.
# Or set an integer to average only that many randomly chosen training images.
N_REF_SAMPLES = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# =========================
# Reproducibility
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "comparison_grids"), exist_ok=True)

# =========================
# Transforms
# =========================
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# =========================
# Helpers
# =========================
def set_dataset_paths(root, client_names):
    paths = {}
    for client in client_names:
        paths[client] = {
            "train": os.path.join(root, client, "train"),
            "val": os.path.join(root, client, "val"),
            "test": os.path.join(root, client, "test"),
        }
    return paths

def image_list_in_dir(root_dir):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = []
    if not os.path.isdir(root_dir):
        return files
    for ext in exts:
        files.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
    return sorted(files)

def match_histogram_rgb(src_rgb, ref_rgb):
    try:
        matched = exposure.match_histograms(src_rgb, ref_rgb, channel_axis=-1)
    except TypeError:
        matched = exposure.match_histograms(src_rgb, ref_rgb, multichannel=True)
    return np.clip(matched, 0, 255).astype(np.uint8)

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

def build_model(num_classes, weights_path=WEIGHTS_PATH):
    model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"Pretrained weights not found: {weights_path}\n"
            f"Download the official EfficientNet-B0 checkpoint and place it there."
        )

    checkpoint = torch.load(weights_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    filtered_state_dict = {}
    for k, v in state_dict.items():
        key = k.replace("module.", "")
        if key.startswith("classifier.1."):
            continue
        filtered_state_dict[key] = v

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)

    print(f"Loaded EfficientNet-B0 weights from: {weights_path}")
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

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
    model_sds = [m.state_dict() for m in models]

    for k in avg_sd.keys():
        if torch.is_floating_point(avg_sd[k]):
            avg_tensor = torch.zeros_like(avg_sd[k])
            for i in range(len(models)):
                avg_tensor += weights[i] * model_sds[i][k].to(avg_tensor.device, dtype=avg_tensor.dtype)
            avg_sd[k] = avg_tensor
        else:
            avg_sd[k] = model_sds[0][k]

    return avg_sd

# =========================
# Reference image: average image from largest client
# =========================
def _list_images_recursively(root_dir):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(root_dir):
        return []
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
    return sorted(files)

def select_reference_client_by_training_size(paths):
    train_counts = {}
    for client in CLIENT_NAMES:
        train_dir = paths[client]["train"]
        ds = datasets.ImageFolder(train_dir)
        train_counts[client] = len(ds)

    reference_client = max(train_counts, key=train_counts.get)
    reference_idx = CLIENT_NAMES.index(reference_client)

    print("\nTraining sample counts per client:")
    for c in CLIENT_NAMES:
        print(f"  {c:15s}: {train_counts[c]}")

    print(f"\nSelected reference client: {reference_client} "
          f"(highest training sample count: {train_counts[reference_client]})")

    return reference_idx, train_counts

def compute_average_reference_image(reference_train_dir, n_samples=None, resize_to=(224, 224)):
    """
    reference_train_dir should be the train folder itself, e.g.
    /.../figshare/train
    containing class subfolders like glioma, pituitary, meningioma, no_tumor.
    """
    img_paths = _list_images_recursively(reference_train_dir)
    if len(img_paths) == 0:
        raise ValueError(f"No training images found in reference train directory: {reference_train_dir}")

    if n_samples is not None and len(img_paths) > n_samples:
        rng = np.random.default_rng(SEED)
        img_paths = list(rng.choice(img_paths, size=n_samples, replace=False))

    acc = None
    count = 0

    for p in img_paths:
        img = Image.open(p).convert("RGB")
        if resize_to is not None:
            img = img.resize(resize_to, resample=Image.BILINEAR)
        arr = np.array(img).astype(np.float32)

        if acc is None:
            acc = arr
        else:
            acc += arr
        count += 1

    avg = (acc / max(count, 1)).astype(np.uint8)
    return avg, len(img_paths)

# =========================
# Comparison grid
# =========================
def save_comparison_grid_for_client(original_split_dir, reference_rgb, client_name, out_dir, split_name="train", sample_index=0):
    files = image_list_in_dir(original_split_dir)
    if len(files) == 0:
        print(f"[VIS] No images found for {client_name} ({split_name})")
        return

    sample_index = max(0, min(sample_index, len(files) - 1))
    sample_path = files[sample_index]
    sample_name = os.path.basename(sample_path)

    orig_img = Image.open(sample_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    orig_np = np.array(orig_img).astype(np.uint8)

    matched_np = match_histogram_rgb(orig_np, reference_rgb)
    diff_np = np.abs(matched_np.astype(np.int16) - orig_np.astype(np.int16)).astype(np.uint8)
    amplified_diff = np.clip(diff_np * 3, 0, 255).astype(np.uint8)

    fig, axs = plt.subplots(3, 1, figsize=(7, 10))

    axs[0].imshow(orig_np)
    axs[0].axis("off")
    axs[0].set_title(f"Original\nIndex: {sample_index} | File: {sample_name}", fontsize=10)

    axs[1].imshow(matched_np)
    axs[1].axis("off")
    axs[1].set_title(f"Histogram Matched\nIndex: {sample_index} | File: {sample_name}", fontsize=10)

    axs[2].imshow(amplified_diff)
    axs[2].axis("off")
    axs[2].set_title(f"Amplified Difference\nIndex: {sample_index} | File: {sample_name}", fontsize=10)

    fig.suptitle(
        f"Histogram Matching Comparison - {client_name} ({split_name})\n"
        f"Image index: {sample_index} | Image name: {sample_name}",
        fontsize=13
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])

    out_path = os.path.join(out_dir, "comparison_grids", f"comparison_{client_name}_{split_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[VIS] Saved comparison grid: {out_path}")
    print(f"[VIS] {client_name} {split_name} sample -> index={sample_index}, file={sample_name}")

# =========================
# Histogram-matched dataset
# =========================
class HistogramMatchedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, reference_rgb=None):
        super().__init__(root=root, transform=None, target_transform=None, loader=datasets.folder.default_loader)
        self.user_transform = transform
        self.reference_rgb = reference_rgb

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path).convert("RGB")
        img_np = np.array(img).astype(np.uint8)

        if self.reference_rgb is not None:
            matched_np = match_histogram_rgb(img_np, self.reference_rgb)
        else:
            matched_np = img_np

        matched_img = Image.fromarray(matched_np)

        if self.user_transform is not None:
            matched_img = self.user_transform(matched_img)

        return matched_img, target

def build_client_datasets(paths_dict, split, transform, reference_rgb):
    ds_list = []
    for client in CLIENT_NAMES:
        split_dir = paths_dict[client][split]
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing directory: {split_dir}")
        ds = HistogramMatchedImageFolder(split_dir, transform=transform, reference_rgb=reference_rgb)
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

# =========================
# Training / evaluation
# =========================
def run_epoch(model, loader, criterion, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

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
            "specificity_macro"
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
                f'{r["specificity_macro"]:.6f}'
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

# =========================
# Main
# =========================
def main():
    paths = set_dataset_paths(DATA_ROOT, CLIENT_NAMES)

    for client in CLIENT_NAMES:
        for split in ["train", "val", "test"]:
            if not os.path.isdir(paths[client][split]):
                raise FileNotFoundError(f"Missing directory: {paths[client][split]}")

    # Pick the client with the highest number of training samples
    reference_client_idx, train_counts = select_reference_client_by_training_size(paths)
    reference_client_name = CLIENT_NAMES[reference_client_idx]
    reference_train_dir = paths[reference_client_name]["train"]

    # Build the average reference image from some or all training samples
    reference_rgb, used_count = compute_average_reference_image(
        reference_train_dir,
        n_samples=N_REF_SAMPLES,          # None = all samples
        resize_to=(IMG_SIZE, IMG_SIZE)
    )

    ref_save_path = os.path.join(OUTPUT_DIR, "reference_average_image.png")
    Image.fromarray(reference_rgb).save(ref_save_path)
    print(f"\nReference average image saved to: {ref_save_path}")
    print(f"Reference dataset: {reference_client_name}")
    print(f"Number of images used for average: {used_count}")

    print("\nSaving one comparison grid per client...")
    for client in CLIENT_NAMES:
        save_comparison_grid_for_client(
            original_split_dir=paths[client]["train"],
            reference_rgb=reference_rgb,
            client_name=client,
            out_dir=OUTPUT_DIR,
            split_name="train",
            sample_index=0
        )

    # Build per-client datasets with histogram matching applied on the fly
    train_datasets, class_names, class_to_idx = build_client_datasets(paths, "train", train_tfms, reference_rgb)
    val_datasets, _, _ = build_client_datasets(paths, "val", eval_tfms, reference_rgb)
    test_datasets, _, _ = build_client_datasets(paths, "test", eval_tfms, reference_rgb)

    train_ds_all = build_combined_dataset(train_datasets)
    val_ds_all = build_combined_dataset(val_datasets)
    test_ds_all = build_combined_dataset(test_datasets)

    num_classes = len(class_names)

    print("\nClasses:", class_names)
    print("Train samples (all clients):", count_samples(train_ds_all))
    print("Val samples   (all clients):", count_samples(val_ds_all))
    print("Test samples  (all clients):", count_samples(test_ds_all))

    for client, ds_tr, ds_va, ds_te in zip(CLIENT_NAMES, train_datasets, val_datasets, test_datasets):
        print(f"{client:15s} | train={len(ds_tr):5d}  val={len(ds_va):5d}  test={len(ds_te):5d}")

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

    global_model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

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
                    train=True
                )
                client_epoch_losses.append(tr_loss)
                client_epoch_accs.append(tr_acc)
                print(f"  Local Epoch {ep + 1}/{LOCAL_EPOCHS} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f}")

            val_loss, val_acc, _, _ = run_epoch(
                local_model,
                val_loader,
                criterion,
                optimizer=None,
                train=False
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
            train=False
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
            save_cm=True
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
                save_cm=True
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
        save_cm=True
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

    print("\nSaved outputs to:")
    print(os.path.join(OUTPUT_DIR, MODEL_NAME))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "comparison_grids"))
    print(os.path.join(OUTPUT_DIR, "reference_average_image.png"))

if __name__ == "__main__":
    main()
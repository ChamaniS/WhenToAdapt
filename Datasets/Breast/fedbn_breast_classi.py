import os
import time
import copy
import random
import json
import csv
import numpy as np
from collections import defaultdict

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
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
SEED = 42
DATA_ROOT = r"C:\Users\csj5\Projects\Data\Breasttumor_classi_renamed"  # parent folder containing client folders
OUTPUT_DIR = "breast_tumor_federated_fedbn"
MODEL_NAME = "efficientnet_b0_breast_tumor_fedbn.pth"

BATCH_SIZE = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
LR = 1e-4
WEIGHT_DECAY = 0
NUM_WORKERS = 0
IMG_SIZE = 224

CLIENT_NAMES = ["BUSBRA", "BUS", "BUSI", "UDIAT"]

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
    try:
        from torchvision.models import EfficientNet_B0_Weights
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
    except Exception:
        model = efficientnet_b0(pretrained=True)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def get_batchnorm_state_keys(model):
    """
    Collect all BatchNorm-related state_dict keys so they remain client-local.
    FedBN keeps BN affine params and BN running stats local.
    """
    bn_keys = set()
    for module_name, module in model.named_modules():
        if isinstance(module, (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            prefix = module_name + "." if module_name else ""
            for suffix in [
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "num_batches_tracked",
            ]:
                bn_keys.add(prefix + suffix)
    return bn_keys

def extract_bn_state(model, bn_keys):
    sd = model.state_dict()
    return {k: sd[k].detach().cpu().clone() for k in bn_keys if k in sd}

def inject_bn_state(model, bn_state):
    sd = model.state_dict()
    for k, v in bn_state.items():
        if k in sd:
            sd[k] = v.detach().clone()
    model.load_state_dict(sd, strict=True)
    return model

def clone_model_with_bn(global_model, bn_state):
    """
    Clone the global model and inject a client-specific BN state.
    """
    model = copy.deepcopy(global_model)
    if bn_state is not None:
        inject_bn_state(model, bn_state)
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

def compute_metrics_from_arrays(targets, preds, class_names):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(targets, preds, labels=labels)
    precision_macro = precision_score(targets, preds, average="macro", zero_division=0)
    recall_macro = recall_score(targets, preds, average="macro", zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    kappa = cohen_kappa_score(targets, preds)
    per_class_specificity, macro_specificity = compute_specificity_from_cm(cm)

    return {
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "kappa": float(kappa),
        "specificity_macro": float(macro_specificity),
        "per_class_specificity": [float(x) for x in per_class_specificity],
        "cm": cm,
    }

def save_confusion_matrix(cm, class_names, title_prefix, save_dir):
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

def average_state_dicts_fedbn(local_state_dicts, weights, bn_keys, reference_state_dict):
    """
    FedBN aggregation:
      - Average all parameters except BatchNorm parameters and BatchNorm running stats.
      - Keep BN-related keys from the reference/global state dict.
    """
    avg_sd = {}
    keys = reference_state_dict.keys()

    for k in keys:
        if k in bn_keys:
            avg_sd[k] = reference_state_dict[k].detach().cpu().clone()
        else:
            avg_sd[k] = sum(
                weights[i] * local_state_dicts[i][k].detach().cpu()
                for i in range(len(local_state_dicts))
            )
    return avg_sd

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
    metrics = compute_metrics_from_arrays(targets, preds, class_names)

    print(f"\n=== {title_prefix.upper()} ===")
    print(f"Loss        : {loss:.4f}")
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {metrics['precision_macro']:.4f}")
    print(f"Recall      : {metrics['recall_macro']:.4f}")
    print(f"F1-score    : {metrics['f1_macro']:.4f}")
    print(f"Kappa       : {metrics['kappa']:.4f}")
    print(f"Specificity : {metrics['specificity_macro']:.4f}")

    print("\nPer-class Specificity:")
    for idx, cls_name in enumerate(class_names):
        print(f"{cls_name:15s}: {metrics['per_class_specificity'][idx]:.4f}")

    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=class_names, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print(metrics["cm"])

    if save_cm:
        save_confusion_matrix(metrics["cm"], class_names, title_prefix, save_dir)

    return {
        "split": title_prefix,
        "loss": float(loss),
        "accuracy": float(acc),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "f1_macro": float(metrics["f1_macro"]),
        "kappa": float(metrics["kappa"]),
        "specificity_macro": float(metrics["specificity_macro"]),
        "per_class_specificity": [float(x) for x in metrics["per_class_specificity"]],
        "cm": metrics["cm"].tolist(),
    }

def evaluate_fedbn_pooled(global_model, client_bn_states, loaders_by_client, criterion, class_names, title_prefix="pooled", save_dir=OUTPUT_DIR, save_cm=True):
    """
    FedBN-correct pooled evaluation:
    each client is evaluated with its own BN state.
    Predictions are pooled across clients.
    """
    total_loss = 0.0
    total_samples = 0
    all_targets = []
    all_preds = []

    print(f"\n=== {title_prefix.upper()} (FEDBN POOLED) ===")

    for client_name in CLIENT_NAMES:
        loader = loaders_by_client[client_name]
        client_model = clone_model_with_bn(global_model, client_bn_states[client_name]).to(DEVICE)

        loss, acc, targets, preds = run_epoch(client_model, loader, criterion, optimizer=None, train=False)
        n = len(loader.dataset)
        total_loss += loss * n
        total_samples += n
        all_targets.extend(targets)
        all_preds.extend(preds)

        print(f"{client_name:15s} | Loss: {loss:.4f} | Acc: {acc:.4f} | Samples: {n}")

    pooled_loss = total_loss / max(total_samples, 1)
    pooled_acc = accuracy_score(all_targets, all_preds)
    metrics = compute_metrics_from_arrays(all_targets, all_preds, class_names)

    print("\nPooled FedBN Metrics:")
    print(f"Loss        : {pooled_loss:.4f}")
    print(f"Accuracy    : {pooled_acc:.4f}")
    print(f"Precision   : {metrics['precision_macro']:.4f}")
    print(f"Recall      : {metrics['recall_macro']:.4f}")
    print(f"F1-score    : {metrics['f1_macro']:.4f}")
    print(f"Kappa       : {metrics['kappa']:.4f}")
    print(f"Specificity : {metrics['specificity_macro']:.4f}")

    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print(metrics["cm"])

    if save_cm:
        save_confusion_matrix(metrics["cm"], class_names, title_prefix, save_dir)

    return {
        "split": title_prefix,
        "loss": float(pooled_loss),
        "accuracy": float(pooled_acc),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "f1_macro": float(metrics["f1_macro"]),
        "kappa": float(metrics["kappa"]),
        "specificity_macro": float(metrics["specificity_macro"]),
        "per_class_specificity": [float(x) for x in metrics["per_class_specificity"]],
        "cm": metrics["cm"].tolist(),
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
                r.get("split", ""),
                f'{float(r.get("loss", 0.0)):.6f}',
                f'{float(r.get("accuracy", 0.0)):.6f}',
                f'{float(r.get("precision_macro", 0.0)):.6f}',
                f'{float(r.get("recall_macro", 0.0)):.6f}',
                f'{float(r.get("f1_macro", 0.0)):.6f}',
                f'{float(r.get("kappa", 0.0)):.6f}',
                f'{float(r.get("specificity_macro", 0.0)):.6f}'
            ])

def save_dict_list_csv(rows, path):
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

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
    plt.plot(rounds, history["global_test_acc"], label="Pooled Test Acc")
    plt.xlabel("Communication Round")
    plt.ylabel("Accuracy")
    plt.title("Pooled FedBN Test Accuracy Across Rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fed_global_test_acc.png"), dpi=300)
    plt.close()

def save_per_round_checkpoint(global_model, client_bn_states, round_idx, round_metrics, pooled_val_result, pooled_test_result, class_names):
    ckpt = {
        "round": round_idx,
        "model_state": global_model.state_dict(),
        "client_bn_states": client_bn_states,
        "pooled_val_metrics": pooled_val_result,
        "pooled_test_metrics": pooled_test_result,
        "class_names": class_names,
        "client_names": CLIENT_NAMES,
    }
    ckpt_path = os.path.join(OUTPUT_DIR, f"global_round_{round_idx}.pth")
    torch.save(ckpt, ckpt_path)
    print("Saved checkpoint:", ckpt_path)

# =========================
# Main
# =========================
def main():
    paths = set_dataset_paths(DATA_ROOT, CLIENT_NAMES)

    # Build per-client datasets
    train_datasets, class_names, class_to_idx = build_client_datasets(paths, "train", train_tfms)
    val_datasets, _, _ = build_client_datasets(paths, "val", eval_tfms)
    test_datasets, _, _ = build_client_datasets(paths, "test", eval_tfms)

    # Combined datasets only for sample counting
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

    # Model / loss
    global_model = build_model(num_classes).to(DEVICE)
    bn_keys = get_batchnorm_state_keys(global_model)

    # Each client keeps its own BN state across rounds
    client_bn_states = {
        client_name: extract_bn_state(global_model, bn_keys)
        for client_name in CLIENT_NAMES
    }

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_wts = copy.deepcopy(global_model.state_dict())

    history = defaultdict(list)
    round_metrics = []
    start_time = time.time()

    for r in range(COMM_ROUNDS):
        print(f"\n==================== Communication Round {r + 1}/{COMM_ROUNDS} ====================")

        local_state_dicts = []
        local_weights = []

        round_train_losses = []
        round_train_accs = []

        # -------- Local training on each client --------
        for client_name in CLIENT_NAMES:
            print(f"\n[Client {client_name}]")

            # Start from global non-BN weights, but restore this client's BN state
            local_model = clone_model_with_bn(global_model, client_bn_states[client_name]).to(DEVICE)
            optimizer = optim.AdamW(local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

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

            # Save client BN state for next round
            client_bn_states[client_name] = extract_bn_state(local_model, bn_keys)

            # Save the full local state dict for FedBN aggregation
            local_state_dicts.append({
                k: v.detach().cpu().clone() for k, v in local_model.state_dict().items()
            })
            local_weights.append(len(train_loader.dataset))

            round_train_losses.append(float(np.mean(client_epoch_losses)))
            round_train_accs.append(float(np.mean(client_epoch_accs)))

        total_train_size = sum(local_weights)
        if total_train_size == 0:
            raise RuntimeError("Total training size across clients is 0. Check your dataset splits.")

        # -------- FedBN aggregation --------
        norm_weights = [w / total_train_size for w in local_weights]
        global_reference_state = {
            k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()
        }
        avg_state = average_state_dicts_fedbn(
            local_state_dicts=local_state_dicts,
            weights=norm_weights,
            bn_keys=bn_keys,
            reference_state_dict=global_reference_state
        )
        global_model.load_state_dict(avg_state, strict=True)
        global_model.to(DEVICE)

        # -------- FedBN-correct pooled validation --------
        pooled_val_result = evaluate_fedbn_pooled(
            global_model=global_model,
            client_bn_states=client_bn_states,
            loaders_by_client=val_loaders,
            criterion=criterion,
            class_names=class_names,
            title_prefix=f"pooled_val_round_{r + 1}",
            save_dir=OUTPUT_DIR,
            save_cm=False
        )

        # -------- Save best model by pooled validation loss --------
        if pooled_val_result["loss"] < best_val_loss:
            best_val_loss = pooled_val_result["loss"]
            best_model_wts = copy.deepcopy(global_model.state_dict())
            torch.save(best_model_wts, os.path.join(OUTPUT_DIR, MODEL_NAME))
            print("\nSaved best global model.")

        rm = {
            "round": r + 1,
            "train_loss": float(np.mean(round_train_losses)),
            "train_acc": float(np.mean(round_train_accs)),
            "val_loss": float(pooled_val_result["loss"]),
            "val_acc": float(pooled_val_result["accuracy"]),
        }

        # =========================
        # FedBN-correct pooled test after this round
        # =========================
        print("\n" + "=" * 30)
        print(f"FEDBN POOLED TEST AFTER ROUND {r + 1}")
        print("=" * 30)

        pooled_test_result = evaluate_fedbn_pooled(
            global_model=global_model,
            client_bn_states=client_bn_states,
            loaders_by_client=test_loaders,
            criterion=criterion,
            class_names=class_names,
            title_prefix=f"pooled_test_round_{r + 1}",
            save_dir=OUTPUT_DIR,
            save_cm=True
        )

        rm["global_test_loss"] = pooled_test_result["loss"]
        rm["global_test_acc"] = pooled_test_result["accuracy"]
        rm["global_test_precision"] = pooled_test_result["precision_macro"]
        rm["global_test_recall"] = pooled_test_result["recall_macro"]
        rm["global_test_f1"] = pooled_test_result["f1_macro"]
        rm["global_test_kappa"] = pooled_test_result["kappa"]
        rm["global_test_specificity"] = pooled_test_result["specificity_macro"]

        print("\n" + "=" * 30)
        print(f"INDIVIDUAL CLIENT TESTS AFTER ROUND {r + 1}")
        print("=" * 30)

        for client_name in CLIENT_NAMES:
            # Build a client-specific evaluation model using the client's BN state
            client_eval_model = clone_model_with_bn(global_model, client_bn_states[client_name]).to(DEVICE)

            client_result = evaluate_loader(
                client_eval_model,
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
            f"Pooled Test Acc: {rm['global_test_acc']:.4f}"
        )

        plot_round_curves(history, OUTPUT_DIR)
        save_per_round_checkpoint(global_model, client_bn_states, r + 1, round_metrics, pooled_val_result, pooled_test_result, class_names)

        # Save per-round summary CSV after every round
        save_dict_list_csv(round_metrics, os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"))
        with open(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"), "w") as f:
            json.dump(round_metrics, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\nFederated training finished in {elapsed / 60:.2f} minutes")

    # Load best weights
    global_model.load_state_dict(best_model_wts)

    # =========================
    # Final evaluation
    # =========================
    final_results = []

    print("\n==============================")
    print("FINAL POOLED FEDBN TEST RESULTS")
    print("==============================")
    pooled_final = evaluate_fedbn_pooled(
        global_model=global_model,
        client_bn_states=client_bn_states,
        loaders_by_client=test_loaders,
        criterion=criterion,
        class_names=class_names,
        title_prefix="pooled_final_test",
        save_dir=OUTPUT_DIR,
        save_cm=True
    )
    final_results.append(pooled_final)

    print("\n==============================")
    print("FINAL TEST RESULTS: EACH CLIENT SEPARATELY")
    print("==============================")
    for client_name in CLIENT_NAMES:
        client_eval_model = clone_model_with_bn(global_model, client_bn_states[client_name]).to(DEVICE)

        result_client = evaluate_loader(
            client_eval_model,
            test_loaders[client_name],
            criterion,
            class_names,
            title_prefix=f"{client_name}_final",
            save_dir=OUTPUT_DIR,
            save_cm=True
        )
        final_results.append(result_client)

    # Save final metrics
    save_metrics_csv(final_results, os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    with open(os.path.join(OUTPUT_DIR, "final_test_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=2)

    # Save final model + BN states
    torch.save(
        {
            "model_state": global_model.state_dict(),
            "client_bn_states": client_bn_states,
            "class_names": class_names,
            "client_names": CLIENT_NAMES,
        },
        os.path.join(OUTPUT_DIR, "global_final_fedbn.pth")
    )

    print("\nSaved outputs to:")
    print(os.path.join(OUTPUT_DIR, MODEL_NAME))
    print(os.path.join(OUTPUT_DIR, "global_final_fedbn.pth"))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "federated_round_metrics.json"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.csv"))
    print(os.path.join(OUTPUT_DIR, "final_test_metrics.json"))

if __name__ == "__main__":
    main()
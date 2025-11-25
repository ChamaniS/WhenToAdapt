# fl_train.py
"""
Federated training across clients for TB CXR classification with optional SCAFFOLD integration.
SCAFFOLD implementation (Option II) based on:
  Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning".
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# avoid repeated torchvision C++ image extension warnings on Windows
from torchvision import io as tv_io
try:
    tv_io.set_image_backend('PIL')
except Exception:
    pass

import copy
import time
import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import timm
import torchvision
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, balanced_accuracy_score,
    cohen_kappa_score, confusion_matrix, accuracy_score
)
import matplotlib.pyplot as plt
from typing import List, Tuple

# ----------------------------
# CONFIG - set these paths
# ----------------------------
CLIENT_ROOTS = [
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Shenzhen",     # <-- update these 4 paths
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Montgomery",
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\TBX11K",
    r"C:\Users\csj5\Projects\Data\Tuberculosis_Data\Pakistan"
]
CLIENT_NAMES = ["Shenzhen", "Montgomery", "TBX11K", "Pakistan"]  # used only for display
OUTPUT_DIR = r"./fl_outputs"
ARCH = "densenet169"   # or other timm / torchvision model name
PRETRAINED = True
IMG_SIZE = 224
BATCH_SIZE = 4          # local batch size (per client)
WORKERS = 4
LOCAL_EPOCHS = 6      # local epochs per communication round (as requested)
COMM_ROUNDS = 10        # number of federated rounds
LR = 1e-4               # local optimizer lr = eta_l
WEIGHT_DECAY = 1e-5
USE_AMP = False         # set True to use amp when CUDA available
PIN_MEMORY = True
DROPOUT_P = 0.5
SEED = 42
CLASS_NAMES = ["normal", "positive"]  # canonical ordering

# --- SCAFFOLD-specific config ---
SCAFFOLD = True         # set True to run SCAFFOLD; False -> FedAvg behavior
ETA_G = 1.0             # global step size used in SCAFFOLD (server update)
# ----------------------------

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Dataset utilities (PathListDataset + multi-client gather)
# ----------------------------
class PathListDataset(Dataset):
    def __init__(self, samples: List[Tuple[str,int]], transform=None, loader=default_loader):
        self.samples = list(samples)
        self.transform = transform
        self.loader = loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label

def gather_samples_from_client_split(client_root: str, split: str, class_names: List[str]):
    split_dir = os.path.join(client_root, split)
    if not os.path.isdir(split_dir):
        raise ValueError(f"Missing {split} in {client_root}")
    samples = []
    canon_map = {c.lower(): i for i,c in enumerate(class_names)}
    for cls_folder in os.listdir(split_dir):
        cls_path = os.path.join(split_dir, cls_folder)
        if not os.path.isdir(cls_path): continue
        key = cls_folder.lower()
        if key not in canon_map:
            print(f"Warning: unknown class folder '{cls_folder}' in {split_dir}; skipping")
            continue
        label = canon_map[key]
        for fn in os.listdir(cls_path):
            if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                samples.append((os.path.join(cls_path, fn), label))
    return samples

def make_multi_client_dataloaders(client_roots, batch_size=BATCH_SIZE, image_size=IMG_SIZE, workers=WORKERS, pin_memory=PIN_MEMORY):
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
        normalize
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])

    train_samples_all, val_samples_all, test_samples_all = [], [], []
    per_client_dataloaders = []
    per_client_test_dsets = {}

    for client_root in client_roots:
        tr = gather_samples_from_client_split(client_root, "train", CLASS_NAMES)
        va = gather_samples_from_client_split(client_root, "val", CLASS_NAMES)
        te = gather_samples_from_client_split(client_root, "test", CLASS_NAMES)
        print(f"[DATA] client {client_root} -> train:{len(tr)} val:{len(va)} test:{len(te)}")
        train_samples_all.extend(tr); val_samples_all.extend(va); test_samples_all.extend(te)
        # per-client dataloaders
        train_ds = PathListDataset(tr, transform=train_tf)
        val_ds = PathListDataset(va, transform=val_tf)
        test_ds = PathListDataset(te, transform=val_tf)
        per_client_dataloaders.append({
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
            "train_ds": train_ds
        })
        per_client_test_dsets[client_root] = test_ds

    combined_train_ds = PathListDataset(train_samples_all, transform=train_tf)
    combined_val_ds = PathListDataset(val_samples_all, transform=val_tf)
    combined_test_ds = PathListDataset(test_samples_all, transform=val_tf)

    dataloaders_combined = {
        "train": DataLoader(combined_train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory),
        "val": DataLoader(combined_val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory),
        "test": DataLoader(combined_test_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)
    }
    sizes = {"train": len(combined_train_ds), "val": len(combined_val_ds), "test": len(combined_test_ds)}
    return dataloaders_combined, sizes, CLASS_NAMES, combined_train_ds, per_client_dataloaders, per_client_test_dsets

def compute_class_weights_from_dataset(dataset):
    targets = [s[1] for s in dataset.samples]
    counts = np.bincount(targets, minlength=len(CLASS_NAMES)).astype(np.float32)
    total = counts.sum()
    weights = total / (counts + 1e-8)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)

def make_weighted_sampler_from_dataset(dataset):
    targets = [s[1] for s in dataset.samples]
    counts = np.bincount(targets, minlength=len(CLASS_NAMES)).astype(np.float32)
    inv_freq = 1.0 / np.maximum(counts, 1.0)
    weights = [float(inv_freq[t]) for t in targets]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

# ----------------------------
# Model + training utilities
# ----------------------------
def create_model(num_classes, arch=ARCH, pretrained=PRETRAINED):
    if arch.startswith("densenet") and hasattr(torchvision.models, arch):
        model = getattr(torchvision.models, arch)(pretrained=pretrained)
        # replace classifier
        if hasattr(model, "classifier"):
            in_ch = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(in_ch, num_classes))
        elif hasattr(model, "fc"):
            in_ch = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(p=DROPOUT_P), nn.Linear(in_ch, num_classes))
        return model
    else:
        # use timm
        model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
        return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def average_models_weighted(models: List[torch.nn.Module], weights: List[float]):
    """FedAvg: average state_dict keys weighted by client weights. models are CPU tensors or on same device."""
    avg_sd = copy.deepcopy(models[0].state_dict())
    for k in avg_sd.keys():
        # sum weighted parameters (move to CPU to avoid GPU mem spikes)
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k].cpu() for i in range(len(models)))
    return avg_sd

# ----------------------------
# Helper: control variate utilities
# ----------------------------
def zeros_like_state_dict_on_cpu(state_dict):
    return {k: torch.zeros_like(v.cpu()) for k,v in state_dict.items()}

def state_dict_to_device(sd, device):
    return {k: v.to(device) for k,v in sd.items()}

# ----------------------------
# Training / evaluation loops
# ----------------------------
def train_local(model, dataloader, criterion, optimizer, device, epochs=LOCAL_EPOCHS, use_amp=False):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type=="cuda") else None
    logs = []
    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(dataloader, desc=f"LocalTrain ep{ep+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                out = model(x)
                loss = criterion(out, y)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            running_loss += float(loss.item()) * x.size(0)
            _, preds = out.max(1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            pbar.set_postfix(loss = running_loss / total if total>0 else 0.0, acc = correct/total if total>0 else 0.0)
        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        logs.append((epoch_loss, epoch_acc))
    return logs

def train_local_scaffold(model, dataloader, criterion, optimizer, device, ci_sd_cpu, c_sd_cpu, epochs=LOCAL_EPOCHS, use_amp=False, eta_l=LR):
    """
    Local training that applies SCAFFOLD correction to gradients: gradient += (c - c_i)
    - ci_sd_cpu: client control variate (CPU state dict)
    - c_sd_cpu: global control variate (CPU state dict)
    Returns:
      - updated local model (moved to CPU)
      - total number of local gradient steps (K)
    """
    model.to(device)
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type=="cuda") else None

    # create device copies (do NOT modify the original CPU storages)
    c_dev = state_dict_to_device(c_sd_cpu, device)
    ci_dev = state_dict_to_device(ci_sd_cpu, device)

    steps_per_epoch = max(1, len(dataloader))
    total_steps = steps_per_epoch * epochs

    model.train()
    step_count = 0
    # mapping from parameter/state_dict name -> parameter tensor
    param_map = {name: p for name, p in model.named_parameters()}

    for ep in range(epochs):
        pbar = tqdm(dataloader, desc=f"SCAFFOLD Local ep{ep+1}/{epochs}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                out = model(x)
                loss = criterion(out, y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # unscale so we can safely modify grads
            else:
                loss.backward()

            # Adjust gradients: for each parameter p, p.grad += (c_param - ci_param) (device tensors)
            with torch.no_grad():
                # iterate keys present in both c_dev and param_map
                for k, c_val in c_dev.items():
                    if k in param_map:
                        p = param_map[k]
                        if p.grad is None:
                            continue
                        # c_dev[k] and ci_dev[k] are device tensors
                        p.grad.add_(c_val - ci_dev[k])

            if scaler:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()

            step_count += 1
            pbar.set_postfix(step=step_count)

    # return model on CPU and total steps
    return model.cpu(), total_steps

@torch.no_grad()
def evaluate_model(model, dataloader, device, criterion=None, return_per_class=False, class_names=None):
    model.to(device)
    model.eval()
    all_y = []; all_pred = []
    total_loss = 0.0; n = 0
    for x, y in tqdm(dataloader, desc="Eval", leave=False):
        x, y = x.to(device), y.to(device)
        out = model(x)
        _, preds = out.max(1)
        all_y.extend(y.cpu().numpy().tolist())
        all_pred.extend(preds.cpu().numpy().tolist())
        if criterion is not None:
            loss = criterion(out, y)
            total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    if n == 0:
        return {}
    acc = accuracy_score(all_y, all_pred)
    prec_macro = precision_score(all_y, all_pred, average="macro", zero_division=0)
    rec_macro = recall_score(all_y, all_pred, average="macro", zero_division=0)
    f1_macro = f1_score(all_y, all_pred, average="macro", zero_division=0)
    bal = balanced_accuracy_score(all_y, all_pred)
    kappa = cohen_kappa_score(all_y, all_pred)
    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "balanced_acc": float(bal),
        "cohen_kappa": float(kappa)
    }
    if criterion is not None:
        metrics["loss"] = float(total_loss / max(1, n))
    if return_per_class:
        if class_names is None:
            raise ValueError("class_names must be provided when return_per_class=True")
        num_classes = len(class_names)
        cm = confusion_matrix(all_y, all_pred, labels=list(range(num_classes)))
        per_class_prec = precision_score(all_y, all_pred, labels=list(range(num_classes)), average=None, zero_division=0)
        per_class_rec = recall_score(all_y, all_pred, labels=list(range(num_classes)), average=None, zero_division=0)
        per_class_f1 = f1_score(all_y, all_pred, labels=list(range(num_classes)), average=None, zero_division=0)
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        tn = cm.sum() - (tp + fp + fn)
        per_class_spec = np.divide(tn, (tn + fp), out=np.zeros_like(tn), where=(tn + fp) != 0)
        support = cm.sum(axis=1).astype(float)
        per_class_acc = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
        metrics.update({
            "confusion_matrix": cm,
            "per_class_precision": [float(x) for x in per_class_prec.tolist()],
            "per_class_recall": [float(x) for x in per_class_rec.tolist()],
            "per_class_f1": [float(x) for x in per_class_f1.tolist()],
            "per_class_specificity": [float(x) for x in per_class_spec.tolist()],
            "per_class_correct": [int(x) for x in tp.tolist()],
            "per_class_accuracy": [float(x) for x in per_class_acc.tolist()],
            "per_class_support": [int(x) for x in support.tolist()]
        })
    return metrics

# ----------------------------
# Federated training main with SCAFFOLD integration
# ----------------------------
def main():
    print("DEVICE:", DEVICE)
    # Create dataloaders per-client and combined loaders
    combined_loaders, combined_sizes, class_names, combined_train_ds, per_client_dataloaders, per_client_test_dsets = make_multi_client_dataloaders(
        CLIENT_ROOTS, batch_size=BATCH_SIZE, image_size=IMG_SIZE, workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE.type=="cuda")
    )
    num_classes = len(class_names)
    print("class names:", class_names)
    # Determine client train sizes for weighting
    client_train_sizes = [len(per_client_dataloaders[i]['train'].dataset) for i in range(len(per_client_dataloaders))]
    total_train = sum(client_train_sizes) if sum(client_train_sizes) > 0 else 1
    print("client train sizes:", client_train_sizes)

    # Initialize global model
    global_model = create_model(num_classes=num_classes, arch=ARCH, pretrained=PRETRAINED).to(DEVICE)
    print(f"Global model {ARCH} created with {count_parameters(global_model):,} trainable params")

    # Initialize SCAFFOLD control variates (explicitly keep them on CPU)
    initial_sd = global_model.state_dict()
    c_global = zeros_like_state_dict_on_cpu(initial_sd)  # CPU tensors
    c_clients = [zeros_like_state_dict_on_cpu(initial_sd) for _ in range(len(per_client_dataloaders))]  # per-client CI on CPU

    # Sanity: assert control variates on CPU
    for k, v in c_global.items():
        assert v.device.type == "cpu"
    for ci in c_clients:
        for k,v in ci.items():
            assert v.device.type == "cpu"

    # Prepare logging
    round_results = []
    global_test_acc_fname = os.path.join(OUTPUT_DIR, "global_test_accuracy_rounds.png")
    global_test_loss_fname = os.path.join(OUTPUT_DIR, "global_test_loss_rounds.png")
    per_client_acc_history = {i: [] for i in range(len(per_client_dataloaders))}
    per_client_loss_history = {i: [] for i in range(len(per_client_dataloaders))}

    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Main federated rounds
    for r in range(COMM_ROUNDS):
        print("\n" + "="*60)
        print(f"COMM ROUND {r+1}/{COMM_ROUNDS}")
        print("="*60)
        local_models = []
        local_deltas = []      # stores yi - x (state_dicts) for aggregation (CPU tensors)
        # store tuples (client_index, delta_ci) so we update the correct client's ci when aggregating
        local_delta_cs = []    # list of (client_idx, delta_ci) with CPU tensors
        weights = []
        round_summary = {"round": r+1}

        # Each client trains locally starting from global weights
        for i, client in enumerate(per_client_dataloaders):
            print(f"\n[CLIENT {i}] {CLIENT_NAMES[i]}: local training (SCAFFOLD={SCAFFOLD})")
            # copy global to local model
            local_model = copy.deepcopy(global_model)
            global_sd_cpu = {k: v.cpu() for k,v in global_model.state_dict().items()}  # CPU copy of global params
            # per-client class weights from its train set
            train_ds = client['train'].dataset
            client_cw = compute_class_weights_from_dataset(train_ds).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=client_cw)
            optimizer = AdamW(local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

            if not SCAFFOLD:
                # Original FedAvg local training
                logs = train_local(local_model, client['train'], criterion, optimizer, DEVICE, epochs=LOCAL_EPOCHS, use_amp=USE_AMP)
                last_train_loss, last_train_acc = logs[-1]
                print(f"[CLIENT {i}] last local epoch loss={last_train_loss:.4f}, acc={last_train_acc:.4f}")
            else:
                # SCAFFOLD local training: adjust gradients by (c - c_i).
                trained_local_model, K_local_steps = train_local_scaffold(
                    local_model, client['train'], criterion, optimizer, DEVICE,
                    ci_sd_cpu=c_clients[i], c_sd_cpu=c_global, epochs=LOCAL_EPOCHS, use_amp=USE_AMP, eta_l=LR
                )

                # Compute final train loss/acc on train loader as approximation
                try:
                    train_metrics = evaluate_model(trained_local_model, client['train'], DEVICE, criterion=criterion)
                    last_train_loss = float(train_metrics.get("loss", np.nan))
                    last_train_acc = float(train_metrics.get("accuracy", np.nan))
                except Exception:
                    last_train_loss, last_train_acc = np.nan, np.nan

                print(f"[CLIENT {i}] last local train loss={last_train_loss:.4f}, acc={last_train_acc:.4f}")
                local_model = trained_local_model  # CPU model returned

                # compute c_i_plus using Option II:
                # c_i_plus = c_i - c + (x - y_i) / (K * eta_l)
                local_sd_cpu = local_model.state_dict()
                K = max(1, int(K_local_steps))
                ci_plus = {}
                # All operands are CPU tensors
                for k in c_clients[i].keys():
                    xi = global_sd_cpu[k]
                    yi = local_sd_cpu[k].cpu()
                    diff = (xi - yi) / float(K * LR)
                    ci_plus[k] = (c_clients[i][k].cpu() - c_global[k].cpu()) + diff

                # compute delta ci = ci_plus - ci (CPU)
                delta_ci = {k: (ci_plus[k] - c_clients[i][k].cpu()) for k in c_clients[i].keys()}

                # append (client_index, delta_ci) for aggregation at server
                local_delta_cs.append((i, delta_ci))

            # validation for that client (local)
            print(f"[CLIENT {i}] local validation")
            local_model_to_eval = local_model.to(DEVICE)
            local_val_metrics = evaluate_model(local_model_to_eval, client['val'], DEVICE, criterion=criterion)
            print(f"[CLIENT {i}] local val acc={local_val_metrics.get('accuracy', np.nan):.4f}, loss={local_val_metrics.get('loss', np.nan):.4f}")
            round_summary[f"client{i}_localval_loss"] = float(local_val_metrics.get("loss", np.nan))
            round_summary[f"client{i}_localval_acc"] = float(local_val_metrics.get("accuracy", np.nan))

            # Move local model to CPU for aggregation
            local_models.append(local_model.cpu())

            # compute Δy = y_i - x (local - global) for server aggregation (CPU)
            local_sd_cpu = local_model.state_dict()
            delta_y = {}
            for k in global_sd_cpu.keys():
                delta_y[k] = (local_sd_cpu[k].cpu() - global_sd_cpu[k].cpu())
            local_deltas.append(delta_y)

            # aggregation weight (based on client train size)
            w = float(client_train_sizes[i]) / float(total_train)
            weights.append(w)
            print(f"[CLIENT {i}] aggregation weight: {w:.4f}")

        # Server aggregation
        if not SCAFFOLD:
            # Weighted FedAvg (original behavior)
            print("\nAggregating local models (FedAvg weighted)")
            avg_state = average_models_weighted(local_models, weights)
            global_model.load_state_dict(avg_state)
            global_model.to(DEVICE)
        else:
            # SCAFFOLD server updates:
            # 1) x <- x + eta_g * average(Δy)  (here Δy = y_i - x)
            # 2) c <- c + (1/N) * sum(delta_ci)
            print("\nAggregating local updates (SCAFFOLD)")
            # average Δy over participating clients (CPU)
            n_clients = float(len(local_deltas))
            if n_clients <= 0:
                print("No local updates received; skipping aggregation")
            else:
                # initialize avg_dy with zeros on CPU
                avg_dy = {k: torch.zeros_like(v.cpu()) for k,v in local_deltas[0].items()}
                for d in local_deltas:
                    for k in d.keys():
                        avg_dy[k] = avg_dy[k] + d[k].cpu()
                for k in avg_dy.keys():
                    avg_dy[k] = avg_dy[k] / n_clients

                # apply server model update: x += eta_g * avg_dy (work on CPU then load to model)
                global_sd = global_model.state_dict()
                for k in global_sd.keys():
                    global_sd[k] = (global_sd[k].cpu() + (ETA_G * avg_dy[k].cpu()))
                global_model.load_state_dict(global_sd)
                global_model.to(DEVICE)

                # update control variate c: c += (1/N_total) * sum(delta_ci)  (all CPU)
                if len(local_delta_cs) > 0:
                    sum_delta_c = {k: torch.zeros_like(v.cpu()) for k,v in c_global.items()}  # CPU zeros
                    # add each client's delta (ensure CPU)
                    for (client_idx, dci) in local_delta_cs:
                        for k in dci.keys():
                            sum_delta_c[k] = sum_delta_c[k] + dci[k].cpu()
                    N_total = float(len(c_clients))
                    for k in c_global.keys():
                        c_global[k] = (c_global[k].cpu() + (sum_delta_c[k].cpu() / N_total)).cpu()

                    # update per-client c_i for participating clients: ci <- ci + delta_ci
                    for (client_idx, dci) in local_delta_cs:
                        if 0 <= client_idx < len(c_clients):
                            for k in c_clients[client_idx].keys():
                                c_clients[client_idx][k] = (c_clients[client_idx][k].cpu() + dci[k].cpu()).cpu()
                else:
                    # no client delta ci (if SCAFFOLD disabled or no clients), do nothing
                    pass

        # Global validation on combined val (concatenate client val datasets)
        print("\nGlobal validation on combined val sets...")
        combined_val_dsets = [per_client_dataloaders[i]['val'].dataset for i in range(len(per_client_dataloaders))]
        combined_val = ConcatDataset(combined_val_dsets)
        combined_val_loader = DataLoader(combined_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE.type=="cuda"))

        # build combined class weights from combined train targets
        combined_train_targets = []
        for i in range(len(per_client_dataloaders)):
            combined_train_targets.extend([s[1] for s in per_client_dataloaders[i]['train'].dataset.samples])
        counts = np.bincount(combined_train_targets, minlength=num_classes).astype(np.float32)
        counts[counts==0] = 1.0
        weights_arr = 1.0 / counts
        weights_arr = weights_arr * (len(weights_arr) / weights_arr.sum())
        combined_class_weights = torch.tensor(weights_arr, dtype=torch.float32).to(DEVICE)
        combined_criterion = nn.CrossEntropyLoss(weight=combined_class_weights)

        global_val_metrics = evaluate_model(global_model, combined_val_loader, DEVICE, criterion=combined_criterion)
        print("Global combined val metrics:", global_val_metrics)
        round_summary["global_val_loss"] = float(global_val_metrics.get("loss", np.nan))
        round_summary["global_val_acc"] = float(global_val_metrics.get("accuracy", np.nan))

        # Global test on combined test set
        print("\nGlobal TEST on combined test (all clients)")
        combined_test_dsets = [per_client_dataloaders[i]['test'].dataset for i in range(len(per_client_dataloaders))]
        combined_test = ConcatDataset(combined_test_dsets)
        combined_test_loader = DataLoader(combined_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEMORY and (DEVICE.type=="cuda"))
        global_test_metrics = evaluate_model(global_model, combined_test_loader, DEVICE, criterion=combined_criterion, return_per_class=True, class_names=class_names)
        print("Global combined TEST metrics summary:", {k: global_test_metrics.get(k) for k in ["accuracy","loss","f1_macro","precision_macro","recall_macro","balanced_acc","cohen_kappa"]})
        round_summary["global_test_loss"] = float(global_test_metrics.get("loss", np.nan))
        round_summary["global_test_acc"] = float(global_test_metrics.get("accuracy", np.nan))

        # --- per-class print & CSV saving (same as original script) ---
        try:
            cm = global_test_metrics.get("confusion_matrix", None)
            per_prec = global_test_metrics.get("per_class_precision", [])
            per_rec = global_test_metrics.get("per_class_recall", [])
            per_f1 = global_test_metrics.get("per_class_f1", [])
            per_spec = global_test_metrics.get("per_class_specificity", [])
            per_acc = global_test_metrics.get("per_class_accuracy", [])
            per_support = global_test_metrics.get("per_class_support", [])
            per_correct = global_test_metrics.get("per_class_correct", [])

            print("\nCombined TEST per-class metrics (order = {}):".format(class_names))
            header = ["Class", "Support", "Correct", "Acc", "Prec", "Rec", "F1", "Spec"]
            print("{:12s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format(*header))
            for ci, cname in enumerate(class_names):
                s = int(per_support[ci]) if ci < len(per_support) else 0
                ccount = int(per_correct[ci]) if ci < len(per_correct) else 0
                acc_val = float(per_acc[ci]) if ci < len(per_acc) else np.nan
                pval = float(per_prec[ci]) if ci < len(per_prec) else np.nan
                rval = float(per_rec[ci]) if ci < len(per_rec) else np.nan
                fval = float(per_f1[ci]) if ci < len(per_f1) else np.nan
                sval = float(per_spec[ci]) if ci < len(per_spec) else np.nan
                print("{:12s} {:8d} {:8d} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format(
                    cname, s, ccount, acc_val, pval, rval, fval, sval
                ))

            combined_rows = []
            if cm is not None:
                for ci, cname in enumerate(class_names):
                    combined_rows.append({
                        "class": cname,
                        "support": int(per_support[ci]) if ci < len(per_support) else 0,
                        "correct": int(per_correct[ci]) if ci < len(per_correct) else 0,
                        "acc": float(per_acc[ci]) if ci < len(per_acc) else np.nan,
                        "prec": float(per_prec[ci]) if ci < len(per_prec) else np.nan,
                        "rec": float(per_rec[ci]) if ci < len(per_rec) else np.nan,
                        "f1": float(per_f1[ci]) if ci < len(per_f1) else np.nan,
                        "spec": float(per_spec[ci]) if ci < len(per_spec) else np.nan
                    })

            macro_row = {
                "class": "macro",
                "support": int(cm.sum()) if cm is not None else sum([r["support"] for r in combined_rows]) if combined_rows else 0,
                "correct": int(np.sum([r["correct"] for r in combined_rows])) if combined_rows else 0,
                "acc": float(global_test_metrics.get("balanced_acc", np.nan)),
                "prec": float(global_test_metrics.get("precision_macro", np.nan)),
                "rec": float(global_test_metrics.get("recall_macro", np.nan)),
                "f1": float(global_test_metrics.get("f1_macro", np.nan)),
                "spec": float(np.mean(per_spec)) if len(per_spec) > 0 else float(np.nan)
            }
            combined_rows.append(macro_row)
            df_combined = pd.DataFrame(combined_rows)
            combined_csv = os.path.join(OUTPUT_DIR, "combined_test_metrics.csv")
            df_combined.to_csv(combined_csv, index=False)
            if cm is not None:
                pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(os.path.join(OUTPUT_DIR, "combined_confusion_matrix.csv"))
            print(f"Saved combined per-class test metrics CSV to: {combined_csv}")
        except Exception as e:
            print("Warning saving/printing combined per-class metrics:", e)

        # Global test per client
        per_client_test_metrics = []
        for i, client in enumerate(per_client_dataloaders):
            print(f"\nGlobal TEST on client {i} ({CLIENT_NAMES[i]}) test set")
            # use class weights computed from that client's train set
            client_train_ds = client['train'].dataset
            client_cw = compute_class_weights_from_dataset(client_train_ds).to(DEVICE)
            client_criterion = nn.CrossEntropyLoss(weight=client_cw)
            cl_metrics = evaluate_model(global_model, client['test'], DEVICE, criterion=client_criterion, return_per_class=True, class_names=class_names)

            # compute mean specificity across classes (if per-class specificity present)
            mean_spec = None
            if "per_class_specificity" in cl_metrics:
                specs = cl_metrics.get("per_class_specificity", [])
                if len(specs) > 0:
                    mean_spec = float(np.mean(specs))

            # Print requested summary metrics for this client
            acc = cl_metrics.get("accuracy", np.nan)
            prec = cl_metrics.get("precision_macro", np.nan)
            rec = cl_metrics.get("recall_macro", np.nan)
            f1 = cl_metrics.get("f1_macro", np.nan)
            kappa = cl_metrics.get("cohen_kappa", np.nan)
            if mean_spec is None and "per_class_specificity" in cl_metrics:
                mean_spec = float(np.mean(cl_metrics.get("per_class_specificity", [np.nan])))

            print(f"[CLIENT {i}] Summary metrics:")
            print(f"  Accuracy       : {acc:.4f}")
            print(f"  Precision (mac): {prec:.4f}")
            print(f"  Recall (mac)   : {rec:.4f}")
            print(f"  F1 (mac)       : {f1:.4f}")
            if mean_spec is not None:
                print(f"  Mean Specificity: {mean_spec:.4f}")
            else:
                print(f"  Mean Specificity: n/a")
            print(f"  Cohen's kappa  : {kappa:.4f}")

            # Per-class table formatted per your requested layout
            if "per_class_precision" in cl_metrics:
                print(f"\n  Per-class metrics (order = {class_names}):")
                header = ["Class", "Support", "Correct", "Acc", "Prec", "Rec", "F1", "Spec"]
                print("    " + "{:12s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format(*header))
                cmc = cl_metrics.get("confusion_matrix", None)
                tp_counts = np.diag(cmc).astype(int) if cmc is not None else [0]*len(class_names)
                supports = cmc.sum(axis=1).astype(int) if cmc is not None else [0]*len(class_names)
                precisions = cl_metrics.get("per_class_precision", [])
                recalls = cl_metrics.get("per_class_recall", [])
                f1s = cl_metrics.get("per_class_f1", [])
                specs = cl_metrics.get("per_class_specificity", [])
                accs = cl_metrics.get("per_class_accuracy", [])
                for ci, cname in enumerate(class_names):
                    s = supports[ci] if ci < len(supports) else 0
                    ccount = tp_counts[ci] if ci < len(tp_counts) else 0
                    acc_val = accs[ci] if ci < len(accs) else np.nan
                    pval = precisions[ci] if ci < len(precisions) else np.nan
                    rval = recalls[ci] if ci < len(recalls) else np.nan
                    fval = f1s[ci] if ci < len(f1s) else np.nan
                    sval = specs[ci] if ci < len(specs) else np.nan
                    print("    {:12s} {:8d} {:8d} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format(
                        cname, int(s), int(ccount), float(acc_val), float(pval), float(rval), float(fval), float(sval)
                    ))

            # Save per-client metrics CSV (match format used for combined)
            try:
                cmc = cl_metrics.get("confusion_matrix", None)
                rows=[]
                if cmc is not None:
                    per_support = cl_metrics.get("per_class_support", [])
                    per_correct = cl_metrics.get("per_class_correct", [])
                    per_accs = cl_metrics.get("per_class_accuracy", [])
                    per_precs = cl_metrics.get("per_class_precision", [])
                    per_recs = cl_metrics.get("per_class_recall", [])
                    per_f1s = cl_metrics.get("per_class_f1", [])
                    per_specs = cl_metrics.get("per_class_specificity", [])
                    for i_c, cname in enumerate(class_names):
                        rows.append({
                            "class": cname,
                            "support": int(per_support[i_c]) if i_c < len(per_support) else int(cmc[i_c].sum()) if cmc is not None else 0,
                            "correct": int(per_correct[i_c]) if i_c < len(per_correct) else int(np.diag(cmc)[i_c]) if cmc is not None else 0,
                            "acc": float(per_accs[i_c]) if i_c < len(per_accs) else np.nan,
                            "prec": float(per_precs[i_c]) if i_c < len(per_precs) else np.nan,
                            "rec": float(per_recs[i_c]) if i_c < len(per_recs) else np.nan,
                            "f1": float(per_f1s[i_c]) if i_c < len(per_f1s) else np.nan,
                            "spec": float(per_specs[i_c]) if i_c < len(per_specs) else np.nan
                        })
                macro = {
                    "class": "macro",
                    "support": int(cmc.sum()) if cmc is not None else 0,
                    "correct": int(np.sum([r["correct"] for r in rows])) if rows else 0,
                    "acc": cl_metrics.get("balanced_acc", np.nan),
                    "prec": cl_metrics.get("precision_macro", np.nan),
                    "rec": cl_metrics.get("recall_macro", np.nan),
                    "f1": cl_metrics.get("f1_macro", np.nan),
                    "spec": float(np.mean(cl_metrics.get("per_class_specificity", [np.nan]))) if "per_class_specificity" in cl_metrics else np.nan
                }
                if rows:
                    rows.append(macro)
                    dfc = pd.DataFrame(rows)
                    safe_name = CLIENT_NAMES[i].replace(" ", "_")
                    client_csv = os.path.join(OUTPUT_DIR, f"{safe_name}_test_metrics_round{r+1}.csv")
                    dfc.to_csv(client_csv, index=False)
                    if cmc is not None:
                        pd.DataFrame(cmc, index=class_names, columns=class_names).to_csv(os.path.join(OUTPUT_DIR, f"{safe_name}_confusion_matrix_round{r+1}.csv"))
                    print(f"Saved per-client metrics CSV to {client_csv}")
            except Exception as e:
                print("Warning saving per-client metrics CSV:", e)

            per_client_test_metrics.append(cl_metrics)
            round_summary[f"client{i}_test_loss"] = float(cl_metrics.get("loss", np.nan))
            round_summary[f"client{i}_test_acc"] = float(cl_metrics.get("accuracy", np.nan))
            per_client_acc_history[i].append(float(cl_metrics.get("accuracy", np.nan)))
            per_client_loss_history[i].append(float(cl_metrics.get("loss", np.nan)))

        # Save checkpoint and per-round summary (also persist SCAFFOLD state)
        ckpt = {
            "round": r+1,
            "model_state": global_model.state_dict(),
            "global_val_metrics": global_val_metrics,
            "global_test_metrics": global_test_metrics,
            "per_client_test_metrics": per_client_test_metrics,
            "client_names": CLIENT_NAMES,
            "class_names": class_names,
            # SCAFFOLD-specific
            "c_global": c_global,
            "c_clients": c_clients
        }
        ckpt_path = os.path.join(OUTPUT_DIR, f"global_round_{r+1}.pth")
        torch.save(ckpt, ckpt_path)
        print("Saved checkpoint:", ckpt_path)

        # Append summary and save CSV
        round_results.append(round_summary)
        df = pd.DataFrame(round_results)
        csv_path = os.path.join(OUTPUT_DIR, "fl_round_results.csv")
        df.to_csv(csv_path, index=False)
        print("Saved per-round summary CSV to", csv_path)

        # Update plots for global test accuracy & loss
        rounds = list(range(1, len(round_results)+1))
        gtest_acc = [rr.get("global_test_acc", 0.0) for rr in round_results]
        gtest_loss = [rr.get("global_test_loss", 0.0) for rr in round_results]

        plt.figure(figsize=(6,4))
        plt.plot(rounds, gtest_acc)
        plt.xlabel("Global Round"); plt.ylabel("Test Accuracy"); plt.title("Global Test Accuracy")
        plt.savefig(global_test_acc_fname); plt.close()

        plt.figure(figsize=(6,4))
        plt.plot(rounds, gtest_loss)
        plt.xlabel("Global Round"); plt.ylabel("Test Loss");plt.title("Global Test Loss")
        plt.savefig(global_test_loss_fname); plt.close()

        # per-client accuracy plot
        per_client_acc_fname = os.path.join(OUTPUT_DIR, "per_client_test_accuracy_rounds.png")
        plt.figure(figsize=(8,5))
        for i, name in enumerate(CLIENT_NAMES):
            plt.plot(range(1, len(per_client_acc_history[i])+1), per_client_acc_history[i], label=name)
        plt.xlabel("Global Round"); plt.ylabel("Test Accuracy");plt.title("Per-client Test Accuracy"); plt.legend()
        plt.savefig(per_client_acc_fname); plt.close()

        per_client_loss_fname = os.path.join(OUTPUT_DIR, "per_client_test_loss_rounds.png")
        plt.figure(figsize=(8,5))
        for i, name in enumerate(CLIENT_NAMES):
            plt.plot(range(1, len(per_client_loss_history[i])+1), per_client_loss_history[i], label=name)
        plt.xlabel("Global Round"); plt.ylabel("Test Loss");plt.title("Per-client Test Loss"); plt.legend()
        plt.savefig(per_client_loss_fname); plt.close()

    # End of rounds
    final_model_path = os.path.join(OUTPUT_DIR, "global_final.pth")
    torch.save({"model_state": global_model.state_dict(), "class_names": class_names, "c_global": c_global, "c_clients": c_clients}, final_model_path)
    print("Federated training finished. Final global model saved to:", final_model_path)

if __name__ == "__main__":
    main()

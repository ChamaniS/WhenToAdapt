# fl_train_fedrep.py
"""
Federated Rep (FedRep) across 4 clients for TB CXR classification.
Representation (shared) + local head (personalized). Aggregation averages only
the representation parameters across clients; heads remain local.

Each client must have this structure:
  <client_root>/
    train/
      normal/ *.png
      positive/ *.png
    val/
      normal/ ...
      positive/ ...
    test/
      normal/ ...
      positive/ ...

Adjust CLIENT_ROOTS and CONFIG below, then run:
  python fl_train_fedrep.py
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
    r"xxxxx\Projects\Data\Tuberculosis_Data\Shenzhen",     # <-- update these 4 paths
    r"xxxxx\Projects\Data\Tuberculosis_Data\Montgomery",
    r"xxxxx\Projects\Data\Tuberculosis_Data\TBX11K",
    r"xxxxx\Projects\Data\Tuberculosis_Data\Pakistan"
]
CLIENT_NAMES = ["Shenzhen", "Montgomery", "TBX11K", "Pakistan"]  # used only for display
OUTPUT_DIR = r"./fl_outputs"
ARCH = "densenet169"   # or other timm / torchvision model name
PRETRAINED = True
IMG_SIZE = 224
BATCH_SIZE = 4          # local batch size (per client)
WORKERS = 4
LOCAL_EPOCHS = 6     # local epochs per communication round (as requested)
COMM_ROUNDS = 10        # number of federated rounds
LR = 1e-4
WEIGHT_DECAY = 1e-5
USE_AMP = False         # set True to use amp when CUDA available
PIN_MEMORY = True
DROPOUT_P = 0.5
SEED = 42
CLASS_NAMES = ["normal", "positive"]  # canonical ordering
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
# Model + training utilities (FedRep adaptation)
# ----------------------------
class RepresentationHeadModel(nn.Module):
    """
    Wrapper model: representation (shared) + local head (personalized).
    representation: feature extractor outputting a 1D feature vector per sample
    head: small classifier mapping features -> num_classes
    """
    def __init__(self, arch: str, num_classes: int, pretrained: bool = True, dropout_p: float = 0.5):
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dropout_p = dropout_p

        # create backbone and representation extractor:
        # for torchvision densenet models: the 'features' module is the encoder, classifier is 'classifier'
        # for timm models: use create_model with num_classes=0 to get features and then add head
        if arch.startswith("densenet") and hasattr(torchvision.models, arch):
            # use torchvision densenet: features -> global_pool -> classifier
            base = getattr(torchvision.models, arch)(pretrained=pretrained)
            # densenet.features is the convolutional extractor
            self.representation = nn.Sequential(
                base.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            # determine output dim
            # run a dummy forward to get feature dim (safe default with attribute)
            if hasattr(base, "classifier"):
                in_ch = base.classifier.in_features
            else:
                # fallback guess (should not happen)
                in_ch = 1664
            self.head = nn.Sequential(
                nn.Dropout(p=self.dropout_p),
                nn.Linear(in_ch, num_classes)
            )
        else:
            # use timm to create a backbone with no classifier
            # many timm models support num_classes=0 or num_classes=None - we request features_only or use create_model with num_classes=0
            try:
                # Try creating a feature-extractor model (no head)
                backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, global_pool="avg")
                # backbone.forward returns features of size (B, embed_dim)
                self.representation = nn.Sequential(backbone)  # single-module wrapper
                # infer embedding dim by checking attribute
                if hasattr(backbone, "num_features"):
                    in_ch = backbone.num_features
                elif hasattr(backbone, "embed_dim"):
                    in_ch = backbone.embed_dim
                else:
                    # last resort guess - 1024
                    in_ch = 1024
                self.head = nn.Sequential(
                    nn.Dropout(p=self.dropout_p),
                    nn.Linear(in_ch, num_classes)
                )
            except Exception as e:
                # fallback: create a timm model with classifier and split it
                model = timm.create_model(arch, pretrained=pretrained, num_classes=num_classes)
                # try to detect classifier attribute
                if hasattr(model, "get_classifier"):
                    # try to pull out penultimate features by setting num_classes=0 would have been ideal
                    # As a safe fallback, we keep the model as-is and treat entire model as representation + small head
                    # We'll replace final classifier with identity and create a separate head
                    classifier_name = model.default_cfg.get("classifier", None) if hasattr(model, "default_cfg") else None
                # Ultimately fallback: use the full model as representation and put an identity head (not ideal)
                self.representation = nn.Sequential(model)
                self.head = nn.Sequential(nn.Flatten(), nn.Linear(1, num_classes))  # will likely fail, but left as fallback

    def forward(self, x):
        # representation -> head
        feat = self.representation(x)
        out = self.head(feat)
        return out

def create_model(num_classes, arch=ARCH, pretrained=PRETRAINED):
    """Return RepresentationHeadModel (representation + head)"""
    model = RepresentationHeadModel(arch=arch, num_classes=num_classes, pretrained=pretrained, dropout_p=DROPOUT_P)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fedrep_average_representation(models: List[torch.nn.Module], weights: List[float]):
    """
    Average ONLY representation parameters across clients (FedRep).
    models: list of model instances (on CPU or same device) - we use state_dicts.
    weights: normalized list summing to 1
    Returns a state_dict containing averaged representation params (and other keys copied from first model).
    This version:
      - detects representation keys (starts with "representation." or fallback heuristics)
      - filters to floating-point tensors only (skips integer buffers)
      - accumulates in float32 and casts back to original dtype before returning
    """
    # get state dicts
    sd0 = copy.deepcopy(models[0].state_dict())
    # Collect representation-like keys
    rep_keys = [k for k in sd0.keys() if k.startswith("representation.")]
    if len(rep_keys) == 0:
        # fallback heuristics
        rep_keys = [k for k in sd0.keys() if "features" in k or "backbone" in k or "rep" in k]

    # If still empty, just return first model's state dict (nothing to average)
    if len(rep_keys) == 0:
        return sd0

    # Filter to floating point keys only (skip integer buffers like num_batches_tracked)
    float_rep_keys = [k for k in rep_keys if sd0[k].dtype.is_floating_point]
    if len(float_rep_keys) == 0:
        # if none are float, return sd0 (safe fallback)
        return sd0

    # Initialize avg_state dict: copy everything from sd0
    avg_sd = {}
    for k in sd0.keys():
        avg_sd[k] = sd0[k].clone()

    # For each float rep key, accumulate weighted sum in float32
    for k in float_rep_keys:
        # accumulate in float32 for numerical stability
        acc = torch.zeros_like(sd0[k], dtype=torch.float32)
        for i, m in enumerate(models):
            sdi = m.state_dict()
            if k not in sdi:
                raise KeyError(f"Key {k} not found in client {i} state_dict during FedRep averaging.")
            acc += float(weights[i]) * sdi[k].to(torch.float32)
        # cast back to original dtype of sd0[k]
        avg_sd[k] = acc.to(sd0[k].dtype)

    return avg_sd

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

@torch.no_grad()
def evaluate_model(model, dataloader, device, criterion=None, return_per_class=False, class_names=None):
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
# Federated training main (FedRep)
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

    # Initialize global model (Representation + head)
    global_model = create_model(num_classes=num_classes, arch=ARCH, pretrained=PRETRAINED).to(DEVICE)
    print(f"Global model {ARCH} created with {count_parameters(global_model):,} trainable params")

    # Prepare logging
    round_results = []
    global_test_acc_fname = os.path.join(OUTPUT_DIR, "global_test_accuracy_rounds.png")
    global_test_loss_fname = os.path.join(OUTPUT_DIR, "global_test_loss_rounds.png")
    per_client_acc_history = {i: [] for i in range(len(per_client_dataloaders))}
    per_client_loss_history = {i: [] for i in range(len(per_client_dataloaders))}

    for r in range(COMM_ROUNDS):
        print("\n" + "="*60)
        print(f"COMM ROUND {r+1}/{COMM_ROUNDS}")
        print("="*60)
        local_models = []
        weights = []
        round_summary = {"round": r+1}
        # Each client trains locally starting from global representation (and a local head)
        for i, client in enumerate(per_client_dataloaders):
            print(f"\n[CLIENT {i}] {CLIENT_NAMES[i]}: local training")
            # create local model by deepcopy so each client has its own head parameters
            local_model = copy.deepcopy(global_model)  # copy start weights (representation + head copy)
            # Option: if you want heads to be reinitialized per client instead of copied, uncomment:
            # local_model.head.apply(lambda m: isinstance(m, nn.Linear) and m.reset_parameters())

            train_ds = client['train'].dataset
            # per-client class weights from its train set
            client_cw = compute_class_weights_from_dataset(train_ds).to(DEVICE)
            criterion = nn.CrossEntropyLoss(weight=client_cw)
            optimizer = AdamW(local_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            # local training
            logs = train_local(local_model, client['train'], criterion, optimizer, DEVICE, epochs=LOCAL_EPOCHS, use_amp=USE_AMP)
            last_train_loss, last_train_acc = logs[-1]
            print(f"[CLIENT {i}] last local epoch loss={last_train_loss:.4f}, acc={last_train_acc:.4f}")
            round_summary[f"client{i}_train_loss"] = float(last_train_loss)
            round_summary[f"client{i}_train_acc"] = float(last_train_acc)
            # local validation
            print(f"[CLIENT {i}] local validation")
            local_val_metrics = evaluate_model(local_model, client['val'], DEVICE, criterion=criterion)
            print(f"[CLIENT {i}] local val acc={local_val_metrics.get('accuracy', np.nan):.4f}, loss={local_val_metrics.get('loss', np.nan):.4f}")
            round_summary[f"client{i}_localval_loss"] = float(local_val_metrics.get("loss", np.nan))
            round_summary[f"client{i}_localval_acc"] = float(local_val_metrics.get("accuracy", np.nan))
            local_models.append(local_model.cpu())  # move to cpu for aggregation to save GPU mem
            w = float(client_train_sizes[i]) / float(total_train)
            weights.append(w)
            print(f"[CLIENT {i}] aggregation weight: {w:.4f}")

        # FedRep aggregation: average representation parameters only, weighted by client sizes
        print("\nAggregating local models (FedRep: average representation only)")
        # normalize weights
        total_w = sum(weights) if sum(weights) != 0 else 1.0
        norm_weights = [w / total_w for w in weights]
        avg_rep_state = fedrep_average_representation(local_models, norm_weights)

        # Load averaged representation into global_model (non-strict so heads are preserved)
        # Note: avg_rep_state contains all keys; we will update only representation keys
        global_state = global_model.state_dict()
        rep_keys = [k for k in avg_rep_state.keys() if k.startswith("representation.")]
        if len(rep_keys) == 0:
            rep_keys = [k for k in avg_rep_state.keys() if "features" in k or "backbone" in k]
        for k in rep_keys:
            if k in global_state and avg_rep_state[k].dtype.is_floating_point:
                global_state[k] = avg_rep_state[k]
        global_model.load_state_dict(global_state, strict=False)
        global_model.to(DEVICE)

        # Sync averaged representation back into local models so subsequent client evaluation uses the shared rep:
        for lm in local_models:
            lm_state = lm.state_dict()
            for k in rep_keys:
                if k in lm_state and avg_rep_state[k].dtype.is_floating_point:
                    lm_state[k] = avg_rep_state[k]
            lm.load_state_dict(lm_state, strict=False)

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

        # --- NEW: print per-class table for combined test and save combined CSV ---
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

            # Save combined per-class CSV (plus macro row)
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
        # --- end NEW block ---

        # Global test per client
        per_client_test_metrics = []
        for i, client in enumerate(per_client_dataloaders):
            print(f"\nGlobal TEST on client {i} ({CLIENT_NAMES[i]}) test set")
            # use class weights computed from that client's train set
            client_train_ds = client['train'].dataset
            client_cw = compute_class_weights_from_dataset(client_train_ds).to(DEVICE)
            client_criterion = nn.CrossEntropyLoss(weight=client_cw)
            # For client test we use the local model that had its head personalized (we synced its representation to the averaged rep above)
            local_model = local_models[i].to(DEVICE)
            cl_metrics = evaluate_model(local_model, client['test'], DEVICE, criterion=client_criterion, return_per_class=True, class_names=class_names)
            # compute mean specificity across classes (if per-class specificity present)
            mean_spec = None
            if "per_class_specificity" in cl_metrics:
                specs = cl_metrics.get("per_class_specificity", [])
                if len(specs) > 0:
                    mean_spec = float(np.mean(specs))
            else:
                # fallback: derive specificity from confusion matrix for binary case
                cm = cl_metrics.get("confusion_matrix", None)
                if cm is not None and cm.shape[0] == 2:
                    # prefer per_class_specificity from evaluate_model; fallback is not ideal
                    mean_spec = None

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

            # Also print per-class table (optional, more detailed)
            if "per_class_precision" in cl_metrics:
                print(f"\n  Per-class metrics (order = {class_names}):")
                header = ["Class", "Support", "Correct", "Acc", "Prec", "Rec", "F1", "Spec"]
                print("    " + "{:12s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s} {:8s}".format(*header))
                cm = cl_metrics.get("confusion_matrix", None)
                tp_counts = np.diag(cm).astype(int) if cm is not None else [0]*len(class_names)
                supports = cm.sum(axis=1).astype(int) if cm is not None else [0]*len(class_names)
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
                    print(f"    {cname:12s} {s:8d} {ccount:8d} {acc_val:8.4f} {pval:8.4f} {rval:8.4f} {fval:8.4f} {sval:8.4f}")

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

        # Save checkpoint and per-round summary
        ckpt = {
            "round": r+1,
            "model_state": global_model.state_dict(),
            "global_val_metrics": global_val_metrics,
            "global_test_metrics": global_test_metrics,
            "per_client_test_metrics": per_client_test_metrics,
            "client_names": CLIENT_NAMES,
            "class_names": class_names
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
    torch.save({"model_state": global_model.state_dict(), "class_names": class_names}, final_model_path)
    print("Federated training finished. Final global model saved to:", final_model_path)

if __name__ == "__main__":
    main()

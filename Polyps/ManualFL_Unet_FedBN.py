import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

from models.UNET import UNET
from dataset import CVCDataset   # must support (img_dir, mask_dir, transform)

# -------------------------
# Settings
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
start_time = time.time()

out_dir = "Outputs"
os.makedirs(out_dir, exist_ok=True)

# -------------------------
# Client dataset directories
# -------------------------
train_img_dirs = [
    r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_imgs",
    r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\train\images",
    r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\images",
    r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\images",
]
train_mask_dirs = [
    r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_masks",
    r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\train\masks",
    r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\masks",
    r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\masks",
]
val_img_dirs = [
    r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_imgs",
    r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\val\images",
    r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\images",
    r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\images"
]
val_mask_dirs = [
    r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_masks",
    r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\val\masks",
    r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\masks",
    r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\masks"
]
test_img_dirs = [
    r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_imgs",
    r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\test\images",
    r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\images",
    r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\images"
]
test_mask_dirs = [
    r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_masks",
    r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\test\masks",
    r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\masks",
    r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\masks"
]
client_names = ["Kvasir", "ETIS", "CVC-Colon","CVC-Clinic"]

# -------------------------
# Utils
# -------------------------
def get_loader(img_dir, mask_dir, transform, batch_size=4, shuffle=True, oversample_factor=None):
    ds = CVCDataset(img_dir, mask_dir, transform=transform)

    if oversample_factor is not None:
        sampler = RandomSampler(ds, replacement=True, num_samples=oversample_factor * len(ds))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def compute_metrics(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()
    TP = (pred * target).sum().item()
    TN = ((1 - pred) * (1 - target)).sum().item()
    FP = (pred * (1 - target)).sum().item()
    FN = ((1 - pred) * target).sum().item()
    dice_with_bg = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    iou_with_bg = (TP + smooth) / (TP + FP + FN + smooth)
    acc = (TP + TN) / (TP + TN + FP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    specificity = (TN + smooth) / (TN + FP + smooth)
    if target.sum() == 0:
        dice_no_bg = 1.0 if pred.sum() == 0 else 0.0
        iou_no_bg = 1.0 if pred.sum() == 0 else 0.0
    else:
        intersection = (pred * target).sum().item()
        dice_no_bg = (2 * intersection + smooth) / (pred.sum().item() + target.sum().item() + smooth)
        iou_no_bg = (intersection + smooth) / (pred.sum().item() + target.sum().item() - intersection + smooth)
    return dict(dice_with_bg=dice_with_bg, dice_no_bg=dice_no_bg,
                iou_with_bg=iou_with_bg, iou_no_bg=iou_no_bg,
                accuracy=acc, precision=precision, recall=recall, specificity=specificity)

def average_metrics(metrics_list):
    avg = {}
    for k in metrics_list[0].keys():
        avg[k] = sum(m[k] for m in metrics_list) / len(metrics_list)
    return avg

def get_loss_fn(device):
    return smp.losses.DiceLoss(mode="binary", from_logits=True).to(device)

# -------------------------
# FedBN helpers
# -------------------------
def fedbn_average(models, weights):
    """Average only non-BN parameters across clients (FedBN)."""
    avg_state = copy.deepcopy(models[0].state_dict())
    for key in avg_state.keys():
        if "bn" in key.lower() or "downsample.1" in key.lower():  # skip BN params
            continue
        avg_state[key] = sum(weights[i] * models[i].state_dict()[key] for i in range(len(models)))
    return avg_state

# -------------------------
# Training / Eval functions
# -------------------------
def train_local(loader, model, loss_fn, opt):
    model.train()
    total_loss, metrics = 0.0, []
    for _ in range(LOCAL_EPOCHS):
        for data, target in tqdm(loader, leave=False):
            data, target = data.to(DEVICE), target.to(DEVICE).unsqueeze(1).float()
            preds = model(data)
            loss = loss_fn(preds, target)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
            metrics.append(compute_metrics(preds.detach(), target))
    avg_metrics = average_metrics(metrics)
    print("Train: " + " | ".join([f"{k}: {v:.4f}" for k,v in avg_metrics.items()]))
    return total_loss / len(loader.dataset), avg_metrics

@torch.no_grad()
def evaluate(loader, model, loss_fn, split="Val"):
    model.eval()
    total_loss, metrics = 0.0, []
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE).unsqueeze(1).float()
        preds = model(data)
        loss = loss_fn(preds, target)
        total_loss += loss.item()
        metrics.append(compute_metrics(preds, target))
    avg_metrics = average_metrics(metrics)
    print(f"{split}: " + " | ".join([f"{k}: {v:.4f}" for k,v in avg_metrics.items()]))
    return total_loss / len(loader.dataset), avg_metrics

# -------------------------
# Plotting
# -------------------------
def plot_metrics(round_metrics, out_dir):
    rounds = list(range(1, len(round_metrics) + 1))

    # Dice_no_bg
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_dice_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round")
    plt.ylabel("Dice")
    plt.title("Per-client Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_no_bg.png"))
    plt.close()

    # IoU_no_bg
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_iou_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Per-client IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_no_bg.png"))
    plt.close()

# -------------------------
# Main FedBN
# -------------------------
def main():
    # Training transform with augmentation
    tr_tf = A.Compose([
        A.Resize(224,224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=[0]*3, std=[1]*3),
        ToTensorV2()
    ])

    # Validation / test transform (no aug)
    val_tf = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=[0]*3, std=[1]*3),
        ToTensorV2()
    ])

    global_model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models, weights = [], []
        total_sz = 0

        # ----- Local Training -----
        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model).to(DEVICE)
            opt = optim.AdamW(local_model.parameters(), lr=1e-4)
            loss_fn = get_loss_fn(DEVICE)

            oversample_factor = 10
            train_loader = get_loader(train_img_dirs[i], train_mask_dirs[i], tr_tf,
                                      oversample_factor=oversample_factor)
            val_loader = get_loader(val_img_dirs[i], val_mask_dirs[i], val_tf, shuffle=False)

            print(f"[Client {client_names[i]}]")
            train_local(train_loader, local_model, loss_fn, opt)
            evaluate(val_loader, local_model, loss_fn, split="Val")

            local_models.append(local_model)
            sz = len(train_loader.dataset); weights.append(sz); total_sz += sz

        # ----- FedBN Aggregation -----
        norm_weights = [w/total_sz for w in weights]
        avg_state = fedbn_average(local_models, norm_weights)
        global_model.load_state_dict(avg_state, strict=False)

        # Sync only non-BN params back to clients
        for lm in local_models:
            lm.load_state_dict(avg_state, strict=False)

        # ----- Testing -----
        rm = {}
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(test_img_dirs[i], test_mask_dirs[i], val_tf, shuffle=False)
            print(f"[Client {client_names[i]}] Test")
            _, test_metrics = evaluate(test_loader, local_models[i], get_loss_fn(DEVICE), split="Test")
            rm[f"client{i}_dice_no_bg"] = test_metrics["dice_no_bg"]
            rm[f"client{i}_iou_no_bg"] = test_metrics["iou_no_bg"]

        round_metrics.append(rm)
        plot_metrics(round_metrics, out_dir)

    end_time = time.time()
    print(f"Total runtime: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main()

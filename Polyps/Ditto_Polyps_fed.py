# ditto_fed_polyp_personal_plots.py
"""
Ditto-style federated polyp segmentation (per-client personalization) with
per-client plots that show each client's PERSONAL model Dice and IoU across rounds.

Copy-paste and run. Tune hyperparams at the top.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# local modules (unchanged)
from models.UNET import UNET
from dataset import CVCDataset   # must support (img_dir, mask_dir, transform)

# -------------------------
# Settings (tune these)
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
BATCH_SIZE = 4
LR_GLOBAL = 1e-4
LR_PERSONAL = 1e-4
PERSONALIZATION_MU = 1.0
OUT_DIR = "Outputs_ditto_personal"
os.makedirs(OUT_DIR, exist_ok=True)

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

start_time = time.time()

def get_loader(img_dir, mask_dir, transform, batch_size=BATCH_SIZE, shuffle=True, num_workers=0):
    ds = CVCDataset(img_dir, mask_dir, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=(DEVICE=="cuda"))

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
    if len(metrics_list) == 0:
        return {}
    avg = {}
    for k in metrics_list[0].keys():
        avg[k] = sum(m[k] for m in metrics_list) / len(metrics_list)
    return avg

def get_loss_fn(device):
    return smp.losses.DiceLoss(mode="binary", from_logits=True).to(device)

def average_models_weighted(models, weights):
    if len(models) == 0:
        raise ValueError("No models to average")
    sum_w = float(sum(weights))
    norm_weights = [w / sum_w for w in weights]
    base_sd = models[0].state_dict()
    avg_sd = {}
    for k in base_sd.keys():
        acc = None
        for m, w in zip(models, norm_weights):
            v = m.state_dict()[k].cpu().to(dtype=torch.float32)
            if acc is None:
                acc = w * v
            else:
                acc += w * v
        try:
            avg_sd[k] = acc.to(dtype=base_sd[k].dtype)
        except Exception:
            avg_sd[k] = acc
    return avg_sd

def l2_distance_params(model_a, model_b):
    total = 0.0
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        total += torch.sum((pa - pb.detach()) ** 2)
    return total

def train_local_ditto(train_loader, local_global_model, personal_model,
                      loss_fn, opt_global, opt_personal, mu, device):
    local_global_model.train()
    personal_model.train()

    for epoch in range(LOCAL_EPOCHS):
        pbar = tqdm(train_loader, desc=f"LocalDitto ep{epoch+1}/{LOCAL_EPOCHS}", leave=False)
        for data, target in pbar:
            data = data.to(device)
            target = target.to(device).unsqueeze(1).float()

            opt_global.zero_grad()
            preds_g = local_global_model(data)
            loss_g = loss_fn(preds_g, target)
            loss_g.backward()
            opt_global.step()

            opt_personal.zero_grad()
            preds_p = personal_model(data)
            loss_p = loss_fn(preds_p, target)
            prox = 0.5 * mu * l2_distance_params(personal_model, local_global_model)
            total_personal_loss = loss_p + prox
            total_personal_loss.backward()
            opt_personal.step()

    return

@torch.no_grad()
def evaluate(loader, model, loss_fn, split="Val", device="cpu"):
    model.eval()
    total_loss = 0.0
    metrics = []
    n = 0
    for data, target in loader:
        data = data.to(device)
        target = target.to(device).unsqueeze(1).float()
        preds = model(data)
        loss = loss_fn(preds, target)
        total_loss += float(loss.item()) * data.size(0)
        metrics.append(compute_metrics(preds, target))
        n += data.size(0)
    avg_metrics = average_metrics(metrics)
    if avg_metrics:
        print(f"{split}: " + " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()]))
    return (total_loss / max(1, n), avg_metrics)

def plot_personal_curves(round_metrics, out_dir, client_names):
    rounds = list(range(1, len(round_metrics) + 1))

    plt.figure(figsize=(6,4))
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_personal_dice_no_bg", 0.0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round")
    plt.ylabel("Dice")
    plt.title("Per-client Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_client_personal_dice_no_bg.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_personal_iou_no_bg", 0.0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Per-client IoU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_client_personal_iou_no_bg.png"))
    plt.close()

def main():
    tr_tf = A.Compose([A.Resize(224,224), A.Normalize(mean=[0]*3,std=[1]*3), ToTensorV2()])
    val_tf = tr_tf
    global_model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    personal_models = [copy.deepcopy(global_model).to(DEVICE) for _ in range(NUM_CLIENTS)]

    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models = []
        weights = []
        total_sz = 0

        for i in range(NUM_CLIENTS):
            print(f"\n[Client {client_names[i]}]")
            local_global = copy.deepcopy(global_model).to(DEVICE)
            personal = personal_models[i].to(DEVICE)

            opt_global = optim.AdamW(local_global.parameters(), lr=LR_GLOBAL)
            opt_personal = optim.AdamW(personal.parameters(), lr=LR_PERSONAL)

            loss_fn = get_loss_fn(DEVICE)

            train_loader = get_loader(train_img_dirs[i], train_mask_dirs[i], tr_tf, batch_size=BATCH_SIZE)
            val_loader = get_loader(val_img_dirs[i], val_mask_dirs[i], val_tf, batch_size=BATCH_SIZE, shuffle=False)
            train_local_ditto(train_loader, local_global, personal, loss_fn, opt_global, opt_personal, PERSONALIZATION_MU, DEVICE)

            print("Local global model eval (client local view):")
            evaluate(val_loader, local_global, loss_fn, split="Val (local_global)", device=DEVICE)

            local_models.append(local_global.cpu())
            sz = len(train_loader.dataset)
            weights.append(sz)
            total_sz += sz
            personal_models[i] = personal.cpu()

        if total_sz == 0:
            total_sz = 1.0
        norm_weights = [w / total_sz for w in weights]
        avg_sd = average_models_weighted(local_models, norm_weights)
        global_model.load_state_dict(avg_sd)
        global_model.to(DEVICE)

        loss_fn = get_loss_fn(DEVICE)
        rm = {}
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(test_img_dirs[i], test_mask_dirs[i], val_tf, batch_size=BATCH_SIZE, shuffle=False)
            personal = personal_models[i].to(DEVICE)
            print(f"[Client {client_names[i]}] Test PERSONAL model")
            _, personal_metrics = evaluate(test_loader, personal, loss_fn, split="Test (personal)", device=DEVICE)

            if personal_metrics:
                rm[f"client{i}_personal_dice_no_bg"] = personal_metrics.get("dice_no_bg", 0.0)
                rm[f"client{i}_personal_iou_no_bg"] = personal_metrics.get("iou_no_bg", 0.0)
            else:
                rm[f"client{i}_personal_dice_no_bg"] = 0.0
                rm[f"client{i}_personal_iou_no_bg"] = 0.0

            print(f"[Client {client_names[i]}] Test GLOBAL model")
            _, global_metrics = evaluate(test_loader, global_model, loss_fn, split="Test (global)", device=DEVICE)
            if global_metrics:
                rm[f"client{i}_global_dice_no_bg"] = global_metrics.get("dice_no_bg", 0.0)
                rm[f"client{i}_global_iou_no_bg"] = global_metrics.get("iou_no_bg", 0.0)
            else:
                rm[f"client{i}_global_dice_no_bg"] = 0.0
                rm[f"client{i}_global_iou_no_bg"] = 0.0

        round_metrics.append(rm)
        plot_personal_curves(round_metrics, OUT_DIR, client_names)
        torch.save(round_metrics, os.path.join(OUT_DIR, "round_metrics_personal.pt"))

    end_time = time.time()
    print(f"Total runtime: {(end_time - start_time):.2f} seconds")

    torch.save(global_model.state_dict(), os.path.join(OUT_DIR, "global_model_final.pth"))
    for i, pm in enumerate(personal_models):
        torch.save(pm.state_dict(), os.path.join(OUT_DIR, f"personal_model_client{i}.pth"))

if __name__ == "__main__":
    main()

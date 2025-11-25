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
PROX_MU = 0.01          # FedProx proximal coefficient
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
def get_loader(img_dir, mask_dir, transform, batch_size=8, shuffle=True, oversample_factor=None):
    ds = CVCDataset(img_dir, mask_dir, transform=transform)
    if oversample_factor is not None:
        sampler = RandomSampler(ds, replacement=True, num_samples=oversample_factor * len(ds))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

# -------------------------
# Corrected compute_metrics (robust, clamped to [0,1])
# -------------------------
def compute_metrics(pred, target, smooth=1e-6):
    """
    pred: raw logits (Bx1xHxW) or (BxHxW) -- we apply sigmoid and threshold at 0.5
    target: binary mask (Bx1xHxW) or (BxHxW) with values in {0,1} or float [0,1]
    Returns dict with dice_with_bg, dice_no_bg, iou_with_bg, iou_no_bg, accuracy, precision, recall, specificity
    """
    # Apply sigmoid then threshold
    probs = torch.sigmoid(pred)
    pred_bin = (probs > 0.5).float()  # Bx1xHxW or BxHxW

    # Ensure target is same shape as pred_bin
    if target.dim() == 3:
        # BxHxW -> Bx1xHxW
        target = target.unsqueeze(1)
    target_bin = (target > 0.5).float()

    # If pred has channel dim 1, squeeze to BxHxW for counting
    if pred_bin.dim() == 4 and pred_bin.size(1) == 1:
        pred_flat = pred_bin.view(pred_bin.size(0), -1)
    else:
        pred_flat = pred_bin.view(pred_bin.size(0), -1)

    if target_bin.dim() == 4 and target_bin.size(1) == 1:
        targ_flat = target_bin.view(target_bin.size(0), -1)
    else:
        targ_flat = target_bin.view(target_bin.size(0), -1)

    # Compute counts per-batch (sum across spatial dims per image)
    # Use float sums to avoid integer issues
    TP = (pred_flat * targ_flat).sum(dim=1).float()  # per-image
    FP = (pred_flat * (1.0 - targ_flat)).sum(dim=1).float()
    FN = ((1.0 - pred_flat) * targ_flat).sum(dim=1).float()
    TN = ((1.0 - pred_flat) * (1.0 - targ_flat)).sum(dim=1).float()

    # Foreground sums
    pred_sum = (TP + FP)  # per-image predicted positives
    targ_sum = (TP + FN)  # per-image target positives
    inter = TP

    # Dice no background (foreground)
    # dice = 2*|A ∩ B| / (|A| + |B|)
    denom_dice_fg = (pred_sum + targ_sum)
    dice_no_bg_per_image = torch.where(
        denom_dice_fg > 0,
        (2.0 * inter + smooth) / (denom_dice_fg + smooth),
        torch.ones_like(denom_dice_fg)  # if both empty, define as 1.0
    )

    # IoU no background (foreground)
    denom_iou_fg = (pred_sum + targ_sum - inter)
    iou_no_bg_per_image = torch.where(
        denom_iou_fg > 0,
        (inter + smooth) / (denom_iou_fg + smooth),
        torch.where((pred_sum == 0) & (targ_sum == 0), torch.ones_like(denom_iou_fg), torch.zeros_like(denom_iou_fg))
    )

    # Now background metrics (treat background as positive)
    back_pred_sum = (TN + FN)
    back_targ_sum = (TN + FP)
    back_inter = TN

    denom_dice_bg = (back_pred_sum + back_targ_sum)
    dice_bg_per_image = torch.where(
        denom_dice_bg > 0,
        (2.0 * back_inter + smooth) / (denom_dice_bg + smooth),
        torch.ones_like(denom_dice_bg)
    )

    denom_iou_bg = (back_pred_sum + back_targ_sum - back_inter)
    iou_bg_per_image = torch.where(
        denom_iou_bg > 0,
        (back_inter + smooth) / (denom_iou_bg + smooth),
        torch.where((back_pred_sum == 0) & (back_targ_sum == 0), torch.ones_like(denom_iou_bg), torch.zeros_like(denom_iou_bg))
    )

    # Accuracy / precision / recall / specificity per image
    total = TP + TN + FP + FN
    acc_per_image = torch.where(total > 0, (TP + TN) / (total + smooth), torch.ones_like(total))

    precision_per_image = torch.where((TP + FP) > 0, (TP + smooth) / (TP + FP + smooth), torch.where(TP == 0, torch.zeros_like(TP), torch.ones_like(TP)))
    recall_per_image = torch.where((TP + FN) > 0, (TP + smooth) / (TP + FN + smooth), torch.where(TP == 0, torch.zeros_like(TP), torch.ones_like(TP)))
    specificity_per_image = torch.where((TN + FP) > 0, (TN + smooth) / (TN + FP + smooth), torch.where(TN == 0, torch.zeros_like(TN), torch.ones_like(TN)))

    # Average across images in batch
    dice_no_bg = float(torch.clamp(dice_no_bg_per_image.mean(), 0.0, 1.0).item())
    iou_no_bg = float(torch.clamp(iou_no_bg_per_image.mean(), 0.0, 1.0).item())
    dice_with_bg = float(torch.clamp(0.5 * (dice_no_bg_per_image + dice_bg_per_image).mean(), 0.0, 1.0).item())
    iou_with_bg = float(torch.clamp(0.5 * (iou_no_bg_per_image + iou_bg_per_image).mean(), 0.0, 1.0).item())
    accuracy = float(torch.clamp(acc_per_image.mean(), 0.0, 1.0).item())
    precision = float(torch.clamp(precision_per_image.mean(), 0.0, 1.0).item())
    recall = float(torch.clamp(recall_per_image.mean(), 0.0, 1.0).item())
    specificity = float(torch.clamp(specificity_per_image.mean(), 0.0, 1.0).item())

    return dict(
        dice_with_bg=dice_with_bg,
        dice_no_bg=dice_no_bg,
        iou_with_bg=iou_with_bg,
        iou_no_bg=iou_no_bg,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity
    )

# -------------------------
# Other helpers (unchanged)
# -------------------------
def average_metrics(metrics_list):
    if not metrics_list:
        return {}
    avg = {}
    for k in metrics_list[0].keys():
        avg[k] = sum(m[k] for m in metrics_list) / len(metrics_list)
    return avg

def get_loss_fn(device):
    return smp.losses.DiceLoss(mode="binary", from_logits=True).to(device)

# --- FedAvg helper (aggregate all parameters, including BN) ---
def average_models_weighted(models, weights):
    """Average all model parameters across clients (FedAvg-style)."""
    avg_sd = copy.deepcopy(models[0].state_dict())
    for k in avg_sd.keys():
        # sum on cpu to avoid GPU memory blowup if many models
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k].cpu() for i in range(len(models)))
    return avg_sd

# -------------------------
# Training / Eval functions (with FedProx)
# -------------------------
def train_local(loader, model, loss_fn, opt, global_state_dict=None, mu=0.0):
    model.train()
    total_loss, metrics = 0.0, []
    use_prox = (mu is not None and mu > 0.0 and global_state_dict is not None)
    if use_prox:
        global_params = {k: v.to(DEVICE) for k, v in global_state_dict.items()}

    for _ in range(LOCAL_EPOCHS):
        for data, target in tqdm(loader, leave=False):
            data, target = data.to(DEVICE), target.to(DEVICE).unsqueeze(1).float()
            preds = model(data)
            loss = loss_fn(preds, target)

            # FedProx proximal term (μ/2 * ||w - w_global||²)
            if use_prox:
                prox = torch.tensor(0.0, device=DEVICE)
                for name, param in model.named_parameters():
                    if name in global_params:
                        prox = prox + torch.sum((param - global_params[name]) ** 2)
                loss = loss + (mu / 2.0) * prox

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())
            metrics.append(compute_metrics(preds.detach(), target))
    avg_metrics = average_metrics(metrics) if metrics else {}
    print("Train: " + " | ".join([f"{k}: {v:.4f}" for k,v in (avg_metrics or {}).items()]))
    return total_loss / max(1, len(loader.dataset)), avg_metrics

@torch.no_grad()
def evaluate(loader, model, loss_fn, split="Val"):
    model.eval()
    total_loss, metrics = 0.0, []
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE).unsqueeze(1).float()
        preds = model(data)
        loss = loss_fn(preds, target)
        total_loss += float(loss.item())
        metrics.append(compute_metrics(preds, target))
    avg_metrics = average_metrics(metrics) if metrics else {}
    print(f"{split}: " + " | ".join([f"{k}: {v:.4f}" for k,v in (avg_metrics or {}).items()]))
    return total_loss / max(1, len(loader.dataset)), avg_metrics

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
    plt.xlabel("Global Round"); plt.ylabel("Dice"); plt.title("Per-client Dice (FedProx)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_no_bg_fedprox.png")); plt.close()
    # IoU_no_bg
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_iou_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round"); plt.ylabel("IoU"); plt.title("Per-client IoU (FedProx)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_no_bg_fedprox.png")); plt.close()

# -------------------------
# Main FedProx + FedAvg (aggregate all layers)
# -------------------------
def main():
    tr_tf = A.Compose([
        A.Resize(224,224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
        A.Normalize(mean=[0]*3, std=[1]*3),
        ToTensorV2()
    ])
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

        global_state_for_prox = {k: v.clone().detach() for k,v in global_model.state_dict().items()}

        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model).to(DEVICE)
            opt = optim.AdamW(local_model.parameters(), lr=1e-4)
            loss_fn = get_loss_fn(DEVICE)

            oversample_factor = 1
            train_loader = get_loader(train_img_dirs[i], train_mask_dirs[i], tr_tf,
                                      oversample_factor=oversample_factor)
            val_loader = get_loader(val_img_dirs[i], val_mask_dirs[i], val_tf, shuffle=False)

            print(f"[Client {client_names[i]}] Local training (FedProx μ={PROX_MU})")
            train_local(train_loader, local_model, loss_fn, opt, global_state_dict=global_state_for_prox, mu=PROX_MU)
            evaluate(val_loader, local_model, loss_fn, split="Val")

            local_models.append(local_model)
            sz = len(train_loader.dataset); weights.append(sz); total_sz += sz

        norm_weights = [w/total_sz for w in weights]
        avg_state = average_models_weighted(local_models, norm_weights)
        global_model.load_state_dict(avg_state, strict=False)

        for lm in local_models:
            lm.load_state_dict(avg_state, strict=False)

        rm = {}
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(test_img_dirs[i], test_mask_dirs[i], val_tf, shuffle=False)
            print(f"[Client {client_names[i]}] Test")
            _, test_metrics = evaluate(test_loader, local_models[i], get_loss_fn(DEVICE), split="Test")
            rm[f"client{i}_dice_no_bg"] = test_metrics.get("dice_no_bg", 0)
            rm[f"client{i}_iou_no_bg"] = test_metrics.get("iou_no_bg", 0)

        round_metrics.append(rm)
        plot_metrics(round_metrics, out_dir)

    print(f"Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

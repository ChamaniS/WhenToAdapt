# ditto_fed_polyp.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# local modules (unchanged)
from models.UNET import UNET
#from models.DuckNet import DuckNet
from dataset import CVCDataset   # must support (img_dir, mask_dir, transform)

# -------------------------
# Settings (tune these)
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 4
LOCAL_EPOCHS = 12           # local passes per communication round
COMM_ROUNDS = 10
BATCH_SIZE = 4
LR_GLOBAL = 1e-4
LR_PERSONAL = 1e-4
PERSONALIZATION_MU = 1.0    # proximal weight for Ditto (mu)
OUT_DIR = "Outputs_ditto"
os.makedirs(OUT_DIR, exist_ok=True)

# paths (leave as you provided)
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

# -------------------------
# Utils (kept your implementations, minor adapt)
# -------------------------
def get_loader(img_dir, mask_dir, transform, batch_size=BATCH_SIZE, shuffle=True):
    ds = CVCDataset(img_dir, mask_dir, transform=transform)
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
    if len(metrics_list) == 0:
        return {}
    avg = {}
    for k in metrics_list[0].keys():
        avg[k] = sum(m[k] for m in metrics_list) / len(metrics_list)
    return avg

def get_loss_fn(device):
    return smp.losses.DiceLoss(mode="binary", from_logits=True).to(device)

def average_models_weighted(models, weights):
    # weighted average of state_dicts, returns averaged state_dict (not loaded)
    avg_sd = copy.deepcopy(models[0].state_dict())
    for k in avg_sd.keys():
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k] for i in range(len(models)))
    return avg_sd

def l2_distance_params(model_a, model_b):
    """sum of squared L2 norms between corresponding parameters (used for proximal)"""
    total = 0.0
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        # pb is treated as constant; detach it to avoid gradients w.r.t. pb
        total += torch.sum((pa - pb.detach()) ** 2)
    return total

# -------------------------
# Training / Eval for Ditto
# -------------------------
def train_local_ditto(train_loader, local_global_model, personal_model,
                      loss_fn, opt_global, opt_personal, mu, device):
    """
    Performs local updates for both the local copy of global model (local_global_model)
    and the personalized model (personal_model) using client's local data.

    local_global_model: contributes to server aggregation (updated and sent)
    personal_model: client's private personalized model (not sent)
    mu: proximal coefficient for personalized objective
    """
    local_global_model.train()
    personal_model.train()

    for epoch in range(LOCAL_EPOCHS):
        for data, target in tqdm(train_loader, leave=False):
            data = data.to(device)
            target = target.to(device).unsqueeze(1).float()

            # ----- update local global model (w_i) -----
            preds_g = local_global_model(data)
            loss_g = loss_fn(preds_g, target)
            opt_global.zero_grad()
            loss_g.backward()
            opt_global.step()

            # ----- update personalized model (p_i) with proximal to local_global_model -----
            preds_p = personal_model(data)
            loss_p = loss_fn(preds_p, target)
            # proximal term: (mu/2) * ||p - w_i||^2
            prox = 0.5 * mu * l2_distance_params(personal_model, local_global_model)
            total_personal_loss = loss_p + prox
            opt_personal.zero_grad()
            total_personal_loss.backward()
            opt_personal.step()

    # return average train loss? for simplicity return last losses (not used for aggregation)
    return

@torch.no_grad()
def evaluate(loader, model, loss_fn, split="Val", device="cpu"):
    model.eval()
    total_loss, metrics = 0.0, []
    n = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device).unsqueeze(1).float()
        preds = model(data)
        loss = loss_fn(preds, target)
        total_loss += loss.item() * data.size(0)
        metrics.append(compute_metrics(preds, target))
        n += data.size(0)
    avg_metrics = average_metrics(metrics)
    if avg_metrics:
        print(f"{split}: " + " | ".join([f"{k}: {v:.4f}" for k,v in avg_metrics.items()]))
    return total_loss / max(1, n), avg_metrics

# -------------------------
# Plotting (record both global and personalized)
# -------------------------
def plot_metrics(round_metrics, out_dir):
    rounds = list(range(1, len(round_metrics) + 1))

    # per-client personalized Dice_no_bg
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_personal_dice_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]} (personal)")
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_global_dice_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, linestyle="--", label=f"{client_names[cid]} (global)")
    plt.xlabel("Global Round")
    plt.ylabel("Dice (no-bg)")
    plt.title("Per-client Dice (global vs personal)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_no_bg_ditto.png"))
    plt.close()

    # IoU_no_bg
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_personal_iou_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]} (personal)")
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_global_iou_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, linestyle="--", label=f"{client_names[cid]} (global)")
    plt.xlabel("Global Round")
    plt.ylabel("IoU (no-bg)")
    plt.title("Per-client IoU (global vs personal)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_no_bg_ditto.png"))
    plt.close()

# -------------------------
# Main Ditto Fed loop
# -------------------------
def main():
    tr_tf = A.Compose([A.Resize(224,224), A.Normalize(mean=[0]*3,std=[1]*3), ToTensorV2()])
    val_tf = tr_tf

    # initialize global model (server model)
    #global_model = DuckNet(input_channels=3, num_classes=1, num_filters=17).to(DEVICE)
    global_model = UNET(in_channels=3, out_channels=1).cuda()
    # initialize personalized models per client as copies of global model
    personal_models = [copy.deepcopy(global_model).to(DEVICE) for _ in range(NUM_CLIENTS)]

    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models = []   # local copies of global models to be sent to server
        weights = []
        total_sz = 0

        for i in range(NUM_CLIENTS):
            print(f"\n[Client {client_names[i]}]")

            # local copy of global model (this will be updated and sent to server)
            local_global = copy.deepcopy(global_model).to(DEVICE)
            # personalized model for this client
            personal = personal_models[i]

            # optimizers: note separate learning rates ok
            opt_global = optim.AdamW(local_global.parameters(), lr=LR_GLOBAL)
            opt_personal = optim.AdamW(personal.parameters(), lr=LR_PERSONAL)

            loss_fn = get_loss_fn(DEVICE)

            train_loader = get_loader(train_img_dirs[i], train_mask_dirs[i], tr_tf, batch_size=BATCH_SIZE)
            val_loader = get_loader(val_img_dirs[i], val_mask_dirs[i], val_tf, batch_size=BATCH_SIZE, shuffle=False)

            # local updates for Ditto (updates both local_global and personal)
            train_local_ditto(train_loader, local_global, personal, loss_fn, opt_global, opt_personal, PERSONALIZATION_MU, DEVICE)

            # evaluate local_global on validation (optional monitoring)
            print("Local global model eval (client local view):")
            evaluate(val_loader, local_global, loss_fn, split="Val (local_global)", device=DEVICE)

            local_models.append(local_global)
            sz = len(train_loader.dataset)
            weights.append(sz)
            total_sz += sz

            # store back personalized model (personal is mutated in-place), so personal_models[i] already updated

        # Server aggregation: aggregate local_models -> new global_model
        norm_weights = [w / total_sz for w in weights]
        avg_sd = average_models_weighted(local_models, norm_weights)
        global_model.load_state_dict(avg_sd)

        # After server updates, we do NOT overwrite personal models (Ditto keeps personal models),
        # but clients should refresh their local copy of global model next round (we do that at start of each client loop).
        # Optionally, you may want to set a "local copy of global" used in proximal to the new server model.
        # In this implementation, the proximal anchors to client's local_global (which after local updates is fine).
        # If you want anchor to server global, you can set personal_model parameters closer to server here (not standard Ditto).

        # ---- Testing ----
        rm = {}
        loss_fn = get_loss_fn(DEVICE)
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(test_img_dirs[i], test_mask_dirs[i], val_tf, batch_size=BATCH_SIZE, shuffle=False)
            print(f"[Client {client_names[i]}] Test GLOBAL model")
            _, global_test_metrics = evaluate(test_loader, global_model, loss_fn, split="Test (global)", device=DEVICE)
            print(f"[Client {client_names[i]}] Test PERSONAL model")
            _, personal_test_metrics = evaluate(test_loader, personal_models[i], loss_fn, split="Test (personal)", device=DEVICE)

            # store metrics for plotting
            if global_test_metrics:
                rm[f"client{i}_global_dice_no_bg"] = global_test_metrics["dice_no_bg"]
                rm[f"client{i}_global_iou_no_bg"] = global_test_metrics["iou_no_bg"]
            else:
                rm[f"client{i}_global_dice_no_bg"] = 0.0
                rm[f"client{i}_global_iou_no_bg"] = 0.0

            if personal_test_metrics:
                rm[f"client{i}_personal_dice_no_bg"] = personal_test_metrics["dice_no_bg"]
                rm[f"client{i}_personal_iou_no_bg"] = personal_test_metrics["iou_no_bg"]
            else:
                rm[f"client{i}_personal_dice_no_bg"] = 0.0
                rm[f"client{i}_personal_iou_no_bg"] = 0.0

        round_metrics.append(rm)
        plot_metrics(round_metrics, OUT_DIR)

    # final timing
    end_time = time.time()
    print(f"Total runtime: {(end_time - start_time):.2f} seconds")

    # Optionally save final global and personal models
    torch.save(global_model.state_dict(), os.path.join(OUT_DIR, "global_model_final.pth"))
    for i, pm in enumerate(personal_models):
        torch.save(pm.state_dict(), os.path.join(OUT_DIR, f"personal_model_client{i}.pth"))

if __name__ == "__main__":
    main()

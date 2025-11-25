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

# --- SCAFFOLD tuning flags (adjusted conservatively to avoid collapse)
USE_SGD = True                     # Recommended: True for SCAFFOLD (use SGD). Set False to keep AdamW.
SGD_LR = 0.01                      # Learning rate when using SGD
SGD_MOM = 0.9
ADAMW_LR = 1e-4                    # If USE_SGD=False, AdamW uses this lr
GRAD_CORRECTION_SCALE = 0.05       # << reduced scale to avoid dominating gradients
C_UPDATE_SCALE = 0.1               # << shrink client c update term (safe default)
DEBUG_SCAFFOLD = False             # Set True to print grad vs correction norms for debugging

out_dir = "Outputs"
os.makedirs(out_dir, exist_ok=True)

train_img_dirs = [
    r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_imgs",
    r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\train\images",
    r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\images",
    r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\images"
]
train_mask_dirs = [
    r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_masks",
    r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\train\masks",
    r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\masks",
    r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\masks"
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


def get_loader(img_dir, mask_dir, transform, batch_size=4, shuffle=True, oversample_factor=None):
    ds = CVCDataset(img_dir, mask_dir, transform=transform)
    if oversample_factor is not None:
        sampler = RandomSampler(ds, replacement=True, num_samples=oversample_factor * len(ds))
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def compute_metrics(pred, target, smooth=1e-6):
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
# SCAFFOLD local training (safer, diagnostic variant)
# -------------------------
def train_local_scaffold(loader, model, loss_fn, opt, global_c, client_c,
                        grad_correction_scale=GRAD_CORRECTION_SCALE,
                        c_update_scale=C_UPDATE_SCALE,
                        debug=DEBUG_SCAFFOLD):
    """
    SCAFFOLD local training with optional scaling + diagnostics.
    - grad_correction_scale: multiply (global_c - client_c) added to gradients
    - c_update_scale: multiply the computed (1/(eta * tau)) * delta_w term before adding to c_i update
    Returns: local_steps, avg_loss_per_batch, avg_metrics, client_c_new (on CPU)
    """
    model.train()
    total_loss = 0.0
    metrics = []
    # keep initial weights to compute delta for control-variate update
    w_init = {name: param.data.clone().detach() for name, param in model.named_parameters()}

    local_steps = 0
    # extract eta from optimizer (assume all param groups same)
    eta = opt.param_groups[0].get('lr', None)
    if eta is None or eta <= 0:
        raise ValueError("Optimizer learning rate must be > 0 for SCAFFOLD control-variate update.")

    num_batches = 0
    for _ in range(LOCAL_EPOCHS):
        for data, target in tqdm(loader, leave=False):
            num_batches += 1
            data, target = data.to(DEVICE), target.to(DEVICE).unsqueeze(1).float()
            preds = model(data)
            loss = loss_fn(preds, target)

            opt.zero_grad()
            loss.backward()

            # compute norm diagnostics if requested
            if debug and (local_steps % 50 == 0):
                grad_norm_sq = 0.0
                corr_norm_sq = 0.0
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    g = param.grad.detach()
                    grad_norm_sq += float(g.norm().cpu().item() ** 2)
                    if name in global_c and name in client_c:
                        corr = (global_c[name].to(g.device) - client_c[name].to(g.device)) * grad_correction_scale
                        corr_norm_sq += float(corr.norm().cpu().item() ** 2)
                grad_norm = grad_norm_sq ** 0.5
                corr_norm = corr_norm_sq ** 0.5
                print(f"[DEBUG] step {local_steps}: grad_norm={grad_norm:.6f}, corr_norm={corr_norm:.6f}")

            # SCAFFOLD correction: add (global_c - client_c) * scale to each param.grad
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if name in global_c and name in client_c:
                        correction = (global_c[name].to(param.grad.device) - client_c[name].to(param.grad.device)) * grad_correction_scale
                        param.grad.add_(correction)

            opt.step()
            total_loss += float(loss.item())
            metrics.append(compute_metrics(preds.detach(), target))
            local_steps += 1

    # compute new client control variate (store on CPU to reduce GPU memory usage)
    client_c_new = {}
    if local_steps > 0:
        # w_final - w_init
        for name, param in model.named_parameters():
            delta_w = (param.data.clone().detach() - w_init[name]).to(DEVICE)
            # classical formula: c_i_new = c_i - c + (1 / (eta * tau)) * (w_i - w)
            # apply optional c_update_scale to shrink this correction if required
            scaled_term = (1.0 / (eta * local_steps)) * delta_w * c_update_scale
            # keep client control variates on CPU to save GPU memory
            client_c_new[name] = (client_c[name].to(DEVICE) - global_c[name].to(DEVICE) + scaled_term).detach().cpu().clone()
    else:
        # no updates: keep client_c unchanged
        for name, _ in model.named_parameters():
            client_c_new[name] = client_c[name].to('cpu').clone().detach()

    avg_metrics = average_metrics(metrics) if metrics else {}
    avg_loss_per_batch = total_loss / max(1, num_batches)
    print("Train (SCAFFOLD): " + " | ".join([f"{k}: {v:.4f}" for k,v in (avg_metrics or {}).items()]))
    return local_steps, avg_loss_per_batch, avg_metrics, client_c_new

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
    plt.xlabel("Global Round"); plt.ylabel("Dice"); plt.title("Per-client Dice (SCAFFOLD)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_no_bg_scaffold.png")); plt.close()
    # IoU_no_bg
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_iou_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round"); plt.ylabel("IoU"); plt.title("Per-client IoU (SCAFFOLD)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_no_bg_scaffold.png")); plt.close()


def main():
    tr_tf = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=[0]*3, std=[1]*3),
        ToTensorV2()
    ])
    val_tf = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=[0]*3, std=[1]*3),
        ToTensorV2()
    ])

    global_model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    # Initialize global control variate c (per-parameter) as zeros (store on CPU for safety)
    global_c = {}
    for name, param in global_model.named_parameters():
        global_c[name] = torch.zeros_like(param.data).cpu()

    # Initialize per-client control variates c_i as zeros (store on CPU)
    client_cs = []
    for i in range(NUM_CLIENTS):
        ci = {name: torch.zeros_like(param.data).cpu() for name, param in global_model.named_parameters()}
        client_cs.append(ci)

    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models_info = []  # store tuples (local_model, client_c_old_cpu, client_c_new_cpu, local_steps, client_dataset_size)
        total_sz = 0

        # broadcast global state and global control variate to clients
        global_state_for_clients = {k: v.clone().detach() for k,v in global_model.state_dict().items()}
        # move global_c to device for sending to clients
        global_c_for_clients = {k: v.to(DEVICE).clone().detach() for k,v in global_c.items()}

        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model).to(DEVICE)
            # Choose optimizer depending on USE_SGD
            if USE_SGD:
                opt = optim.SGD(local_model.parameters(), lr=SGD_LR, momentum=SGD_MOM)
            else:
                opt = optim.AdamW(local_model.parameters(), lr=ADAMW_LR)
            loss_fn = get_loss_fn(DEVICE)

            oversample_factor = 1
            train_loader = get_loader(train_img_dirs[i], train_mask_dirs[i], tr_tf,
                                      oversample_factor=oversample_factor)
            val_loader = get_loader(val_img_dirs[i], val_mask_dirs[i], val_tf, shuffle=False)

            print(f"[Client {client_names[i]}] Local training (SCAFFOLD)")
            # run local training with SCAFFOLD correction
            # move client's stored c (CPU) to device inside train function via client_cs[i]
            local_steps, train_loss, train_metrics, client_c_new_cpu = train_local_scaffold(
                train_loader, local_model, loss_fn, opt,
                global_c=global_c_for_clients, client_c={k: v.to(DEVICE) for k,v in client_cs[i].items()},
                grad_correction_scale=GRAD_CORRECTION_SCALE,
                c_update_scale=C_UPDATE_SCALE,
                debug=DEBUG_SCAFFOLD
            )

            # keep old and new control variates on CPU
            client_c_old_cpu = client_cs[i]
            client_cs[i] = {name: client_c_new_cpu[name].cpu().clone().detach() for name in client_c_new_cpu.keys()}

            evaluate(val_loader, local_model, loss_fn, split="Val")

            client_dataset_size = len(train_loader.dataset)
            local_models_info.append((local_model, client_c_old_cpu, client_cs[i], local_steps, client_dataset_size))
            total_sz += client_dataset_size

        # Server: aggregate models (FedAvg) and update global control variate
        # Build model list for averaging
        model_list_for_avg = [info[0] for info in local_models_info]
        weights = [info[4] for info in local_models_info]
        norm_weights = [w/total_sz for w in weights]
        avg_state = average_models_weighted(model_list_for_avg, norm_weights)
        global_model.load_state_dict(avg_state, strict=False)

        # Update global control variate c:
        # Weighted average: c = c + sum( weight_i * (c_i_new - c_i_old) )
        c_delta_accum = {name: torch.zeros_like(param.data).cpu() for name, param in global_model.named_parameters()}
        for idx, (local_model, client_c_old_cpu, client_c_new_cpu, local_steps, client_sz) in enumerate(local_models_info):
            w = norm_weights[idx]
            for name in c_delta_accum.keys():
                # ensure CPU tensors
                c_delta_accum[name] += w * (client_c_new_cpu[name].cpu() - client_c_old_cpu[name].cpu())
        for name in global_c.keys():
            global_c[name] = (global_c[name] + c_delta_accum[name]).clone().detach()

        # After server aggregation, overwrite local models with global parameters (optional useful)
        for lm in model_list_for_avg:
            lm.load_state_dict(avg_state, strict=False)

        # Evaluate per-client on test sets (use the aggregated global_model for a fair global evaluation)
        rm = {}
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(test_img_dirs[i], test_mask_dirs[i], val_tf, shuffle=False)
            print(f"[Client {client_names[i]}] Test")
            # Use global_model (aggregated) for testing to see global performance.
            _, test_metrics = evaluate(test_loader, global_model, get_loss_fn(DEVICE), split="Test")
            rm[f"client{i}_dice_no_bg"] = test_metrics.get("dice_no_bg", 0)
            rm[f"client{i}_iou_no_bg"] = test_metrics.get("iou_no_bg", 0)

        round_metrics.append(rm)
        plot_metrics(round_metrics, out_dir)

    print(f"Total runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

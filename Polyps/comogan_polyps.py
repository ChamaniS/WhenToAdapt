import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")

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
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

from models.DuckNet import DuckNet
from dataset import CVCDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
IMG_SIZE = 224
start_time = time.time()

out_dir = "Outputs_polyp_comogan_single"
os.makedirs(out_dir, exist_ok=True)


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


DESIRED_FILES = {
    "Kvasir": "cju2hw5gjlr5h0988so2qqres.png",
    "ETIS": "111.png",
    "CVC-Colon": "140.PNG",
    "CVC-Clinic": "100.PNG"
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def image_list_in_dir(d):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(d):
        return []
    out = []
    for root, _, files in os.walk(d):
        for fn in files:
            if fn.lower().endswith(exts):
                out.append(os.path.join(root, fn))
    return sorted(out)

def find_file_case_insensitive(dirpath, desired_name):
    if not os.path.isdir(dirpath):
        return None
    desired_lower = desired_name.lower()
    files = [f for f in os.listdir(dirpath) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))]
    for f in files:
        if f.lower() == desired_lower:
            return f
    desired_base = os.path.splitext(desired_lower)[0]
    for f in files:
        if os.path.splitext(f.lower())[0] == desired_base:
            return f
    return None


def harmonize_client_validation(client_img_dir, client_name, out_base):
    dst_base = ensure_dir(os.path.join(out_base, "harmonized", client_name, "val"))
    if not os.path.isdir(client_img_dir):
        print(f"[HARM] no val img dir for {client_name} at {client_img_dir}")
        return dst_base

    imgs = image_list_in_dir(client_img_dir)
    print(f"[HARM] (stub) found {len(imgs)} images for {client_name}")
    for idx, p in enumerate(imgs):
        try:
            img = Image.open(p).convert("RGB")
            img = ImageOps.autocontrast(img, cutoff=0)
            seed = sum(bytearray(os.path.basename(p).encode("utf-8"))) % 1000
            ctr = 0.9 + (seed % 11) * 0.02
            bri = 0.95 + ((seed//7) % 9) * 0.02
            img = ImageEnhance.Contrast(img).enhance(ctr)
            img = ImageEnhance.Brightness(img).enhance(bri)
            arr = np.array(img).astype(np.float32)
            r_off = ((seed % 11) - 5) * 1.2
            g_off = (((seed//7) % 9) - 4) * 0.8
            b_off = (((seed//35) % 7) - 3) * 1.0
            arr[...,0] = np.clip(arr[...,0] + r_off, 0, 255)
            arr[...,1] = np.clip(arr[...,1] + g_off, 0, 255)
            arr[...,2] = np.clip(arr[...,2] + b_off, 0, 255)
            out_img = Image.fromarray(arr.astype(np.uint8))
            base = os.path.basename(p)
            name, ext = os.path.splitext(base)
            dst = os.path.join(dst_base, f"{name}_harm{ext}")
            out_img.save(dst)
            if idx < 3:
                print(f"[HARM] wrote {dst} (ctr={ctr:.2f}, bri={bri:.2f}, offs={(r_off,g_off,b_off)})")
        except Exception as e:
            print("[HARM] error processing", p, ":", e)
    print("[HARM] harmonized saved to", dst_base)
    return dst_base

def make_comparison_grid_single_filenames(orig_dir, hm_dir, client_name, out_base, desired_filenames):
    base = ensure_dir(os.path.join(out_base, "ComparisonGrid"))
    for desired in desired_filenames:
        found = find_file_case_insensitive(orig_dir, desired)
        if found is None:
            print(f"[VIS] desired file '{desired}' not found in original dir for {client_name}: {orig_dir}. Skipping.")
            continue

        orig_path = os.path.join(orig_dir, found)

        name, ext = os.path.splitext(found)
        expected_hm_name = f"{name}_harm{ext}"
        hm_found = find_file_case_insensitive(hm_dir, expected_hm_name) if hm_dir and os.path.isdir(hm_dir) else None
        if hm_found:
            hm_path = os.path.join(hm_dir, hm_found)
        else:
            alt = find_file_case_insensitive(hm_dir, found) if hm_dir and os.path.isdir(hm_dir) else None
            if alt:
                hm_path = os.path.join(hm_dir, alt)
            else:
                print(f"[VIS] harmonized file for '{found}' not found in {hm_dir}. Falling back to original as harmonized (diff will be zero).")
                hm_path = None

        try:
            orig = Image.open(orig_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            if hm_path and os.path.exists(hm_path):
                hm = Image.open(hm_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            else:
                hm = orig.copy()

            orig_np = np.array(orig).astype(np.uint8)
            hm_np = np.array(hm).astype(np.uint8)

            diff = np.abs(hm_np.astype(np.float32) - orig_np.astype(np.float32))
            amplified = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
            if amplified.max() < 8 and diff.max() > 0:
                amplified = (diff / (diff.max() + 1e-8) * 255.0).astype(np.uint8)

            fig, axs = plt.subplots(3, 1, figsize=(4, 9))
            axs[0].imshow(orig_np); axs[0].axis('off'); axs[0].set_title(f"Raw: {found}", fontsize=9)
            axs[1].imshow(hm_np); axs[1].axis('off'); axs[1].set_title("CoMoGAN-stub harmonized", fontsize=9)
            axs[2].imshow(amplified); axs[2].axis('off'); axs[2].set_title("Amplified diff", fontsize=9)
            plt.tight_layout()
            out_png = os.path.join(base, f"comparison_{client_name}_{name}.png")
            plt.savefig(out_png, dpi=150)
            plt.close(fig)
            print(f"[VIS] Saved single-file comparison for {client_name}: {out_png}")
        except Exception as e:
            print(f"[VIS] failed to build/save grid for {client_name}, file {found}: {e}")

def get_loader(img_dir, mask_dir, transform, batch_size=4, shuffle=True):
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
    avg_sd = copy.deepcopy(models[0].state_dict())
    for k in avg_sd.keys():
        tensor = sum(weights[i] * models[i].state_dict()[k] for i in range(len(models)))
        avg_sd[k] = tensor
    return avg_sd

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
    return total_loss / max(1, len(loader.dataset)), avg_metrics

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
    return total_loss / max(1, len(loader.dataset)), avg_metrics

def plot_metrics(round_metrics, out_dir):
    rounds = list(range(1, len(round_metrics) + 1))
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_dice_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round"); plt.ylabel("Dice"); plt.title("Per-client Dice")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "dice_no_bg_ducknetbs4.png")); plt.close()

def main():
    tr_tf = A.Compose([A.Resize(IMG_SIZE,IMG_SIZE), A.Normalize(mean=[0]*3,std=[1]*3), ToTensorV2()])
    val_tf = tr_tf

    global_model = DuckNet(input_channels=3, num_classes=1, num_filters=17).to(DEVICE)
    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models, weights = [], []
        total_sz = 0

        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model).to(DEVICE)
            opt = optim.AdamW(local_model.parameters(), lr=1e-4)
            loss_fn = get_loss_fn(DEVICE)

            train_loader = get_loader(train_img_dirs[i], train_mask_dirs[i], tr_tf)
            val_loader = get_loader(val_img_dirs[i], val_mask_dirs[i], val_tf, shuffle=False)

            print(f"[Client {client_names[i]}]")

            try:
                hm_dir = harmonize_client_validation(val_img_dirs[i], client_names[i], out_dir)
                desired = DESIRED_FILES.get(client_names[i])
                if desired:
                    make_comparison_grid_single_filenames(val_img_dirs[i], hm_dir, client_names[i], out_dir, [desired])
                else:
                    print(f"[VIS] no desired filename for {client_names[i]} — skipping grid.")
            except Exception as e:
                print("[HARM] error/harmonize skipped:", e)

            train_local(train_loader, local_model, loss_fn, opt)
            evaluate(val_loader, local_model, loss_fn, split="Val")

            local_models.append(local_model)
            sz = len(train_loader.dataset); weights.append(sz); total_sz += sz

        if total_sz == 0:
            print("Warning: total train size 0; skipping aggregation for this round")
        else:
            norm_weights = [w/total_sz for w in weights]
            avg_sd = average_models_weighted(local_models, norm_weights)
            global_model.load_state_dict(avg_sd)

        rm = {}
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(test_img_dirs[i], test_mask_dirs[i], val_tf, shuffle=False)
            print(f"[Client {client_names[i]}] Test")
            _, test_metrics = evaluate(test_loader, global_model, get_loss_fn(DEVICE), split="Test")
            rm[f"client{i}_dice_no_bg"] = test_metrics.get("dice_no_bg", 0.0)
            rm[f"client{i}_iou_no_bg"] = test_metrics.get("iou_no_bg", 0.0)
        round_metrics.append(rm)
        plot_metrics(round_metrics, out_dir)

    end_time = time.time()
    print(f"Total runtime: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main()

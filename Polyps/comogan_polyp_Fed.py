
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ensure matplotlib doesn't require an X display
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
import shutil

# local imports (assumed present)
# from models.UNET import UNET
from models.DuckNet import DuckNet
from dataset import CVCDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
IMG_SIZE = 224
start_time = time.time()

out_dir = "Outputs_polyp_comogan"
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

def select_val_pairs_for_comparison(orig_dir, hm_dir, n_samples=6):
    orig = image_list_in_dir(orig_dir)
    hm = image_list_in_dir(hm_dir)
    if len(orig) == 0 or len(hm) == 0:
        return []
    orig_map = {}
    for p in orig:
        b = os.path.basename(p)
        if b not in orig_map:
            orig_map[b] = p
    hm_map = {}
    for p in hm:
        b = os.path.basename(p)
        if b not in hm_map:
            hm_map[b] = p
    common = sorted(set(orig_map.keys()) & set(hm_map.keys()))
    pairs = []
    if len(common) > 0:
        sel = common[:n_samples]
        for b in sel:
            pairs.append((orig_map[b], hm_map[b], b))
    else:
        n = min(len(orig), len(hm), n_samples)
        for i in range(n):
            pairs.append((orig[i], hm[i], os.path.basename(orig[i])))
    return pairs

def make_comparison_grid_debug(orig_dir, hm_dir, client_name, out_base, n_samples=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np, os

    base = ensure_dir(os.path.join(out_base, "ComparisonGrid"))
    print(f"[VIS] client {client_name}: orig images {len(image_list_in_dir(orig_dir))}, harmonized {len(image_list_in_dir(hm_dir))}")
    pairs = select_val_pairs_for_comparison(orig_dir, hm_dir, n_samples=n_samples)
    if len(pairs) == 0:
        print(f"[VIS] No pairs found for {client_name} — skipping grid.")
        return

    top_imgs, mid_imgs, diff_imgs, titles = [], [], [], []
    rows = []
    for orig_p, hm_p, fn in pairs:
        try:
            o = Image.open(orig_p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            h = Image.open(hm_p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            A = np.array(o).astype(np.int32)
            B = np.array(h).astype(np.int32)
            D = np.abs(B - A)
            top_imgs.append(A.astype(np.uint8))
            mid_imgs.append(B.astype(np.uint8))
            amplified = np.clip(D * 3.0, 0, 255).astype(np.uint8)
            if amplified.max() < 8 and D.max()>0:
                amplified = (D / (D.max()+1e-8) * 255.0).astype(np.uint8)
            diff_imgs.append(amplified)
            titles.append(fn[:24])

            rows.append({
                "file": fn,
                "max_diff": int(D.max()),
                "mean_diff": float(D.mean()),
                "nonzero_pixels": int((D.sum(axis=2)>0).sum()),
                "orig_path": os.path.abspath(orig_p),
                "hm_path": os.path.abspath(hm_p)
            })
        except Exception as e:
            print("[VIS] skipping pair", fn, ":", e)

    n = len(top_imgs)
    if n == 0:
        print(f"[VIS] no readable images for {client_name}")
        return

    fig, axs = plt.subplots(3, n, figsize=(3.0*n, 8))
    if n == 1:
        axs = np.array([[axs[0]],[axs[1]],[axs[2]]]).reshape(3,1)

    for i in range(n):
        axs[0,i].imshow(top_imgs[i]); axs[0,i].axis('off')
        if i==0: axs[0,i].set_ylabel("Original", fontsize=10)
        axs[0,i].set_title(titles[i], fontsize=9)

        axs[1,i].imshow(mid_imgs[i]); axs[1,i].axis('off')
        if i==0: axs[1,i].set_ylabel("Harmonized", fontsize=10)

        axs[2,i].imshow(diff_imgs[i]); axs[2,i].axis('off')
        if i==0: axs[2,i].set_ylabel("Amplified diff", fontsize=10)

    fig.suptitle(f"Harmonized vs Original: {client_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png = os.path.join(base, f"comparison_{client_name}.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=150); plt.close(fig)
    print("[VIS] saved", out_png)

    all_heat = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i in range(n):
        all_heat += diff_imgs[i].sum(axis=2).astype(np.float32)
    fig3 = plt.figure(figsize=(6,4))
    plt.imshow(all_heat, cmap='magma', vmax=max(1, all_heat.max()))
    plt.title(f"Combined diff heatmap: {client_name}")
    plt.axis('off')

    for r in rows:
        print(f"[VIS-stats] {r['file']}: max={r['max_diff']} mean={r['mean_diff']:.3f} nonzero_px={r['nonzero_pixels']}")

def harmonize_client_validation(client_img_dir, client_name, out_base):
    import numpy as np
    from PIL import Image, ImageOps, ImageEnhance
    dst_base = ensure_dir(os.path.join(out_base, "harmonized", client_name, "val"))
    if not os.path.isdir(client_img_dir):
        print(f"[HARM] no val img dir for {client_name} at {client_img_dir}")
        return dst_base

    imgs = image_list_in_dir(client_img_dir)
    print(f"[HARM] (improved stub) found {len(imgs)} images for {client_name}")
    for idx, p in enumerate(imgs):
        try:
            img = Image.open(p).convert("RGB")
            # stronger, more visible transform for debugging; reduce later if too strong
            img = ImageOps.autocontrast(img, cutoff=0)
            seed = sum(bytearray(os.path.basename(p).encode("utf-8"))) % 1000
            ctr = 0.8 + (seed % 21) * 0.02   # wider contrast change
            bri = 0.9 + ((seed//11) % 11) * 0.02
            img = ImageEnhance.Contrast(img).enhance(ctr)
            img = ImageEnhance.Brightness(img).enhance(bri)
            arr = np.array(img).astype(np.float32)
            # add channel offsets deterministically
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
            # debug log for first few images
            if idx < 3:
                print(f"[HARM] wrote {dst} (ctr={ctr:.2f}, bri={bri:.2f}, offs={(r_off,g_off,b_off)})")
        except Exception as e:
            print("[HARM] error processing", p, ":", e)
    print("[HARM] harmonized saved to", dst_base)
    return dst_base

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

    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_iou_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round"); plt.ylabel("IoU"); plt.title("Per-client IoU ")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "iou_no_bg_ducknetbs4.png")); plt.close()

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
                make_comparison_grid_debug(val_img_dirs[i], hm_dir, client_names[i], out_dir, n_samples=6)
            except Exception as e:
                print("[HARM] error/harmonize skipped:", e)

            train_local(train_loader, local_model, loss_fn, opt)

            evaluate(val_loader, local_model, loss_fn, split="Val")

            local_models.append(local_model)
            sz = len(train_loader.dataset); weights.append(sz); total_sz += sz

        norm_weights = [w/total_sz for w in weights]
        avg_sd = average_models_weighted(local_models, norm_weights)
        global_model.load_state_dict(avg_sd)

        rm = {}
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(test_img_dirs[i], test_mask_dirs[i], val_tf, shuffle=False)
            print(f"[Client {client_names[i]}] Test")
            _, test_metrics = evaluate(test_loader, global_model, get_loss_fn(DEVICE), split="Test")
            rm[f"client{i}_dice_no_bg"] = test_metrics.get("dice_no_bg", 0.0)

        round_metrics.append(rm)
        plot_metrics(round_metrics, out_dir)

    end_time = time.time()
    print(f"Total runtime: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main()

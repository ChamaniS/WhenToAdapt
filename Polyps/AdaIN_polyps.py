
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import copy
import time
import random
import shutil
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from PIL import Image

from models.UNET import UNET
from dataset import CVCDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLIENTS = 4
LOCAL_EPOCHS = 12
COMM_ROUNDS = 10
start_time = time.time()

out_dir = "FL_Outputs_AdaIN_Polyps"
os.makedirs(out_dir, exist_ok=True)

train_img_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_imgs",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\train\images",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\images",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\images",
]
train_mask_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_masks",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\train\masks",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\masks",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\masks",
]
val_img_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_imgs",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\val\images",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\images",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\images"
]
val_mask_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_masks",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\val\masks",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\masks",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\masks"
]
test_img_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_imgs",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\test\images",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\images",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\images"
]
test_mask_dirs = [
    r"xxxxx\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_masks",
    r"xxxxx\Projects\Data\ETIS-Larib polyp\rearranged\test\masks",
    r"xxxxx\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\masks",
    r"xxxxx\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\masks"
]
client_names = ["Kvasir", "ETIS", "CVC-Colon","CVC-Clinic"]


HARMONIZE = True
STYLE_SOURCE = "first_client"
HARMONIZE_VAL_TEST = True
IMG_SIZE = 224
SEED = 42


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def list_images_in_dir(d):
    if not os.path.isdir(d):
        return []
    return [f for f in os.listdir(d) if f.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff"))]

from torchvision.datasets.folder import default_loader
from torchvision import transforms
to_tensor_no_norm = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor()])
to_pil_from_tensor = transforms.ToPILImage()

def image_load_as_tensor(path):
    img = default_loader(path)
    t = to_tensor_no_norm(img)
    return t

def compute_channel_mean_std(tensor):
    c = tensor.view(tensor.shape[0], -1)
    mu = c.mean(dim=1)
    std = c.std(dim=1, unbiased=False)
    std = torch.clamp(std, min=1e-6)
    return mu, std

def adain_transfer(content_t, style_t):
    mu_c, std_c = compute_channel_mean_std(content_t)
    mu_s, std_s = compute_channel_mean_std(style_t)
    mu_c = mu_c[:, None, None]; std_c = std_c[:, None, None]
    mu_s = mu_s[:, None, None]; std_s = std_s[:, None, None]
    normalized = (content_t - mu_c) / std_c
    out = normalized * std_s + mu_s
    out = torch.clamp(out, 0.0, 1.0)
    return out


def pick_style_image_paths(base_img_dirs: List[str]) -> Dict[int, str]:
    rng = random.Random(SEED)
    client_train_images = {}
    for i, d in enumerate(base_img_dirs):
        imgs = []
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.lower().endswith((".png",".jpg",".jpeg",".bmp",".tif",".tiff")):
                    imgs.append(os.path.join(d, fn))
        client_train_images[i] = imgs

    style_map = {}
    if STYLE_SOURCE == "first_client":
        ref_imgs = client_train_images.get(0, [])
        if len(ref_imgs) == 0:
            raise RuntimeError("No images found in first client for style reference")
        ref = ref_imgs[len(ref_imgs)//2]
        for i in range(len(base_img_dirs)):
            style_map[i] = ref
    elif STYLE_SOURCE == "per_client":
        for i in range(len(base_img_dirs)):
            imgs = client_train_images.get(i, [])
            style_map[i] = (imgs[len(imgs)//2] if len(imgs) > 0 else None)
    elif STYLE_SOURCE == "random_global":
        all_imgs = []
        for lst in client_train_images.values():
            all_imgs.extend(lst)
        if len(all_imgs) == 0:
            raise RuntimeError("No images found for random_global style")
        ref = rng.choice(all_imgs)
        for i in range(len(base_img_dirs)):
            style_map[i] = ref
    else:
        raise ValueError("Unknown STYLE_SOURCE")
    return style_map

def create_harmonized_round(round_idx: int,
                            src_img_dirs: List[str],
                            src_mask_dirs: List[str],
                            out_base_dir: str = out_dir) -> Tuple[List[str], List[str], Dict[int,str]]:

    style_map = pick_style_image_paths(src_img_dirs)
    hr_base = os.path.join(out_base_dir, f"harmonized_round_{round_idx}")
    if os.path.exists(hr_base):
        shutil.rmtree(hr_base)
    harmonized_img_dirs = []
    harmonized_mask_dirs = []
    for i in range(len(src_img_dirs)):
        client_name = client_names[i] if i < len(client_names) else f"client{i}"
        dest_client_root = os.path.join(hr_base, client_name)
        ensure_dir(dest_client_root)
        dest_img_dir = os.path.join(dest_client_root, "images")
        dest_mask_dir = os.path.join(dest_client_root, "masks")
        ensure_dir(dest_img_dir); ensure_dir(dest_mask_dir)

        if os.path.isdir(src_mask_dirs[i]):
            for fn in list_images_in_dir(src_mask_dirs[i]):
                src_mask_fp = os.path.join(src_mask_dirs[i], fn)
                dst_mask_fp = os.path.join(dest_mask_dir, fn)
                try:
                    shutil.copy2(src_mask_fp, dst_mask_fp)
                except Exception as e:
                    print(f"[HARM] warning copying mask {src_mask_fp}: {e}")

        style_path = style_map.get(i, None)
        for fn in list_images_in_dir(src_img_dirs[i]) if os.path.isdir(src_img_dirs[i]) else []:
            src_img_fp = os.path.join(src_img_dirs[i], fn)
            dst_img_fp = os.path.join(dest_img_dir, fn)
            if style_path is None:
                try:
                    shutil.copy2(src_img_fp, dst_img_fp)
                except Exception as e:
                    print(f"[HARM] fallback copy failed for {src_img_fp}: {e}")
                continue
            try:
                content_t = image_load_as_tensor(src_img_fp)
                style_t = image_load_as_tensor(style_path)
                hm_t = adain_transfer(content_t, style_t)
                pil = to_pil_from_tensor(hm_t)
                pil.save(dst_img_fp)
            except Exception as e:
                print(f"[HARM] failed harmonizing {src_img_fp} (copying). Err: {e}")
                try:
                    shutil.copy2(src_img_fp, dst_img_fp)
                except Exception as e2:
                    print(f"[HARM] also failed copying {src_img_fp}: {e2}")

        harmonized_img_dirs.append(dest_img_dir)
        harmonized_mask_dirs.append(dest_mask_dir)
    return harmonized_img_dirs, harmonized_mask_dirs, style_map


def create_harmonized_val_vis(orig_val_dir: str, dest_val_dir: str, style_path: str):
    ensure_dir(dest_val_dir)
    for fn in list_images_in_dir(orig_val_dir):
        src_fp = os.path.join(orig_val_dir, fn)
        dst_fp = os.path.join(dest_val_dir, fn)
        try:
            if style_path is None:
                shutil.copy2(src_fp, dst_fp)
            else:
                ct = image_load_as_tensor(src_fp); st = image_load_as_tensor(style_path)
                hm = adain_transfer(ct, st)
                to_pil_from_tensor(hm).save(dst_fp)
        except Exception as e:
            try:
                shutil.copy2(src_fp, dst_fp)
            except Exception as e2:
                print(f"[VIS-HARM] failed copying {src_fp}: {e2}")


def select_pairs_flat(orig_dir: str, hm_dir: str, n_samples: int = 7) -> List[Tuple[str,str,str]]:
    if not os.path.isdir(orig_dir) or not os.path.isdir(hm_dir):
        return []
    orig_files = set(list_images_in_dir(orig_dir))
    hm_files = set(list_images_in_dir(hm_dir))
    common = list(orig_files.intersection(hm_files))
    if len(common) == 0:
        return []
    rng = random.Random(SEED)
    rng.shuffle(common)
    chosen = common[:min(n_samples, len(common))]
    return [(os.path.join(orig_dir, fn), os.path.join(hm_dir, fn), fn) for fn in chosen]

def make_comparison_grid_flat(original_dir: str, harmonized_dir: str, client_name: str, out_base: str, n_samples: int = 7):
    base_dest = os.path.join(out_base, "ComparisonGrid")
    ensure_dir(base_dest)
    pairs = select_pairs_flat(original_dir, harmonized_dir, n_samples=n_samples)
    if len(pairs) == 0:
        print(f"[VIS] no matching pairs for comparison for {client_name} (orig={original_dir}, hm={harmonized_dir})")
        return
    top_imgs = []
    mid_imgs = []
    diff_imgs = []
    titles = []
    for orig_p, hm_p, fn in pairs:
        try:
            orig = np.array(Image.open(orig_p).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
            hm = np.array(Image.open(hm_p).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
            if orig.dtype != np.uint8:
                orig = np.clip(orig, 0, 255).astype(np.uint8)
            if hm.dtype != np.uint8:
                hm = np.clip(hm, 0, 255).astype(np.uint8)
            top_imgs.append(orig)
            mid_imgs.append(hm)
            diff = np.abs(hm.astype(np.float32) - orig.astype(np.float32))
            amplified = diff * 3.0
            amplified = np.clip(amplified, 0, 255).astype(np.uint8)
            if amplified.max() < 8:
                if diff.max() > 0:
                    amplified = (diff / (diff.max() + 1e-8) * 255.0).astype(np.uint8)
                else:
                    amplified = np.zeros_like(amplified, dtype=np.uint8)
            diff_imgs.append(amplified)
            titles.append(fn)
        except Exception as e:
            print(f"[VIS] skipping pair {fn} due to error: {e}")
    n = len(top_imgs)
    if n == 0:
        print(f"[VIS] no readable pairs for {client_name}")
        return
    fig, axs = plt.subplots(3, n, figsize=(2.5 * n, 7.5))
    if n == 1:
        axs = np.array([[axs[0]],[axs[1]],[axs[2]]]) if hasattr(axs, '__len__') else np.array([[axs]])
        axs = axs.reshape(3,1)
    col_titles = [t[:24] for t in titles]
    for i in range(n):
        axs[0, i].imshow(top_imgs[i]); axs[0, i].axis('off')
        if i == 0:
            axs[0, i].set_ylabel("Original", fontsize=10)
        axs[0, i].set_title(col_titles[i], fontsize=9)

        axs[1, i].imshow(mid_imgs[i]); axs[1, i].axis('off')
        if i == 0:
            axs[1, i].set_ylabel("Harmonized", fontsize=10)

        axs[2, i].imshow(diff_imgs[i]); axs[2, i].axis('off')
        if i == 0:
            axs[2, i].set_ylabel("Amplified diff", fontsize=10)
    fig.suptitle(f"Harmonized vs Original: {client_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png = os.path.join(base_dest, f"comparison_{client_name}.png")
    try:
        plt.savefig(out_png); plt.close(fig)
        print(f"[VIS] Saved comparison grid for {client_name} at {out_png}")
    except Exception as e:
        print("[VIS] failed saving comparison grid:", e)
        plt.close(fig)


def get_loader(img_dir, mask_dir, transform, batch_size=8, shuffle=True):
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
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k] for i in range(len(models)))
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
    avg_metrics = average_metrics(metrics) if metrics else {}
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
    avg_metrics = average_metrics(metrics) if metrics else {}
    print(f"{split}: " + " | ".join([f"{k}: {v:.4f}" for k,v in avg_metrics.items()]))
    return total_loss / max(1, len(loader.dataset)), avg_metrics


def plot_metrics(round_metrics, out_dir):
    rounds = list(range(1, len(round_metrics) + 1))
    # Dice_no_bg
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_dice_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round"); plt.ylabel("Dice"); plt.title("Per-client Dice"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice_no_bg_unet.png")); plt.close()

    # IoU_no_bg
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_iou_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round"); plt.ylabel("IoU"); plt.title("Per-client IoU"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_no_bg_unet.png")); plt.close()


def main():
    tr_tf = A.Compose([A.Resize(IMG_SIZE,IMG_SIZE), A.Normalize(mean=[0]*3,std=[1]*3), ToTensorV2()])
    val_tf = tr_tf
    global_model = UNET(in_channels=3, out_channels=1).cuda()
    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models, weights = [], []
        total_sz = 0

        if HARMONIZE:
            print("[HARM] creating harmonized directories for round", r+1)
            harmonized_train_imgs, harmonized_train_masks, style_map = create_harmonized_round(r+1, train_img_dirs, train_mask_dirs, out_base_dir=out_dir)

            harmonized_val_imgs = []
            harmonized_val_masks = []
            harmonized_test_imgs = []
            harmonized_test_masks = []

            for i in range(NUM_CLIENTS):
                client_name = client_names[i] if i < len(client_names) else f"client{i}"
                client_round_root = os.path.join(out_dir, f"harmonized_round_{r+1}", client_name)
                ensure_dir(client_round_root)

                val_dest_img_dir = os.path.join(client_round_root, "val_images")
                val_dest_mask_dir = os.path.join(client_round_root, "val_masks")
                ensure_dir(val_dest_img_dir); ensure_dir(val_dest_mask_dir)
                if HARMONIZE_VAL_TEST:
                    style_path = style_map.get(i, None)
                    if os.path.isdir(val_img_dirs[i]):
                        for fn in list_images_in_dir(val_img_dirs[i]):
                            src_fp = os.path.join(val_img_dirs[i], fn)
                            dst_fp = os.path.join(val_dest_img_dir, fn)
                            try:
                                if style_path is None:
                                    shutil.copy2(src_fp, dst_fp)
                                else:
                                    ct = image_load_as_tensor(src_fp); st = image_load_as_tensor(style_path)
                                    hm = adain_transfer(ct, st)
                                    to_pil_from_tensor(hm).save(dst_fp)
                            except Exception:
                                shutil.copy2(src_fp, dst_fp)
                    if os.path.isdir(val_mask_dirs[i]):
                        for fn in list_images_in_dir(val_mask_dirs[i]):
                            shutil.copy2(os.path.join(val_mask_dirs[i], fn), os.path.join(val_dest_mask_dir, fn))
                else:
                    if os.path.isdir(val_img_dirs[i]):
                        for fn in list_images_in_dir(val_img_dirs[i]):
                            shutil.copy2(os.path.join(val_img_dirs[i], fn), os.path.join(val_dest_img_dir, fn))
                    if os.path.isdir(val_mask_dirs[i]):
                        for fn in list_images_in_dir(val_mask_dirs[i]):
                            shutil.copy2(os.path.join(val_mask_dirs[i], fn), os.path.join(val_dest_mask_dir, fn))

                harmonized_val_imgs.append(val_dest_img_dir)
                harmonized_val_masks.append(val_dest_mask_dir)

                # Test
                test_dest_img_dir = os.path.join(client_round_root, "test_images")
                test_dest_mask_dir = os.path.join(client_round_root, "test_masks")
                ensure_dir(test_dest_img_dir); ensure_dir(test_dest_mask_dir)
                if HARMONIZE_VAL_TEST:
                    style_path = style_map.get(i, None)
                    if os.path.isdir(test_img_dirs[i]):
                        for fn in list_images_in_dir(test_img_dirs[i]):
                            src_fp = os.path.join(test_img_dirs[i], fn)
                            dst_fp = os.path.join(test_dest_img_dir, fn)
                            try:
                                if style_path is None:
                                    shutil.copy2(src_fp, dst_fp)
                                else:
                                    ct = image_load_as_tensor(src_fp); st = image_load_as_tensor(style_path)
                                    hm = adain_transfer(ct, st)
                                    to_pil_from_tensor(hm).save(dst_fp)
                            except Exception:
                                shutil.copy2(src_fp, dst_fp)
                    if os.path.isdir(test_mask_dirs[i]):
                        for fn in list_images_in_dir(test_mask_dirs[i]):
                            shutil.copy2(os.path.join(test_mask_dirs[i], fn), os.path.join(test_dest_mask_dir, fn))
                else:
                    if os.path.isdir(test_img_dirs[i]):
                        for fn in list_images_in_dir(test_img_dirs[i]):
                            shutil.copy2(os.path.join(test_img_dirs[i], fn), os.path.join(test_dest_img_dir, fn))
                    if os.path.isdir(test_mask_dirs[i]):
                        for fn in list_images_in_dir(test_mask_dirs[i]):
                            shutil.copy2(os.path.join(test_mask_dirs[i], fn), os.path.join(test_dest_mask_dir, fn))

                harmonized_test_imgs.append(test_dest_img_dir)
                harmonized_test_masks.append(test_dest_mask_dir)

                val_vis_dir = os.path.join(out_dir, f"harmonized_round_{r+1}", "val_vis", client_name)
                if os.path.exists(val_vis_dir):
                    shutil.rmtree(val_vis_dir)
                style_path = style_map.get(i, None)
                if os.path.isdir(val_img_dirs[i]):
                    create_harmonized_val_vis(val_img_dirs[i], val_vis_dir, style_path)
                else:
                    ensure_dir(val_vis_dir)
                try:
                    make_comparison_grid_flat(val_img_dirs[i], val_vis_dir, client_name, os.path.join(out_dir, f"harmonized_round_{r+1}"), n_samples=7)
                except Exception as e:
                    print(f"[VIS] failed making grid for {client_name}: {e}")

            used_train_img_dirs = harmonized_train_imgs
            used_train_mask_dirs = harmonized_train_masks
            used_val_img_dirs = harmonized_val_imgs
            used_val_mask_dirs = harmonized_val_masks
            used_test_img_dirs = harmonized_test_imgs
            used_test_mask_dirs = harmonized_test_masks

        else:
            used_train_img_dirs = train_img_dirs
            used_train_mask_dirs = train_mask_dirs
            used_val_img_dirs = val_img_dirs
            used_val_mask_dirs = val_mask_dirs
            used_test_img_dirs = test_img_dirs
            used_test_mask_dirs = test_mask_dirs

        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model).to(DEVICE)
            opt = optim.AdamW(local_model.parameters(), lr=1e-4)
            loss_fn = get_loss_fn(DEVICE)

            train_loader = get_loader(used_train_img_dirs[i], used_train_mask_dirs[i], tr_tf, batch_size=8, shuffle=True)
            val_loader = get_loader(used_val_img_dirs[i], used_val_mask_dirs[i], val_tf, batch_size=8, shuffle=False)

            print(f"[Client {client_names[i]}] training on {used_train_img_dirs[i]}")
            train_local(train_loader, local_model, loss_fn, opt)
            evaluate(val_loader, local_model, loss_fn, split="Val")

            local_models.append(local_model)
            sz = len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 0
            weights.append(sz); total_sz += sz

        if total_sz == 0:
            print("Warning: total train size 0; skipping aggregation for this round")
        else:
            norm_weights = [w / total_sz for w in weights]
            print("[FedAvg] Aggregating with normalized weights:", norm_weights)
            global_model.load_state_dict(average_models_weighted(local_models, norm_weights))

        rm = {}
        for i in range(NUM_CLIENTS):
            test_loader = get_loader(used_test_img_dirs[i], used_test_mask_dirs[i], val_tf, batch_size=8, shuffle=False)
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

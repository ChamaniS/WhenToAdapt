import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import os, copy, time, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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
    r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\images",
    r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\images",
]
train_mask_dirs = [
    r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_masks",
    r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\train\masks",
    r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\masks",
    r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\masks",
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
# Additional imports & helpers for saving visualizations
# -------------------------
import numpy as np
from PIL import Image
import shutil
from skimage import exposure  # for histogram matching
import torchvision.transforms.functional as TF  # used by dataset fallback

def _unnormalize_image(tensor, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)):
    """
    tensor: torch tensor CxHxW normalized by mean/std
    returns uint8 HxWx3 numpy array
    """
    arr = tensor.cpu().numpy()
    if arr.ndim == 3:
        c, h, w = arr.shape
        arr = arr.transpose(1,2,0)
    else:
        # unexpected shape
        arr = arr
    arr = arr * np.array(std).reshape(1,1,3) + np.array(mean).reshape(1,1,3)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr

def _mask_to_uint8(mask_tensor):
    """
    mask_tensor: torch tensor (1,H,W) or (H,W)
    returns uint8 HxW
    """
    m = mask_tensor.cpu().numpy()
    if m.ndim == 3:
        m = np.squeeze(m, axis=0)
    m = (m > 0.5).astype(np.uint8) * 255
    return m

def save_image(arr, path):
    Image.fromarray(arr).save(path)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# -------------------------
# Utils
# -------------------------
def get_loader(img_dir, mask_dir, transform, batch_size=1, shuffle=True, return_filenames=False):
    """
    Try to create CVCDataset with `return_filename` or `return_filenames` if supported.
    If not supported, return a standard dataset. Returns a DataLoader.
    """
    try:
        # try common keyword name
        if return_filenames:
            ds = CVCDataset(img_dir, mask_dir, transform=transform, return_filename=True)
        else:
            ds = CVCDataset(img_dir, mask_dir, transform=transform)
    except TypeError:
        try:
            if return_filenames:
                ds = CVCDataset(img_dir, mask_dir, transform=transform, return_filenames=True)
            else:
                ds = CVCDataset(img_dir, mask_dir, transform=transform)
        except TypeError:
            # dataset doesn't accept those kwargs
            ds = CVCDataset(img_dir, mask_dir, transform=transform)
            # return_filenames will be effectively ignored
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

def average_models_weighted(models, weights):
    avg_sd = copy.deepcopy(models[0].state_dict())
    for k in avg_sd.keys():
        avg_sd[k] = sum(weights[i] * models[i].state_dict()[k] for i in range(len(models)))
    return avg_sd

# -------------------------
# Histogram Matching Helpers (Harmonization Method 2)
# -------------------------
def compute_reference_image(ref_img_dir, n_samples=64, resize_to=(224,224)):
    """
    Compute a reference image by averaging up to n_samples images from ref_img_dir.
    Returns uint8 HxWx3 numpy array.
    """
    files = sorted([f for f in os.listdir(ref_img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    files = files[:min(len(files), n_samples)]
    if len(files) == 0:
        raise ValueError(f"No images found in reference dir: {ref_img_dir}")
    acc = None
    count = 0
    for fn in files:
        p = os.path.join(ref_img_dir, fn)
        img = Image.open(p).convert("RGB")
        if resize_to is not None:
            img = img.resize(resize_to, resample=Image.BILINEAR)
        arr = np.array(img).astype(np.float32)
        if acc is None:
            acc = arr
        else:
            acc += arr
        count += 1
    avg = (acc / count).astype(np.uint8)
    return avg

def histogram_match_dataset(src_img_dir, src_mask_dir, dst_img_dir, dst_mask_dir, reference_image):
    """
    Create a histogram-matched copy of images in src_img_dir to dst_img_dir using reference_image.
    Masks are copied from src_mask_dir to dst_mask_dir unchanged.
    """
    ensure_dir(dst_img_dir); ensure_dir(dst_mask_dir)
    img_files = sorted([f for f in os.listdir(src_img_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])
    for fn in img_files:
        src_img_p = os.path.join(src_img_dir, fn)
        dst_img_p = os.path.join(dst_img_dir, fn)
        # load
        img = np.array(Image.open(src_img_p).convert("RGB"))
        # match histogram
        try:
            # recent skimage uses channel_axis arg
            matched = exposure.match_histograms(img, reference_image, channel_axis=-1)
        except TypeError:
            try:
                matched = exposure.match_histograms(img, reference_image, multichannel=True)
            except Exception:
                matched = img
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        Image.fromarray(matched).save(dst_img_p)
        # copy mask if exists (same filename)
        src_mask_p = os.path.join(src_mask_dir, fn)
        if os.path.exists(src_mask_p):
            shutil.copy(src_mask_p, os.path.join(dst_mask_dir, fn))
        else:
            # try without extension or alternative extensions
            base, _ = os.path.splitext(fn)
            found = False
            for ext in ('.png','.jpg','.jpeg','.bmp'):
                alt = os.path.join(src_mask_dir, base + ext)
                if os.path.exists(alt):
                    shutil.copy(alt, os.path.join(dst_mask_dir, base + ext))
                    found = True
                    break
            # if not found, leave mask folder missing that file

def create_histogram_matched_datasets(train_img_dirs, train_mask_dirs,
                                      val_img_dirs, val_mask_dirs,
                                      test_img_dirs, test_mask_dirs,
                                      client_names, out_base,
                                      reference_client_idx=0,
                                      n_ref_samples=64,
                                      resize_ref=(224,224)):
    """
    Creates harmonized copies of all clients' train/val/test images under:
      out_base/HistMatch/<client>/{train_images, train_masks, val_images, ...}
    Returns the lists of new directories to use.
    """
    base = os.path.join(out_base, "HistMatch")
    ensure_dir(base)

    # compute reference image from chosen client's train images
    ref_dir = train_img_dirs[reference_client_idx]
    print(f"[HistMatch] Computing reference image from {ref_dir} (n={n_ref_samples}) ...")
    ref_img = compute_reference_image(ref_dir, n_samples=n_ref_samples, resize_to=resize_ref)

    # save reference image for inspection
    ref_save = os.path.join(base, "reference_image.png")
    Image.fromarray(ref_img).save(ref_save)
    print(f"[HistMatch] Reference image saved to {ref_save}")

    train_img_hm_dirs = []
    train_mask_hm_dirs = []
    val_img_hm_dirs = []
    val_mask_hm_dirs = []
    test_img_hm_dirs = []
    test_mask_hm_dirs = []

    for i, cname in enumerate(client_names):
        client_base = os.path.join(base, cname)
        train_dst_img = os.path.join(client_base, "train_images")
        train_dst_mask = os.path.join(client_base, "train_masks")
        val_dst_img = os.path.join(client_base, "val_images")
        val_dst_mask = os.path.join(client_base, "val_masks")
        test_dst_img = os.path.join(client_base, "test_images")
        test_dst_mask = os.path.join(client_base, "test_masks")
        print(f"[HistMatch] Harmonizing client {cname} train/val/test ...")
        histogram_match_dataset(train_img_dirs[i], train_mask_dirs[i], train_dst_img, train_dst_mask, ref_img)
        histogram_match_dataset(val_img_dirs[i], val_mask_dirs[i], val_dst_img, val_dst_mask, ref_img)
        histogram_match_dataset(test_img_dirs[i], test_mask_dirs[i], test_dst_img, test_dst_mask, ref_img)

        train_img_hm_dirs.append(train_dst_img); train_mask_hm_dirs.append(train_dst_mask)
        val_img_hm_dirs.append(val_dst_img); val_mask_hm_dirs.append(val_dst_mask)
        test_img_hm_dirs.append(test_dst_img); test_mask_hm_dirs.append(test_dst_mask)

    print("[HistMatch] Done creating histogram-matched datasets.")
    return train_img_hm_dirs, train_mask_hm_dirs, val_img_hm_dirs, val_mask_hm_dirs, test_img_hm_dirs, test_mask_hm_dirs

# -------------------------
# Functions to save harmonized examples + test predictions
# -------------------------
def save_transformed_samples(img_dir, mask_dir, transform, client_name, out_base, n_samples=8, prefix="harmonized"):
    ds = CVCDataset(img_dir, mask_dir, transform=transform)
    dest = os.path.join(out_base, f"{client_name}", prefix)
    ensure_dir(dest)
    num = min(n_samples, len(ds))
    for i in range(num):
        try:
            img_t, mask_t = ds[i]  # expected transform returns tensors via ToTensorV2
        except Exception as e:
            # If CVCDataset returns (img, mask) as numpy arrays, handle accordingly
            item = ds[i]
            if isinstance(item, tuple) and len(item) >= 2:
                img_t, mask_t = item[0], item[1]
            else:
                raise e
        # If mask_t is dict or other, attempt to get mask
        # Normalize assumptions: transform includes Normalize and ToTensorV2
        # If img_t is numpy HxWxC (albumentations without ToTensorV2), handle that
        if isinstance(img_t, np.ndarray):
            # HxWxC in 0-255
            img_arr = img_t.astype(np.uint8)
            save_image(img_arr, os.path.join(dest, f"{client_name}_img_{i}.png"))
        else:
            img_arr = _unnormalize_image(img_t)
            save_image(img_arr, os.path.join(dest, f"{client_name}_img_{i}.png"))

        # mask
        if isinstance(mask_t, np.ndarray):
            m_arr = (mask_t > 0.5).astype(np.uint8) * 255
            save_image(m_arr, os.path.join(dest, f"{client_name}_mask_{i}.png"))
        else:
            m_arr = _mask_to_uint8(mask_t)
            save_image(m_arr, os.path.join(dest, f"{client_name}_mask_{i}.png"))

def save_test_predictions(global_model, test_loader, client_name, out_base, round_num, max_to_save=16):
    """
    Save predicted masks (as .png) for test images for a client.
    - If the test_loader yields filenames (3rd element in batch) they will be used.
    - Otherwise, attempts to use dataset images list (e.g. dataset.images or dataset.common_basenames) to preserve order.
    - Falls back to idx_b naming if none available.
    """
    global_model.eval()
    latest_dir = os.path.join(out_base, "TestPreds", client_name, "latest")

    # Clear latest dir then recreate
    if os.path.exists(latest_dir):
        shutil.rmtree(latest_dir)
    ensure_dir(latest_dir)

    # Try to get ordered filename list from dataset if supported
    dataset_filenames = None
    try:
        ds = test_loader.dataset
        # dataset may expose images or common_basenames etc.
        if hasattr(ds, "images") and isinstance(ds.images, list):
            dataset_filenames = ds.images  # filenames with extensions, ordered per dataset
        elif hasattr(ds, "common_basenames") and isinstance(ds.common_basenames, list):
            # try to reconstruct filenames by preserving extension using masks if available
            # as a fallback, use basename only (may not have extension)
            dataset_filenames = ds.common_basenames
        elif hasattr(ds, "common") and isinstance(ds.common, list):
            dataset_filenames = ds.common
    except Exception:
        dataset_filenames = None

    saved = 0
    global_model.to(DEVICE)
    with torch.no_grad():
        # If DataLoader yields (data, target, filenames) we use those names
        for batch_idx, batch in enumerate(test_loader):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                data, target, fnames = batch
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                data, target = batch[0], batch[1]
                fnames = None
            else:
                raise RuntimeError("Unexpected batch format from test_loader")

            # standardize target dims: BxH x W -> Bx1xHxW
            if target.dim() == 3:
                target = target.unsqueeze(1)
            elif target.dim() == 4:
                pass
            else:
                # unexpected target dims, try to convert
                try:
                    target = target.unsqueeze(1)
                except Exception:
                    pass

            data = data.to(DEVICE)
            preds = global_model(data)
            probs = torch.sigmoid(preds)
            bin_mask = (probs > 0.5).float()

            bsz = data.size(0)
            for b in range(bsz):
                mask_t = bin_mask[b].cpu()
                mask_arr = _mask_to_uint8(mask_t)   # convert to 0/255 uint8

                # choose filename priority:
                # 1) fnames from batch (if provided)
                # 2) dataset_filenames using global sample index
                # 3) fallback to client_idx naming
                if fnames is not None:
                    try:
                        orig_name = fnames[b]
                        # if orig_name is a path, take basename
                        orig_name = os.path.basename(orig_name)
                        base, _ = os.path.splitext(orig_name)
                        fname = f"{base}_pred.png"
                    except Exception:
                        fname = f"{client_name}_pred_mask_{batch_idx}_{b}.png"
                elif dataset_filenames is not None:
                    # compute global index = batch_idx * batch_size + b
                    global_idx = batch_idx * test_loader.batch_size + b
                    if global_idx < len(dataset_filenames):
                        orig_name = dataset_filenames[global_idx]
                        # if dataset_filenames are basenames without ext, keep them; else use as-is
                        base = os.path.splitext(orig_name)[0]
                        fname = f"{base}_pred.png"
                    else:
                        fname = f"{client_name}_pred_mask_{batch_idx}_{b}.png"
                else:
                    fname = f"{client_name}_pred_mask_{batch_idx}_{b}.png"

                save_image(mask_arr, os.path.join(latest_dir, fname))
                saved += 1
                if saved >= max_to_save:
                    break
            if saved >= max_to_save:
                break

    print(f"Saved {saved} prediction masks for {client_name} in {latest_dir}")

def make_comparison_grid_and_histograms_updated(img_dir, mask_dir, val_transform, visual_transform,
                                                client_name, out_base, n_samples=7, diff_amp=4.0,
                                                reference_client_idx=0, n_ref_samples=50):
    """
    Comparison grid:
     - Top row: ORIGINAL raw images from train_img_dirs (resized for display via visual_transform)
     - Middle row: HISTOGRAM-MATCHED images (match to reference image)
     - Bottom row: Amplified absolute color difference (orig_resized vs matched)
    Behavior:
     - If Outputs/HistMatch/reference_image.png exists it will be used as reference.
     - Else: computes an average reference from train_img_dirs[reference_client_idx] (first n_ref_samples).
    """
    base_dest = os.path.join(out_base, "HarmonizedSamples", client_name)
    diffs_dest = os.path.join(base_dest, "diffs")
    hist_dest = os.path.join(base_dest, "histograms")
    ensure_dir(base_dest); ensure_dir(diffs_dest); ensure_dir(hist_dest)

    # Use train_img_dirs for raw originals (guaranteed)
    try:
        client_idx = client_names.index(client_name)
        raw_dir = train_img_dirs[client_idx]
    except Exception:
        raw_dir = img_dir
        print(f"[make_comparison_grid] Warning: client_name '{client_name}' not found -> using provided img_dir as raw_dir")

    fnames = sorted([f for f in os.listdir(raw_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])[:n_samples]
    if len(fnames) == 0:
        print(f"No images found in {raw_dir} for {client_name}")
        return

    # Try to load precomputed reference image
    ref_path = os.path.join(out_base, "HistMatch", "reference_image.png")
    reference_img = None
    if os.path.exists(ref_path):
        try:
            reference_img = np.array(Image.open(ref_path).convert("RGB"))
            print(f"[make_comparison_grid] Using existing reference image: {ref_path}")
        except Exception as e:
            print(f"[make_comparison_grid] Could not read reference image: {e}. Will compute reference on-the-fly.")

    top_imgs, mid_imgs, diff_color_imgs, short_names = [], [], [], []

    # Determine display target size using visual_transform on the first raw image
    first_img = Image.open(os.path.join(raw_dir, fnames[0])).convert("RGB")
    first_np = np.array(first_img)
    vis_out0 = visual_transform(image=first_np)
    vis_img0 = vis_out0['image']
    if isinstance(vis_img0, np.ndarray):
        target_h, target_w = vis_img0.shape[0], vis_img0.shape[1]
    else:
        tmp = np.array(vis_img0)
        target_h, target_w = tmp.shape[0], tmp.shape[1]

    # If no reference image loaded, compute an average reference from reference_client_idx (resized to target size)
    if reference_img is None:
        try:
            ref_dir = train_img_dirs[reference_client_idx]
            ref_files = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(('.png','.jpg','.jpeg'))])[:n_ref_samples]
            if len(ref_files) == 0:
                raise RuntimeError("no files in ref_dir")
            ref_stack = []
            for rf in ref_files:
                p = os.path.join(ref_dir, rf)
                im = Image.open(p).convert("RGB").resize((target_w, target_h), resample=Image.BILINEAR)
                ref_stack.append(np.array(im).astype(np.float32))
            ref_mean = np.mean(np.stack(ref_stack, axis=0), axis=0)
            reference_img = np.clip(ref_mean, 0, 255).astype(np.uint8)
            # save reference for later
            ref_out_dir = os.path.join(out_base, "HistMatch")
            ensure_dir(ref_out_dir)
            save_image(reference_img, os.path.join(ref_out_dir, "reference_image.png"))
            print(f"[make_comparison_grid] Computed and saved reference image from {ref_dir}")
        except Exception as e:
            print(f"[make_comparison_grid] Failed to compute reference image: {e}. Mid-row will fallback to val_transform per-image.")
            reference_img = None

    for fname in fnames:
        raw_path = os.path.join(raw_dir, fname)
        orig_pil = Image.open(raw_path).convert("RGB")
        orig_np = np.array(orig_pil)  # raw uint8

        # TOP: original resized using visual_transform (keeps same display size)
        vis_out = visual_transform(image=orig_np)
        vis_img = vis_out['image']
        top = vis_img.astype(np.uint8) if isinstance(vis_img, np.ndarray) else np.array(vis_img).astype(np.uint8)
        # ensure top is target size
        if top.shape[0] != target_h or top.shape[1] != target_w:
            top = np.array(Image.fromarray(top).resize((target_w, target_h), resample=Image.BILINEAR)).astype(np.uint8)

        # MIDDLE: histogram-matched version of the SAME raw image (match then resize to target size)
        if reference_img is not None:
            try:
                # Try skimage API with channel_axis first (newer versions)
                try:
                    matched = exposure.match_histograms(orig_np, reference_img, channel_axis=-1)
                except TypeError:
                    matched = exposure.match_histograms(orig_np, reference_img, multichannel=True)
                matched = np.clip(matched, 0, 255).astype(np.uint8)
                # resize matched to target size for display consistency
                mid = np.array(Image.fromarray(matched).resize((target_w, target_h), resample=Image.BILINEAR)).astype(np.uint8)
                applied_hist = True
            except Exception as e:
                print(f"[make_comparison_grid] histogram matching failed for {fname}: {e}. Falling back to val_transform.")
                applied_hist = False
        else:
            applied_hist = False

        if not applied_hist:
            # fallback: use val_transform (resize + normalize -> unnormalize)
            vt_out = val_transform(image=orig_np)
            img_t = vt_out['image']
            if isinstance(img_t, torch.Tensor):
                mid = _unnormalize_image(img_t)
            elif isinstance(img_t, np.ndarray):
                hnp = img_t
                if hnp.dtype != np.uint8:
                    if hnp.max() <= 1.5:
                        hnp = (np.clip(hnp,0,1)*255).astype(np.uint8)
                    else:
                        hnp = np.clip(hnp,0,255).astype(np.uint8)
                mid = hnp.astype(np.uint8)
            else:
                mid = top.copy()
            # ensure mid size
            if mid.shape[0] != target_h or mid.shape[1] != target_w:
                mid = np.array(Image.fromarray(mid).resize((target_w, target_h), resample=Image.BILINEAR)).astype(np.uint8)

        # DIFF: absolute color diff between top and mid
        diff_color = np.abs(top.astype(int) - mid.astype(int)).astype(np.uint8)
        amp = np.clip((diff_color.astype(float) * diff_amp), 0, 255).astype(np.uint8)

        # Save matched and diffs
        matched_dir = os.path.join(base_dest, "matched")
        ensure_dir(matched_dir)
        save_image(mid, os.path.join(matched_dir, f"matched_{fname}"))
        save_image(top, os.path.join(diffs_dest, f"orig_resized_{fname}"))
        save_image(diff_color, os.path.join(diffs_dest, f"diff_color_{fname}"))
        save_image(amp, os.path.join(diffs_dest, f"diff_color_amp_{fname}"))

        # Save histogram plot only in hist_dest
        fig, ax = plt.subplots(1,3, figsize=(12,3))
        colors = ['r','g','b']
        for ch in range(3):
            ax[ch].hist(top[:,:,ch].ravel(), bins=256, alpha=0.5, label='orig', color=colors[ch])
            ax[ch].hist(mid[:,:,ch].ravel(), bins=256, alpha=0.7, label='matched', color=colors[ch], histtype='step')
            ax[ch].legend(fontsize=6)
            ax[ch].set_title(['R','G','B'][ch])
        plt.tight_layout()
        hist_path = os.path.join(hist_dest, f"hist_{fname}.png")
        plt.savefig(hist_path)
        plt.close(fig)

        # Collect for grid
        top_imgs.append(top)
        mid_imgs.append(mid)
        diff_color_imgs.append(amp)
        short_names.append(fname if len(fname) <= 20 else fname[:17] + "...")

    # Build 3-row grid (Original | Histogram-Matched | Amplified Color Diff)
    n = len(top_imgs)
    fig_w = max(3 * n, 10)
    fig_h = 6
    fig, axs = plt.subplots(3, n, figsize=(fig_w, fig_h))
    if n == 1:
        axs = np.array([[axs[0]],[axs[1]],[axs[2]]])

    fig.suptitle(f"Harmonized (Histogram matching using average of Kvasir samples) vs. Original: {client_name}", fontsize=16, y=0.98)
    fig.text(0.035, 0.82, "Original\nimages", fontsize=12, va='center', rotation='vertical')
    fig.text(0.035, 0.50, "Histogram-\nmatched \n (Average)", fontsize=12, va='center', rotation='vertical')
    fig.text(0.035, 0.18, "Amplified\nDifference", fontsize=12, va='center', rotation='vertical')

    for i in range(n):
        axs[0, i].imshow(top_imgs[i]); axs[0, i].axis('off'); axs[0, i].set_title(short_names[i], fontsize=8)
        axs[1, i].imshow(mid_imgs[i]); axs[1, i].axis('off')
        axs[2, i].imshow(diff_color_imgs[i]); axs[2, i].axis('off')

    plt.tight_layout(rect=[0.06, 0.03, 0.99, 0.90])

    plt.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.03, wspace=0.02, hspace=0.06)
    grid_path = os.path.join(base_dest, f"comparison_grid_{client_name}.png")
    plt.savefig(grid_path, dpi=150)
    plt.close(fig)

    print(f"Saved comparison grid: {grid_path}")
    print(f"Saved matched images in: {os.path.join(base_dest, 'matched')}")
    print(f"Saved color diffs in: {diffs_dest} (raw + amplified).")
    print(f"Saved RGB histograms in: {hist_dest} (histograms only).")


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
    plt.savefig(os.path.join(out_dir, "dice_no_bg_harm2.png"))
    plt.close()

    # IoU_no_bg
    plt.figure()
    for cid in range(NUM_CLIENTS):
        vals = [rm.get(f"client{cid}_iou_no_bg", 0) for rm in round_metrics]
        plt.plot(rounds, vals, label=f"{client_names[cid]}")
    plt.xlabel("Global Round")
    plt.ylabel("IoU")
    plt.title("Per-client IoU ")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iou_no_bg_harm2.png"))
    plt.close()

# -------------------------
# Main FedAvg
# -------------------------
def main():
    # -------------------------
    # Basic preprocessing + normalization + consistent augmentation
    # -------------------------
    tr_tf = A.Compose([
        A.RandomResizedCrop(256, 256, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=0.6),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.35, rotate_limit=25, p=0.6),

        A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.15),

        A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomGamma(gamma_limit=(60, 140), p=0.3),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MotionBlur(blur_limit=7),
            A.MedianBlur(blur_limit=3)
        ], p=0.2),
        A.GaussNoise(var_limit=(5.0, 50.0), p=0.2),

        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    # Validation / test: only resize + normalize
    val_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # VISUAL transforms used to create original-resized images for comparison (no Normalize/ToTensor)
    visual_val_tf = A.Compose([A.Resize(224, 224)])

    # -------------------------
    # Optionally create histogram-matched copies and use those paths
    # -------------------------
    USE_HIST_MATCH = True  # set False to use original dirs
    if USE_HIST_MATCH:
        (train_img_dirs_used, train_mask_dirs_used,
         val_img_dirs_used, val_mask_dirs_used,
         test_img_dirs_used, test_mask_dirs_used) = create_histogram_matched_datasets(
            train_img_dirs, train_mask_dirs,
            val_img_dirs, val_mask_dirs,
            test_img_dirs, test_mask_dirs,
            client_names, out_dir,
            reference_client_idx=0  # use client 0 (Kvasir) as reference
        )
    else:
        train_img_dirs_used = train_img_dirs; train_mask_dirs_used = train_mask_dirs
        val_img_dirs_used = val_img_dirs; val_mask_dirs_used = val_mask_dirs
        test_img_dirs_used = test_img_dirs; test_mask_dirs_used = test_mask_dirs

    # -------------------------
    # Save harmonized/augmented samples and comparison grids BEFORE training
    # -------------------------
    print("Saving harmonized samples (val-style) and augmented samples (train-style) for each client...")
    visuals_base = os.path.join(out_dir, "HarmonizedSamples")
    for i in range(NUM_CLIENTS):
        cname = client_names[i]
        # harmonized (deterministic - val transform) - save transformed images and masks (first 7)
        save_transformed_samples(val_img_dirs_used[i], val_mask_dirs_used[i], val_tf, cname, visuals_base, n_samples=7, prefix="harmonized")
        # augmented (train transform - random) - also save first 7 augmented examples if you want
        save_transformed_samples(train_img_dirs_used[i], train_mask_dirs_used[i], tr_tf, cname, visuals_base, n_samples=7, prefix="augmented")

        # Create and save the comparison grid (top: original resized, middle: harmonized, bottom: amplified diff)
        make_comparison_grid_and_histograms_updated(val_img_dirs_used[i], val_mask_dirs_used[i], val_tf, visual_val_tf, cname, visuals_base, n_samples=7)
    print("Saved harmonized/augmented samples and comparison grids to Outputs/HarmonizedSamples/")

    # -------------------------
    # Federated training loop (FedAvg)
    # -------------------------
    global_model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    round_metrics = []

    for r in range(COMM_ROUNDS):
        print(f"\n[Comm Round {r+1}/{COMM_ROUNDS}]")
        local_models, weights = [], []
        total_sz = 0

        for i in range(NUM_CLIENTS):
            local_model = copy.deepcopy(global_model).to(DEVICE)
            opt = optim.AdamW(local_model.parameters(), lr=1e-4)
            loss_fn = get_loss_fn(DEVICE)

            train_loader = get_loader(train_img_dirs_used[i], train_mask_dirs_used[i], tr_tf)
            val_loader = get_loader(val_img_dirs_used[i], val_mask_dirs_used[i], val_tf, shuffle=False)

            print(f"[Client {client_names[i]}]")
            train_local(train_loader, local_model, loss_fn, opt)
            evaluate(val_loader, local_model, loss_fn, split="Val")

            local_models.append(local_model)
            sz = len(train_loader.dataset); weights.append(sz); total_sz += sz

        # FedAvg aggregation
        norm_weights = [w/total_sz for w in weights]
        global_model.load_state_dict(average_models_weighted(local_models, norm_weights))

        # ---- Testing: evaluate global model on each client's test set and save latest predictions ----
        rm = {}
        for i in range(NUM_CLIENTS):
            # request filenames if possible to preserve order/names
            test_loader = get_loader(test_img_dirs_used[i], test_mask_dirs_used[i], val_tf, shuffle=False)
            print(f"[Client {client_names[i]}] Test")
            _, test_metrics = evaluate(test_loader, global_model, get_loss_fn(DEVICE), split="Test")
            rm[f"client{i}_dice_no_bg"] = test_metrics["dice_no_bg"]
            rm[f"client{i}_iou_no_bg"] = test_metrics["iou_no_bg"]

            # save predicted binary masks into latest folder for this client (overwritten each round)
            save_test_predictions(global_model, test_loader, client_names[i], out_dir, round_num=(r+1), max_to_save=int(len(test_loader.dataset)))

        round_metrics.append(rm)
        plot_metrics(round_metrics, out_dir)

    end_time = time.time()
    print(f"Total runtime: {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main()

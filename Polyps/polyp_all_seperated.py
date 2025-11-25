CUDA_LAUNCH_BLOCKING=1
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CUDA_LAUNCH_BLOCKING=1
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.UNET import UNET
#from models.DuckNet import DuckNet
from dataset import CVCDataset
import torch.nn as nn
import segmentation_models_pytorch as smp
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# Metric functions (same as before)
# -------------------------------
def compute_metrics(pred, target, smooth=1e-6):
    """
    pred: raw logits or probabilities (B,1,H,W) or (B,1) etc.
    target: binary ground-truth (B, H, W) or (B,1,H,W)
    Returns foreground dice/iou as before (dice_no_bg / iou_no_bg),
    and class-averaged (foreground+background)/2 as dice_with_bg / iou_with_bg.
    """
    # convert to binary tensors
    pred_prob = torch.sigmoid(pred) if pred.dtype != torch.bool and pred.max() > 1 else pred
    pred_bin = (pred_prob > 0.5).float()
    target_bin = (target > 0.5).float()

    # ensure channel dims: (B,1,H,W)
    if pred_bin.dim() == target_bin.dim() - 1:
        pred_bin = pred_bin.unsqueeze(1)
    if target_bin.dim() == pred_bin.dim() - 1:
        target_bin = target_bin.unsqueeze(1)

    # flatten to compute totals
    pred_f = pred_bin.view(-1)
    targ_f = target_bin.view(-1)

    TP = (pred_f * targ_f).sum().item()
    TN = ((1 - pred_f) * (1 - targ_f)).sum().item()
    FP = (pred_f * (1 - targ_f)).sum().item()
    FN = ((1 - pred_f) * targ_f).sum().item()

    # foreground (no background) Dice / IoU
    dice_fg = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    iou_fg = (TP + smooth) / (TP + FP + FN + smooth)

    # background class: treat background as positive
    # intersection for background is TN, predicted_bg_sum = TN + FN, target_bg_sum = TN + FP
    dice_bg = (2 * TN + smooth) / (2 * TN + FP + FN + smooth)
    iou_bg = (TN + smooth) / (TN + FP + FN + smooth)

    # averaged-over-classes (foreground + background)/2
    dice_with_bg = 0.5 * (dice_fg + dice_bg)
    iou_with_bg = 0.5 * (iou_fg + iou_bg)

    acc = (TP + TN) / (TP + TN + FP + FN + smooth)
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)
    specificity = (TN + smooth) / (TN + FP + smooth)

    return {
        "dice_with_bg": dice_with_bg,
        "dice_no_bg": dice_fg,
        "iou_with_bg": iou_with_bg,
        "iou_no_bg": iou_fg,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }


def aggregate_metrics(metrics_list):
    agg = {}
    if not metrics_list:
        return agg
    for key in metrics_list[0].keys():
        agg[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
    return agg


# -------------------------------
# Dataset helpers
# -------------------------------
def build_concat_dataset(dataset_class, img_dirs, mask_dirs, transform=None):
    datasets = []
    for img_dir, mask_dir in zip(img_dirs, mask_dirs):
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            datasets.append(dataset_class(img_dir, mask_dir, transform=transform))
        else:
            print(f"[WARN] Skipping missing dataset: {img_dir} or {mask_dir}")
    if not datasets:
        raise RuntimeError("No datasets found for provided directories.")
    return ConcatDataset(datasets)

def build_single_dataset(dataset_class, img_dir, mask_dir, transform=None):
    if os.path.exists(img_dir) and os.path.exists(mask_dir):
        return dataset_class(img_dir, mask_dir, transform=transform)
    else:
        print(f"[WARN] Missing dataset for: {img_dir} or {mask_dir}")
        return None

def get_loaders(dataset_class, train_img_dirs, train_mask_dirs,
                val_img_dirs, val_mask_dirs,
                test_img_dirs, test_mask_dirs,
                batch_size, train_transform, val_transform,
                num_workers):

    train_ds = build_concat_dataset(dataset_class, train_img_dirs, train_mask_dirs, transform=train_transform)
    val_ds = build_concat_dataset(dataset_class, val_img_dirs, val_mask_dirs, transform=val_transform)
    test_ds = build_concat_dataset(dataset_class, test_img_dirs, test_mask_dirs, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader


# -------------------------------
# Loss selector
# -------------------------------
def get_loss_fn(net, device):
    return smp.losses.DiceLoss(mode="binary", from_logits=True)


# -------------------------------
# Train / Val / Test loops
# -------------------------------
def train(train_loader, model, optimizer, scheduler, loss_fn):
    loop = tqdm(train_loader)
    total_loss, total_correct = 0.0, 0.0
    all_metrics = []

    for data, targets in loop:
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)
        preds = model(data)

        if preds.shape[1] == 1:  # binary case
            loss = loss_fn(preds, targets.unsqueeze(1).float())
        else:  # multi-class case
            loss = loss_fn(preds, targets.long())

        preds_label = torch.argmax(preds, dim=1) if preds.shape[1] > 1 else (torch.sigmoid(preds) > 0.5).long().squeeze(1)
        total_correct += (preds_label == targets).float().mean().item()
        total_loss += loss.item()

        # ✅ Metrics (binary assumes channel dim for targets)
        if preds.shape[1] == 1:
            batch_metrics = compute_metrics(preds, targets.unsqueeze(1))
            all_metrics.append(batch_metrics)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

    scheduler.step()

    # safer averaging over batches
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_acc = 100. * (total_correct / num_batches) if num_batches > 0 else 0.0
    avg_metrics = aggregate_metrics(all_metrics) if all_metrics else {}

    return avg_loss, avg_acc, avg_metrics


def eval_performance(loader, model, loss_fn):
    val_running_loss = 0.0
    val_running_correct = 0.0
    all_metrics = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            predictions = model(x)

            if predictions.shape[1] == 1:  # binary case
                loss = loss_fn(predictions, y.unsqueeze(1).float())
            else:  # multi-class case
                loss = loss_fn(predictions, y.long())

            preds = torch.argmax(predictions, dim=1) if predictions.shape[1] > 1 else (torch.sigmoid(predictions) > 0.5).long().squeeze(1)
            val_running_correct += torch.mean((preds == y).float()).item()
            val_running_loss += loss.item()

            if predictions.shape[1] == 1:
                batch_metrics = compute_metrics(predictions, y.unsqueeze(1))
                all_metrics.append(batch_metrics)

    num_batches = len(loader)
    epoch_loss = val_running_loss / num_batches if num_batches > 0 else 0.0
    epoch_acc = 100. * (val_running_correct / num_batches) if num_batches > 0 else 0.0
    avg_metrics = aggregate_metrics(all_metrics) if all_metrics else {}

    model.train()
    return epoch_loss, epoch_acc, avg_metrics


def test(loader, model, loss_fn):
    test_running_loss = 0.0
    test_running_correct = 0.0
    all_metrics = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            predictions = model(x)

            if predictions.shape[1] == 1:  # binary case
                loss = loss_fn(predictions, y.unsqueeze(1).float())
            else:  # multi-class case
                loss = loss_fn(predictions, y.long())

            preds = torch.argmax(predictions, dim=1) if predictions.shape[1] > 1 else (torch.sigmoid(predictions) > 0.5).long().squeeze(1)
            test_running_correct += torch.mean((preds == y).float()).item()
            test_running_loss += loss.item()

            if predictions.shape[1] == 1:
                batch_metrics = compute_metrics(predictions, y.unsqueeze(1))
                all_metrics.append(batch_metrics)

    num_batches = len(loader)
    epoch_loss = test_running_loss / num_batches if num_batches > 0 else 0.0
    epoch_acc = 100. * (test_running_correct / num_batches) if num_batches > 0 else 0.0
    avg_metrics = aggregate_metrics(all_metrics) if all_metrics else {}

    model.train()
    return epoch_loss, epoch_acc, avg_metrics


# -------------------------------
# Utils
# -------------------------------
def print_results(mode, loss, acc, metrics):
    print(f"\n[{mode}] Loss: {loss:.4f} | Acc: {acc:.2f}%")

    if metrics:
        ordered_keys = [
            "dice_with_bg",
            "dice_no_bg",
            "iou_with_bg",
            "iou_no_bg",
            "accuracy",
            "precision",
            "recall",
            "specificity"
        ]
        for k in ordered_keys:
            if k in metrics:
                print(f"{k}: {metrics[k]:.4f}", end=" | ")
        print()


# -------------------------------
# Main
# -------------------------------
def main():
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 4
    NUM_EPOCHS =120
    NUM_WORKERS = 1

    dataset_class = CVCDataset

    # -------------------------------
    # Train/Val/Test dataset paths
    # -------------------------------
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

    # -------------------------------
    # Transforms
    # -------------------------------
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0] * 3, std=[1] * 3),
        ToTensorV2(),
    ])
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0] * 3, std=[1] * 3),
        ToTensorV2(),
    ])

    train_loader, val_loader, _ = get_loaders(
        dataset_class,
        train_img_dirs, train_mask_dirs,
        val_img_dirs, val_mask_dirs,
        test_img_dirs, test_mask_dirs,
        BATCH_SIZE, train_transform, val_transform, NUM_WORKERS
    )

    # -------------------------------
    # Model
    # -------------------------------
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    #model = DuckNet(input_channels=3, num_classes=1, num_filters=17).to(DEVICE)

    for param in model.parameters():
        param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO]: {total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO]: {total_trainable_params:,} trainable parameters.")

    loss_fn = get_loss_fn(model, DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)

    # -------------------------------
    # Training loop (same as before)
    # -------------------------------
    best_iou = 0
    os.makedirs("BestModels", exist_ok=True)
    for epoch in range(NUM_EPOCHS):
        print(f"[INFO]: Epoch {epoch + 1} of {NUM_EPOCHS}")

        train_loss, train_acc, train_metrics = train(train_loader, model, optimizer, scheduler, loss_fn)
        val_loss, val_acc, val_metrics = eval_performance(val_loader, model, loss_fn)

        print_results("TRAIN", train_loss, train_acc, train_metrics)
        print_results("VAL", val_loss, val_acc, val_metrics)

        if val_metrics and val_metrics.get("iou_no_bg", 0) > best_iou:
            best_iou = val_metrics["iou_no_bg"]
            torch.save(model.state_dict(), 'BestModels/best_model_centralized_unet.pth')
            print("Model saved!")

    print("[INFO]: Loading the best centralized model for per-client testing...")
    best_path = 'BestModels/best_model_centralized_unet.pth'
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Expected saved model at {best_path} but not found.")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.to(DEVICE)

    # -------------------------------
    # Per-client testing (4 clients)
    # -------------------------------
    num_clients = min(len(test_img_dirs), len(test_mask_dirs))
    for i in range(num_clients):
        client_name = f"Client_{i+1}"
        img_dir = test_img_dirs[i]
        mask_dir = test_mask_dirs[i]
        ds = build_single_dataset(dataset_class, img_dir, mask_dir, transform=val_transform)
        if ds is None:
            print(f"[SKIP] {client_name}: missing test dataset.")
            continue

        loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
        test_loss, test_acc, test_metrics = test(loader, model, loss_fn)
        print(f"\n===== Results for {client_name} =====")
        print_results(f"{client_name} TEST", test_loss, test_acc, test_metrics)

    print("[INFO]: All client tests complete.")

if __name__ == "__main__":
    main()

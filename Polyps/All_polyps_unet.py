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
#from models.UNET import UNET
from models.DuckNet import DuckNet
from dataset import CVCDataset
import torch.nn as nn
import segmentation_models_pytorch as smp
DEVICE = "cuda"

# -------------------------------
# Metric functions (same as before)
# -------------------------------
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

    return {
        "dice_with_bg": dice_with_bg,
        "dice_no_bg": dice_no_bg,
        "iou_with_bg": iou_with_bg,
        "iou_no_bg": iou_no_bg,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
    }

def aggregate_metrics(metrics_list):
    agg = {}
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
    return ConcatDataset(datasets)


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
    #out_channels = net.conv.out_channels if hasattr(net, "conv") else net.outc.out_channels
    #if out_channels == 1:
        #return nn.BCEWithLogitsLoss().to(device)
     return smp.losses.DiceLoss(mode="binary", from_logits=True)
    #else:
        #return nn.CrossEntropyLoss().to(device)


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

    N = len(train_loader.dataset)
    avg_loss = total_loss / N
    avg_acc = 100. * total_correct / N
    avg_metrics = aggregate_metrics(all_metrics) if all_metrics else {}

    return avg_loss, avg_acc, avg_metrics


def eval_performance(loader, model, loss_fn):
    val_running_loss = 0.0
    val_running_correct = 0.0
    all_metrics = []

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

    dataset_size = len(loader.dataset)
    epoch_loss = val_running_loss / dataset_size
    epoch_acc = 100. * (val_running_correct / dataset_size)
    avg_metrics = aggregate_metrics(all_metrics) if all_metrics else {}

    return epoch_loss, epoch_acc, avg_metrics


def test(loader, model, loss_fn):
    test_running_loss = 0.0
    test_running_correct = 0.0
    all_metrics = []

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

    dataset_size = len(loader.dataset)
    epoch_loss = test_running_loss / dataset_size
    epoch_acc = 100. * (test_running_correct / dataset_size)
    avg_metrics = aggregate_metrics(all_metrics) if all_metrics else {}

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
    BATCH_SIZE = 1
    NUM_EPOCHS =120
    NUM_WORKERS = 1
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256

    dataset_class = CVCDataset

    # -------------------------------
    # Train/Val/Test dataset paths
    # -------------------------------
    train_img_dirs = [
        r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\images",
        r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\images",
        r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_imgs",
        r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\train\images"
    ]
    train_mask_dirs = [
        r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\train\masks",
        r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\train\masks",
        r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\train_masks",
        r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\train\masks"
    ]

    val_img_dirs = [
        r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\images",
        r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\images",
        r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_imgs",
        r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\val\images"
    ]
    val_mask_dirs = [
        r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\val\masks",
        r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\val\masks",
        r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\val_masks",
        r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\val\val\masks"
    ]

    test_img_dirs = [
        r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\images",
        r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\images",
        r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_imgs",
        r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\test\images"
    ]
    test_mask_dirs = [
        r"C:\Users\csj5\Projects\Data\CVC_ClinicDB\PNG\cvc-clinic\rearranged\test\masks",
        r"C:\Users\csj5\Projects\Data\CVC-ColonDB\CVC-colon\rearranged\test\masks",
        r"C:\Users\csj5\Projects\Data\kvasir-seg\Kvasir-SEG\centralized_Kvasir-SEG\test_masks",
        r"C:\Users\csj5\Projects\Data\ETIS-Larib polyp\rearranged\test\masks"
    ]

    # -------------------------------
    # Transforms
    # -------------------------------
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_loader, val_loader, test_loader = get_loaders(
        dataset_class,
        train_img_dirs, train_mask_dirs,
        val_img_dirs, val_mask_dirs,
        test_img_dirs, test_mask_dirs,
        BATCH_SIZE, train_transform, val_transform, NUM_WORKERS
    )

    # -------------------------------
    # Model
    # -------------------------------
    #model = UNET(in_channels=3, out_channels=1).cuda()
    model = DuckNet(input_channels=3, num_classes=1, num_filters=34).cuda()

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
    for epoch in range(NUM_EPOCHS):
        print(f"[INFO]: Epoch {epoch + 1} of {NUM_EPOCHS}")

        train_loss, train_acc, train_metrics = train(train_loader, model, optimizer, scheduler, loss_fn)
        val_loss, val_acc, val_metrics = eval_performance(val_loader, model, loss_fn)

        print_results("TRAIN", train_loss, train_acc, train_metrics)
        print_results("VAL", val_loss, val_acc, val_metrics)

        if val_metrics and val_metrics.get("iou_no_bg", 0) > best_iou:
            best_iou = val_metrics["iou_no_bg"]
            os.makedirs("BestModels", exist_ok=True)
            torch.save(model.state_dict(), 'BestModels/best_model_centralized.pth')
            print("Model saved!")

    print("[INFO]: Testing the best model...")
    model.load_state_dict(torch.load('BestModels/best_model_centralized.pth'))
    test_loss, test_acc, test_metrics = test(test_loader, model, loss_fn)
    print_results("TEST", test_loss, test_acc, test_metrics)


if __name__ == "__main__":
    main()

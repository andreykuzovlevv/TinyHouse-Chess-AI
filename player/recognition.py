import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms

# ----------------------------
# Defaults
# ----------------------------
IMG_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15
RANDOM_SEED = 42


def build_datasets(data_dir: Path, img_size: int, val_split: float, seed: int):
    """
    Assumes directory structure:
      dataset_labeled/squares/
        W_PAWN/
        W_HORSE/
        ...
        B_KING/
    Optionally: EMPTY/ (if you labeled empty cells as a class)
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02
            ),
            transforms.RandomRotation(8, fill=(0, 0, 0)),
            transforms.RandomAffine(
                degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=0
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Full dataset for train/val split
    full_ds = datasets.ImageFolder(str(data_dir), transform=None)

    # Indices
    num_items = len(full_ds)
    num_val = max(1, int(num_items * val_split))
    num_train = num_items - num_val

    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(
        range(num_items), [num_train, num_val], generator=g
    )

    # Two dataset views with different transforms
    train_ds = datasets.ImageFolder(str(data_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(data_dir), transform=val_tf)

    train_sub = Subset(
        train_ds, train_idx.indices if hasattr(train_idx, "indices") else train_idx
    )
    val_sub = Subset(
        val_ds, val_idx.indices if hasattr(val_idx, "indices") else val_idx
    )

    return train_sub, val_sub, train_ds.classes, train_ds.class_to_idx


def build_model(num_classes: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(p=0.20), nn.Linear(in_features, num_classes))
    return model


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    per_class_correct = {c: 0 for c in class_names}
    per_class_total = {c: 0 for c in class_names}

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        for y, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            per_class_total[class_names[y]] += 1
            if y == p:
                per_class_correct[class_names[y]] += 1

    per_class_acc = {
        c: (
            (per_class_correct[c] / per_class_total[c])
            if per_class_total[c] > 0
            else 0.0
        )
        for c in class_names
    }

    return running_loss / total, correct / total, per_class_acc


def main():
    ap = argparse.ArgumentParser(description="Train TinyHouse square classifier")
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("player/dataset_labeled/squares"),
        help="Path to labeled squares root (folders per class)",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("player/models"),
        help="Where to save model and artifacts",
    )
    ap.add_argument("--imgsz", type=int, default=IMG_SIZE)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--wd", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--val_split", type=float, default=VAL_SPLIT)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Repro
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Data
    train_ds, val_ds, class_names, class_to_idx = build_datasets(
        args.data, args.imgsz, args.val_split, args.seed
    )

    # Dataloaders (Windows-safe)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model = build_model(num_classes=len(class_names))
    model.to(device)

    # Optim / loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # AMP if CUDA available
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc, per_class = evaluate(
            model, val_loader, criterion, device, class_names
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        # Save best
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            out_path = args.outdir / "square_cls.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "class_to_idx": class_to_idx,
                    "img_size": args.imgsz,
                },
                out_path,
            )

        # Write metrics each epoch
        with open(args.outdir / "val_metrics.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "per_class_acc": per_class,
                    "class_names": class_names,
                },
                f,
                indent=2,
            )

    # Persist mapping separately too
    with open(args.outdir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Saved model to: {args.outdir / 'square_cls.pt'}")


if __name__ == "__main__":
    main()

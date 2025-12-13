# train_square_classifier.py
# Usage (PowerShell, from project folder with your venv):
#   pip install torch torchvision pillow
#   python .\train_square_classifier.py
#
# Outputs:
#   - Augmented dataset on disk: dataset_labeled/squares_aug/
#   - Checkpoint for your loader: models/squares_cls.pt

import os, shutil, random, time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
from PIL import Image
import torchvision.transforms as T

# ---------- Fixed settings (no knobs) ----------
DATA_SRC = "dataset_labeled/squares"  # your labeled folders (W_PAWN, EMPTY, ...)
DATA_AUG = "dataset_labeled/squares_aug"  # rebuilt each run
MODEL_OUT = "models/squares_cls3.pt"
IMG_SIZE = 128
EPOCHS = 40
BATCH = 32
LR = 3e-4
WEIGHT_DEC = 1e-4
PATIENCE = 8
SEED = 42
NUM_WORKERS = min(4, os.cpu_count() or 0)
AUG_MULT = 4  # per original image, write 4 augmented copies (on-disk)

random.seed(SEED)
torch.manual_seed(SEED)


def rebuild_augmented_dataset():
    src = Path(DATA_SRC)
    dst = Path(DATA_AUG)

    if not src.is_dir():
        raise SystemExit(f"Missing dataset folder: {src}")

    # Re-create destination
    if dst.exists():
        shutil.rmtree(dst)
    for cls in sorted([d for d in src.iterdir() if d.is_dir()]):
        (dst / cls.name).mkdir(parents=True, exist_ok=True)

    # Aug pipeline for file copies (small, square-preserving tweaks)
    aug = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomAffine(
                degrees=5, translate=(0.06, 0.06), scale=(0.94, 1.06), shear=4, fill=0
            ),
            T.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.04, hue=0.02),
            T.RandomPerspective(distortion_scale=0.05, p=0.5),
        ]
    )

    count = 0
    for cls in sorted([d for d in src.iterdir() if d.is_dir()]):
        out_dir = dst / cls.name
        for p in sorted(cls.iterdir()):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            # Copy original (preserve as-is but resized to IMG_SIZE for consistency)
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            im_r = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            im_r.save(out_dir / p.name)

            # Write AUG_MULT augmented variants
            stem = p.stem
            ext = ".png"
            for k in range(AUG_MULT):
                im_aug = aug(im)
                im_aug.save(out_dir / f"{stem}_aug{k}{ext}")
            count += 1
    print(f"[AUG] Built augmented dataset at {dst} (from {count} originals).")


def stratified_split(dataset, val_frac=0.2):
    # dataset is ImageFolder; dataset.samples = [(path, class_idx), ...]
    per_class = {}
    for idx, (_, c) in enumerate(dataset.samples):
        per_class.setdefault(c, []).append(idx)
    train_idx, val_idx = [], []
    for c, idxs in per_class.items():
        n = len(idxs)
        v = max(1, int(round(n * val_frac)))
        random.shuffle(idxs)
        val_idx += idxs[:v]
        train_idx += idxs[v:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def make_samplers(train_subset, num_classes):
    # Weighted sampling to balance classes
    # Count per-class in train subset
    counts = [0] * num_classes
    for _, c in [train_subset.dataset.samples[i] for i in train_subset.indices]:
        counts[c] += 1
    class_w = [0 if c == 0 else 1.0 / c for c in counts]
    sample_w = [
        class_w[c]
        for _, c in [train_subset.dataset.samples[i] for i in train_subset.indices]
    ]
    sampler = WeightedRandomSampler(
        sample_w, num_samples=len(sample_w), replacement=True
    )
    return sampler, counts


def train():
    os.makedirs("models", exist_ok=True)

    rebuild_augmented_dataset()

    # Datasets and transforms
    train_tf = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.20),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    full = datasets.ImageFolder(DATA_AUG, transform=train_tf)
    class_names = full.classes  # folder names (sorted)
    num_classes = len(class_names)
    if "EMPTY" not in class_names:
        print("[WARN] 'EMPTY' class folder not found. It should be present.")
    print(f"[DATA] Classes ({num_classes}): {class_names}")

    # Split (stratified 80/20)
    train_subset, val_subset = stratified_split(full, val_frac=0.2)
    # Swap val transform on the val subset (Dataset object is shared)
    val_subset.dataset.transform = val_tf

    # Sampler for train
    sampler, counts = make_samplers(train_subset, num_classes)
    print(f"[DATA] Train counts per class (pre-sampling): {counts}")

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Model: ResNet-18 + Dropout(0.2) + Linear(num_classes)
    # Try pretrained; fall back to None if not available (offline).
    try:
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except Exception:
        backbone = models.resnet18(weights=None)
    in_feat = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Dropout(p=0.20),
        nn.Linear(in_feat, num_classes),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)

    # Optimizer / loss / schedule
    opt = torch.optim.AdamW(backbone.parameters(), lr=LR, weight_decay=WEIGHT_DEC)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_acc, best_state, epochs_no_improve = 0.0, None, 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        # ---- Train ----
        backbone.train()
        tr_loss, tr_ok, tr_n = 0.0, 0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = backbone(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
            tr_ok += (logits.argmax(1) == y).sum().item()
            tr_n += x.size(0)
        sched.step()

        # ---- Val ----
        backbone.eval()
        va_loss, va_ok, va_n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = backbone(x)
                loss = criterion(logits, y)
                va_loss += loss.item() * x.size(0)
                va_ok += (logits.argmax(1) == y).sum().item()
                va_n += x.size(0)

        tr_loss /= max(1, tr_n)
        va_loss /= max(1, va_n)
        tr_acc = tr_ok / max(1, tr_n)
        va_acc = va_ok / max(1, va_n)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{EPOCHS} "
            f"| train loss {tr_loss:.4f} acc {tr_acc:.3f} "
            f"| val loss {va_loss:.4f} acc {va_acc:.3f} "
            f"| {dt:.1f}s"
        )

        # Early stopping on best val acc
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu() for k, v in backbone.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"[EARLY STOP] no improvement for {PATIENCE} epochs.")
                break

    if best_state is None:
        best_state = {k: v.cpu() for k, v in backbone.state_dict().items()}

    # Save checkpoint exactly for your loader
    ckpt = {
        "class_names": class_names,
        "img_size": IMG_SIZE,
        "model_state": best_state,
    }
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    torch.save(ckpt, MODEL_OUT)
    print(f"[OK] Saved classifier checkpoint to {MODEL_OUT}")
    print(f"[BEST] val acc: {best_acc:.3f}")


if __name__ == "__main__":
    train()

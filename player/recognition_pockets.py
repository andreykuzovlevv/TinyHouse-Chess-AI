# train_pockets.py
# Usage (PowerShell, in your venv, from the project root that contains dataset_labeled/):
#   pip install --upgrade ultralytics opencv-python
#   python train_pockets.py

import os
import shutil
import textwrap
from ultralytics import YOLO

# Fixed relative paths (use forward slashes; Python on Windows accepts them)
DATASET_DIR = "dataset_labeled/pockets"
IMAGES_DIR = "dataset_labeled/pockets/images"
LABELS_DIR = "dataset_labeled/pockets/labels"
CLASSES_TXT = "dataset_labeled/pockets/classes.txt"
DATA_YAML = "pockets_data.yaml"
BEST_WEIGHTS = "runs/pockets/train/weights/best.pt"
FINAL_WEIGHTS = "models/pockets.pt"


def main():
    # Basic checks
    if not os.path.isdir(IMAGES_DIR):
        raise SystemExit(f"Missing: {IMAGES_DIR}")
    if not os.path.isdir(LABELS_DIR):
        raise SystemExit(f"Missing: {LABELS_DIR}")
    if not os.path.isfile(CLASSES_TXT):
        raise SystemExit(f"Missing: {CLASSES_TXT}")

    # Read classes (one per line, already ordered P,H,F,W,K)
    with open(CLASSES_TXT, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    if not names:
        raise SystemExit("classes.txt is empty")

    # Write minimal data.yaml (train=val=images for tiny data)
    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write(
            textwrap.dedent(
                f"""
            path: {DATASET_DIR}
            train: images
            val: images
            names:
        """
            ).lstrip()
        )
        for i, n in enumerate(names):
            f.write(f"  {i}: {n}\n")

    # Train a tiny YOLO; short schedule to overfit tiny set (intentional).
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=60,
        patience=20,
        imgsz=512,
        batch=4,
        seed=42,
        pretrained=True,
        cache=True,
        project="runs/pockets",
        name="train",
        exist_ok=True,  # reuse the same folder on reruns
        verbose=True,
    )

    # Save final weights to a stable path
    os.makedirs("models", exist_ok=True)
    if not os.path.isfile(BEST_WEIGHTS):
        raise SystemExit(f"best.pt not found at {BEST_WEIGHTS}")
    shutil.copy2(BEST_WEIGHTS, FINAL_WEIGHTS)
    print(f"[OK] Model saved to {FINAL_WEIGHTS}")

    # Smoke test on the training images (writes annotated PNGs)
    pred_model = YOLO(FINAL_WEIGHTS)
    results = pred_model.predict(
        source=IMAGES_DIR,
        save=True,
        project="runs/pockets",
        name="smoke",
        exist_ok=True,
        conf=0.25,
        imgsz=512,
        verbose=False,
    )

    # Compact summary
    for r in results:
        counts = {}
        if r.boxes is not None and len(r.boxes):
            for c in r.boxes.cls.tolist():
                counts[int(c)] = counts.get(int(c), 0) + 1
        print(f"{os.path.basename(r.path)} -> {counts}")

    print("[READY] Use models/pockets.pt in your app.")


if __name__ == "__main__":
    main()

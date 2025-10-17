# test_pockets.py
# Usage (PowerShell, from the project folder):
#   python .\test_pockets.py path\to\some_image.png
# If you don't pass an image path, it will try one from dataset_labeled/pockets/images/.

import os
import sys
from ultralytics import YOLO

MODEL_PATH = "models/pockets.pt"
DEFAULT_IMAGE = "dataset_labeled/pockets/images/1760699820879.png"


def main():
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"Missing model: {MODEL_PATH}")

    # Pick source: CLI arg or default
    if len(sys.argv) >= 2:
        source = sys.argv[1]
    else:
        source = DEFAULT_IMAGE

    if not os.path.exists(source):
        raise SystemExit(f"Image not found: {source}")

    model = YOLO(MODEL_PATH)

    # Run prediction and save annotated image(s)
    results = model.predict(
        source=source,  # file path or folder
        conf=0.25,  # confidence threshold
        imgsz=512,  # inference size
        save=True,  # write annotated images
        project="runs/pockets",  # output root
        name="test_vis",  # subfolder
        exist_ok=True,  # reuse folder on reruns
        verbose=False,
    )

    # Report where the file(s) were written and a compact detection summary
    # Ultralytics returns a list of Result objects (one per image)
    if not results:
        print("No results.")
        return

    # Save dir is shared; use the first result
    out_dir = results[0].save_dir  # e.g., runs/pockets/test_vis
    print(f"[OK] Annotated image(s) saved to: {out_dir}")

    # Print detections by image
    for r in results:
        counts = {}
        if r.boxes is not None and len(r.boxes):
            # r.names maps class id -> class name loaded from the model
            for c in r.boxes.cls.tolist():
                cid = int(c)
                counts[r.names[cid]] = counts.get(r.names[cid], 0) + 1
        fname = os.path.basename(getattr(r, "path", "image"))
        print(f"{fname} -> {counts}")


if __name__ == "__main__":
    main()

# dataset_capture_min_keys.py
# Usage:
#   python dataset_capture_min_keys.py --calib tinyhouse_calibration.json --out dataset --side white
#
# Hotkeys:
#   F6  : capture bottom-left square
#   F7  : capture square to the right of bottom-left
#   F8  : capture OUR pockets area
#   ESC : quit

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import keyboard
import mss
import numpy as np

FILES = "abcd"
RANKS = "1234"


def make_square(file_idx: int, rank_idx: int) -> str:
    return FILES[file_idx] + RANKS[rank_idx]


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def as_mss(self):
        return {
            "left": int(self.x),
            "top": int(self.y),
            "width": int(self.w),
            "height": int(self.h),
        }


@dataclass
class Calibration:
    board: Rect
    our_pockets: Rect


def load_calibration(path: Path) -> Calibration:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return Calibration(
        board=Rect(*d["board"]),
        our_pockets=Rect(*d["our_pockets"]),
    )


def grab_rect(sct: mss.mss, r: Rect) -> np.ndarray:
    raw = sct.grab(r.as_mss())
    return cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)


def cell_crop(board_bgr: np.ndarray, row: int, col: int) -> np.ndarray:
    """Crop a cell by screen row/col (row 0=top, col 0=left)."""
    H, W = board_bgr.shape[:2]
    cw, ch = W / 4.0, H / 4.0
    x1 = int(round(col * cw))
    x2 = int(round((col + 1) * cw))
    y1 = int(round(row * ch))
    y2 = int(round((row + 1) * ch))
    return board_bgr[y1:y2, x1:x2]


def name_for_rowcol(row: int, col: int, white_bottom: bool) -> str:
    """
    Map a screen row/col to canonical square name a1..d4 for filenames.
    white_bottom=True => bottom of screen is rank 1.
    """
    if white_bottom:
        # screen row 3 => rank1, row 0 => rank4
        rank_idx = 3 - row
        file_idx = col
    else:
        # flipped: screen row 0 => rank1, row 3 => rank4
        rank_idx = row
        file_idx = 3 - col
    return make_square(file_idx, rank_idx)


def ensure_dirs(base: Path):
    (base / "squares").mkdir(parents=True, exist_ok=True)
    (base / "pockets_ours").mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="TinyHouse targeted capture")
    ap.add_argument(
        "--calib",
        type=Path,
        default=Path("tinyhouse_calibration.json"),
        help="Path to tinyhouse_calibration.json",
    )
    ap.add_argument(
        "--out", type=Path, default=Path("dataset"), help="Output directory"
    )
    ap.add_argument(
        "--side",
        choices=["white", "black"],
        default="white",
        help="Which side is at the bottom of the screen (affects square naming)",
    )
    args = ap.parse_args()

    calib = load_calibration(args.calib)
    white_bottom = args.side == "white"

    ensure_dirs(args.out)
    sct = mss.mss()

    print("Ready. F6=bottom-left square, F7=right of it, F8=pockets, ESC=quit.")

    def capture_bottom_left():
        ts = int(time.time() * 1000)
        board = grab_rect(sct, calib.board)
        # bottom-left on SCREEN is row=3, col=0
        row, col = 3, 0
        crop = cell_crop(board, row, col)
        sq_name = name_for_rowcol(row, col, white_bottom)
        out_path = args.out / "squares" / f"{ts}_{sq_name}_bl.png"
        cv2.imwrite(str(out_path), crop)
        print(f"Saved bottom-left square -> {out_path.name}")

    def capture_right_of_bl():
        ts = int(time.time() * 1000)
        board = grab_rect(sct, calib.board)
        # right of bottom-left is same row, col=1
        row, col = 3, 1
        crop = cell_crop(board, row, col)
        sq_name = name_for_rowcol(row, col, white_bottom)
        out_path = args.out / "squares" / f"{ts}_{sq_name}_br.png"
        cv2.imwrite(str(out_path), crop)
        print(f"Saved right-of-bottom-left square -> {out_path.name}")

    def capture_our_pockets():
        ts = int(time.time() * 1000)
        ours = grab_rect(sct, calib.our_pockets)
        out_path = args.out / "pockets_ours" / f"{ts}.png"
        cv2.imwrite(str(out_path), ours)
        print(f"Saved pockets -> {out_path.name}")

    # Simple polling with debounce
    while True:
        if keyboard.is_pressed("esc"):
            break

        if keyboard.is_pressed("f6"):
            capture_bottom_left()
            time.sleep(0.25)

        if keyboard.is_pressed("f7"):
            capture_right_of_bl()
            time.sleep(0.25)

        if keyboard.is_pressed("f8"):
            capture_our_pockets()
            time.sleep(0.25)

        time.sleep(0.01)


if __name__ == "__main__":
    main()

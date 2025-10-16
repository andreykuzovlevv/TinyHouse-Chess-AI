# vision.py
import os
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import cv2
import mss
import numpy as np
from cairosvg import svg2png

# ---------- Data models ----------


@dataclass
class BoardState:
    # "a1" -> "w_p" / "b_k" / "empty" etc. Using your asset codes.
    pieces: Dict[str, str]
    pockets: Dict[str, Dict[str, int]]  # fill later
    side_to_move: Optional[str]  # fill from UI later; keep None for now


# ---------- Utility ----------

FILES_BY_CODE = {
    "w_p": "w_p.svg",
    "w_h": "w_h.svg",
    "w_f": "w_f.svg",
    "w_w": "w_w.svg",
    "w_k": "w_k.svg",
    "b_p": "b_p.svg",
    "b_h": "b_h.svg",
    "b_f": "b_f.svg",
    "b_w": "b_w.svg",
    "b_k": "b_k.svg",
}

ALG_FILES = [f"{f}{r}" for r in range(1, 4) for f in "abcd"]  # a1..h8


def square_name(file_idx: int, rank_idx: int) -> str:
    return "abcd"[file_idx] + str(rank_idx + 1)


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def norm_uint8(img: np.ndarray) -> np.ndarray:
    out = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)


# ---------- ScreenReader ----------


class ScreenReader:
    """
    - locate_board(): one-time board rectangle and tile size by maximizing
      correlation with an 8x8 checkerboard model.
    - read_boardstate(): classify each square via template match to your SVGs.
    """

    def __init__(
        self,
        assets_dir: str = "assets",
        downscale_for_search: int = 1400,
    ):
        self.assets_dir = assets_dir
        self._mss = mss.mss()
        self._board_rect: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h
        self._tile: Optional[int] = None
        self._square_rois: Optional[Dict[str, Tuple[int, int, int, int]]] = None
        self._tmpl: Dict[str, np.ndarray] = {}
        self._downscale_for_search = downscale_for_search
        self._orientation_white_at_bottom: Optional[bool] = None

    # ----- Public API -----

    def read_boardstate(self) -> BoardState:
        if self._board_rect is None:
            self.locate_board()

        board_bgr = self._grab_region(self._board_rect)
        if self._tile is None:
            self._tile = round(self._board_rect[2] / 8)
            self._build_square_rois()

        # Lazy-load templates at the correct tile size
        if not self._tmpl:
            self._load_templates(self._tile)

        pieces = self._classify_all_squares(board_bgr)

        # Orientation inference (once per session)
        if self._orientation_white_at_bottom is None:
            self._orientation_white_at_bottom = self._infer_orientation(pieces)

        # Map (file,rank) to algebraic names with proper orientation
        mapped = self._map_grid_to_algebraic(pieces)

        # side_to_move is unknown from pure board pixels; keep None for now
        # (If you have a UI indicator, add a detector and set it here.)
        return BoardState(pieces=mapped, pockets={"w": {}, "b": {}}, side_to_move=None)

    # ----- Board finding -----

    def locate_board(self):
        """Find the 8x8 area with strongest checkerboard correlation."""
        full = self._grab_fullscreen()
        H, W = full.shape[:2]

        # Downscale for speed
        scale = 1.0
        if max(H, W) > self._downscale_for_search:
            scale = self._downscale_for_search / max(H, W)
            small = cv2.resize(
                full, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            small = full.copy()

        gray = to_gray(small)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        best_score = -1e9
        best_rect_small = None

        # Candidate square sizes (tiles) in the downscaled image
        # We assume tile between 36 and 140 px (tune for your UI).
        for tile in range(36, 141, 4):
            board_w = tile * 8
            board_h = tile * 8
            if board_w > gray.shape[1] or board_h > gray.shape[0]:
                continue

            # Sliding window stride ~ one tile
            stride = max(tile // 2, 12)
            for y in range(0, gray.shape[0] - board_h + 1, stride):
                row = gray[y : y + board_h, :]
                for x in range(0, gray.shape[1] - board_w + 1, stride):
                    crop = row[:, x : x + board_w]

                    # Compute mean per cell and correlate with +/- checkerboard
                    means = self._cell_means(crop, tile)
                    score = self._checkerboard_score(means)
                    if score > best_score:
                        best_score = score
                        best_rect_small = (x, y, board_w, board_h)

        if best_rect_small is None:
            raise RuntimeError("Board not found")

        # Map back to full resolution
        x, y, w, h = best_rect_small
        X = int(round(x / scale))
        Y = int(round(y / scale))
        Wb = int(round(w / scale))
        Hb = int(round(h / scale))

        # Snap to nearest 8×8 grid to remove drift
        tile_full = round(Wb / 8)
        Wb = Hb = tile_full * 8
        self._board_rect = (X, Y, Wb, Hb)
        self._tile = tile_full
        self._build_square_rois()

    # ----- Templates & classification -----

    def _load_templates(self, tile: int):
        """Rasterize SVGs to tile size and keep grayscale versions."""
        for code, filename in FILES_BY_CODE.items():
            path = os.path.join(self.assets_dir, filename)
            if not os.path.exists(path):
                continue
            png_bytes = svg2png(url=path, output_width=tile, output_height=tile)
            nparr = np.frombuffer(png_bytes, np.uint8)
            png = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)  # may include alpha
            if png.shape[2] == 4:
                # Composite on mid-gray background to reduce square-color bias
                alpha = png[:, :, 3:4] / 255.0
                rgb = png[:, :, :3].astype(np.float32)
                bg = np.full_like(rgb, 200, dtype=np.float32)
                comp = (alpha * rgb + (1 - alpha) * bg).astype(np.uint8)
                gray = to_gray(comp)
            else:
                gray = to_gray(png)
            self._tmpl[code] = norm_uint8(gray)

    def _classify_all_squares(self, board_bgr: np.ndarray) -> List[List[str]]:
        t = self._tile
        pieces_grid = []
        board_gray = to_gray(board_bgr)
        # Local contrast normalization improves stability
        board_gray = cv2.equalizeHist(board_gray)

        for r in range(8):
            row = []
            for f in range(8):
                y0, x0 = r * t, f * t
                roi = board_gray[y0 : y0 + t, x0 : x0 + t]
                lab = self._classify_square(roi)
                row.append(lab)
            pieces_grid.append(row)
        return pieces_grid

    def _classify_square(self, roi_gray: np.ndarray) -> str:
        best, label = 0.0, "empty"
        # quick empty check: low edge energy likely empty
        edges = cv2.Canny(roi_gray, 50, 150)
        if edges.mean() < 10:  # tuned threshold
            return "empty"

        for code, templ in self._tmpl.items():
            # TM_CCOEFF_NORMED is robust to brightness offsets
            res = cv2.matchTemplate(roi_gray, templ, cv2.TM_CCOEFF_NORMED)
            s = float(res.max())
            if s > best:
                best, label = s, code

        return label if best > 0.63 else "empty"  # tune threshold per sprite set

    # ----- Orientation and mapping -----

    def _infer_orientation(self, grid: List[List[str]]) -> bool:
        """Return True if White is at bottom (ranks 1..2)."""
        ys_white = []
        ys_black = []
        for r in range(8):
            for f in range(8):
                lab = grid[r][f]
                if lab.startswith("w_"):
                    ys_white.append(r)
                elif lab.startswith("b_"):
                    ys_black.append(r)
        if not ys_white or not ys_black:
            # Fall back to assume white bottom
            return True
        # Lower r means closer to top; bottom means larger r
        return np.mean(ys_white) > np.mean(ys_black)

    def _map_grid_to_algebraic(self, grid: List[List[str]]) -> Dict[str, str]:
        mapped = {}
        if self._orientation_white_at_bottom:
            # r=0 is rank 8, r=7 is rank 1
            for r in range(8):
                for f in range(8):
                    sq = square_name(f, 7 - r)
                    mapped[sq] = grid[r][f]
        else:
            # Black at bottom: r=0 is rank1
            for r in range(8):
                for f in range(8):
                    sq = square_name(7 - f, r + 0)
                    mapped[sq] = grid[r][f]
        return mapped

    # ----- Helpers -----

    def _build_square_rois(self):
        x, y, w, h = self._board_rect
        t = round(w / 8)
        self._tile = t
        self._square_rois = {}
        # Store screen rectangles for clicker
        for r in range(8):
            for f in range(8):
                sx = x + f * t
                sy = y + r * t
                self._square_rois[(f, r)] = (sx, sy, t, t)

    def board_to_pixels(self, algebraic: str) -> Tuple[int, int]:
        """Center pixel of a given algebraic square."""
        # Build if necessary
        if self._square_rois is None:
            self._build_square_rois()
        # Convert alg to (f,r) in screen grid coordinates (top-left origin)
        f = "abcdefgh".index(algebraic[0])
        r = int(algebraic[1]) - 1
        # Reverse mapping to screen grid depends on orientation
        if self._orientation_white_at_bottom:
            sr = 7 - r
            sf = f
        else:
            sr = r
            sf = 7 - f
        x, y, w, h = self._square_rois[(sf, sr)]
        return x + w // 2, y + h // 2

    def _grab_fullscreen(self) -> np.ndarray:
        mon = self._mss.monitors[0]  # full virtual screen
        sct = self._mss.grab(mon)
        img = np.array(sct)[:, :, :3]  # BGRA → BGR
        return img

    def _grab_region(self, rect_xywh) -> np.ndarray:
        x, y, w, h = rect_xywh
        sct = self._mss.grab({"left": x, "top": y, "width": w, "height": h})
        return np.array(sct)[:, :, :3]

    @staticmethod
    def _cell_means(crop: np.ndarray, tile: int) -> np.ndarray:
        h, w = crop.shape[:2]
        means = np.zeros((8, 8), dtype=np.float32)
        for r in range(8):
            for f in range(8):
                y0 = r * tile
                x0 = f * tile
                roi = crop[y0 : y0 + tile, x0 : x0 + tile]
                means[r, f] = float(roi.mean())
        return means

    @staticmethod
    def _checkerboard_score(means: np.ndarray) -> float:
        # +1/-1 ideal checkerboard
        patt = np.fromfunction(
            lambda r, f: ((r + f) % 2) * 2 - 1, (8, 8), dtype=int
        ).astype(np.float32)
        means_z = (means - means.mean()) / (means.std() + 1e-6)
        return float((means_z * patt).sum())

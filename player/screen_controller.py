from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np
import pyautogui
from PIL import Image
import torch
from torchvision import transforms

from tinyhouse import (
    Color,
    Piece,
    PieceType,
    Square,
    File,
    Rank,
    pt_code,
    color_of,
    type_of,
    make_square,
    file_of,
    rank_of,
    square_to_str,
    str_to_square,
    code_from_piece,
    FILES_BY_PIECE,
)

# ---- Constants / thresholds tuned for “yellow highlight”, no animations ----
# rgb(186, 202, 73)
# rgb(247, 245, 125)


# Occupancy heuristic (gradient energy per pixel)
GRAD_MIN = 5.0  # tune if necessary

# Template matching
TM_THRESH = 0.70  # accept best match if above this

# Calibration file default
CALIB_PATH = Path("tinyhouse_calibration.json")


@dataclass
class Rect:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    def inset(self, px: int) -> "Rect":
        return Rect(
            self.x + px, self.y + px, max(1, self.w - 2 * px), max(1, self.h - 2 * px)
        )


@dataclass
class _Tmpl:
    img: np.ndarray  # uint8, grayscale
    mask: np.ndarray  # uint8, 0 or 255
    scale: float


@dataclass
class Calibration:
    board: Rect
    our_pockets: Rect  # left panel area for our reserve


class ScreenController:
    def __init__(
        self,
        calib_path: Path = CALIB_PATH,
        cls_model_path: str = "models/square_cls.pt",
    ):
        pyautogui.FAILSAFE = False
        self._sct = mss.mss()
        self._calib_path = Path(calib_path)
        self.calib: Calibration = self._load_or_prompt_calibration(self._calib_path)

        # Precompute cell geometry
        self._cell_w = self.calib.board.w / 4.0
        self._cell_h = self.calib.board.h / 4.0

        # ---- Square-classifier model (PyTorch) ----
        self._cls_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cls_model, self._cls_class_names, self._cls_tf = (
            self._load_square_classifier(cls_model_path, self._cls_device)
        )

        # Establish our side from screen (bottom-left piece at game start)
        self._my_side: Color = self._detect_my_side()

    # --------------------------- Public API ---------------------------

    @property
    def my_side(self) -> Color:
        return self._my_side

    def detect_move(self) -> Optional[str]:
        """
        Poll the board for a new opponent move based on highlight(s), then
        return it in engine string format:
            - normal/capture: "a2a3"
            - drop:           "P@b2"
            - promotion:      "a3a4=H" (H|W|F)
        Updates the internal mirror position when a move is reported.
        Returns None if no move is currently detected.
        """
        img = self._grab_rect(self.calib.board)

        highlighted = self._detect_highlight_cells(img)
        print(highlighted)
        if not highlighted:
            return None

        # Map image-cell indices to board Squares (White perspective)
        hl_squares = [self._cell_index_to_square(i) for i in highlighted]

        if len(hl_squares) == 1:
            # Drop by opponent onto hl_squares[0]
            to_sq = hl_squares[0]
            # Identify piece type on the destination cell (now occupied by the dropped piece)
            pt, _col = self._classify_cell_piece(img, highlighted[0])

            if pt == PieceType.NO_PIECE_TYPE:
                return None

            mv = f"{pt_code(pt)}@{square_to_str(to_sq)}"
            self._apply_drop(to_sq, pt)
            self._side_to_move = self._side_to_move.other()
            return mv

        if len(hl_squares) == 2:
            a, b = hl_squares
            # Decide which is origin vs destination.
            # Heuristic: origin cell is empty AFTER the move; destination is occupied.
            occ_a = self._cell_occupied(img, self._square_to_cell_index(a))
            occ_b = self._cell_occupied(img, self._square_to_cell_index(b))

            if occ_a and not occ_b:
                # Rare, but handle inverted shade quirks by re-grab once
                time.sleep(0.02)
                img = self._grab_rect(self.calib.board)
                occ_a = self._cell_occupied(img, self._square_to_cell_index(a))
                occ_b = self._cell_occupied(img, self._square_to_cell_index(b))

            if occ_a and not occ_b:
                # Still inverted; fall back to board mirror: origin must contain opponent piece
                if self._is_side_piece_at(self._side_to_move.other(), a):
                    from_sq, to_sq = a, b
                elif self._is_side_piece_at(self._side_to_move.other(), b):
                    from_sq, to_sq = b, a
                else:
                    # As last resort, pick by gradient magnitude (higher => occupied => destination)
                    from_sq, to_sq = (b, a) if occ_a else (a, b)
            else:
                from_sq, to_sq = (a, b) if not occ_a and occ_b else (b, a)

            # Determine if this was a promotion by opponent:
            moving_piece = self._board.get(from_sq)
            was_pawn = (
                moving_piece is not None and self._ptype(moving_piece) == PieceType.PAWN
            )
            dest_last_rank = rank_of(to_sq) == (
                Rank.RANK_4
                if self._side_to_move.other() == Color.WHITE
                else Rank.RANK_1
            )
            promo_suffix = ""
            if was_pawn and dest_last_rank:
                # Identify which piece now sits on to_sq (must be one of H/W/F)
                pt_to, _col_to = self._classify_cell_piece(
                    img, self._square_to_cell_index(to_sq)
                )
                if pt_to in (PieceType.HORSE, PieceType.WAZIR, PieceType.FERZ):
                    promo_suffix = f"={self._pt_code(pt_to)}"

            mv = f"{square_to_str(from_sq)}{square_to_str(to_sq)}{promo_suffix}"
            self._apply_move(from_sq, to_sq, promo_suffix)
            self._side_to_move = self._side_to_move.other()
            return mv

        # 3+ highlighted cells should not occur with animations off; ignore frame.
        return None

    def execute_ui_move(self, move_str: str) -> None:
        """
        Perform our move on the UI.
        Supported:
          - "a2a3"
          - "a3a4=H" (promotion)
          - "P@b2"   (drop)
        """
        if "@" in move_str:
            # Drop: "<P|H|F|W>@e2"
            code, sq = move_str.split("@", 1)
            pt = self._code_to_pt(code[0])
            to_sq = str_to_square(sq)
            self._do_drop_on_ui(pt, to_sq)
            self._apply_drop(to_sq, pt, mover=self._side_to_move)  # our drop
            self._side_to_move = self._side_to_move.other()
            return

        # Normal/promotion
        if "=" in move_str:
            main, promo_code = move_str.split("=")
            promo_pt = self._code_to_pt(promo_code[0])
        else:
            main, promo_pt = move_str, None

        from_str, to_str = main[:2], main[2:4]
        from_sq = str_to_square(from_str)
        to_sq = str_to_square(to_str)

        self._drag_board_cell(from_sq, to_sq)

        if promo_pt is not None:
            # Click the right piece in the promotion popup
            self._choose_promotion_piece(promo_pt)

        self._apply_move(
            from_sq,
            to_sq,
            f"={self._pt_code(promo_pt)}" if promo_pt else "",
            mover=self._side_to_move,
        )
        self._side_to_move = self._side_to_move.other()

    # ------------------------ Calibration (one-time) ------------------------

    def calibrate_interactive(self) -> None:
        """
        CLI-guided calibration. Hover mouse and press Enter when prompted.
        Saves JSON to self._calib_path. Call once before playing if needed.
        """
        print("\n=== Tinyhouse calibration ===")
        print(
            "Put a game on your screen. Ensure the full 4x4 board is visible. Animations off."
        )
        print("You will hover specific corners and press ENTER each time.")

        def get_point(prompt: str) -> Tuple[int, int]:
            input(prompt)
            x, y = pyautogui.position()
            print(f"  captured: ({x}, {y})")
            return x, y

        # Board rect
        print("\nBoard rectangle:")
        bx1, by1 = get_point("Hover TOP-LEFT corner of the board, press ENTER...")
        bx2, by2 = get_point("Hover BOTTOM-RIGHT corner of the board, press ENTER...")

        # Our pockets (left of board, our color)
        print("\nOur pockets rectangle (area where OUR reserve icons appear):")
        ox1, oy1 = get_point("Hover TOP-LEFT of our pockets area, press ENTER...")
        ox2, oy2 = get_point("Hover BOTTOM-RIGHT of our pockets area, press ENTER...")

        self.calib = Calibration(
            board=Rect(min(bx1, bx2), min(by1, by2), abs(bx2 - bx1), abs(by2 - by1)),
            our_pockets=Rect(
                min(ox1, ox2), min(oy1, oy2), abs(ox2 - ox1), abs(oy2 - oy1)
            ),
        )
        self._save_calibration(self._calib_path, self.calib)

        print("Calibration saved.")

    # ------------------------ Internals: capture & mapping ------------------------

    def _grab_rect(self, r: Rect) -> np.ndarray:
        shot = self._sct.grab({"left": r.x, "top": r.y, "width": r.w, "height": r.h})
        img = np.array(shot)  # BGRA
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _square_center_xy(self, s: Square) -> Tuple[int, int]:
        """
        Return screen coordinates (int x,y) for the center of square s (a1..d4),
        taking into account board flip based on self._my_side.
        """
        f = int(file_of(s))  # 0..3
        r = int(rank_of(s))  # 0..3

        bx, by, bw, bh = (
            self.calib.board.x,
            self.calib.board.y,
            self.calib.board.w,
            self.calib.board.h,
        )

        if self._my_side == Color.WHITE:
            cx = bx + int((f + 0.5) * self._cell_w)
            cy = by + int(bh - (r + 0.5) * self._cell_h)
        else:
            # Board is flipped for us: 'a1' is top-right on screen
            cx = bx + int(bw - (f + 0.5) * self._cell_w)
            cy = by + int((r + 0.5) * self._cell_h)
        return cx, cy

    def _square_to_cell_index(self, s: Square) -> int:
        """0..15 cell index laid out row-major from top-left of the board image array."""
        # independent of self._my_side: this is about the image array indexing (top-left origin)
        # Row-major: row 0 = top rank on screen
        # Compute the on-screen row/col for a given chess square under our orientation.
        if self._my_side == Color.WHITE:
            row = 3 - int(rank_of(s))
            col = int(file_of(s))
        else:
            row = int(rank_of(s))
            col = 3 - int(file_of(s))
        return row * 4 + col

    def _cell_index_to_square(self, idx: int) -> Square:
        """Inverse of _square_to_cell_index for the current orientation."""
        row, col = divmod(idx, 4)
        if self._my_side == Color.WHITE:
            r = 3 - row
            f = col
        else:
            r = row
            f = 3 - col
        return make_square(File(f), Rank(r))

    # ------------------------ Internals: detection ------------------------
    def _load_square_classifier(self, ckpt_path: str, device: torch.device):
        """
        Load square classifier exported by train_square_classifier.py.
        Returns: (model.eval(), class_names: List[str], transform)
        """
        ckpt = torch.load(ckpt_path, map_location=device)
        class_names = ckpt["class_names"]
        img_size = ckpt.get("img_size", 128)

        # Build the same backbone head shape as in training
        from torchvision import models

        model = models.resnet18(weights=None)
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(p=0.20),
            torch.nn.Linear(model.fc.in_features, len(class_names)),
        )
        model.load_state_dict(ckpt["model_state"])
        model.to(device).eval()

        tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        return model, class_names, tf

    def _detect_highlight_cells(self, board_img_bgr: np.ndarray) -> List[int]:
        """
        Return ONLY opponent-highlighted cell indices (0..15).
        Steps:
        1) Corner-sample each cell to detect highlight (two known highlight colors).
        2) Keep only highlighted cells that currently contain an OPPONENT piece
            according to _classify_cell_piece(...).
        """

        # --- Convert target highlight colors to LAB once ---
        def rgb_to_lab(rgb):
            bgr = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0, :]

        lab1 = rgb_to_lab((186, 202, 73))
        lab2 = rgb_to_lab((247, 245, 125))

        def delta_e(labA, labB) -> float:
            d = labA - labB
            return float(np.sqrt(np.dot(d, d)))

        lab_board = cv2.cvtColor(board_img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        h, w = lab_board.shape[:2]
        cw = int(self._cell_w)
        ch = int(self._cell_h)

        inset = max(6, int(min(self._cell_w, self._cell_h) * 0.08))
        patch = max(6, int(min(self._cell_w, self._cell_h) * 0.14))
        DELTA_E_THR = 18.0
        MIN_CORNERS_MATCH = 2

        # 1) raw highlight detection (corner sampling)
        raw_hl: List[int] = []
        for row in range(4):
            for col in range(4):
                x1 = int(col * self._cell_w)
                y1 = int(row * self._cell_h)
                x2 = min(w, x1 + cw)
                y2 = min(h, y1 + ch)

                # four corner patches
                tl = (x1 + inset, y1 + inset)
                tr = (max(x1, x2 - inset - patch), y1 + inset)
                bl = (x1 + inset, max(y1, y2 - inset - patch))
                br = (max(x1, x2 - inset - patch), max(y1, y2 - inset - patch))
                corners = [tl, tr, bl, br]

                matches = 0
                for cx, cy in corners:
                    cx2 = min(w, cx + patch)
                    cy2 = min(h, cy + patch)
                    if cx >= cx2 or cy >= cy2:
                        continue
                    mean_lab = lab_board[cy:cy2, cx:cx2].reshape(-1, 3).mean(axis=0)
                    if (
                        min(delta_e(mean_lab, lab1), delta_e(mean_lab, lab2))
                        <= DELTA_E_THR
                    ):
                        matches += 1

                if matches >= MIN_CORNERS_MATCH:
                    raw_hl.append(row * 4 + col)

        if not raw_hl:
            return []

        # 2) keep ONLY opponent-highlighted cells
        opponent = self._my_side.other()
        opp_in_hl = False
        for idx in raw_hl:
            pt, col = self._classify_cell_piece(board_img_bgr, idx)
            if col is not None and col == self.my_side:
                return []
            elif col is not None and col == self.my_side.other():
                opp_in_hl = True

        if opp_in_hl:
            return raw_hl

        # If none of the highlighted cells show an opponent piece, it’s our own highlight -> ignore
        return []

    def _cell_occupied(self, board_img_bgr: np.ndarray, cell_idx: int) -> bool:
        """
        Edge/gradient heuristic: occupied cells exhibit stronger gradients.
        """
        row, col = divmod(cell_idx, 4)
        x1 = int(col * self._cell_w)
        y1 = int(row * self._cell_h)
        x2 = int(min(board_img_bgr.shape[1], x1 + self._cell_w))
        y2 = int(min(board_img_bgr.shape[0], y1 + self._cell_h))

        cell = board_img_bgr[y1:y2, x1:x2]
        if cell.size == 0:
            return False
        g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
        mag = cv2.add(np.abs(gx), np.abs(gy)).astype(np.float32)
        score = float(mag.mean())
        return score >= GRAD_MIN

    def _classify_cell_piece(self, board_img_bgr: np.ndarray, cell_idx: int):
        # ---- crop cell (screen row/col from cell_idx) ----
        row, col = divmod(cell_idx, 4)
        x1 = int(col * self._cell_w)
        y1 = int(row * self._cell_h)
        x2 = int(min(board_img_bgr.shape[1], x1 + self._cell_w))
        y2 = int(min(board_img_bgr.shape[0], y1 + self._cell_h))
        cell = board_img_bgr[y1:y2, x1:x2]
        if cell.size == 0:
            return PieceType.NO_PIECE_TYPE, None

        # ---- model inference (RGB -> PIL -> transform) ----
        from PIL import Image

        cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(cell_rgb)

        x = self._cls_tf(im).unsqueeze(0).to(self._cls_device)
        with torch.no_grad():
            logits = self._cls_model(x)
            probs = torch.softmax(logits, dim=1)[0]

        # ---- confidence + margin gating ----
        conf_min = 0.72
        margin_min = 0.10
        top2 = torch.topk(probs, k=min(2, probs.numel()))
        top1_conf = float(top2.values[0])
        top1_idx = int(top2.indices[0])
        top2_conf = float(top2.values[1]) if top2.indices.numel() > 1 else 0.0
        if top1_conf < conf_min or (top1_conf - top2_conf) < margin_min:
            return PieceType.NO_PIECE_TYPE, None

        label = self._cls_class_names[top1_idx]
        color, pt = self._parse_cls_name(label)
        if pt == PieceType.NO_PIECE_TYPE:
            return PieceType.NO_PIECE_TYPE, None
        return pt, color

    def _parse_cls_name(self, name: str) -> Tuple[Optional[Color], PieceType]:
        """
        Map folder/class name like 'W_PAWN', 'B_HORSE', ... -> (Color, PieceType).
        """
        s = name.strip().upper()
        color = (
            Color.WHITE
            if s.startswith("W_")
            else (Color.BLACK if s.startswith("B_") else None)
        )
        t = s.split("_", 1)[-1] if "_" in s else s
        if t == "PAWN":
            pt = PieceType.PAWN
        elif t == "HORSE":
            pt = PieceType.HORSE
        elif t == "FERZ":
            pt = PieceType.FERZ
        elif t == "WAZIR":
            pt = PieceType.WAZIR
        elif t == "KING":
            pt = PieceType.KING
        else:
            pt = PieceType.NO_PIECE_TYPE
        return color, pt

    # ------------------------ Internals: my side / templates ------------------------

    def _detect_my_side(self) -> Color:
        """
        Inspect bottom-left board corner on screen. If that piece is white, we are WHITE; else BLACK.
        This is robust at the initial position.
        """
        # bottom-left in screen coords = Square a1 for us visually
        # We don’t know side yet, so infer by sampling both K and k templates broadly.
        # Practical approach: classify the bottom-left cell by color.
        # Take cell index of bottom-left in screen coordinates:
        # row=3, col=0
        board_img = self._grab_rect(self.calib.board)
        cell_idx = 3 * 4 + 0
        pt, col = self._classify_cell_piece(board_img, cell_idx)

        if col is None:
            raise RuntimeError("Unable to recognize piece color on bottom-left square.")

        return Color.WHITE if col == Color.WHITE else Color.BLACK

    # ------------------------ Internals: UI actions ------------------------

    def _drag_board_cell(self, from_sq: Square, to_sq: Square) -> None:
        fx, fy = self._square_center_xy(from_sq)
        tx, ty = self._square_center_xy(to_sq)
        pyautogui.moveTo(fx, fy, duration=0)
        pyautogui.mouseDown()
        pyautogui.moveTo(tx, ty, duration=0)
        pyautogui.mouseUp()

    def _choose_promotion_piece(self, pt: PieceType) -> None:
        """
        Find the template of the promotion piece inside promo_area and click it.
        """
        r = self.calib.promo_area
        img = self._grab_rect(r)
        # Choose correct color = our color (we are promoting on our turn)
        piece = Piece(int(pt) if self._my_side == Color.WHITE else int(pt) + 8)
        tmpl = self._templates.get(piece)
        if tmpl is None:
            # If we lack color-specific asset, try color-agnostic search over both
            cand = [Piece(int(pt)), Piece(int(pt) + 8)]
            for p in cand:
                tmpl = self._templates.get(p)
                if tmpl is not None:
                    piece = p
                    break
        if tmpl is None:
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, tw = tmpl.shape[:2]
        if gray.shape[0] < th or gray.shape[1] < tw:
            return
        res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(res)
        if score < TM_THRESH:
            return
        px = r.x + max_loc[0] + tw // 2
        py = r.y + max_loc[1] + th // 2
        pyautogui.moveTo(px, py, duration=0)
        pyautogui.click()

    def _do_drop_on_ui(self, pt: PieceType, to_sq: Square) -> None:
        """
        Drag the requested piece from our pockets to the board.
        """
        # Find the piece icon in our pockets region
        r = self.calib.our_pockets
        img = self._grab_rect(r)

        # Our color, our piece
        piece = Piece(int(pt) if self._my_side == Color.WHITE else int(pt) + 8)
        tmpl = self._templates.get(piece)

        # Fallback: if exact color template missing, accept either color (UI may tint icons differently)
        search_pieces = [piece]
        if tmpl is None:
            search_pieces = [Piece(int(pt)), Piece(int(pt) + 8)]

        found = None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for p in search_pieces:
            t = self._templates.get(p)
            if t is None:
                continue
            th, tw = t.shape[:2]
            if gray.shape[0] < th or gray.shape[1] < tw:
                continue
            res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score >= TM_THRESH:
                _, _, _, max_loc = cv2.minMaxLoc(res)
                found = (max_loc[0] + tw // 2, max_loc[1] + th // 2)
                break

        if found is None:
            # As a degrade, click center of pockets to pick first piece, then drag.
            sx = r.x + r.w // 3
            sy = r.y + r.h // 3
        else:
            sx = r.x + found[0]
            sy = r.y + found[1]

        tx, ty = self._square_center_xy(to_sq)

        pyautogui.moveTo(sx, sy, duration=0)
        pyautogui.mouseDown()
        pyautogui.moveTo(tx, ty, duration=0)
        pyautogui.mouseUp()

    # ------------------------ Persistence ------------------------

    def _load_or_prompt_calibration(self, path: Path) -> Calibration:
        if not path.exists():
            self.calibrate_interactive()
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Calibration(
                board=Rect(*data["board"]),
                our_pockets=Rect(*data["our_pockets"]),
            )

    def _save_calibration(self, path: Path, calib: Calibration) -> None:
        data = {
            "board": [calib.board.x, calib.board.y, calib.board.w, calib.board.h],
            "our_pockets": [
                calib.our_pockets.x,
                calib.our_pockets.y,
                calib.our_pockets.w,
                calib.our_pockets.h,
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

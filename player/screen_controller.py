from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import mss
import numpy as np
import pyautogui
import torch
from torchvision import transforms
from PIL import Image

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
    code_to_pt,
    FILES_BY_PIECE,
)

# ---- Constants / thresholds ----

# Occupancy heuristic (gradient energy per pixel)
GRAD_MIN = 5.0  # tune if necessary

# Template matching
TM_THRESH = 0.70  # accept best match if above this

# Calibration file default
CALIB_PATH = Path("tinyhouse_calibration.json")

_DET_CLASS_BY_PT = {
    PieceType.PAWN: 0,  # "P"
    PieceType.HORSE: 1,  # "H"
    PieceType.FERZ: 2,  # "F"
    PieceType.WAZIR: 3,  # "W"
    PieceType.KING: 4,  # "K"
}


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
class Calibration:
    board: Rect
    our_pockets: Rect  # left panel area for our reserve


class ScreenController:
    def __init__(
        self,
        calib_path: Path = CALIB_PATH,
        cls_model_path: str = "models/square_cls.pt",
        pockets_model_path: str = "models/pockets.pt",
    ):
        pyautogui.FAILSAFE = False
        self._sct = mss.mss()
        self._calib_path = Path(calib_path)
        self.calib: Calibration = self._load_or_prompt_calibration(self._calib_path)

        # Precompute cell geometry
        self._cell_w = self.calib.board.w / 4.0
        self._cell_h = self.calib.board.h / 4.0

        # ---- Square-classifier model (PyTorch) ----
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._cls_model, self._cls_class_names, self._cls_tf = (
            self._load_square_classifier(cls_model_path, self._device)
        )
        self._pckts_model, self._pckts_class_names = self._load_pockets_detector(
            pockets_model_path, self._device
        )

        # Establish our side from screen (bottom-left piece at game start)
        self._my_side: Color = self._detect_my_side()

        self._opp_pawns: List[Square] = [
            Square.SQ_D3 if self._my_side == Color.WHITE else Square.SQ_A2
        ]

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

            if to_sq not in self._opp_pawns and pt == PieceType.PAWN:
                self._opp_pawns.append(to_sq)

            return mv

        if len(hl_squares) == 2:
            a, b = hl_squares
            # Decide which is origin vs destination.
            # Heuristic: origin cell is empty AFTER the move; destination is occupied.
            ptA, cA = self._classify_cell_piece(img, self._square_to_cell_index(a))
            ptB, cB = self._classify_cell_piece(img, self._square_to_cell_index(b))

            if ptA == PieceType.NO_PIECE_TYPE and ptB == PieceType.NO_PIECE_TYPE:
                return None

            if ptA != PieceType.NO_PIECE_TYPE and ptB != PieceType.NO_PIECE_TYPE:
                return None

            from_sq, to_sq = (a, b) if not int(ptA) and int(ptB) else (b, a)
            print("from: ", from_sq.to_string(), "\nto: ", to_sq.to_string())

            # Determine if this was a promotion by opponent:
            was_pawn = True if from_sq in self._opp_pawns else False

            dest_last_rank = rank_of(to_sq) == (
                Rank.RANK_4 if self._my_side.other() == Color.WHITE else Rank.RANK_1
            )
            promo_suffix = ""
            if was_pawn and dest_last_rank:
                # Identify which piece now sits on to_sq (must be one of H/W/F)
                pt_to, _col_to = self._classify_cell_piece(
                    img, self._square_to_cell_index(to_sq)
                )
                if pt_to in (PieceType.HORSE, PieceType.WAZIR, PieceType.FERZ):
                    promo_suffix = f"={pt_code(pt_to)}"

            print("was pawn: ", was_pawn, ". last rank: ", dest_last_rank)

            mv = f"{square_to_str(from_sq)}{square_to_str(to_sq)}{promo_suffix}"

            if was_pawn:
                self._opp_pawns.remove(from_sq)
                if not dest_last_rank:
                    self._opp_pawns.append(to_sq)

            return mv

        if len(hl_squares) > 2:
            raise Exception("Number of Highlighted squares was more then 2")

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
            pt = code_to_pt(code[0])
            to_sq = str_to_square(sq)
            self._do_drop_on_ui(pt, to_sq)
            return

        # Normal/promotion
        if "=" in move_str:
            main, promo_code = move_str.split("=")
            promo_pt = code_to_pt(promo_code[0])
        else:
            main, promo_pt = move_str, None

        from_str, to_str = main[:2], main[2:4]
        from_sq = str_to_square(from_str)
        to_sq = str_to_square(to_str)

        self._drag_board_cell(from_sq, to_sq)

        if promo_pt is not None:
            # Click the right piece in the promotion popup
            self._choose_promotion_piece(promo_pt, to_sq)

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

    # --- Mapping: TinyHouse PieceType -> detector class index (no color) ---

    def _load_pockets_detector(self, model_path: str, device: "torch.device"):
        """
        Load Ultralytics YOLO detector for pockets. Returns (model, class_names).
        Expects 5 classes: P,H,F,W,K in that exact order.
        """
        from ultralytics import YOLO

        model = YOLO(model_path)
        # move to device (ultralytics manages device internally but we can hint via model.to)
        try:
            model.to(str(device))
        except Exception:
            pass
        # names mapping (index -> string)
        names = []
        if hasattr(model, "names") and model.names:
            # dict or list depending on version
            if isinstance(model.names, dict):
                # ensure 0..N order
                names = [model.names[i] for i in sorted(model.names.keys())]
            else:
                names = list(model.names)
        else:
            names = ["P", "H", "F", "W", "K"]  # fallback
        return model, names

    def _detect_pocket_piece_center(
        self, want_pt: PieceType, conf_thres: float = 0.40
    ) -> tuple[int, int] | None:
        """
        Detect the requested piece in the OUR pockets ROI and return (sx, sy) in screen coordinates.
        Returns None if not found. One retry at a lower conf if empty.
        """
        # Grab ROI (BGR -> RGB)
        r = self.calib.our_pockets  # expects (x,y,w,h) or object with .x/.y
        roi_bgr = self._grab_rect(r)
        if roi_bgr is None or roi_bgr.size == 0:
            return None
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

        # YOLO inference on ndarray
        results = self._pckts_model.predict(
            source=roi_rgb,
            conf=conf_thres,
            verbose=False,
        )
        if not results:
            return None

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            if conf_thres > 0.10:
                return self._detect_pocket_piece_center(want_pt, conf_thres=0.10)
            return None

        det_idx = _DET_CLASS_BY_PT.get(want_pt, None)
        if det_idx is None:
            return None

        boxes = res.boxes
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        match_idxs = np.where(cls == det_idx)[0]
        if match_idxs.size == 0:
            if conf_thres > 0.10:
                return self._detect_pocket_piece_center(want_pt, conf_thres=0.10)
            return None

        best_local = match_idxs[np.argmax(conf[match_idxs])]
        x1, y1, x2, y2 = xyxy[best_local]
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        # Map ROI -> screen
        x0 = getattr(r, "x", None)
        y0 = getattr(r, "y", None)
        if x0 is None or y0 is None:  # assume tuple (x,y,w,h)
            x0, y0 = r[0], r[1]

        sx = int(round(x0 + cx))
        sy = int(round(y0 + cy))
        return sx, sy

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

        cell_rgb = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(cell_rgb)

        x = self._cls_tf(im).unsqueeze(0).to(self._device)
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

    def _square_to_screen_col(self, s: Square) -> int:
        """
        Return the on-screen column index (0..3, left->right on screen) for a square.
        Uses the same orientation logic as _square_to_cell_index (which already accounts for self._my_side).
        """
        idx = self._square_to_cell_index(s)  # row-major from top-left of the screen
        _row, col = divmod(idx, 4)
        return col

    def _choose_promotion_piece(self, pt: PieceType, sq: Square) -> None:
        """
        Click the requested promotion piece in the 2x2 popup overlay.
        Popup is aligned to the board grid and always occupies the top two rows on screen.
        Piece layout in popup:
            (row=0, col=left)  -> FERZ
            (row=0, col=right) -> WAZIR
            (row=1, col=left)  -> HORSE
            (row=1, col=right) -> empty
        """
        time.sleep(0.2)
        # --- Determine which two on-screen columns the popup spans ---
        promo_col = self._square_to_screen_col(sq)  # 0..3 (left->right on screen)

        if promo_col == 0:
            popup_cols = (0, 1)
        elif promo_col == 1:
            popup_cols = (1, 2)
        else:  # promo_col in {2, 3}
            popup_cols = (2, 3)

        # Popup rows are always the top two rows on screen
        popup_rows = (0, 1)

        # --- Map requested piece type to (row, col) within the popup ---
        # top-left(FERZ), top-right(WAZIR), bottom-left(HORSE), bottom-right empty
        if pt == PieceType.FERZ:
            target_row, target_col = popup_rows[0], popup_cols[0]
        elif pt == PieceType.WAZIR:
            target_row, target_col = popup_rows[0], popup_cols[1]
        elif pt == PieceType.HORSE:
            target_row, target_col = popup_rows[1], popup_cols[0]
        else:
            # No other piece types are present in the popup (PAWN/KING not valid here).
            return

        # --- Convert on-screen (row, col) -> pixel center and click ---
        bx, by, bw, bh = (
            self.calib.board.x,
            self.calib.board.y,
            self.calib.board.w,
            self.calib.board.h,
        )
        # Centers of board cells in screen coordinates, top-left origin
        cx = bx + int((target_col + 0.5) * self._cell_w)
        cy = by + int((target_row + 0.5) * self._cell_h)

        pyautogui.moveTo(cx, cy, duration=0)
        pyautogui.click()

    def _do_drop_on_ui(self, pt: PieceType, to_sq: Square) -> None:
        """
        Drag the requested piece from our pockets to the board.
        """
        # 1) detect the pocket piece center (screen coords)
        pick = self._detect_pocket_piece_center(pt, conf_thres=0.40)

        # Fallback if detector didn’t find it: use center of pockets region
        if pick is None:
            raise Exception("No piece detected in the pocket")
        else:
            sx, sy = pick

        # 2) drag to target square center
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

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

from tinyhouse import (
    Color,
    Piece,
    PieceType,
    Square,
    File,
    Rank,
    SQUARE_NB,
    make_square,
    file_of,
    rank_of,
    square_to_str,
    str_to_square,
    code_from_piece,
    FILES_BY_PIECE,
)

# ---- Constants / thresholds tuned for “yellow highlight”, no animations ----
HL_H_LO, HL_H_HI = 18, 60  # Hue range for yellow/orange-ish
HL_S_MIN, HL_V_MIN = 80, 120  # Saturation / Value minima in [0..255]
HL_FRACTION = 0.04  # ≥ 4% of a cell’s pixels flagged => highlighted

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
        pieces_dir: str = "assets",
        calib_path: Path = CALIB_PATH,
        start_fen: str = "fhwk/3p/P3/KWHF w 1",
    ):
        pyautogui.FAILSAFE = False
        self._sct = mss.mss()
        self._calib_path = Path(calib_path)
        self.calib: Calibration = self._load_or_prompt_calibration(self._calib_path)

        # Precompute cell geometry
        self._cell_w = self.calib.board.w / 4.0
        self._cell_h = self.calib.board.h / 4.0

        # Rasterize/load templates for all piece PNGs
        self._templates: Dict[Piece, np.ndarray] = self._load_templates(pieces_dir)

        # Establish our side from screen (bottom-left piece at game start)
        self._my_side: Color = self._detect_my_side()
        print(self.my_side.to_string())

        # Internal board mirror
        self._board: Dict[Square, Optional[Piece]] = {}
        self._load_board_from_fen(start_fen)

        # Running clock side-to-move mirror. We only need it for promotion direction.
        # main.py’s engine owns the true state; here we keep it consistent via eng.play(mv) in main.
        self._side_to_move: Color = Color.WHITE if (" w " in start_fen) else Color.BLACK

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
            pt, _col = self._classify_cell_piece(img, self._square_to_cell_index(to_sq))
            if pt == PieceType.NO_PIECE_TYPE:
                # Fallback: try best-of-three rescans quickly
                time.sleep(0.03)
                img = self._grab_rect(self.calib.board)
                pt, _col = self._classify_cell_piece(
                    img, self._square_to_cell_index(to_sq)
                )
                if pt == PieceType.NO_PIECE_TYPE:
                    return None
            mv = f"{self._pt_code(pt)}@{square_to_str(to_sq)}"
            self._apply_drop(
                to_sq, pt, mover=self._side_to_move.other()
            )  # opponent’s drop
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
            self._apply_move(
                from_sq, to_sq, promo_suffix, mover=self._side_to_move.other()
            )
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

    def _detect_highlight_cells(self, board_img_bgr: np.ndarray) -> List[int]:
        """
        Return indices (0..15) of cells whose pixel fraction classified as yellow ≥ HL_FRACTION.
        """
        hsv = cv2.cvtColor(board_img_bgr, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        cw, ch = int(self._cell_w), int(self._cell_h)

        mask = cv2.inRange(
            hsv,
            (HL_H_LO, HL_S_MIN, HL_V_MIN),
            (HL_H_HI, 255, 255),
        )

        highlighted: List[int] = []
        for row in range(4):
            for col in range(4):
                x1 = int(col * self._cell_w)
                y1 = int(row * self._cell_h)
                x2 = min(w, x1 + cw)
                y2 = min(h, y1 + ch)
                cell = mask[y1:y2, x1:x2]
                frac = float(np.count_nonzero(cell)) / float(cell.size + 1e-6)
                if frac >= HL_FRACTION:
                    highlighted.append(row * 4 + col)
        return highlighted

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
        row, col = divmod(cell_idx, 4)
        x1 = int(col * self._cell_w)
        y1 = int(row * self._cell_h)
        x2 = int(min(board_img_bgr.shape[1], x1 + self._cell_w))
        y2 = int(min(board_img_bgr.shape[0], y1 + self._cell_h))
        cell = board_img_bgr[y1:y2, x1:x2]
        if cell.size == 0:
            return PieceType.NO_PIECE_TYPE, None

        # Optional but useful: skip clearly empty cells fast
        if not self._occupied_by_gradient(cell):
            return PieceType.NO_PIECE_TYPE, None

        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

        best_score = -1.0
        best_piece = None

        for p, (tmpl, mask) in self._templates.items():
            th, tw = tmpl.shape[:2]
            ch, cw = cell_gray.shape[:2]
            if ch < th or cw < tw:
                # Template larger than cell; skip
                continue

            res = cv2.matchTemplate(cell_gray, tmpl, cv2.TM_CCORR_NORMED, mask=mask)
            score = float(res.max())
            if score > best_score:
                best_score = score
                best_piece = p

        if best_piece is None or best_score < 0.85:
            return PieceType.NO_PIECE_TYPE, None

        return self._ptype(best_piece), self._color(best_piece)

    def _occupied_by_gradient(self, cell_bgr: np.ndarray) -> bool:
        g = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(g, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_16S, 0, 1, ksize=3)
        mag = cv2.add(np.abs(gx), np.abs(gy)).astype(np.float32)
        return float(mag.mean()) >= GRAD_MIN

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

    def _load_templates(self, pieces_dir: str):
        """
        Load PNGs with alpha → (gray, mask) tuples.
        No resizing, no blur. Sprites must fit inside a board cell.
        """
        templates = {}

        for piece, filename in FILES_BY_PIECE.items():
            path = Path(pieces_dir) / filename
            if not path.exists():
                continue

            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue

            # Handle LA (2ch), RGB (3ch), RGBA (4ch), or GRAY
            if raw.ndim == 2:
                gray = raw.astype(np.uint8)
                alpha = np.full_like(gray, 255, dtype=np.uint8)
            elif raw.shape[2] == 2:
                gray = raw[:, :, 0].astype(np.uint8)
                alpha = raw[:, :, 1].astype(np.uint8)
            elif raw.shape[2] == 3:
                gray = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2GRAY)
                alpha = np.full(gray.shape, 255, dtype=np.uint8)
            else:  # 4 channels
                bgr = raw[:, :, :3]
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                alpha = raw[:, :, 3].astype(np.uint8)

            # Binary mask from alpha (no morphology if you don’t want it)
            _, mask = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)

            # Zero the template background (optional but helps)
            gray = (gray * (mask > 0)).astype(np.uint8)

            templates[piece] = (gray, mask)

        return templates

    # ------------------------ Internals: board mirror ------------------------

    def _ptype(self, p: Piece) -> PieceType:
        return PieceType(int(p) & 7)

    def _color(self, p: Piece) -> Color:
        return Color(int(p) >> 3)

    def _pt_code(self, pt: PieceType) -> str:
        return {
            PieceType.PAWN: "P",
            PieceType.HORSE: "H",
            PieceType.FERZ: "F",
            PieceType.WAZIR: "W",
            PieceType.KING: "K",
        }.get(pt, "?")

    def _code_to_pt(self, ch: str) -> PieceType:
        return {
            "P": PieceType.PAWN,
            "H": PieceType.HORSE,
            "F": PieceType.FERZ,
            "W": PieceType.WAZIR,
            "K": PieceType.KING,
        }.get(ch.upper(), PieceType.PAWN)

    def _is_side_piece_at(self, side: Color, sq: Square) -> bool:
        p = self._board.get(sq)
        return p is not None and self._color(p) == side

    def _load_board_from_fen(self, fen: str) -> None:
        """
        Minimal 4x4 FEN parser for this variant. Ignores reserves/turn counters.
        Example: "fhwk/3p/P3/KWHF w 1"
        """
        board_part = fen.split()[0]
        ranks = board_part.split("/")
        if len(ranks) != 4:
            raise ValueError("Expected 4 ranks in FEN")
        for r_idx, rank_str in enumerate(
            reversed(ranks)
        ):  # r=0 is rank1 (bottom for WHITE)
            file_idx = 0
            for ch in rank_str:
                if ch.isdigit():
                    file_idx += int(ch)
                else:
                    sq = make_square(File(file_idx), Rank(r_idx))
                    p = self._piece_from_code(ch)
                    self._board[sq] = p
                    file_idx += 1
            while file_idx < 4:
                sq = make_square(File(file_idx), Rank(r_idx))
                self._board[sq] = None
                file_idx += 1

    def _piece_from_code(self, ch: str) -> Piece:
        ch = ch.strip()
        if not ch:
            return Piece(0)
        # Reuse tinyhouse’s mapping via code string
        for piece, fname in FILES_BY_PIECE.items():
            if code_from_piece(piece) == ch:
                return piece
        # Fallback heuristic
        upper = ch.upper()
        pt_map = {
            "P": PieceType.PAWN,
            "H": PieceType.HORSE,
            "F": PieceType.FERZ,
            "W": PieceType.WAZIR,
            "K": PieceType.KING,
        }
        pt = pt_map.get(upper, PieceType.PAWN)
        base = int(pt)
        return Piece(base if ch.isupper() else base + 8)

    def _apply_move(
        self, from_sq: Square, to_sq: Square, promo_suffix: str, mover: Color
    ) -> None:
        # capture if any
        self._board[to_sq] = self._board.get(from_sq)
        self._board[from_sq] = None

        if promo_suffix:
            pt = self._code_to_pt(promo_suffix[1])  # "=H" -> "H"
            base = int(pt)
            promoted = Piece(base if mover == Color.WHITE else base + 8)
            self._board[to_sq] = promoted

    def _apply_drop(self, to_sq: Square, pt: PieceType, mover: Color) -> None:
        base = int(pt)
        dropped = Piece(base if mover == Color.WHITE else base + 8)
        self._board[to_sq] = dropped

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

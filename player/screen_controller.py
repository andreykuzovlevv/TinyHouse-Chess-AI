# screen_controller.py
from __future__ import annotations

import json
import time
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Set

import cv2
import mss
import numpy as np
import pyautogui

from tinyhouse import (
    Color,
    PieceType,
    Square,
    File,
    Rank,
    make_square,
    square_to_str,
    str_to_square,
    code_to_pt,
    pt_code,
    file_of,
    rank_of,
    relative_rank,
)

# ---------------------------- Configuration ----------------------------

_DEFAULT_CONFIG_PATH = Path("./screen_config.json")
_DEFAULT_ASSETS_DIR = Path("./assets")  # optional: templates for piece classification

# Yellow highlight (HSV) – tuned for typical “bright yellow” overlays.
# You can widen these if needed.
_YELLOW_LOWER_1 = np.array([18, 120, 140], dtype=np.uint8)
_YELLOW_UPPER_1 = np.array([40, 255, 255], dtype=np.uint8)

# If your client uses a different hue, adjust here.
_HIGHLIGHT_MIN_AREA_FRACTION = 0.12  # ≥12% of a square considered highlighted
_STABILITY_MS = 80  # require ~2 frames of stability before emitting

# Edge/texture threshold to decide "empty vs piece"
_EDGE_EMPTY_THRESHOLD = 12.0  # lower means "emptier square"

# Mouse timings
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.01


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


# ---------------------------- Utility helpers ----------------------------


def _now_ms() -> int:
    return int(time.time() * 1000)


def _crop(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    return img[y : y + h, x : x + w]


def _laplacian_energy(gray: np.ndarray) -> float:
    # Measures edge/texture content.
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.mean(np.abs(lap)))


def _square_rects(
    board: BBox, my_side: Color
) -> Dict[Square, Tuple[int, int, int, int]]:
    """Map logical Square -> pixel rect on screen, given orientation."""
    sq_size_w = board.w // 4
    sq_size_h = board.h // 4
    rects: Dict[Square, Tuple[int, int, int, int]] = {}

    for r in range(4):  # logical rank: 0..3 (RANK_1..RANK_4)
        for f in range(4):  # logical file: 0..3 (a..d)
            sq = make_square(File(f), Rank(r))

            if my_side == Color.WHITE:
                # bottom of screen shows logical rank 1 (r==0)
                screen_row = 3 - r  # y increases downward on screen
                screen_col = f
            else:
                # 180° rotation for black-at-bottom orientation
                screen_row = r
                screen_col = 3 - f

            x = board.x + screen_col * sq_size_w
            y = board.y + screen_row * sq_size_h
            rects[sq] = (x, y, sq_size_w, sq_size_h)

    return rects


def _rank_is_last_for(color: Color, r: Rank) -> bool:
    return (color == Color.WHITE and r == Rank.RANK_4) or (
        color == Color.BLACK and r == Rank.RANK_1
    )


def _pt_letter(pt: PieceType) -> str:
    # Always uppercase for engine drop/promo encoding
    return pt_code(pt)


# ---------------------------- Piece templates (optional) ----------------------------


class PieceClassifier:
    """
    Template-based piece classifier.
    - If you provide 32–64 px templates per piece (white+black), place them under ./assets like:
        assets/
          w_p.png, w_h.png, w_f.png, w_w.png, w_k.png
          b_p.png, b_h.png, b_f.png, b_w.png, b_k.png
    - If assets are not found, classifier gracefully returns None and the controller
      will fall back to internal state where possible.
    """

    def __init__(self, assets_dir: Path = _DEFAULT_ASSETS_DIR) -> None:
        self.templates: Dict[str, np.ndarray] = {}
        self._load(assets_dir)

    def _load(self, assets_dir: Path) -> None:
        if not assets_dir.exists():
            return
        for side in ("w", "b"):
            for code, name in (
                ("p", "pawn"),
                ("h", "horse"),
                ("f", "ferz"),
                ("w", "wazir"),
                ("k", "king"),
            ):
                p = assets_dir / f"{side}_{code}.png"
                if p.exists():
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        self.templates[f"{side}_{code}"] = img

    def classify(
        self, square_crop_bgr: np.ndarray
    ) -> Optional[Tuple[Color, PieceType]]:
        if not self.templates:
            return None

        gray = cv2.cvtColor(square_crop_bgr, cv2.COLOR_BGR2GRAY)
        # Resize to a standard size for matching.
        target = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)

        best_name = None
        best_score = -1.0
        for name, templ in self.templates.items():
            templ_resized = cv2.resize(templ, (48, 48), interpolation=cv2.INTER_AREA)
            # Normalized correlation
            res = cv2.matchTemplate(target, templ_resized, cv2.TM_CCOEFF_NORMED)
            score = float(res[0][0])
            if score > best_score:
                best_score = score
                best_name = name

        if best_name is None or best_score < 0.5:
            return None

        side, code = best_name.split("_", 1)
        color = Color.WHITE if side == "w" else Color.BLACK
        pt_map = {
            "p": PieceType.PAWN,
            "h": PieceType.HORSE,
            "f": PieceType.FERZ,
            "w": PieceType.WAZIR,
            "k": PieceType.KING,
        }
        return (color, pt_map[code])


# ---------------------------- Screen capture ----------------------------


class Grabber:
    def __init__(self) -> None:
        self.sct = mss.mss()

    def grab(self, bbox: BBox) -> np.ndarray:
        shot = self.sct.grab(
            {"left": bbox.x, "top": bbox.y, "width": bbox.w, "height": bbox.h}
        )
        img = np.array(shot)  # BGRA
        return img[:, :, :3].copy()  # BGR


# ---------------------------- ScreenController ----------------------------


class ScreenController:
    """
    Responsibilities:
      - Calibrate and store the board bounding box.
      - Detect side to move (our color) by bottom-left piece at start.
      - Detect opponent moves via yellow highlights.
      - Keep an internal board model for disambiguation and promotion handling.
      - Execute our moves by clicking.

    Configuration persisted to screen_config.json:
      {
        "board_bbox": [x, y, w, h],
        "reserve_clicks": { "W": [x, y], "F": [x, y], "H": [x, y], "P": [x, y] },  # optional for drops
        "promotion_panel_bbox": [x, y, w, h],                                      # optional
        "promotion_clicks": { "W": [x, y], "F": [x, y], "H": [x, y] }              # optional fast path
      }
    """

    def __init__(
        self,
        config_path: Path = _DEFAULT_CONFIG_PATH,
        assets_dir: Path = _DEFAULT_ASSETS_DIR,
    ):
        self.config_path = Path(config_path)
        self.assets_dir = Path(assets_dir)

        self._grab = Grabber()
        self._classifier = PieceClassifier(self.assets_dir)

        self.board_bbox: Optional[BBox] = None
        self.square_rect: Dict[Square, Tuple[int, int, int, int]] = {}

        self.my_side: Color = Color.WHITE  # set after detection
        self._opponent: Color = Color.BLACK

        # Internal board state: mapping Square -> Optional[(Color, PieceType)]
        self._board: Dict[Square, Optional[Tuple[Color, PieceType]]] = {
            sq: None for sq in list(Square) if sq != Square.SQ_NONE
        }

        # Reserves (counts) – not strictly required but handy if you want additional checks
        self._reserve: Dict[Color, Dict[PieceType, int]] = {
            Color.WHITE: {
                PieceType.PAWN: 0,
                PieceType.HORSE: 0,
                PieceType.FERZ: 0,
                PieceType.WAZIR: 0,
                PieceType.KING: 0,
            },
            Color.BLACK: {
                PieceType.PAWN: 0,
                PieceType.HORSE: 0,
                PieceType.FERZ: 0,
                PieceType.WAZIR: 0,
                PieceType.KING: 0,
            },
        }

        # Optional UI elements for drops/promo
        self._reserve_clicks: Dict[str, Tuple[int, int]] = {}
        self._promotion_panel: Optional[BBox] = None
        self._promotion_clicks: Dict[str, Tuple[int, int]] = {}

        # Move highlight debounce
        self._prev_hs: Set[Square] = set()
        self._prev_hs_ts = 0
        self._last_emitted_hs: Set[Square] = set()

        # Load config or calibrate interactively once.
        self._load_or_calibrate()

        # Compute squares mapping and detect our side (from the bottom-left corner).
        self._finalize_orientation_and_grid()

        # Initialize internal board from known initial layout (relative to our orientation).
        self._init_board_from_start()

    # ----------------- public API used by main.py -----------------

    def detect_move(self) -> Optional[str]:
        """Return an engine move string (e.g., 'a2a3', 'a3a4=H', 'P@b2') for the opponent; None if nothing new."""
        if self.board_bbox is None:
            return None

        img = self._grab.grab(self.board_bbox)
        hs = self._highlight_squares(img)

        # No highlights
        if not hs:
            self._prev_hs = set()
            self._prev_hs_ts = _now_ms()
            return None

        # Debounce / stability
        now = _now_ms()
        if hs != self._prev_hs:
            self._prev_hs = hs
            self._prev_hs_ts = now
            return None

        if (now - self._prev_hs_ts) < _STABILITY_MS:
            return None

        # Avoid re-emitting the same highlight pattern
        if hs == self._last_emitted_hs:
            return None

        # Interpret move
        move_str: Optional[str] = None
        if len(hs) == 2:
            frm, to = self._resolve_from_to(img, hs)
            if frm is None or to is None:
                return None

            # Promotion check: if opponent pawn moved to last rank, determine piece at destination.
            mover = self._board.get(frm)
            promo_suffix = ""
            if (
                mover
                and mover[1] == PieceType.PAWN
                and _rank_is_last_for(self._opponent, rank_of(to))
            ):
                # Classify piece now at destination (should be the promoted piece)
                crop = _crop(img, self.square_rect[to])
                classified = self._classifier.classify(crop)
                if classified:
                    _, pt = classified
                    promo_suffix = f"={_pt_letter(pt)}"

            move_str = square_to_str(frm) + square_to_str(to) + promo_suffix

            # Update internal board
            self._apply_move_string(move_str, self._opponent)

        elif len(hs) == 1:
            # Drop: a single highlighted square contains the dropped piece.
            (to,) = tuple(hs)
            crop = _crop(img, self.square_rect[to])
            classified = self._classifier.classify(crop)
            if not classified:
                # Fallback: if classifier unavailable, heuristic based on shape cannot be robust; abort gracefully.
                return None
            _, pt = classified
            move_str = f"{_pt_letter(pt)}@{square_to_str(to)}"

            self._apply_move_string(move_str, self._opponent)

        else:
            # Spurious or UI transitional states
            return None

        # record emitted highlight signature
        self._last_emitted_hs = hs
        return move_str

    def execute_ui_move(self, move_str: str) -> None:
        """Click on squares to perform our move; handle drops and promotion UI if configured."""
        if "@" in move_str:
            # Drop: e.g., "P@b2"
            piece_letter, dest = move_str.split("@", 1)
            dest_sq = str_to_square(dest)
            self._click_drop_then_square(piece_letter, dest_sq)
            self._apply_move_string(move_str, self.my_side)
            return

        # Normal or promotion: "a2a3" or "a3a4=H"
        promo_pt: Optional[str] = None
        if "=" in move_str:
            move_part, promo_part = move_str.split("=", 1)
            promo_pt = promo_part.strip().upper()
        else:
            move_part = move_str

        frm = str_to_square(move_part[:2])
        to = str_to_square(move_part[2:4])

        self._click_square(frm)
        self._click_square(to)

        if promo_pt:
            # If explicit promotion clicks configured, use them; else try to find in panel via templates.
            self._choose_promotion(promo_pt)

        self._apply_move_string(move_str, self.my_side)

    # ----------------- calibration & setup -----------------

    def _load_or_calibrate(self) -> None:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            bx, by, bw, bh = data["board_bbox"]
            self.board_bbox = BBox(bx, by, bw, bh)

            self._reserve_clicks = {
                k: tuple(v) for k, v in data.get("reserve_clicks", {}).items()
            }
            if "promotion_panel_bbox" in data:
                px, py, pw, ph = data["promotion_panel_bbox"]
                self._promotion_panel = BBox(px, py, pw, ph)
            self._promotion_clicks = {
                k: tuple(v) for k, v in data.get("promotion_clicks", {}).items()
            }
            return

        # Interactive one-time calibration (console prompts)
        print("[Calibration] Hover TOP-LEFT of the 4×4 board and press Enter...")
        input()
        x1, y1 = pyautogui.position()
        print("[Calibration] Hover BOTTOM-RIGHT of the 4×4 board and press Enter...")
        input()
        x2, y2 = pyautogui.position()
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        # Snap to multiples of 4
        w -= w % 4
        h -= h % 4
        self.board_bbox = BBox(x, y, w, h)

        self._save_config()

    def _save_config(self) -> None:
        if not self.board_bbox:
            return
        data = {
            "board_bbox": [
                self.board_bbox.x,
                self.board_bbox.y,
                self.board_bbox.w,
                self.board_bbox.h,
            ],
            "reserve_clicks": {k: list(v) for k, v in self._reserve_clicks.items()},
            "promotion_clicks": {k: list(v) for k, v in self._promotion_clicks.items()},
        }
        if self._promotion_panel:
            data["promotion_panel_bbox"] = [
                self._promotion_panel.x,
                self._promotion_panel.y,
                self._promotion_panel.w,
                self._promotion_panel.h,
            ]
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _finalize_orientation_and_grid(self) -> None:
        assert self.board_bbox is not None
        # Provisional: compute rects assuming WHITE; we may flip after detecting my_side.
        self.square_rect = _square_rects(self.board_bbox, Color.WHITE)

        # Decide my_side by inspecting bottom-left on screen.
        # We test both WHITE and BLACK orientations and choose the one consistent with start layout.
        # Heuristic: if bottom-left (screen) looks like a King for our side in white orientation, pick WHITE; else BLACK.
        # More robust: classify the bottom-left crop and pick its color.
        img = self._grab.grab(self.board_bbox)

        # Bottom-left screen square in WHITE orientation is logical a1.
        bl_rect_white = self.square_rect[make_square(File.FILE_A, Rank.RANK_1)]
        bl_crop = _crop(img, bl_rect_white)

        classified = self._classifier.classify(bl_crop)
        if classified:
            color, _ = classified
            # If bottom-left classified color is WHITE => we are WHITE at bottom
            self.my_side = Color.WHITE if color == Color.WHITE else Color.BLACK
        else:
            # Fallback heuristic: use brightness to guess presence of a white piece icon (often lighter).
            mean_v = cv2.cvtColor(bl_crop, cv2.COLOR_BGR2HSV)[:, :, 2].mean()
            self.my_side = Color.WHITE if mean_v > 128 else Color.BLACK

        self._opponent = self.my_side.other()
        # Recompute rects with final orientation
        self.square_rect = _square_rects(self.board_bbox, self.my_side)

    def _init_board_from_start(self) -> None:
        """Initialize internal 4×4 board from the Tinyhouse starting layout, relative to logical coordinates."""
        # Clear
        for sq in self._board.keys():
            self._board[sq] = None

        # Logical layout (not screen): White back rank a1..d1: K W U F; pawn at a2.
        # Black back rank a4..d4: F U W K; pawn at d3.
        # We'll set logical squares independent of orientation. Move application handles side-to-move and rotation is only for clicking/detection.

        def put(file_char: str, rank_num: int, color: Color, pt: PieceType):
            sq = str_to_square(file_char + str(rank_num))
            self._board[sq] = (color, pt)

        # White
        put("a", 1, Color.WHITE, PieceType.KING)
        put("b", 1, Color.WHITE, PieceType.WAZIR)
        put("c", 1, Color.WHITE, PieceType.HORSE)
        put("d", 1, Color.WHITE, PieceType.FERZ)
        put("a", 2, Color.WHITE, PieceType.PAWN)

        # Black
        put("a", 4, Color.BLACK, PieceType.FERZ)
        put("b", 4, Color.BLACK, PieceType.HORSE)
        put("c", 4, Color.BLACK, PieceType.WAZIR)
        put("d", 4, Color.BLACK, PieceType.KING)
        put("d", 3, Color.BLACK, PieceType.PAWN)

    # ----------------- highlight detection & interpretation -----------------

    def _highlight_squares(self, board_bgr: np.ndarray) -> Set[Square]:
        hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, _YELLOW_LOWER_1, _YELLOW_UPPER_1)
        mask = cv2.medianBlur(mask, 5)

        hits: Set[Square] = set()
        for sq, rect in self.square_rect.items():
            crop_mask = _crop(mask, rect)
            frac = float(np.count_nonzero(crop_mask)) / float(crop_mask.size)
            if frac >= _HIGHLIGHT_MIN_AREA_FRACTION:
                hits.add(sq)
        return hits

    def _resolve_from_to(
        self, board_bgr: np.ndarray, highlighted: Set[Square]
    ) -> Tuple[Optional[Square], Optional[Square]]:
        sqs = list(highlighted)
        s1, s2 = sqs[0], sqs[1]

        # Prefer using internal state: from-square should currently contain an opponent piece; to-square may be empty or occupied by our piece (capture).
        s1_occ = self._board.get(s1)
        s2_occ = self._board.get(s2)
        cands = []
        if s1_occ and s1_occ[0] == self._opponent:
            cands.append((s1, s2))
        if s2_occ and s2_occ[0] == self._opponent:
            cands.append((s2, s1))
        if len(cands) == 1:
            return cands[0]

        # Fallback: edge energy — from-square tends to be emptier after move, to-square has a piece.
        crop1 = cv2.cvtColor(_crop(board_bgr, self.square_rect[s1]), cv2.COLOR_BGR2GRAY)
        crop2 = cv2.cvtColor(_crop(board_bgr, self.square_rect[s2]), cv2.COLOR_BGR2GRAY)
        e1 = _laplacian_energy(crop1)
        e2 = _laplacian_energy(crop2)

        if e1 < e2 - 1.0:  # margin
            return (s1, s2)
        if e2 < e1 - 1.0:
            return (s2, s1)

        # If still ambiguous, try destination contains opponent piece after capture?
        # As last resort, return deterministic order to avoid None.
        return (s1, s2)

    # ----------------- move application to internal model -----------------

    def _apply_move_string(self, move: str, mover: Color) -> None:
        """Update internal board state and reserves given a move string."""
        if "@" in move:
            # Drop: "P@b2"
            letter, dst = move.split("@", 1)
            pt = code_to_pt(letter)
            dst_sq = str_to_square(dst)
            # Remove from mover's reserve if tracking
            self._reserve[mover][pt] = max(0, self._reserve[mover][pt] - 1)
            # Place on board
            self._board[dst_sq] = (mover, pt)
            return

        promo_pt: Optional[PieceType] = None
        if "=" in move:
            base, promo = move.split("=", 1)
            promo_pt = code_to_pt(promo.strip())
        else:
            base = move

        frm = str_to_square(base[:2])
        to = str_to_square(base[2:4])

        moving = self._board.get(frm)
        if moving is None:
            # If we somehow missed a prior update, try to continue gracefully.
            return

        # Capture?
        captured = self._board.get(to)
        if captured is not None:
            cap_color, cap_pt = captured
            if cap_color != mover:
                self._reserve[mover][cap_pt] += 1

        # Move / promote
        if promo_pt:
            self._board[to] = (mover, promo_pt)
        else:
            self._board[to] = moving

        self._board[frm] = None

    # ----------------- clicking primitives -----------------

    def _center_of(self, rect: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x, y, w, h = rect
        return (x + w // 2, y + h // 2)

    def _click_xy(self, x: int, y: int) -> None:
        pyautogui.moveTo(x, y, duration=0)
        pyautogui.click()

    def _click_square(self, sq: Square) -> None:
        rect = self.square_rect[sq]
        cx, cy = self._center_of(rect)
        self._click_xy(cx, cy)

    def _click_drop_then_square(self, piece_letter: str, dst: Square) -> None:
        # Prefer a configured reserve click (single-click select), then click destination.
        target = self._reserve_clicks.get(piece_letter.upper())
        if not target:
            raise RuntimeError(
                f"No reserve click configured for piece '{piece_letter}'. "
                f"Add reserve_clicks['{piece_letter}'] = [x, y] to {self.config_path}"
            )
        self._click_xy(int(target[0]), int(target[1]))
        time.sleep(0.02)
        self._click_square(dst)

    def _choose_promotion(self, piece_letter: str) -> None:
        piece_letter = piece_letter.upper()

        # Fast path: explicit click coordinates
        if piece_letter in self._promotion_clicks:
            x, y = self._promotion_clicks[piece_letter]
            self._click_xy(int(x), int(y))
            return

        # Slow path: search inside a configured promotion panel by template matching
        if not self._promotion_panel:
            # If not configured, we can't auto-select; leave to default UI if any.
            return

        panel_img = self._grab.grab(self._promotion_panel)
        # Reuse templates from classifier
        wanted_map = {"W": "w_w", "F": "w_f", "H": "w_h"}
        # If we are black promoting, look for black templates instead (many UIs show our color)
        key = wanted_map[piece_letter]
        if self.my_side == Color.BLACK and f"b_{key[2:]}" in self._classifier.templates:
            key = f"b_{key[2:]}"

        templ = self._classifier.templates.get(key)
        if templ is None:
            return

        panel_gray = cv2.cvtColor(panel_img, cv2.COLOR_BGR2GRAY)
        # Scale template to a reasonable size relative to panel height
        scale = min(1.5, max(0.5, self._promotion_panel.h / 200.0))
        t = cv2.resize(templ, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(panel_gray, t, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(res)
        th, tw = t.shape[:2]
        center = (
            self._promotion_panel.x + max_loc[0] + tw // 2,
            self._promotion_panel.y + max_loc[1] + th // 2,
        )
        self._click_xy(center[0], center[1])

    # ----------------- optional helpers to extend config -----------------

    def set_reserve_click(self, piece_letter: str, x: int, y: int) -> None:
        self._reserve_clicks[piece_letter.upper()] = (x, y)
        self._save_config()

    def set_promotion_click(self, piece_letter: str, x: int, y: int) -> None:
        self._promotion_clicks[piece_letter.upper()] = (x, y)
        self._save_config()

    def set_promotion_panel(self, x: int, y: int, w: int, h: int) -> None:
        self._promotion_panel = BBox(x, y, w, h)
        self._save_config()

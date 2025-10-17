# labeler_pockets_pygame.py
# Pockets detector labeler (draw bounding boxes, assign piece type).
# Uses YOLO format for labels; copies images to dataset_labeled and writes txt labels.
#
# Run:
#   python labeler_pockets_pygame.py
#
# Folders:
#   Raw pockets images:  player/dataset/pockets_ours/
#   Labeled output:      player/dataset_labeled/pockets/images and .../labels
#
# Hotkeys:
#   1..5 : select class (P,H,F,W,K)
#   S    : save (copy image + write YOLO labels)
#   N    : next image
#   B    : previous image
#   Del  : delete selected box
#   Esc  : quit
#
# Mouse:
#   Left-drag on image  : draw a new box
#   Left-click a box    : select it
#   Right-click a box   : delete it

import os
import sys
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import pygame

# --------- Paths ---------
SRC_DIR = Path("dataset/pockets_ours")
OUT_IMG_DIR = Path("dataset_labeled/pockets/images")
OUT_LBL_DIR = Path("dataset_labeled/pockets/labels")
CLASSES_TXT = Path("dataset_labeled/pockets/classes.txt")
ASSETS_DIR = Path("assets")

# --------- Classes (no color) ---------
CLASSES = ["P", "H", "F", "W", "K"]  # 0..4
KEY_TO_CLASS_INDEX = {
    pygame.K_1: 0,
    pygame.K_2: 1,
    pygame.K_3: 2,
    pygame.K_4: 3,
    pygame.K_5: 4,
}

# Map classes to your provided asset files (use white icons for visuals)
ASSET_BY_CLASS = {
    "P": "w_p.png",
    "H": "w_h.png",
    "F": "w_f.png",
    "W": "w_w.png",
    "K": "w_k.png",
}

# --------- UI constants ---------
WIN_W, WIN_H = 1100, 800
BG = (18, 18, 22)
PANEL_BG = (28, 28, 34)
IMG_BG = (20, 20, 26)
TEXT = (230, 230, 235)
ACCENT = (70, 120, 255)
BOX_COLOR = (0, 200, 120)
BOX_SEL = (255, 160, 60)
BTN_BG = (40, 40, 48)
BTN_HOVER = (64, 64, 76)
BTN_BORDER = (90, 90, 100)

TOP_PAD = 10
SIDE_PAD = 12
FOOTER_H = 160
BTN_H = 90
BTN_PAD = 10


@dataclass
class BBox:
    cls_idx: int
    x1: int
    y1: int
    x2: int
    y2: int

    def normalized(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        # YOLO: cx, cy, w, h normalized
        x_min = min(self.x1, self.x2)
        y_min = min(self.y1, self.y2)
        w = abs(self.x2 - self.x1)
        h = abs(self.y2 - self.y1)
        cx = x_min + w / 2.0
        cy = y_min + h / 2.0
        return (cx / img_w, cy / img_h, w / img_w, h / img_h)

    def contains(self, x: int, y: int) -> bool:
        x_min = min(self.x1, self.x2)
        y_min = min(self.y1, self.y2)
        x_max = max(self.x1, self.x2)
        y_max = max(self.y1, self.y2)
        return x_min <= x <= x_max and y_min <= y <= y_max


class PocketsLabeler:
    def __init__(self):
        self.files: List[Path] = self._list_images(SRC_DIR)
        if not self.files:
            print(f"No images in {SRC_DIR}")
            sys.exit(0)

        OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
        OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)
        if not CLASSES_TXT.exists():
            with open(CLASSES_TXT, "w", encoding="utf-8") as f:
                for c in CLASSES:
                    f.write(c + "\n")

        pygame.init()
        pygame.display.set_caption("Pockets Labeler")
        self.win = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 20)
        self.font_small = pygame.font.SysFont(None, 16)
        self.font_big = pygame.font.SysFont(None, 24)

        # Layout rects
        self.img_area = pygame.Rect(
            SIDE_PAD, TOP_PAD, WIN_W - 2 * SIDE_PAD, WIN_H - FOOTER_H - TOP_PAD - 8
        )
        self.footer = pygame.Rect(
            SIDE_PAD, self.img_area.bottom + 6, WIN_W - 2 * SIDE_PAD, FOOTER_H
        )

        # Load class icons
        self.icons = {}
        for c in CLASSES:
            p = ASSETS_DIR / ASSET_BY_CLASS[c]
            if not p.exists():
                raise FileNotFoundError(f"Missing asset: {p}")
            self.icons[c] = pygame.image.load(str(p)).convert_alpha()

        # Build class buttons
        self.class_btns = self._make_class_buttons()

        # State
        self.idx = 0
        self.current_image: Optional[pygame.Surface] = None
        self.current_img_path: Optional[Path] = None
        self.current_img_orig: Optional[pygame.Surface] = None
        self.current_scale = 1.0
        self.current_offset = (0, 0)  # top-left in img_area
        self.boxes: List[BBox] = []
        self.selected = -1
        self.current_class = 0
        self.dragging = False
        self.drag_start = (0, 0)
        self.temp_box: Optional[BBox] = None

        self._load_current()

    # ---------- file ops ----------
    def _list_images(self, base: Path) -> List[Path]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        return sorted([p for p in base.glob("*") if p.suffix.lower() in exts])

    def _label_path_for(self, src: Path) -> Path:
        return OUT_LBL_DIR / (src.stem + ".txt")

    def _copy_image_if_needed(self, src: Path) -> Path:
        dst = OUT_IMG_DIR / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        return dst

    # ---------- image/labels load/save ----------
    def _load_current(self):
        self.current_img_path = self.files[self.idx]
        img = pygame.image.load(str(self.current_img_path)).convert()
        self.current_img_orig = img
        # Fit into img_area
        iw, ih = img.get_width(), img.get_height()
        scale = min(self.img_area.width / iw, self.img_area.height / ih, 1.0)
        self.current_scale = scale
        disp = pygame.transform.smoothscale(img, (int(iw * scale), int(ih * scale)))
        ox = self.img_area.x + (self.img_area.width - disp.get_width()) // 2
        oy = self.img_area.y + (self.img_area.height - disp.get_height()) // 2
        self.current_offset = (ox, oy)
        self.current_image = disp

        # Load existing boxes if any
        self.boxes = []
        self.selected = -1
        lbl = self._label_path_for(self.current_img_path)
        if lbl.exists():
            with open(lbl, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            iw0, ih0 = iw, ih
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls_idx = int(parts[0])
                xc = float(parts[1]) * iw0
                yc = float(parts[2]) * ih0
                w = float(parts[3]) * iw0
                h = float(parts[4]) * ih0
                x1 = int(xc - w / 2)
                y1 = int(yc - h / 2)
                x2 = int(xc + w / 2)
                y2 = int(yc + h / 2)
                self.boxes.append(BBox(cls_idx, x1, y1, x2, y2))

    def _save_current(self):
        if self.current_img_path is None or self.current_img_orig is None:
            return
        # Ensure image is copied once
        self._copy_image_if_needed(self.current_img_path)
        # Write YOLO labels
        iw, ih = self.current_img_orig.get_width(), self.current_img_orig.get_height()
        out = []
        for b in self.boxes:
            cx, cy, w, h = b.normalized(iw, ih)
            out.append(f"{b.cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        with open(
            self._label_path_for(self.current_img_path), "w", encoding="utf-8"
        ) as f:
            f.write("\n".join(out))

    # ---------- coords ----------
    def _to_image_coords(self, x: int, y: int) -> Tuple[int, int]:
        ox, oy = self.current_offset
        ix = int((x - ox) / self.current_scale)
        iy = int((y - oy) / self.current_scale)
        return ix, iy

    def _to_display_coords(self, x: int, y: int) -> Tuple[int, int]:
        ox, oy = self.current_offset
        dx = int(x * self.current_scale + ox)
        dy = int(y * self.current_scale + oy)
        return dx, dy

    # ---------- UI build ----------
    def _make_class_buttons(self):
        btns = []
        cols = 5
        btn_w = (self.footer.width - (cols - 1) * BTN_PAD) // cols
        for i, cls_name in enumerate(CLASSES):
            row = i // cols
            col = i % cols
            x = self.footer.x + col * (btn_w + BTN_PAD)
            y = self.footer.y + row * (BTN_H + BTN_PAD) + 40
            rect = pygame.Rect(x, y, btn_w, BTN_H)
            icon = self.icons[cls_name]
            # fit icon to button
            max_w = rect.width - 20
            max_h = rect.height - 20
            iw, ih = icon.get_width(), icon.get_height()
            scale = min(max_w / iw, max_h / ih, 1.0)
            icon_scaled = pygame.transform.smoothscale(
                icon, (int(iw * scale), int(ih * scale))
            )
            btns.append((rect, cls_name, icon_scaled))
        return btns

    # ---------- interactions ----------
    def _select_box_at(self, mx: int, my: int):
        if self.current_img_orig is None:
            return
        ix, iy = self._to_image_coords(mx, my)
        self.selected = -1
        # topmost last
        for i in range(len(self.boxes) - 1, -1, -1):
            if self.boxes[i].contains(ix, iy):
                self.selected = i
                break

    def _delete_selected(self):
        if 0 <= self.selected < len(self.boxes):
            self.boxes.pop(self.selected)
            self.selected = -1

    # ---------- main loop ----------
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key in KEY_TO_CLASS_INDEX:
                        self.current_class = KEY_TO_CLASS_INDEX[event.key]
                    elif event.key == pygame.K_s:
                        self._save_current()
                    elif event.key == pygame.K_n:
                        self._save_current()
                        if self.idx + 1 < len(self.files):
                            self.idx += 1
                            self._load_current()
                    elif event.key == pygame.K_b:
                        self._save_current()
                        if self.idx - 1 >= 0:
                            self.idx -= 1
                            self._load_current()
                    elif event.key == pygame.K_DELETE:
                        self._delete_selected()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mx, my = event.pos
                        # Click on class button?
                        clicked_class = self._hit_class_button(mx, my)
                        if clicked_class is not None:
                            self.current_class = clicked_class
                        elif self._point_in_image(mx, my):
                            # start drawing
                            ix, iy = self._to_image_coords(mx, my)
                            self.dragging = True
                            self.drag_start = (ix, iy)
                            self.temp_box = BBox(self.current_class, ix, iy, ix, iy)
                        else:
                            # maybe select a box if clicked over image margins
                            self._select_box_at(mx, my)

                    elif event.button == 3:
                        # right-click delete selected if over it
                        mx, my = event.pos
                        self._select_box_at(mx, my)
                        self._delete_selected()

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1 and self.dragging:
                        mx, my = event.pos
                        ix, iy = self._to_image_coords(mx, my)
                        self.dragging = False
                        if self.temp_box is not None:
                            self.temp_box.x2 = ix
                            self.temp_box.y2 = iy
                            # discard tiny boxes
                            if (
                                abs(self.temp_box.x2 - self.temp_box.x1) >= 3
                                and abs(self.temp_box.y2 - self.temp_box.y1) >= 3
                            ):
                                self.boxes.append(self.temp_box)
                                self.selected = len(self.boxes) - 1
                            self.temp_box = None

                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging and self.temp_box is not None:
                        mx, my = event.pos
                        ix, iy = self._to_image_coords(mx, my)
                        self.temp_box.x2 = ix
                        self.temp_box.y2 = iy

            self._draw()
            pygame.display.flip()
            self.clock.tick(60)

        # Auto-save on exit
        self._save_current()
        pygame.quit()

    # ---------- drawing ----------
    def _point_in_image(self, mx: int, my: int) -> bool:
        if self.current_image is None:
            return False
        rect = pygame.Rect(
            self.current_offset[0],
            self.current_offset[1],
            self.current_image.get_width(),
            self.current_image.get_height(),
        )
        return rect.collidepoint(mx, my)

    def _hit_class_button(self, mx: int, my: int) -> Optional[int]:
        for i, (rect, cls_name, icon) in enumerate(self.class_btns):
            if rect.collidepoint(mx, my):
                return i
        return None

    def _draw(self):
        self.win.fill(BG)
        # Image area
        pygame.draw.rect(self.win, IMG_BG, self.img_area, border_radius=8)
        # Footer
        pygame.draw.rect(self.win, PANEL_BG, self.footer, border_radius=8)

        if self.current_image is not None:
            self.win.blit(self.current_image, self.current_offset)

        # Draw boxes
        if self.current_img_orig is not None:
            for i, b in enumerate(self.boxes):
                self._draw_box(b, selected=(i == self.selected))
            if self.temp_box is not None:
                self._draw_box(self.temp_box, selected=False)

        # Header text
        head = f"{self.idx+1}/{len(self.files)}  |  Class: {CLASSES[self.current_class]}  |  S=Save  N=Next  B=Prev  Del=Delete"
        surf = self.font_big.render(head, True, TEXT)
        self.win.blit(surf, (self.img_area.x + 8, self.img_area.y + 8))

        # Filename
        name = self.files[self.idx].name
        fn = self.font.render(name, True, TEXT)
        self.win.blit(fn, (self.img_area.x + 8, self.img_area.y + 36))

        # Class buttons
        mx, my = pygame.mouse.get_pos()
        for i, (rect, cls_name, icon) in enumerate(self.class_btns):
            hovered = rect.collidepoint(mx, my)
            pygame.draw.rect(
                self.win,
                BTN_HOVER if hovered or i == self.current_class else BTN_BG,
                rect,
                border_radius=8,
            )
            pygame.draw.rect(self.win, BTN_BORDER, rect, width=1, border_radius=8)
            # icon
            ix = rect.x + (rect.width - icon.get_width()) // 2
            iy = rect.y + (rect.height - icon.get_height()) // 2
            self.win.blit(icon, (ix, iy))
            # label under icon
            lbl = self.font.render(cls_name, True, TEXT)
            self.win.blit(lbl, (rect.centerx - lbl.get_width() // 2, rect.bottom + 4))

    def _draw_box(self, b: BBox, selected: bool):
        color = BOX_SEL if selected else BOX_COLOR
        # convert image coords to display coords
        x1, y1 = self._to_display_coords(b.x1, b.y1)
        x2, y2 = self._to_display_coords(b.x2, b.y2)
        rect = pygame.Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
        pygame.draw.rect(self.win, color, rect, width=2)
        cap = self.font_small.render(CLASSES[b.cls_idx], True, color)
        cap_bg = pygame.Surface((cap.get_width() + 6, cap.get_height() + 2))
        cap_bg.fill((0, 0, 0))
        cap_bg.set_alpha(140)
        self.win.blit(cap_bg, (rect.x, rect.y - cap.get_height() - 4))
        self.win.blit(cap, (rect.x + 3, rect.y - cap.get_height() - 3))


if __name__ == "__main__":
    try:
        PocketsLabeler().run()
    except KeyboardInterrupt:
        pygame.quit()
        sys.exit(0)

# actions.py
import ctypes, time

def click(x, y):
    # SetCursorPos + SendInput MOUSEEVENTF_LEFTDOWN/UP
    # (pyautogui works too if you prefer)
    ...

def execute_ui_move(mv, sq2pix, promo_piece=None):
    if "@” in mv:         # drop
        p, sq = mv.split("@")
        click(*pocket_roi[p]) ; time.sleep(0.05)
        click(*center_of(sq2pix[sq]))
    elif len(mv) in (4,5):  # e2e4[Q]
        a,b = mv[:2], mv[2:4]
        click(*center_of(sq2pix[a])); time.sleep(0.03)
        click(*center_of(sq2pix[b]))
        if len(mv)==5:
            click(*promotion_menu_roi[mv[4].upper()])

# main.py
import time
from vision import ScreenReader
from engine_bridge import Engine
from actions import execute_ui_move

SEARCH_DEPTH = 9
ENGINE_PATH = "../engine_main.exe"


def main():
    sr = ScreenReader()  # holds ROIs, templates, orientation
    eng = Engine(ENGINE_PATH)

    # Initial scan
    prev = sr.read_boardstate()  # pieces, pockets, side_to_move
    while True:
        cur = sr.read_boardstate()

        # Opp move arrived?
        if cur.side_to_move != prev.side_to_move:
            # Either detect explicit move delta, or just trust UI and move on
            prev = cur
            continue

        # Our turn
        fen = to_fen(cur)  # your variant FEN
        eng.position(fen)
        bm = eng.go(SEARCH_DEPTH)["move"]  # "e2e4", "N@f3", "e1g1", "e7e8q", ...

        execute_ui_move(bm, sr.board_to_pixels)  # clicks
        time.sleep(0.15)
        # Verify board changed accordingly
        prev = sr.read_boardstate()


if __name__ == "__main__":
    main()

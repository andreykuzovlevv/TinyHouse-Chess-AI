import time
from tinyhouse import Color
from screen_controller import ScreenController
from engine_bridge import Engine

SEARCH_DEPTH = 7
ENGINE_PATH = r"..\engine_main.exe"

# Set this to your variant's start position FEN
START_FEN = "fhwk/3p/P3/KWHF w 1"


def main():
    sc = ScreenController()

    while True:
        mv = sc.detect_move()
        print(mv)

        time.sleep(0.1)


if __name__ == "__main__":
    main()

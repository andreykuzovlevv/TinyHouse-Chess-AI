import time
from tinyhouse import Color
from screen_controller import ScreenController
from engine_bridge import Engine

SEARCH_DEPTH = 7
ENGINE_PATH = r"..\engine_main.exe"

START_FEN = "fhwk/3p/P3/KWHF w 1"


def main() -> None:
    sc = ScreenController()
    eng = Engine(ENGINE_PATH)

    eng.newgame(START_FEN)

    # If we are Black and it is White to move, we must wait for the opponent first.
    our_turn = Color.WHITE == sc.my_side
    print("My side: ", sc.my_side.to_string())

    while True:
        if not our_turn:
            mv = sc.detect_move()
            if mv is None:
                time.sleep(0.1)
                continue

            # Advance engine with opponent move. If the engine rejects the
            # detected move (bad frame / bad detection), retry detection a few
            # times instead of crashing.
            max_retries = 5
            attempt = 0
            success = False
            while attempt < max_retries:
                try:
                    ok = eng.play(mv)
                except TimeoutError as e:
                    print("Engine timeout while playing move:", e)
                    ok = False
                if ok:
                    success = True
                    break

                # rejected -> re-detect and retry
                attempt += 1
                print(
                    f"Engine rejected move '{mv}', retrying detection ({attempt}/{max_retries})..."
                )
                time.sleep(0.2)
                mv = sc.detect_move()
                if mv is None:
                    time.sleep(0.1)
                    continue

            if not success:
                print(
                    "Failed to apply detected move after retries; continuing to wait."
                )
                # keep waiting for a valid detection
                continue

            our_turn = True
            continue

        # ---- Our turn ----
        # Ask engine for a move from its current internal Position

        res = eng.go(SEARCH_DEPTH)
        mv = res["move"]
        print("Engine move: ", mv)
        if mv == "none" or not mv:
            # No legal move (checkmate/stalemate). Stop.
            break

        # Execute on UI
        sc.execute_ui_move(mv)

        # Advance engine with our move to keep internal state consistent
        eng.play(mv)

        our_turn = False


if __name__ == "__main__":
    main()

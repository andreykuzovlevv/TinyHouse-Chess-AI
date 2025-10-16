from engine_bridge import Engine

ENGINE_PATH = "../engine_main.exe"

fen = "fhwk/3p/P3/KWHF w 1"

eng = Engine(ENGINE_PATH)

eng.isready()
eng.newgame(fen)

# If we’re Black and White plays first:
eng.play("a2a3")  # example opponent move observed on screen

bm = eng.go(9)
print("engine move:", bm["move"])
eng.play(bm["move"])  # keep internal state in sync after you click it on UI

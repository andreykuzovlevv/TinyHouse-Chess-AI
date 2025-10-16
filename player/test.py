from engine_bridge import Engine

ENGINE_PATH = "../engine_main.exe"

eng = Engine(ENGINE_PATH)

fen = "fhwk/3p/P3/KWHF w 1"

eng.position(fen)

bm = eng.go(9)["move"]

print(bm)

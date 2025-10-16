# engine_bridge.py
import subprocess, threading, queue, sys


class Engine:
    def __init__(self, path):
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.outq = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.p.stdout:
            self.outq.put(line.rstrip("\n"))

    def position(self, fen: str):
        self.p.stdin.write(f"position {fen}\n")
        self.p.stdin.flush()

    def go(self, depth: int) -> dict:
        self.p.stdin.write(f"go depth {depth}\n")
        self.p.stdin.flush()
        best = {}
        while True:
            line = self.outq.get()
            if line.startswith("bestmove"):
                # e.g., "bestmove e2e4 score 34"
                toks = line.split()
                best["move"] = toks[1]
                if "score" in toks:
                    best["score"] = toks[toks.index("score") + 1]
                return best

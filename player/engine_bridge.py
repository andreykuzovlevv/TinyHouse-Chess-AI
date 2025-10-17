# engine_bridge.py
import subprocess, threading, queue, time


class Engine:
    def __init__(self, path):
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.outq = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        for line in self.p.stdout:
            self.outq.put(line.rstrip("\n"))

    def _send(self, s: str):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _expect(self, prefix: str, timeout: float = 5.0) -> str:
        t0 = time.time()
        while True:
            try:
                line = self.outq.get(timeout=0.05)
            except queue.Empty:
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"timeout waiting for '{prefix}'")
                continue
            if line.startswith(prefix):
                return line
            # discard other lines

    # --- Public API: each call holds lock until it gets its reply ---

    def newgame(self, fen: str):
        with self.lock:
            self._send(f"newgame {fen}")
            # optional ack; safe to proceed immediately

    def play(self, move: str) -> bool:
        with self.lock:
            self._send(f"play {move}")
            line = self._expect("played", timeout=3.0)
            return line.startswith("played")

    def go(self, depth: int) -> dict:
        with self.lock:
            self._send(f"go depth {depth}")
            best = {}
            while True:
                line = self.outq.get()
                if line.startswith("bestmove"):
                    toks = line.split()
                    best["move"] = toks[1]
                    if "score" in toks:
                        best["score"] = toks[toks.index("score") + 1]
                    return best
                # ignore "info ..." lines

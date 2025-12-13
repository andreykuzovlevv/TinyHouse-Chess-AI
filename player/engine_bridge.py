# engine_bridge.py
import subprocess, threading, queue, time


class Engine:
    def __init__(self, path) -> None:
        self.p = subprocess.Popen(
            [path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        self.outq = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        for line in self.p.stdout:
            line = line.rstrip("\n")
            print(line, flush=True)
            self.outq.put(line)

    def _send(self, s: str) -> None:
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _expect(self, prefix: str, timeout: float = 5.0) -> str:
        t0 = time.time()
        while True:
            try:
                line = self.outq.get(timeout=0.2)
            except queue.Empty:
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"timeout waiting for '{prefix}'")
                continue
            if line.startswith(prefix):
                return line
            # discard other lines

    # --- Public API: each call holds lock until it gets its reply ---

    def newgame(self, fen: str) -> None:
        with self.lock:
            self._send(f"newgame {fen}")
            # optional ack; safe to proceed immediately

    def play(self, move: str) -> bool:
        with self.lock:
            self._send(f"play {move}")
            # Wait for either a confirmation 'played ...' or an 'info string illegal'
            t0 = time.time()
            timeout = 3.0
            while True:
                try:
                    line = self.outq.get(timeout=0.2)
                except queue.Empty:
                    if time.time() - t0 > timeout:
                        raise TimeoutError(
                            "timeout waiting for 'played' or 'info string illegal'"
                        )
                    continue
                if line.startswith("played"):
                    return True
                if line.startswith("info string illegal"):
                    return False
                # ignore other lines

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
                # capture evals from "info ... score <n>" lines so caller can see latest
                if line.startswith("info"):
                    toks = line.split()
                    if "score" in toks:
                        try:
                            best["score"] = toks[toks.index("score") + 1]
                        except Exception:
                            pass
                    # continue waiting for bestmove

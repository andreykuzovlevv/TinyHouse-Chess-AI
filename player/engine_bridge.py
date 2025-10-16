# engine_bridge.py
import subprocess, threading, queue, sys, time


class Engine:
    def __init__(self, path):
        # Keep one long-lived process/session.
        self.p = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,  # line buffered
        )
        self.outq = queue.Queue()
        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def _reader(self):
        for line in self.p.stdout:
            self.outq.put(line.rstrip("\n"))

    def _send(self, s: str):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _expect_line(self, prefix: str, timeout: float = 5.0) -> str:
        """Read lines until one starts with `prefix`. Returns that line."""
        t0 = time.time()
        while True:
            try:
                line = self.outq.get(timeout=0.05)
            except queue.Empty:
                if time.time() - t0 > timeout:
                    raise TimeoutError(f"waiting for '{prefix}' timed out")
                continue
            if line.startswith(prefix):
                return line
            # Drop other 'info string ...' lines silently

    # --- Protocol commands ---

    def isready(self) -> bool:
        self._send("isready")
        line = self._expect_line("readyok")
        return True

    def quit(self):
        try:
            self._send("quit")
        except Exception:
            pass

    def newgame(self, fen: str):
        self._send(f"newgame {fen}")
        # Optional: wait for acknowledgment
        # self._expect_line("ok")

    def play(self, move: str) -> bool:
        self._send(f"play {move}")
        # Engine will reply either "played <move>" or "illegal"
        line = self._expect_line("played", timeout=3.0)
        # If you prefer strict checking, parse 'played <move>' and compare.
        return True

    # Legacy: allow setting a full position in one shot (useful for tests)
    def position(self, fen: str):
        self._send(f"position {fen}")
        # optional ack:
        # self._expect_line("info string position set")

    def go(self, depth: int) -> dict:
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
            # else: consume other 'info ...' lines

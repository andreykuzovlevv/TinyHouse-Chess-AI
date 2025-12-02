![Screenshot](game_screenshot.png)

# Tiny House Chess game variant with AI
The initial goal of this project was to solve a small chess variant called **“tinyhouse”** on chess.com.  
It’s played on a 4×4 board, and the full rules can be found in [`TINYHOUSE_rules.txt`](./TINYHOUSE_rules.txt).



## Engine

When implementing the game engine, I took inspiration from the official Stockfish repository, mainly around position representation and board logic (see [`position.h`](./src/core/position.cc) / [`position.cc`](./src/core/position.h)).

For the AI, I went with a classic **negamax** search, implemented in [`minmax.cc`](./src/minmax/minmax.cc).  
At the moment there is:

- No parallel processing
- No move pruning

Despite that, the engine already reaches decent search speeds (exact benchmarks still to be measured, but there’s room for them here).

Current performance:
- **Depth 9 plies**: ~3–5 seconds  
- **Depth 12 plies**: ~5–20 seconds

## UI

I built a simple game UI using **SDL3** where you can:

- Choose which side to play
- Watch the AI slowly corner you while the evaluation number climbs upward until the game ends

You can try it via:

```

ui_chess_game/build/tinychess.exe

```

## Player Automation

I also built an automation bot in Python.

Using basic machine learning, it is trained to:

- Recognize pieces on the board
- Detect which move was last played

The main script lives in:

```

player/main.py

```

## Results

Tests on chess.com show that the bot is able to beat essentially any human player, although the top players still manage to put up a good and entertaining fight.

## Running & Maintenance

I haven’t organized setup instructions or maintenance documentation yet.  
If this becomes necessary for others to run or contribute to the project, I’ll structure that part properly.


// engine_main.cpp
#include <algorithm>
#include <deque>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include "minmax/minmax.h"

using namespace tiny;

static inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}
static inline bool starts_with(const std::string& s, const char* pfx) {
    return s.rfind(pfx, 0) == 0;
}
static inline std::vector<std::string> split_ws(const std::string& s) {
    std::istringstream       is(s);
    std::vector<std::string> out;
    std::string              tok;
    while (is >> tok) out.push_back(tok);
    return out;
}

// Match a move by comparing to_string(m) with the user text.
// This avoids having to write a separate move parser, and stays
// perfectly in sync with your move-encoding (NORMAL/PROMOTION/DROP).
static bool find_move_by_text(Position& pos, const std::string& text, Move& out) {
    MoveList<LEGAL> moves(pos);
    for (int i = 0; i < moves.size(); ++i) {
        const Move m = moves[i];
        if (to_string(m) == text) {
            out = m;
            return true;
        }
    }
    return false;
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Bitboards::init();
    Position::init();

    Position              pos;
    std::deque<StateInfo> states;
    states.emplace_back();  // initial stateinfo frame

    auto set_fen = [&](const std::string& fen) {
        states.clear();
        states.emplace_back();
        pos.set(fen, &states.back());
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;

        if (line == "isready") {
            std::cout << "readyok\n" << std::flush;
            continue;
        }
        if (line == "quit" || line == "exit") break;

        // newgame <FEN>
        if (starts_with(line, "newgame")) {
            // Preserve everything after the first space as FEN (with spaces)
            auto p = line.find(' ');
            if (p == std::string::npos) {
                std::cout << "info string error: newgame requires FEN\n" << std::flush;
                continue;
            }
            std::string fen = trim(line.substr(p + 1));
            try {
                set_fen(fen);
            } catch (...) {
                std::cout << "info string error: bad FEN\n" << std::flush;
            }
            continue;
        }

        // position <FEN>   (kept for testing)
        if (starts_with(line, "position")) {
            auto p = line.find(' ');
            if (p == std::string::npos) {
                std::cout << "info string error: position requires FEN\n" << std::flush;
                continue;
            }
            std::string fen = trim(line.substr(p + 1));
            try {
                set_fen(fen);
                std::cout << "info string position set\n" << std::flush;
            } catch (...) {
                std::cout << "info string error: bad FEN\n" << std::flush;
            }
            continue;
        }

        // play <MOVE>  e.g., "a2a3", "a7a8=H", "P@b2"
        if (starts_with(line, "play")) {
            auto p = line.find(' ');
            if (p == std::string::npos) {
                std::cout << "info string error: play requires move\n" << std::flush;
                continue;
            }
            std::string mtxt = trim(line.substr(p + 1));

            Move m = MOVE_NONE;
            if (!find_move_by_text(pos, mtxt, m)) {
                std::cout << "info string illegal\n" << std::flush;
                continue;
            }
            states.emplace_back();
            pos.do_move(m, states.back());
            std::cout << "played " << to_string(m) << "\n" << std::flush;
            continue;
        }

        // go depth N
        if (starts_with(line, "go")) {
            auto toks  = split_ws(line);
            int  depth = 9;
            for (size_t i = 1; i + 1 < toks.size(); ++i) {
                if (toks[i] == "depth") {
                    try {
                        depth = std::stoi(toks[i + 1]);
                    } catch (...) {
                    }
                }
            }

            MoveList<LEGAL> rootMoves(pos);
            if (rootMoves.size() == 0) {
                std::cout << "bestmove none score 0\n" << std::flush;
                continue;
            }

            SearchResult res = search_best_move(pos, depth);
            if (res.bestMove == MOVE_NONE) {
                std::cout << "bestmove none score 0\n" << std::flush;
            } else {
                std::cout << "bestmove " << to_string(res.bestMove) << " score " << res.score
                          << "\n"
                          << std::flush;
            }
            // IMPORTANT: do not auto-advance here; Python will send 'play' after clicking.
            continue;
        }

        if (line == "d") {
            std::cout << pos << std::flush;
            continue;
        }

        std::cout << "info string unknown command\n" << std::flush;
    }
    return 0;
}

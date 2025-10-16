#include <algorithm>
#include <cctype>
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
    size_t b = s.find_last_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
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

int main() {
    // Fast IO for pipe use from Python
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Bitboards::init();
    Position::init();

    Position              pos;
    std::deque<StateInfo> states;
    states.emplace_back();

    std::string line;
    while (std::getline(std::cin, line)) {
        line = trim(line);
        if (line.empty()) continue;

        // isready / quit
        if (line == "isready") {
            std::cout << "readyok\n" << std::flush;
            continue;
        }
        if (line == "quit" || line == "exit") {
            break;
        }

        // position: accept both "position <fen>" and "position fen <fen>"
        if (starts_with(line, "position")) {
            auto toks = split_ws(line);
            if (toks.size() < 2) {
                std::cout << "info string error: position requires FEN\n" << std::flush;
                continue;
            }
            size_t fenStart = 1;

            // Rebuild FEN from remaining tokens verbatim (spaces significant)
            size_t posAfterCmd = 0;
            for (size_t i = 0, k = 0; i < line.size(); ++i) {
                if (std::isspace(static_cast<unsigned char>(line[i]))) {
                    if (++k == fenStart) {
                        posAfterCmd = i + 1;
                        break;
                    }
                    while (i + 1 < line.size() &&
                           std::isspace(static_cast<unsigned char>(line[i + 1])))
                        ++i;
                }
            }
            std::string fen = trim(line.substr(posAfterCmd));

            states.clear();
            states.emplace_back();
            try {
                pos.set(fen, &states.back());
                std::cout << "info string position set\n" << std::flush;
            } catch (...) {
                std::cout << "info string error: bad FEN\n" << std::flush;
            }
            continue;
        }

        // go depth N
        if (starts_with(line, "go")) {
            auto toks  = split_ws(line);
            int  depth = 9;  // default
            for (size_t i = 1; i + 1 < toks.size(); ++i) {
                if (toks[i] == "depth") {
                    try {
                        depth = std::stoi(toks[i + 1]);
                    } catch (...) {
                    }
                }
            }

            // Terminal checks
            MoveList<LEGAL> rootMoves(pos);
            if (rootMoves.size() == 0) {
                // No legal moves
                std::cout << "bestmove none score 0\n" << std::flush;
                continue;
            }

            SearchResult res = search_best_move(pos, depth);
            if (res.bestMove == MOVE_NONE) {
                std::cout << "bestmove none score 0\n" << std::flush;
                continue;
            }

            std::cout << "bestmove " << to_string(res.bestMove) << " score " << res.score << "\n"
                      << std::flush;

            // If you want the engine to advance its internal state after returning
            // a move (so Python can call "go" repeatedly without resending 'position'),
            // uncomment the following lines:
            // states.emplace_back();
            // pos.do_move(res.bestMove, states.back());

            continue;
        }

        // Optional helpers for debugging from a terminal
        if (line == "d") {
            std::cout << pos << std::flush;
            continue;
        }

        std::cout << "info string unknown command\n" << std::flush;
    }
    return 0;
}

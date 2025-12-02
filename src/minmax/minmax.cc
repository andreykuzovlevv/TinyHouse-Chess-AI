#include "minmax.h"

namespace tiny {

// Material-only evaluation, side-to-move perspective.
// Positive means the side to move is better.
Value evaluate(const Position& pos) {
    // Board material
    Value diff = 0;
    for (Square s = SQ_A1; s <= SQ_D4; ++s) diff += piece_value(pos.piece_on(s));

    // Pocket material (drops)
    // Only PAWN/HORSE/FERZ/WAZIR are used in pockets
    for (PieceType pt = PAWN; pt <= WAZIR; ++pt) {
        diff += int(pos.pocket(WHITE).count(pt)) * type_value(pt) * 0.7;
        diff -= int(pos.pocket(BLACK).count(pt)) * type_value(pt) * 0.7;
    }

    // Perspective: return score for side to move
    if (pos.side_to_move() == BLACK) diff = -diff;

    return diff;
}

// Core negamax with alpha-beta pruning.
// Returns a score from the perspective of the side to move in 'pos'.
Value negamax(Position& pos, int depth, int alpha, int beta, int ply) {
    // Repetition draw
    if (pos.is_draw(ply)) return VALUE_DRAW;

    if (depth == 0) return evaluate(pos);

    MoveList<LEGAL> moves(pos);

    // No legal moves: terminal
    if (moves.size() == 0) {
        if (pos.checkers()) {
            // Checkmate: side to move loses
            return -VALUE_MATE + ply;  // prefer quicker mates against us
        } else {
            // Stalemate: side to move WINS
            return +VALUE_MATE - ply;  // prefer quicker wins for us
        }
    }

    int best = -VALUE_INFINITE;  // track true best value found

    for (const Move& m : moves) {
        StateInfo st;
        pos.do_move(m, st);

        // Negamax flip and window flip
        int score = -negamax(pos, depth - 1, -beta, -alpha, ply + 1);

        pos.undo_move(m);

        if (score > best) best = score;
        if (score > alpha) {
            alpha = score;
            // beta cutoff
            if (alpha >= beta) break;
        }
    }

    return best;
}

// Returns the best move and its score for the current position.
SearchResult search_best_move(Position& pos, int depth) {
    MoveList<LEGAL> moves(pos);

    // Handle immediate terminals at root
    if (moves.size() == 0) {
        int terminalScore =
            pos.checkers() ? (-VALUE_MATE /* + ply=0 */) : (+VALUE_MATE /* - ply=0 */);
        return {MOVE_NONE, terminalScore};
    }
    if (pos.is_draw(/*ply=*/0)) return {MOVE_NONE, VALUE_DRAW};

    int alpha = -VALUE_MATE;
    int beta  = +VALUE_MATE;

    Move bestMove  = MOVE_NONE;
    int  bestScore = -VALUE_MATE;

    for (const Move& m : moves) {
        StateInfo st;
        pos.do_move(m, st);

        int score = -negamax(pos, depth - 1, -beta, -alpha, 1);

        pos.undo_move(m);

        if (score > bestScore) {
            bestScore = score;
            bestMove  = m;
        }
        if (score > alpha) {
            alpha = score;
        }
    }

    return {bestMove, bestScore};
}
}  // namespace tiny

#include "movegen.h"

#include <cassert>
#include <initializer_list>

#include "bitboard.h"
#include "position.h"

namespace tiny {

namespace {

template <Direction offset>
inline Move* splat_pawn_moves(Move* moveList, Bitboard to_bb) {
    while (to_bb) {
        Square to   = pop_lsb(to_bb);
        *moveList++ = Move(to - offset, to);
    }
    return moveList;
}

inline Move* splat_moves(Move* moveList, Square from, Bitboard to_bb) {
    while (to_bb) *moveList++ = Move(from, pop_lsb(to_bb));
    return moveList;
}

template <GenType Type, Direction D>
Move* make_promotions(Move* moveList, [[maybe_unused]] Square to) {
    *moveList++ = Move::make<PROMOTION>(to - D, to, HORSE);
    *moveList++ = Move::make<PROMOTION>(to - D, to, FERZ);
    *moveList++ = Move::make<PROMOTION>(to - D, to, WAZIR);

    return moveList;
}

template <Color Us, GenType Type>
Move* generate_pawn_moves(const Position& pos, Move* moveList, Bitboard target) {
    constexpr Color    Them     = ~Us;
    constexpr Bitboard TRank3BB = (Us == WHITE ? Rank3BB : Rank2BB);
    // constexpr Bitboard  TRank3BB = (Us == WHITE ? Rank3BB : Rank6BB);
    constexpr Direction Up      = pawn_push(Us);
    constexpr Direction UpRight = (Us == WHITE ? NORTH_EAST : SOUTH_WEST);
    constexpr Direction UpLeft  = (Us == WHITE ? NORTH_WEST : SOUTH_EAST);

    const Bitboard emptySquares = ~pos.pieces();
    const Bitboard enemies      = Type == EVASIONS ? pos.checkers() : pos.pieces(Them);

    Bitboard pawnsOn3    = pos.pieces(Us, PAWN) & TRank3BB;
    Bitboard pawnsNotOn3 = pos.pieces(Us, PAWN) & ~TRank3BB;

    // Single pawn pushes, no promotions
    Bitboard b1 = shift<Up>(pawnsNotOn3) & emptySquares;

    if constexpr (Type == EVASIONS)  // Consider only blocking squares
    {
        b1 &= target;
    }

    moveList = splat_pawn_moves<Up>(moveList, b1);

    // Promotions
    if (pawnsOn3) {
        Bitboard b1 = shift<UpRight>(pawnsOn3) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsOn3) & enemies;
        Bitboard b3 = shift<Up>(pawnsOn3) & emptySquares;

        if constexpr (Type == EVASIONS) b3 &= target;

        while (b1) moveList = make_promotions<Type, UpRight>(moveList, pop_lsb(b1));

        while (b2) moveList = make_promotions<Type, UpLeft>(moveList, pop_lsb(b2));

        while (b3) moveList = make_promotions<Type, Up>(moveList, pop_lsb(b3));
    }

    // Standard captures
    if constexpr (Type == EVASIONS || Type == NON_EVASIONS) {
        Bitboard b1 = shift<UpRight>(pawnsNotOn3) & enemies;
        Bitboard b2 = shift<UpLeft>(pawnsNotOn3) & enemies;

        moveList = splat_pawn_moves<UpRight>(moveList, b1);
        moveList = splat_pawn_moves<UpLeft>(moveList, b2);
    }

    return moveList;
}

template <Color Us, PieceType Pt>
Move* generate_moves(const Position& pos, Move* moveList, Bitboard target) {
    static_assert(Pt != KING && Pt != PAWN, "Unsupported piece type in generate_moves()");

    Bitboard bb = pos.pieces(Us, Pt);

    while (bb) {
        Square   from = pop_lsb(bb);
        Bitboard b    = attacks_bb<Pt>(from, pos.pieces()) & target;

        moveList = splat_moves(moveList, from, b);
    }

    return moveList;
}

template <Color Us, GenType Type>
Move* generate_all(const Position& pos, Move* moveList) {
    static_assert(Type != LEGAL, "Unsupported type in generate_all()");

    const Square ksq = pos.square<KING>(Us);
    Bitboard     target;

    // Skip generating non-king moves when in double check
    if (Type != EVASIONS || !more_than_one(pos.checkers())) {
        if constexpr (Type == EVASIONS) {
            // We are in EVASIONS and NOT double check here
            Bitboard checkers = pos.checkers();
            Square   checker  = lsb(checkers);

            Bitboard checkerBB    = square_bb(checker);
            bool     horseChecker = (pos.pieces(HORSE) & checkerBB) != 0;

            if (horseChecker) {
                // Block on the leg or capture the horse
                // NOTE: horse_leg_bb expects (attacker, king)
                target = horse_leg_bb(checker, ksq) | checkerBB;
            } else {
                target = checkerBB;
            }
        } else {
            // NON_EVASIONS: any square not occupied by our pieces
            target = ~pos.pieces(Us);
        }

        // Generate DROP moves from pocket
        // - Can drop on any empty square
        // - In EVASIONS, restrict to 'target' blocking set
        // - Pawns cannot be dropped on last rank (promotion rank)
        const Bitboard emptySquares = ~pos.pieces();
        Bitboard       dropMask     = (Type == EVASIONS ? (target & emptySquares) : emptySquares);

        const Pocket pk = pos.pocket(Us);

        auto gen_drops_for = [&](PieceType pt, Bitboard mask) {
            if (pk.count(pt) == 0) return;
            Bitboard to_bb = mask;
            while (to_bb) {
                Square to   = pop_lsb(to_bb);
                *moveList++ = Move::make<DROP>(to, to, pt);
            }
        };

        moveList = generate_moves<Us, WAZIR>(pos, moveList, target);
        moveList = generate_moves<Us, FERZ>(pos, moveList, target);
        moveList = generate_moves<Us, HORSE>(pos, moveList, target);
        moveList = generate_pawn_moves<Us, Type>(pos, moveList, target);

        // Pawns: exclude last rank
        Bitboard pawnMask = dropMask & (Us == WHITE ? ~Rank4BB : ~Rank1BB);
        gen_drops_for(PAWN, pawnMask);

        // Other pieces: HORSE, FERZ, WAZIR
        gen_drops_for(HORSE, dropMask);
        gen_drops_for(WAZIR, dropMask);
        gen_drops_for(FERZ, dropMask);
    }

    Bitboard b = attacks_bb<KING>(ksq) & (Type == EVASIONS ? ~pos.pieces(Us) : target);

    moveList = splat_moves(moveList, ksq, b);

    return moveList;
}

}  // namespace

// <EVASIONS>     Generates all pseudo-legal check evasions
// <NON_EVASIONS> Generates all pseudo-legal captures and non-captures
//
// Returns a pointer to the end of the move list.
template <GenType Type>
Move* generate(const Position& pos, Move* moveList) {
    static_assert(Type != LEGAL, "Unsupported type in generate()");
    assert((Type == EVASIONS) == bool(pos.checkers()));

    Color us = pos.side_to_move();

    return us == WHITE ? generate_all<WHITE, Type>(pos, moveList)
                       : generate_all<BLACK, Type>(pos, moveList);
}

// Explicit template instantiations
template Move* generate<EVASIONS>(const Position&, Move*);
template Move* generate<NON_EVASIONS>(const Position&, Move*);

// generate<LEGAL> generates all the legal moves in the given position

template <>
Move* generate<LEGAL>(const Position& pos, Move* moveList) {
    Color    us     = pos.side_to_move();
    Bitboard pinned = pos.blockers_for_king(us) & pos.pieces(us);
    Square   ksq    = pos.square<KING>(us);
    Move*    cur    = moveList;

    moveList =
        pos.checkers() ? generate<EVASIONS>(pos, moveList) : generate<NON_EVASIONS>(pos, moveList);
    while (cur != moveList)
        if (((pinned & cur->from_sq()) || cur->from_sq() == ksq) && !pos.legal(*cur))
            *cur = *(--moveList);
        else
            ++cur;

    return moveList;
}

}  // namespace tiny

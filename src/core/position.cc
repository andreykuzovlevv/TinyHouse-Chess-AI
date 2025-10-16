#include "position.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>
#include <utility>

#include "bitboard.h"
#include "misc.h"

using std::string;

namespace tiny {

namespace Zobrist {

Key psq[PIECE_NB][SQUARE_NB];
Key side;
Key pocket[COLOR_NB][PIECE_TYPE_NB][3];  // [color][pieceType][count 0,1,2]
}  // namespace Zobrist

namespace {

static constexpr Piece Pieces[] = {W_PAWN, W_HORSE, W_FERZ, W_WAZIR, W_KING,
                                   B_PAWN, B_HORSE, B_FERZ, B_WAZIR, B_KING};

// Piece to character mapping for FEN input/output
// Indices map to Piece enum: [0]=NO_PIECE, [1]=W_PAWN('P'), [2]=W_HORSE('H'),
// [3]=W_FERZ('F'), [4]=W_WAZIR('W'), [5]=W_KING('K'), then lowercase for black.
constexpr std::string_view PieceToChar = " PHFWK   phfwk  ";
}  // namespace

std::string square_string(Square s) {
    return std::string{char('a' + file_of(s)), char('1' + rank_of(s))};
}

// Returns an ASCII representation of the position (with pockets)
std::ostream& operator<<(std::ostream& os, const Position& pos) {
    // Top pocket for black
    {
        os << "\n+[" << pos.pocket(BLACK).to_string(BLACK) << "]+\n";
    }

    os << " +---+---+---+---+\n";

    for (Rank r = RANK_4; r >= RANK_1; --r) {
        for (File f = FILE_A; f <= FILE_D; ++f)
            os << " | " << PieceToChar[pos.piece_on(make_square(f, r))];

        os << " | " << (1 + r) << "\n +---+---+---+---+\n";
    }

    os << "   a   b   c   d\n"
       << "+[" << pos.pocket(WHITE).to_string(WHITE) << "]+\n";
    //    << "\nFen: " << pos.fen() << "\nKey: " << std::hex << std::uppercase << std::setfill('0')
    //    << std::setw(16) << pos.key() << std::setfill(' ') << std::dec << "\nCheckers: [";

    // for (Bitboard b = pos.checkers(); b;) os << square_string(pop_lsb(b));
    // os << "]\n\n";

    return os;
}

// Implements Marcel van Kervinck's cuckoo algorithm to detect repetition of positions
// for 3-fold repetition draws. The algorithm uses two hash tables with Zobrist hashes
// to allow fast detection of recurring positions. For details see:
// http://web.archive.org/web/20201107002606/https://marcelk.net/2013-04-06/paper/upcoming-rep-v2.pdf

// First and second hash functions for indexing the cuckoo tables
inline int H1(Key h) { return h & 0x7ff; }
inline int H2(Key h) { return (h >> 16) & 0x7ff; }

// Cuckoo tables with Zobrist hashes of valid reversible moves, and the moves themselves
std::array<Key, 2048>  cuckoo;
std::array<Move, 2048> cuckooMove;

// Initializes at startup the various arrays used to compute hash keys
void Position::init() {
    PRNG rng(1070372);

    for (Piece pc : Pieces)
        for (Square s = SQ_A1; s <= SQ_D4; ++s) Zobrist::psq[pc][s] = rng.rand<Key>();
    // pawns on these squares will promote
    std::fill_n(Zobrist::psq[W_PAWN] + SQ_A4, 4, 0);
    std::fill_n(Zobrist::psq[B_PAWN] + SQ_A1, 4, 0);

    Zobrist::side = rng.rand<Key>();

    // Initialize pocket Zobrist keys
    for (Color c = WHITE; c <= BLACK; ++c)
        for (PieceType pt = PAWN; pt <= WAZIR; ++pt)
            for (int count = 0; count < 3; ++count) Zobrist::pocket[c][pt][count] = rng.rand<Key>();

    // Prepare the cuckoo tables
    cuckoo.fill(0);
    cuckooMove.fill(Move::none());
    int count = 0;
    for (Piece pc : Pieces)
        for (Square s1 = SQ_A1; s1 <= SQ_D4; ++s1)
            for (Square s2 = Square(s1 + 1); s2 <= SQ_D4; ++s2)
                if ((type_of(pc) != PAWN) && (attacks_bb(type_of(pc), s1, 0) & s2)) {
                    Move move = Move(s1, s2);
                    Key  key  = Zobrist::psq[pc][s1] ^ Zobrist::psq[pc][s2] ^ Zobrist::side;
                    int  i    = H1(key);
                    while (true) {
                        std::swap(cuckoo[i], key);
                        std::swap(cuckooMove[i], move);
                        if (move == Move::none())  // Arrived at empty slot?
                            break;
                        i = (i == H1(key)) ? H2(key) : H1(key);  // Push victim to alternative slot
                    }
                    count++;
                }
    std::cout << count << " reversible moves in cuckoo hash\n";
}

// Initializes the position object with the given FEN string.
// This function is not very robust - make sure that input FENs are correct,
// this is assumed to be the responsibility of the GUI.
Position& Position::set(const string& fenStr, StateInfo* si) {
    /*
   A FEN string defines a particular position using only the ASCII character set.

   A FEN string contains six fields separated by a space. The fields are:

   1) Piece placement (from white's perspective). Each rank is described, starting
      with rank 8 and ending with rank 1. Within each rank, the contents of each
      square are described from file A through file H. Following the Standard
      Algebraic Notation (SAN), each piece is identified by a single letter taken
      from the standard English names. White pieces are designated using upper-case
      letters ("PNBRQK") whilst Black uses lowercase ("pnbrqk"). Blank squares are
      noted using digits 1 through 8 (the number of blank squares), and "/"
      separates ranks.

   2) Active color. "w" means white moves next, "b" means black.

   3) NO Castling 4) NO En passant

   5) Halfmove clock. This is the number of halfmoves since the last pawn advance
      or capture. This is used to determine if a draw can be claimed under the
      fifty-move rule.

   6) Fullmove number. The number of the full move. It starts at 1, and is
      incremented after Black's move.
*/
    unsigned char      token;
    size_t             idx;
    Square             sq = SQ_A4;
    std::istringstream ss(fenStr);

    std::memset(this, 0, sizeof(Position));
    std::memset(si, 0, sizeof(StateInfo));
    st = si;

    ss >> std::noskipws;

    // 1. Piece placement
    while ((ss >> token) && !isspace(token)) {
        if (isdigit(token))
            sq += (token - '0') * EAST;  // Advance the given number of files

        else if (token == '/')
            sq += 2 * SOUTH;

        else if ((idx = PieceToChar.find(token)) != string::npos) {
            put_piece(Piece(idx), sq);
            ++sq;
        }
    }

    // 1.5. Pocket pieces (optional, format: [black_pocket] [white_pocket])
    // Skip whitespace and check for pocket info
    while (ss.peek() == ' ') ss.ignore();
    if (ss.peek() == '[') {
        // Parse black pocket
        ss.ignore();  // skip '['
        while (ss >> token && token != ']') {
            if (token >= 'a' && token <= 'z') {
                // Convert lowercase to piece type and add to black pocket
                PieceType pt = PieceType(token - 'a' + PAWN);
                if (pt >= PAWN && pt <= WAZIR) pockets[BLACK].inc(pt);
            }
        }

        // Parse white pocket
        while (ss.peek() == ' ') ss.ignore();
        if (ss.peek() == '[') {
            ss.ignore();  // skip '['
            while (ss >> token && token != ']') {
                if (token >= 'A' && token <= 'Z') {
                    // Convert uppercase to piece type and add to white pocket
                    PieceType pt = PieceType(token - 'A' + PAWN);
                    if (pt >= PAWN && pt <= WAZIR) pockets[WHITE].inc(pt);
                }
            }
        }
    }

    // 2. Active color
    ss >> token;
    sideToMove = (token == 'w' ? WHITE : BLACK);
    ss >> token;

    // 5-6. Halfmove clock and fullmove number
    ss >> std::skipws >> gamePly;

    // Convert from fullmove starting from 1 to gamePly starting from 0,
    // handle also common incorrect FEN with fullmove = 0.
    gamePly = std::max(2 * (gamePly - 1), 0) + (sideToMove == BLACK);
    set_state();

    assert(pos_is_ok());

    return *this;
}

// Returns a FEN representation of the position. In case of
// Chess960 the Shredder-FEN notation is used. This is mainly a debugging function.
string Position::fen() const {
    int                emptyCnt;
    std::ostringstream ss;

    for (Rank r = RANK_4; r >= RANK_1; --r) {
        for (File f = FILE_A; f <= FILE_D; ++f) {
            for (emptyCnt = 0; f <= FILE_D && empty(make_square(f, r)); ++f) ++emptyCnt;

            if (emptyCnt) ss << emptyCnt;

            if (f <= FILE_D) ss << PieceToChar[piece_on(make_square(f, r))];
        }

        if (r > RANK_1) ss << '/';
    }

    // Add pocket information if not empty
    std::string blackPocket = pockets[BLACK].to_string(BLACK);
    std::string whitePocket = pockets[WHITE].to_string(WHITE);
    if (!blackPocket.empty() || !whitePocket.empty()) {
        ss << " [" << blackPocket << "] [" << whitePocket << "]";
    }

    ss << (sideToMove == WHITE ? " w " : " b ");

    ss << " " << 1 + (gamePly - (sideToMove == BLACK)) / 2;

    return ss.str();
}

// Sets king attacks to detect if a move gives check
void Position::set_check_info() const {
    update_slider_blockers(WHITE);
    update_slider_blockers(BLACK);

    Square ksq = square<KING>(~sideToMove);

    st->checkSquares[PAWN] = attacks_bb<PAWN>(ksq, ~sideToMove);

    // For HORSE, squares that give check depend on the occupancy of the leg
    // adjacent to the HORSE origin, not the king square. Build reverse-attacker
    // set by filtering pseudo-origins with an empty leg square in current occupancy.
    {
        Bitboard result     = 0;
        Bitboard candidates = attacks_bb<HORSE>(ksq);  // pseudo origins around king
        Bitboard occ        = pieces();

        while (candidates) {
            Square   origin = pop_lsb(candidates);
            Bitboard leg    = horse_leg_bb(origin, ksq);
            if (!(occ & leg)) result |= square_bb(origin);
        }

        st->checkSquares[HORSE] = result;
    }
    st->checkSquares[FERZ]  = attacks_bb<FERZ>(ksq);
    st->checkSquares[WAZIR] = attacks_bb<WAZIR>(ksq);
    st->checkSquares[KING]  = 0;
}

// Computes the hash keys of the position, and other
// data that once computed is updated incrementally as moves are made.
// The function is only used when a new position is set up
void Position::set_state() const {
    st->key        = 0;
    st->checkersBB = attackers_to(square<KING>(sideToMove)) & pieces(~sideToMove);

    set_check_info();

    for (Bitboard b = pieces(); b;) {
        Square s  = pop_lsb(b);
        Piece  pc = piece_on(s);
        st->key ^= Zobrist::psq[pc][s];
    }

    // Add pocket contents to hash
    for (Color c = WHITE; c <= BLACK; ++c)
        for (PieceType pt = PAWN; pt <= WAZIR; ++pt)
            st->key ^= Zobrist::pocket[c][pt][pockets[c].count(pt)];

    if (sideToMove == BLACK) st->key ^= Zobrist::side;
}

// Computes a bitboard of all pieces which attack a given square.
// Slider attacks use the occupied bitboard to indicate occupancy.
Bitboard Position::attackers_to(Square s, Bitboard occupied) const {
    // HORSE needs special reverse handling: whether a HORSE on origin attacks s
    // depends on the occupancy of its leg adjacent to the origin square.
    Bitboard horseAttackers = 0;
    {
        Bitboard candidates = attacks_bb<HORSE>(s);  // pseudo origins around s
        candidates &= pieces(HORSE);

        while (candidates) {
            Square origin = pop_lsb(candidates);
            if (!(occupied & horse_leg_bb(origin, s))) horseAttackers |= square_bb(origin);
        }
    }

    return (attacks_bb<FERZ>(s) & pieces(FERZ)) | (attacks_bb<WAZIR>(s) & pieces(WAZIR)) |
           (attacks_bb<PAWN>(s, BLACK) & pieces(WHITE, PAWN)) |
           (attacks_bb<PAWN>(s, WHITE) & pieces(BLACK, PAWN)) | horseAttackers |
           (attacks_bb<KING>(s) & pieces(KING));
}

bool Position::attackers_to_exist(Square s, Bitboard occupied, Color c) const {
    // Pawns: squares from which a pawn of color c would attack s
    if (attacks_bb<PAWN>(s, ~c) & pieces(c, PAWN)) {
        return true;
    }

    // Horse (xiangqi): reverse-origin with leg-block check against occupied
    {
        Bitboard candidates = attacks_bb<HORSE>(s) & pieces(c, HORSE);
        while (candidates) {
            Square origin = pop_lsb(candidates);
            if (!(occupied & horse_leg_bb(origin, s))) {
                return true;
            }
        }
    }

    // Ferz: diagonal king-step (no occupancy needed)
    if (attacks_bb<FERZ>(s) & pieces(c, FERZ)) {
        return true;
    }

    // Wazir: orthogonal king-step (no occupancy needed)
    if (attacks_bb<WAZIR>(s) & pieces(c, WAZIR)) {
        return true;
    }

    // King: standard king steps (no occupancy needed)
    if (attacks_bb<KING>(s) & pieces(c, KING)) {
        return true;
    }

    return false;
}

inline Bitboard Position::pinners_on_leg(Color us, Square legSq) const {
    Bitboard res = 0;
    Bitboard ps  = pinners(us);
    Square   ksq = square<KING>(us);

    while (ps) {
        Square h = pop_lsb(ps);
        if (horse_leg_bb(h, ksq) & square_bb(legSq)) res |= square_bb(h);
    }
    return res;
}

// Tests whether a pseudo-legal move is legal
bool Position::legal(Move m) const {
    assert(m.is_ok());

    Color  us   = sideToMove;
    Square from = m.from_sq();
    Square to   = m.to_sq();

    assert(color_of(moved_piece(m)) == us);
    assert(piece_on(square<KING>(us)) == make_piece(us, KING));

    // King moves: destination must not be attacked
    if (type_of(piece_on(from)) == KING) return !attackers_to_exist(to, pieces() ^ from, ~us);

    // If our moving piece is *not* a blocker of a HORSE attack on our king, any pseudo-legal move
    // is fine
    if (!(blockers_for_king(us) & square_bb(from))) return true;

    // Our piece sits on a HORSE leg => it is pinned. Only legal option: capture the corresponding
    // pinner, and only if no second HORSE uses the same leg.
    Piece toPc = piece_on(to);
    if (type_of(toPc) != HORSE || color_of(toPc) == us) return false;

    // Filter to pinners that use *this* leg
    Bitboard legPinners = pinners_on_leg(us, from);

    // Must capture a pinner on this leg, and that set must be singleton
    if (!(legPinners & square_bb(to))) return false;

    return !more_than_one(legPinners);
}

// Calculates st->blockersForKing[c],
// which store respectively the pieces preventing king of color c from being in check
void Position::update_slider_blockers(Color c) const {
    Square ksq = square<KING>(c);

    st->blockersForKing[c] = 0;

    // Enemy horses that geometrically attack ksq (reverse pseudo)
    Bitboard snipers = (attacks_bb<HORSE>(ksq) & pieces(HORSE)) & pieces(~c);

    // Ignore snipers themselves in occupancy
    Bitboard occupancy = pieces() ^ snipers;

    while (snipers) {
        Square sniperSq = pop_lsb(snipers);

        // Leg square required for this horse to attack the king
        Bitboard leg = horse_leg_bb(sniperSq, ksq);

        // If that leg square is occupied by piece, it's a blocker
        Bitboard b = leg & occupancy;
        if (b) {
            st->blockersForKing[c] |= b;
            st->pinners[c] |= sniperSq;
        }
    }
}

// Tests whether a pseudo-legal move gives a check
bool Position::gives_check(Move m) const {
    assert(m.is_ok());
    assert(color_of(moved_piece(m)) == sideToMove);

    Square from = m.from_sq();
    Square to   = m.to_sq();

    PieceType movedPieceType = m.type_of() != DROP ? type_of(piece_on(from)) : m.drop_piece();

    // Is there a direct check?
    if (check_squares(movedPieceType) & to) return true;

    // Is there a discovered check?
    if (blockers_for_king(~sideToMove) & from) return true;

    switch (m.type_of()) {
        case NORMAL:
            return false;

        case PROMOTION:
            return attacks_bb(m.promotion_type(), to, pieces() ^ from) & pieces(~sideToMove, KING);

        case DROP:
            return m.drop_piece() == PAWN
                       ? attacks_bb<PAWN>(to, sideToMove) & pieces(~sideToMove, KING)
                       : attacks_bb(m.drop_piece(), to, pieces()) & pieces(~sideToMove, KING);

        default:
            return false;
    }
}

// Makes a move, and saves all information necessary
// to a StateInfo object. The move is assumed to be legal. Pseudo-legal
// moves should be filtered out before this function is called.
// If a pointer to the TT table is passed, the entry for the new position
// will be prefetched
void Position::do_move(Move m, StateInfo& newSt, bool givesCheck) {
    assert(m.is_ok());
    assert(&newSt != st);

    Key k = st->key ^ Zobrist::side;

    // Copy some fields of the old state to our new StateInfo object except the
    // ones which are going to be recalculated from scratch anyway and then switch
    // our state pointer to point to the new (ready to be updated) state.
    std::memcpy(&newSt, st, offsetof(StateInfo, key));
    newSt.previous = st;
    st             = &newSt;

    // Increment ply counters. In particular, rule50 will be reset to zero later on
    // in case of a capture or a pawn move.
    ++gamePly;

    Color  us       = sideToMove;
    Color  them     = ~us;
    Square from     = m.from_sq();
    Square to       = m.to_sq();
    Piece  pc       = m.type_of() != DROP ? piece_on(from) : make_piece(sideToMove, m.drop_piece());
    Piece  captured = piece_on(to);

    assert(color_of(pc) == us);
    assert(captured == NO_PIECE || color_of(captured) == them);
    assert(type_of(captured) != KING);

    if (captured) {
        // Add piece to pocket
        // Check if captured piece is a promoted pawn
        st->capturedWasPromotedPawn = is_promoted_pawn(to);
        if (st->capturedWasPromotedPawn) {
            pocket_add_captured(PAWN, us);
            clear_promoted(to);

        } else {
            pocket_add_captured(type_of(captured), us);
        }

        // Update board and piece lists
        remove_piece(to);

        k ^= Zobrist::psq[captured][to];
    }

    // Update hash key for board piece movement
    if (m.type_of() != DROP) k ^= Zobrist::psq[pc][from] ^ Zobrist::psq[pc][to];

    // If drop just put the piece
    if (m.type_of() != DROP) {
        move_piece(from, to);
    } else {
        put_piece(pc, to);
        // Update pocket and hash for drop: decrement pocket count
        PieceType dpt = m.drop_piece();
        PieceType pt  = dpt;
        int       cnt = pockets[us].count(pt);
        // Toggle out previous count, then toggle in new count-1
        k ^= Zobrist::pocket[us][pt][cnt];
        pocket_remove(dpt, us);
        k ^= Zobrist::pocket[us][pt][cnt - 1];
    }

    // If the moving piece is a pawn do some special extra work
    if (type_of(pc) == PAWN) {
        if (m.type_of() == PROMOTION) {
            Piece     promotion     = make_piece(us, m.promotion_type());
            PieceType promotionType = type_of(promotion);

            assert(relative_rank(us, to) == RANK_4);
            assert(type_of(promotion) >= HORSE && type_of(promotion) <= WAZIR);

            remove_piece(to);
            put_piece(promotion, to);
            track_promoted_pawn(to);

            // if (type_of(piece_on(to)) == PAWN) {
            //     printf("\n\n\nPROMOTED TO PAWN!!!!\n\n\n");
            // }

            // Update hash keys
            // Zobrist::psq[pc][to] is zero, so we don't need to clear it
            k ^= Zobrist::psq[promotion][to];
            // Defensive: ensure board holds the promoted piece (not a pawn)
            if (type_of(piece_on(to)) != promotionType) {
                assert(0);
                // Replace whatever ended up on 'to' with the correct promoted piece
                remove_piece(to);
                put_piece(promotion, to);
            }
        }
    }

    // Set capture piece
    st->capturedPiece = captured;

    // Calculate checkers bitboard (if move gives check)
    st->checkersBB = givesCheck ? attackers_to(square<KING>(them)) & pieces(us) : 0;

    sideToMove = ~sideToMove;

    // Update king attacks used for fast check detection
    set_check_info();

    // Update the key with the final value
    st->key = k;

    // Calculate the repetition info. It is the ply distance from the previous
    // occurrence of the same position, negative in the 3-fold case, or zero
    // if the position was not repeated.
    st->repetition = 0;

    StateInfo* stp = st->previous->previous;
    for (int i = 4; i <= gamePly; i += 2) {
        stp = stp->previous->previous;
        if (stp->key == st->key) {
            st->repetition = stp->repetition ? -i : i;
            break;
        }
    }
    assert(board[SQ_B4] != W_PAWN);
    assert(pos_is_ok());
    assert(from != SQ_NONE);
}

// Unmakes a move. When it returns, the position should
// be restored to exactly the same state as before the move was made.
void Position::undo_move(Move m) {
    assert(m.is_ok());

    // Reverse side to move and ply first
    sideToMove = ~sideToMove;

    Color  us   = sideToMove;
    Color  them = ~us;
    Square from = m.from_sq();
    Square to   = m.to_sq();
    Piece  pc   = piece_on(to);

    assert(type_of(st->capturedPiece) != KING);

    // If it was a promotion, revert the promoted piece back to a pawn
    if (m.type_of() == PROMOTION) {
        assert(relative_rank(us, to) == RANK_4);
        assert(type_of(pc) == m.promotion_type());
        assert(type_of(pc) >= HORSE && type_of(pc) <= WAZIR);

        remove_piece(to);
        pc = make_piece(us, PAWN);
        put_piece(pc, to);
        clear_promoted(to);
    }

    // If it was a drop, remove the dropped piece from the board and restore pocket
    if (m.type_of() == DROP) {
        PieceType dpt = m.drop_piece();
        remove_piece(to);
        // Restore pocket count and hash will be restored by state rewind
        pocket_add_captured(dpt, us);  // add back to pocket
    } else {
        // Otherwise it was a board move: move the piece back
        move_piece(to, from);
    }

    // If there was a capture, restore the captured piece on 'to'
    if (st->capturedPiece) {
        PieceType restoredType;
        if (st->capturedWasPromotedPawn) {
            restoredType = PAWN;
            track_promoted_pawn(to);  // re-tag the square as promoted pawn again
        } else {
            restoredType = type_of(st->capturedPiece);
        }
        put_piece(make_piece(them, type_of(st->capturedPiece)), to);
        // Remove from pocket of side who captured in do_move (which is 'us' now)
        pocket_remove(restoredType, us);
    }

    // Restore previous state pointer and ply
    st = st->previous;
    --gamePly;

    assert(pos_is_ok());
}

// Tests whether the position is drawn by repetition. It does not detect stalemates.
bool Position::is_draw(int ply) const { return is_repetition(ply); }

// Return a draw score if a position repeats once earlier but strictly
// after the root, or repeats twice before or at the root.
bool Position::is_repetition(int ply) const { return st->repetition && st->repetition < ply; }

bool Position::is_threefold_game() const {
    // Count occurrences of current key since last irreversible.
    int              cnt = 0;
    const StateInfo* s   = st;
    const auto       key = st->key;

    while (s) {
        if (s->key == key) {
            if (++cnt >= 3) return true;
        }
        s = s->previous;
    }
    return false;
}

// Performs some consistency checks for the position object
// and raise an assert if something wrong is detected.
// This is meant to be helpful when debugging.
bool Position::pos_is_ok() const {
    constexpr bool Fast = true;  // Quick (default) or full check?

    if ((sideToMove != WHITE && sideToMove != BLACK) || piece_on(square<KING>(WHITE)) != W_KING ||
        piece_on(square<KING>(BLACK)) != B_KING)
        assert(0 && "pos_is_ok: Default");

    if (Fast) return true;

    if (pieceCount[W_KING] != 1 || pieceCount[B_KING] != 1 ||
        attackers_to_exist(square<KING>(~sideToMove), pieces(), sideToMove)) {
        printf("King in check:\n%s\n",
               Bitboards::pretty(square_bb(square<KING>(~sideToMove))).c_str());

        assert(0 && "pos_is_ok: Kings");
    }

    if (pieceCount[W_PAWN] > 2 || pieceCount[B_PAWN] > 2) assert(0 && "pos_is_ok: Pawns");

    if ((pieces(WHITE) & pieces(BLACK)) || (pieces(WHITE) | pieces(BLACK)) != pieces() ||
        popcount(pieces(WHITE)) > 9 || popcount(pieces(BLACK)) > 9)
        assert(0 && "pos_is_ok: Bitboards");

    for (PieceType p1 = PAWN; p1 <= KING; ++p1)
        for (PieceType p2 = PAWN; p2 <= KING; ++p2)
            if (p1 != p2 && (pieces(p1) & pieces(p2))) assert(0 && "pos_is_ok: Bitboards");

    for (Piece pc : Pieces)
        if (pieceCount[pc] != popcount(pieces(color_of(pc), type_of(pc))) ||
            pieceCount[pc] != std::count(board, board + SQUARE_NB, pc))
            assert(0 && "pos_is_ok: Pieces");
    return true;
}

}  // namespace tiny

#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#include <cassert>
#include <cstdint>
#include <ostream>
#include <string>

namespace tiny {

constexpr int MAX_MOVES = 128;
constexpr int MAX_PLY   = 256;

using Bitboard = uint16_t;  // 16-bit board
using Key      = uint64_t;

enum Color : uint8_t { WHITE, BLACK, COLOR_NB };

// Piece types: pack into 8 slots (index 0 reserved)
enum PieceType : std::int8_t {
    NO_PIECE_TYPE = 0,
    PAWN          = 1,  // P
    HORSE         = 2,  // U (xiangqi horse)
    FERZ          = 3,  // F
    WAZIR         = 4,  // W
    KING          = 5,  // K

    ALL_PIECES    = 0,
    PIECE_TYPE_NB = 8
};

// Pieces: color*8 + type; keep 16 total to preserve XOR-8 tricks
enum Piece : std::int8_t {
    NO_PIECE = 0,

    // White  (indices 1..7)
    W_PAWN  = PAWN,
    W_HORSE = HORSE,
    W_FERZ  = FERZ,
    W_WAZIR = WAZIR,
    W_KING  = KING,

    // Black  (add 8)
    B_PAWN  = PAWN + 8,
    B_HORSE = HORSE + 8,
    B_FERZ  = FERZ + 8,
    B_WAZIR = WAZIR + 8,
    B_KING  = KING + 8,

    PIECE_NB = 16
};

using Value = int;

constexpr Value PawnValue  = 100;
constexpr Value HorseValue = 200;
constexpr Value FerzValue  = 200;
constexpr Value WazirValue = 250;

inline constexpr Value type_value(PieceType pt) {
    switch (pt) {
        case PAWN:
            return PawnValue;
        case HORSE:
            return HorseValue;
        case FERZ:
            return FerzValue;
        case WAZIR:
            return WazirValue;
        default:
            return 0;  // KING, NO_PIECE_TYPE
    }
}

inline constexpr Value piece_value(Piece p) {
    switch (p) {
        case W_PAWN:
            return +PawnValue;
        case W_HORSE:
            return +HorseValue;
        case W_FERZ:
            return +FerzValue;
        case W_WAZIR:
            return +WazirValue;
        case B_PAWN:
            return -PawnValue;
        case B_HORSE:
            return -HorseValue;
        case B_FERZ:
            return -FerzValue;
        case B_WAZIR:
            return -WazirValue;
        default:
            return 0;  // kings and empty
    }
}

constexpr Value START_MATERIAL = PawnValue + HorseValue + FerzValue + WazirValue;
constexpr Value EVAL_MAX       = (HorseValue + FerzValue + WazirValue + WazirValue) * 2;

constexpr Value VALUE_MATE     = 2200;
constexpr Value VALUE_ZERO     = 0;
constexpr Value VALUE_DRAW     = 0;
constexpr Value VALUE_NONE     = 2202;
constexpr Value VALUE_INFINITE = 2201;

enum Square : int8_t {
    SQ_A1,
    SQ_B1,
    SQ_C1,
    SQ_D1,  // 0..3
    SQ_A2,
    SQ_B2,
    SQ_C2,
    SQ_D2,  // 4..7
    SQ_A3,
    SQ_B3,
    SQ_C3,
    SQ_D3,  // 8..11
    SQ_A4,
    SQ_B4,
    SQ_C4,
    SQ_D4,  // 12..15
    SQ_NONE,

    SQUARE_ZERO = 0,
    SQUARE_NB   = 16
};

enum DirectionIndex : uint8_t { DIR_N = 0, DIR_E = 1, DIR_S = 2, DIR_W = 3, DIR_NB = 4 };

enum Direction : int8_t {
    NORTH = 4,
    EAST  = 1,
    SOUTH = -NORTH,
    WEST  = -EAST,

    NORTH_EAST = NORTH + EAST,
    SOUTH_EAST = SOUTH + EAST,
    SOUTH_WEST = SOUTH + WEST,
    NORTH_WEST = NORTH + WEST
};

enum File : int8_t { FILE_A, FILE_B, FILE_C, FILE_D, FILE_NB };
enum Rank : int8_t { RANK_1, RANK_2, RANK_3, RANK_4, RANK_NB };

#define ENABLE_INCR_OPERATORS_ON(T)                             \
    constexpr T& operator++(T& d) { return d = T(int(d) + 1); } \
    constexpr T& operator--(T& d) { return d = T(int(d) - 1); }

ENABLE_INCR_OPERATORS_ON(PieceType)
ENABLE_INCR_OPERATORS_ON(Square)
ENABLE_INCR_OPERATORS_ON(File)
ENABLE_INCR_OPERATORS_ON(Rank)
ENABLE_INCR_OPERATORS_ON(Color)

#undef ENABLE_INCR_OPERATORS_ON

constexpr Direction operator+(Direction d1, Direction d2) { return Direction(int(d1) + int(d2)); }
constexpr Direction operator*(int i, Direction d) { return Direction(i * int(d)); }

// Additional operators to add a Direction to a Square
constexpr Square  operator+(Square s, Direction d) { return Square(int(s) + int(d)); }
constexpr Square  operator-(Square s, Direction d) { return Square(int(s) - int(d)); }
constexpr Square& operator+=(Square& s, Direction d) { return s = s + d; }
constexpr Square& operator-=(Square& s, Direction d) { return s = s - d; }

// Toggle color
constexpr Color operator~(Color c) { return Color(c ^ BLACK); }

constexpr Square make_square(File f, Rank r) { return Square((r << 2) + f); }

constexpr bool is_ok(Square s) { return s >= SQ_A1 && s <= SQ_D4; }

constexpr Piece operator~(Piece pc) { return Piece(pc ^ 8); }

constexpr Piece make_piece(Color c, PieceType pt) { return Piece((c << 3) + pt); }

constexpr PieceType type_of(Piece pc) { return PieceType(pc & 7); }

constexpr Color color_of(Piece pc) {
    assert(pc != NO_PIECE);
    return Color(pc >> 3);
}

constexpr File file_of(Square s) { return File(s & 3); }

constexpr Rank rank_of(Square s) { return Rank(s >> 2); }

constexpr Rank relative_rank(Color c, Rank r) {
    // For White: same rank.
    // For Black: flipped vertically (0↔3, 1↔2)
    return c == WHITE ? r : Rank((RANK_NB - 1) - int(r));
}

constexpr Rank relative_rank(Color c, Square s) { return relative_rank(c, rank_of(s)); }

constexpr Direction pawn_push(Color c) { return c == WHITE ? NORTH : SOUTH; }

class Pocket {
   public:
    constexpr Pocket() : data(0) {}

    // Read 2-bit counter for a piece type (PAWN..WAZIR)
    constexpr uint8_t count(PieceType pt) const {
        return (pt >= PAWN && pt <= WAZIR) ? uint8_t((data >> (2 * (pt - PAWN))) & 0x3) : 0;
    }

    // Set 2-bit counter clamped to [0,2]
    constexpr void set_count(PieceType pt, uint8_t c) {
        if (pt < PAWN || pt > WAZIR) return;
        const uint8_t shift = uint8_t(2 * (pt - PAWN));
        const uint8_t mask  = uint8_t(0x3u << shift);
        const uint8_t val   = uint8_t((c > 2 ? 2 : c) << shift);
        data                = uint8_t((data & ~mask) | val);
    }

    // Increment up to 2
    constexpr void inc(PieceType pt) {
        const uint8_t c = count(pt);
        if (c < 2) set_count(pt, uint8_t(c + 1));
    }

    // Decrement down to 0
    constexpr void dec(PieceType pt) {
        const uint8_t c = count(pt);
        if (c > 0) set_count(pt, uint8_t(c - 1));
    }

    // Render pocket as concatenated piece codes in order P,H,F,W; color selects case
    inline std::string to_string(Color c) const {
        std::string s;
        const char  codesUpper[4] = {'P', 'H', 'F', 'W'};
        const char  codesLower[4] = {'p', 'h', 'f', 'w'};
        const char* codes         = (c == WHITE ? codesUpper : codesLower);
        for (PieceType pt = PAWN; pt <= WAZIR; ++pt) {
            for (int k = 0, n = count(pt); k < n; ++k) s.push_back(codes[pt - PAWN]);
        }
        return s;
    }

   protected:
    uint8_t data;
};

// Based on a congruential pseudo-random number generator
constexpr Key make_key(uint64_t seed) {
    return seed * 6364136223846793005ULL + 1442695040888963407ULL;
}

enum MoveType { NORMAL, PROMOTION = 1 << 14, DROP = 2 << 14 };

// Stockfish-compatible 16-bit move container.
class Move {
   public:
    Move() = default;  // Defaulted ctor: leaves 'data' uninitialized on purpose (matches SF style).

    // Construct directly from the packed 16-bit payload.
    constexpr explicit Move(std::uint16_t d) : data(d) {}

    // Normal move: from -> to, with type = NORMAL
    constexpr Move(Square from, Square to) : data((std::uint16_t(from) << 6) + std::uint16_t(to)) {}

    // Generic factory for SPECIAL moves (PROMOTION/DROP).
    // Matches Stockfish's pattern: Move::make<PROMOTION>(from, to, pt);
    // For DROP, pass from == to (conventionally) or any value; only 'to' matters.
    template <MoveType T>
    static constexpr Move make(Square from, Square to, PieceType pt) {
        static_assert(T == PROMOTION || T == DROP, "Only PROMOTION or DROP are valid here");
        // Map aux payload into bits 12..13:
        //  - PROMOTION: encode (pt - HORSE) in [0..3]  (HORSE, FERZ, WAZIR; 4th value
        //  reserved/invalid)
        //  - DROP:      encode (pt - PAWN)  in [0..3]  (PAWN, HORSE, FERZ, WAZIR)
        const std::uint16_t aux =
            (T == PROMOTION) ? (std::uint16_t(std::uint16_t(pt) - std::uint16_t(HORSE)) << 12)
                             : (std::uint16_t(std::uint16_t(pt) - std::uint16_t(PAWN)) << 12);

        return Move(std::uint16_t(T) + aux + (std::uint16_t(from) << 6) + std::uint16_t(to));
    }

    // Accessors
    constexpr Square from_sq() const {
        assert(is_ok());
        return Square((data >> 6) &
                      0x3F);  // mask 6 bits (compatible with 0..63; we only use 0..15)
    }

    constexpr Square to_sq() const {
        assert(is_ok());
        return Square(data & 0x3F);
    }

    // Low 12 bits (to + from) — handy for TT indexing or move ordering keys.
    constexpr int from_to() const { return data & 0x0FFF; }

    // Special move kind.
    constexpr MoveType type_of() const { return MoveType(data & (3u << 14)); }

    // Promotion target: valid only if type_of() == PROMOTION.
    // Decodes bits 12..13 with HORSE as base.
    constexpr PieceType promotion_type() const {
        assert(type_of() == PROMOTION);
        return PieceType(((data >> 12) & 0x3) + std::uint16_t(HORSE));
    }

    // Drop piece: valid only if type_of() == DROP.
    // Decodes bits 12..13 with PAWN as base.
    constexpr PieceType drop_piece() const {
        assert(type_of() == DROP);
        return PieceType(((data >> 12) & 0x3) + std::uint16_t(PAWN));
    }

    // “Is a real move-like value” (not none/null sentinels). Matches Stockfish’s semantics.
    constexpr bool is_ok() const { return data != none().data && data != null().data; }

    // Null & none sentinels.
    static constexpr Move null() {
        return Move(65);
    }  // 0b0000'0000'0100'0001 (from=1,to=1), flags=0
    static constexpr Move none() { return Move(0); }

    // Comparisons / truthiness
    constexpr bool     operator==(const Move& m) const { return data == m.data; }
    constexpr bool     operator!=(const Move& m) const { return data != m.data; }
    constexpr explicit operator bool() const { return data != 0; }  // false only for Move::none()

    // Raw bits (for hashing, serialization, etc.)
    constexpr std::uint16_t raw() const { return data; }

    struct MoveHash {
        std::size_t operator()(const Move& m) const { return make_key(m.data); }
    };

    // Allow streaming via free function operator<< defined in namespace scope
    friend inline std::ostream& operator<<(std::ostream& os, const Move& m);

   protected:
    std::uint16_t data;
};

inline std::string to_string(Move m);
inline const char* pt_code(PieceType pt);
inline void        square_to_cstr(Square s, char out[3]);

inline std::ostream& operator<<(std::ostream& os, const Move& m) { return os << to_string(m); }

inline std::string to_string(Move m) {
    char from[3], to[3];
    square_to_cstr(m.from_sq(), from);
    square_to_cstr(m.to_sq(), to);

    switch (m.type_of()) {
        case NORMAL:
            return std::string(from) + to;  // e.g. "a2a3"
        case PROMOTION:
            return std::string(from) + to + "=" + pt_code(m.promotion_type());  // "a3a4=H"
        case DROP:
            return std::string(pt_code(m.drop_piece())) + "@" + to;  // "P@b2"
        default:
            return "??";
    }
}

inline const char* pt_code(PieceType pt) {
    switch (pt) {
        case PAWN:
            return "P";
        case HORSE:
            return "H";
        case FERZ:
            return "F";
        case WAZIR:
            return "W";
        case KING:
            return "K";
        default:
            return "?";
    }
}

inline void square_to_cstr(Square s, char out[3]) {
    if (s < 0 || s >= SQUARE_NB) {
        out[0] = out[1] = '-';
        out[2]          = 0;
        return;
    }
    out[0] = char('a' + (int(s) % 4));
    out[1] = char('1' + (int(s) / 4));
    out[2] = 0;
}

}  // namespace tiny

#endif  // #ifndef TYPES_H_INCLUDED
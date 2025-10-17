# tinyhouse.py
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, IntEnum

# --------- Core enums / constants (mirror C++) ---------


class Color(IntEnum):
    WHITE = 0
    BLACK = 1
    COLOR_NB = 2

    def other(self) -> "Color":
        return Color(self ^ 1)

    def to_string(self) -> str:
        match self:
            case Color.WHITE:
                return "white"
            case Color.BLACK:
                return "black"
            case _:
                return "unknown"


class PieceType(IntEnum):
    NO_PIECE_TYPE = 0
    PAWN = 1  # P
    HORSE = 2  # U (xiangqi horse)
    FERZ = 3  # F
    WAZIR = 4  # W
    KING = 5  # K

    ALL_PIECES = 0
    PIECE_TYPE_NB = 8

    def to_string(self) -> str:
        match self:
            case PieceType.PAWN:
                return "pawn"
            case PieceType.HORSE:
                return "horse"
            case PieceType.FERZ:
                return "ferz"
            case PieceType.WAZIR:
                return "wazir"
            case PieceType.KING:
                return "king"
            case PieceType.NO_PIECE_TYPE:
                return "none"
            case _:
                return "unknown"


class Piece(IntEnum):
    NO_PIECE = 0

    # White
    W_PAWN = PieceType.PAWN
    W_HORSE = PieceType.HORSE
    W_FERZ = PieceType.FERZ
    W_WAZIR = PieceType.WAZIR
    W_KING = PieceType.KING

    # Black (add 8)
    B_PAWN = PieceType.PAWN + 8
    B_HORSE = PieceType.HORSE + 8
    B_FERZ = PieceType.FERZ + 8
    B_WAZIR = PieceType.WAZIR + 8
    B_KING = PieceType.KING + 8

    PIECE_NB = 16

    def to_string(self) -> str:
        if self == Piece.NO_PIECE:
            return "none"
        color = "white" if self < 8 else "black"
        piece_type = PieceType(self & 7).to_string()
        return f"{color} {piece_type}"


class Square(IntEnum):
    (
        SQ_A1,
        SQ_B1,
        SQ_C1,
        SQ_D1,
        SQ_A2,
        SQ_B2,
        SQ_C2,
        SQ_D2,
        SQ_A3,
        SQ_B3,
        SQ_C3,
        SQ_D3,
        SQ_A4,
        SQ_B4,
        SQ_C4,
        SQ_D4,
        SQ_NONE,
    ) = range(17)

    def to_string(self) -> str:
        if self == Square.SQ_NONE:
            return "none"
        files = "abcd"
        ranks = "1234"
        file_idx = int(self) % 4
        rank_idx = int(self) // 4
        return f"{files[file_idx]}{ranks[rank_idx]}"


SQUARE_ZERO = Square.SQ_A1
SQUARE_NB = 16


class File(IntEnum):
    FILE_A, FILE_B, FILE_C, FILE_D, FILE_NB = range(5)


class Rank(IntEnum):
    RANK_1, RANK_2, RANK_3, RANK_4, RANK_NB = range(5)


def make_square(f: File, r: Rank) -> Square:
    return Square((int(r) << 2) + int(f))


def type_of(p: Piece) -> PieceType:
    return PieceType(int(p) & 7)


def color_of(p: Piece) -> Color:
    assert p != Piece.NO_PIECE
    return Color(int(p) >> 3)


def file_of(s: Square) -> File:
    return File(int(s) & 3)


def rank_of(s: Square) -> Rank:
    return Rank(int(s) >> 2)


def relative_rank(c: Color, r: Rank) -> Rank:
    return r if c == Color.WHITE else Rank((int(Rank.RANK_NB) - 1) - int(r))


def relative_rank_sq(c: Color, s: Square) -> Rank:
    return relative_rank(c, rank_of(s))


# --------- Move (16-bit, Stockfish-like) ---------


class MoveType(Enum):
    NORMAL = 0
    PROMOTION = 1
    DROP = 2


@dataclass
class Move:
    """Represents a move with its origin, destination, and type."""

    from_square: Square
    to_square: Square
    move_type: MoveType


# --------- Square / code helpers ---------


def square_to_str(s: Square) -> str:
    f = "abcd"[int(file_of(s))]
    r = "1234"[int(rank_of(s))]
    return f + r


def str_to_square(s: str) -> Square:
    s = s.strip()
    if len(s) != 2:
        return Square.SQ_NONE
    f = "abcd".find(s[0])
    r = "1234".find(s[1])
    if f < 0 or r < 0:
        return Square.SQ_NONE
    return make_square(File(f), Rank(r))


def pt_code(pt: PieceType) -> str:
    return {
        PieceType.PAWN: "P",
        PieceType.HORSE: "H",
        PieceType.FERZ: "F",
        PieceType.WAZIR: "W",
        PieceType.KING: "K",
    }.get(pt, "?")


def code_to_pt(ch: str) -> PieceType:
    c = ch.strip()
    if not c:
        return PieceType.NO_PIECE_TYPE
    c = c[0]
    return {
        "P": PieceType.PAWN,
        "H": PieceType.HORSE,
        "F": PieceType.FERZ,
        "W": PieceType.WAZIR,
        "K": PieceType.KING,
        "p": PieceType.PAWN,
        "h": PieceType.HORSE,
        "f": PieceType.FERZ,
        "w": PieceType.WAZIR,
        "k": PieceType.KING,
    }.get(c, PieceType.NO_PIECE_TYPE)


def piece_from_code(ch: str) -> Piece:
    pt = code_to_pt(ch)
    if pt == PieceType.NO_PIECE_TYPE:
        return Piece.NO_PIECE
    is_white = ch.isupper()
    base = int(pt)
    return Piece(base if is_white else base + 8)


def code_from_piece(p: Piece) -> str:
    pt = type_of(p)
    letter = pt_code(pt)
    return letter if color_of(p) == Color.WHITE else letter.lower()


FILES_BY_PIECE = {
    Piece.W_PAWN: "w_p.png",
    Piece.W_HORSE: "w_h.png",
    Piece.W_FERZ: "w_f.png",
    Piece.W_WAZIR: "w_w.png",
    Piece.W_KING: "w_k.png",
    Piece.B_PAWN: "b_p.png",
    Piece.B_HORSE: "b_h.png",
    Piece.B_FERZ: "b_f.png",
    Piece.B_WAZIR: "b_w.png",
    Piece.B_KING: "b_k.png",
}

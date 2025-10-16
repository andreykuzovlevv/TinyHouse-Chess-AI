
#define SDL_MAIN_USE_CALLBACKS 1

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3_image/SDL_image.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <deque>
#include <future>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "../include/colors.h"
#include "core/movegen.h"
#include "core/position.h"
#include "core/types.h"
#include "helpers.h"
#include "minmax/minmax.h"

using namespace tiny;
using namespace Colors;

// ---------- Config ----------
#define WINDOW_W 1420
#define WINDOW_H 1080

static const std::string START_FEN = "fhwk/3p/P3/KWHF w 1";

static const int charsize = SDL_DEBUG_TEXT_FONT_CHARACTER_SIZE;

// Board and UI layout constants (logical)
struct UIConf {
    // Layout constants (static, not computed dynamically)
    int boardSizePx = 600;  // square board edge
    int marginPx    = 24;
    int squarePx    = boardSizePx / 4;

    int pocketWidth = boardSizePx / 8;
    int rightPanel  = boardSizePx / 2;

    // Compute total content width = pocket + margin + board
    int totalContentWidth = pocketWidth + marginPx + boardSizePx + marginPx + rightPanel;

    // Center that content horizontally in window
    float startX = (WINDOW_W - totalContentWidth) / 2.0f;
    float startY = (WINDOW_H - boardSizePx) / 2.0f;

    // Pocket on the left
    SDL_FRect leftUIRect = {
        startX,              // x
        startY,              // y
        (float)pocketWidth,  // w
        (float)boardSizePx   // h (same as board)
    };

    // Board to the right of pocket (with margin)
    SDL_FRect boardRect = {
        startX + pocketWidth + marginPx,  // x
        startY,                           // y
        (float)boardSizePx,               // w
        (float)boardSizePx                // h
    };

    // Pocket on the left
    SDL_FRect rightUIRect = {
        boardRect.x + boardRect.w + marginPx,  // x
        startY,                                // y
        (float)rightPanel,                     // w
        (float)boardSizePx                     // h (same as board)
    };
};

// ---------- App/Game state ----------
enum class Phase { SideSelect, Playing, PromotionPick, GameOver };

enum class GameResult { None, CheckMate, Stalemate, Draw };

struct PromotionUI {
    // Visible when a user clicked a (from,to) that has multiple promotions.
    Square                   from = SQ_NONE;
    Square                   to   = SQ_NONE;
    std::vector<Move>        options;  // each is PROMOTION with chosen pt
    std::array<SDL_FRect, 3> rects{};  // up to 3 (HORSE, FERZ, WAZIR)
    bool                     visible = false;
};

struct LastMoveVis {
    std::optional<Square> from{};
    std::optional<Square> to{};
};

struct AsyncAI {
    bool                      thinking = false;
    std::future<SearchResult> fut;

    std::optional<Value> lastEval;  // numeric score from search_best_move
};

typedef struct {
    // SDL
    SDL_Window*   window   = nullptr;
    SDL_Renderer* renderer = nullptr;

    // Engine
    std::deque<StateInfo> states;
    Position              pos;
    std::vector<Move>     legalMoves;
    Color                 humanSide = WHITE;

    // Game flow
    Phase      phase        = Phase::SideSelect;
    bool       boardFlipped = false;  // in addition to side orientation; toggled by 'F'
    GameResult gameResult   = GameResult::None;
    Color      winner       = WHITE;  // when game over (checkmate) or stalemate winner

    // Selection
    std::optional<Square>    selectedSq{};
    std::optional<PieceType> selectedDropPiece{};  // when choosing to drop from pocket
    PromotionUI              promo{};
    LastMoveVis              lastMove{};

    // AI
    AsyncAI ai;
    int     searchDepth = 9;

    // Rendering
    UIConf                           ui{};
    std::array<SDL_Texture*, TEX_NB> textures{};
    bool texturesLoaded = false;  // if false, we fallback to primitive drawing

    // Timing
    Uint64 lastTicks = 0;
} AppState;

// ---------- Texture loading ----------
static SDL_Texture* load_texture(SDL_Renderer* r, const char* path) {
    SDL_Texture* texture  = NULL;
    char*        svg_path = NULL;

    SDL_asprintf(&svg_path, "%s%s", SDL_GetBasePath(),
                 path); /* allocate a string of the full file path */

    // Load the texture
    texture = IMG_LoadTexture(r, svg_path);
    if (!texture) {
        SDL_Log("Couldn't create static texture: %s", SDL_GetError());
    }

    SDL_free(svg_path); /* done with this, the file is loaded. */

    return texture;
}

static void load_all_textures(AppState* as) {
    const char* paths[TEX_NB] = {
        "assets/w_p.svg", "assets/w_h.svg", "assets/w_f.svg", "assets/w_w.svg", "assets/w_k.svg",
        "assets/b_p.svg", "assets/b_h.svg", "assets/b_f.svg", "assets/b_w.svg", "assets/b_k.svg",
    };
    bool ok = true;
    for (int i = 0; i < TEX_NB; ++i) {
        as->textures[i] = load_texture(as->renderer, paths[i]);
        ok              = ok && (as->textures[i] != nullptr);
    }
    as->texturesLoaded = ok;
}

// ---------- Board / UI math ----------
static bool point_in_rect(float x, float y, const SDL_FRect& r) {
    return (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h);
}

// Map mouse (x,y) to a board Square, or SQ_NONE if outside board
static Square screen_to_square(const AppState* as, float x, float y) {
    const SDL_FRect& B = as->ui.boardRect;
    if (!point_in_rect(x, y, B)) return SQ_NONE;
    const int q   = as->ui.squarePx;
    int       col = int((x - B.x) / q);
    int       row = int((y - B.y) / q);
    if (col < 0 || col >= 4 || row < 0 || row >= 4) return SQ_NONE;

    // Orientation: base on human side (white bottom or black bottom) and flip toggle
    bool whiteView = (as->humanSide == WHITE);
    if (as->boardFlipped) whiteView = !whiteView;

    int file = whiteView ? col : (3 - col);
    int rank = whiteView ? (3 - row) : row;  // row 0 at top of screen; rank 3 is top for white view

    return make_square(File(file), Rank(rank));
}

static SDL_FRect square_rect(const AppState* as, Square s) {
    const int q         = as->ui.squarePx;
    bool      whiteView = (as->humanSide == WHITE);
    if (as->boardFlipped) whiteView = !whiteView;

    int file = file_of(s);
    int rank = rank_of(s);

    int col = whiteView ? file : (3 - file);
    int row = whiteView ? (3 - rank) : rank;

    SDL_FRect r;
    r.x = as->ui.boardRect.x + col * q;
    r.y = as->ui.boardRect.y + row * q;
    r.w = (float)q;
    r.h = (float)q;
    return r;
}

// ---------- Move utilities ----------
static std::vector<Move> legal_moves(const Position& pos) {
    MoveList<LEGAL>   ml(pos);
    std::vector<Move> out;
    out.reserve(ml.size());
    for (int i = 0; i < ml.size(); ++i) out.push_back(ml[i]);
    return out;
}

static std::vector<Move> filter_moves_from(const Position& pos, Square from) {
    std::vector<Move> ms = legal_moves(pos);
    std::vector<Move> out;
    out.reserve(ms.size());
    for (auto m : ms)
        if (m.type_of() == NORMAL || m.type_of() == PROMOTION) {
            if (m.from_sq() == from) out.push_back(m);
        }
    return out;
}

static std::vector<Move> filter_moves_from_to(AppState* as, Square from, Square to) {
    std::vector<Move> out;
    for (auto m : as->legalMoves)
        if (m.from_sq() == from && m.to_sq() == to) out.push_back(m);
    return out;
}

static std::vector<Move> filter_drop_moves(const Position& pos, PieceType pt) {
    std::vector<Move> ms = legal_moves(pos);
    std::vector<Move> out;
    for (auto m : ms)
        if (m.type_of() == DROP && m.drop_piece() == pt) out.push_back(m);
    return out;
}

static std::optional<Move> find_drop_to(const Position& pos, PieceType pt, Square to) {
    auto drops = filter_drop_moves(pos, pt);
    for (auto m : drops)
        if (m.to_sq() == to) return m;
    return std::nullopt;
}

static bool is_terminal(const Position& pos, bool& isMate, Color& winner, bool& isThreefold) {
    MoveList<LEGAL> root(pos);
    if (pos.is_threefold_game()) {
        isThreefold = true;
        return true;
    }
    if (root.size() == 0) {
        isMate = pos.checkers() != 0;
        if (isMate)
            winner = ~pos.side_to_move();
        else
            winner = pos.side_to_move();  // stalemated player wins per rules
        return true;
    }
    return false;
}

// ---------- Rendering ----------
static void draw_rect(SDL_Renderer* r, const SDL_FRect& rc, const DrawColor& color) {
    color.set_sdl_color(r);
    SDL_RenderFillRect(r, &rc);
}

static void draw_outline(SDL_Renderer* r, const SDL_FRect& rc, const DrawColor& color,
                         float thickness = 3.0f) {
    // Draw 4 thin rects as outline
    color.set_sdl_color(r);
    SDL_FRect t = rc;
    // top
    t.h = thickness;
    SDL_RenderFillRect(r, &t);
    // bottom
    t.y = rc.y + rc.h - thickness;
    t.h = thickness;
    SDL_RenderFillRect(r, &t);
    // left
    t   = rc;
    t.w = thickness;
    SDL_RenderFillRect(r, &t);
    // right
    t.x = rc.x + rc.w - thickness;
    t.w = thickness;
    SDL_RenderFillRect(r, &t);
}

// Filled disc
static void draw_filled_circle(SDL_Renderer* r, const SDL_FPoint& c, float radius,
                               const DrawColor& color) {
    color.set_sdl_color(r);
    const int yMax = (int)ceilf(radius);
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
    for (int iy = -yMax; iy <= yMax; ++iy) {
        float dy = (float)iy;
        float dx = sqrtf(std::max(0.0f, radius * radius - dy * dy));
        float y  = c.y + dy;

        // draw horizontal span
        SDL_RenderLine(r, c.x - dx, y, c.x + dx, y);
    }
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);
}

// Ring with thickness (outer radius = radius, inner radius = radius - thickness)
static void draw_ring(SDL_Renderer* r, const SDL_FPoint& c, float radius, float thickness,
                      const DrawColor& color) {
    if (thickness <= 0.0f) return;
    color.set_sdl_color(r);
    float     rOuter = radius;
    float     rInner = std::max(0.0f, radius - thickness);
    const int yMax   = (int)ceilf(rOuter);
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
    for (int iy = -yMax; iy <= yMax; ++iy) {
        float dy = (float)iy;
        float yo = rOuter * rOuter - dy * dy;
        if (yo < 0.0f) continue;
        float xo = sqrtf(yo);

        float yi = rInner * rInner - dy * dy;
        float xi = yi > 0.0f ? sqrtf(yi) : 0.0f;

        float y = c.y + dy;
        // left arc segment
        SDL_RenderLine(r, c.x - xo, y, c.x - xi, y);
        // right arc segment
        SDL_RenderLine(r, c.x + xi, y, c.x + xo, y);
    }
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);
}

static void draw_text(SDL_Renderer* r, const char* text, float x, float y, const DrawColor& color) {
    // Use SDL_RenderDebugText for proper text rendering
    color.set_sdl_color(r);
    SDL_RenderDebugText(r, x, y, text);
}

static void draw_board(AppState* as) {
    SDL_Renderer* r = as->renderer;
    for (int rank = 0; rank < 4; ++rank) {
        for (int file = 0; file < 4; ++file) {
            Square    s    = make_square(File(file), Rank(rank));
            SDL_FRect rc   = square_rect(as, s);
            bool      dark = ((file + rank) & 1);
            draw_rect(r, rc, dark ? Colors::BoardDark : Colors::BoardLight);
        }
    }
}

static void draw_last_move(AppState* as) {
    SDL_Renderer* r = as->renderer;
    if (as->lastMove.from) {
        draw_rect(r, square_rect(as, *as->lastMove.from), Colors::LastMoveFrom);
    }
    if (as->lastMove.to) {
        draw_rect(r, square_rect(as, *as->lastMove.to), Colors::LastMoveTo);
    }
}

static void draw_check(AppState* as) {
    // Highlight side-to-move king if in check
    if (as->pos.checkers() == 0) return;
    Square ksq = as->pos.square<KING>(as->pos.side_to_move());
    draw_rect(as->renderer, square_rect(as, ksq), Colors::CheckHighlight);
}

static void draw_piece_texture(SDL_Renderer* r, SDL_Texture* tex, const SDL_FRect& dst) {
    if (!tex) return;
    SDL_RenderTexture(r, tex, nullptr, &dst);
}

static void draw_piece_fallback(SDL_Renderer* r, Color c, PieceType pt, const SDL_FRect& cell) {
    // Simple circle marker; different radii per type to distinguish.
    float     rad        = std::min(cell.w, cell.h) * (0.35f + 0.05f * (pt - 1));
    DrawColor pieceColor = (c == WHITE ? DrawColor(240, 240, 240) : DrawColor(30, 30, 30));
    draw_filled_circle(r, SDL_FPoint{cell.x + cell.w / 2, cell.y + cell.h / 2}, rad, pieceColor);
}

static void draw_pieces(AppState* as) {
    SDL_Renderer* r = as->renderer;
    for (int s = SQUARE_ZERO; s < SQUARE_NB; ++s) {
        Square sq = Square(s);
        Piece  pc = as->pos.piece_on(sq);
        if (pc == NO_PIECE) continue;
        SDL_FRect rc = square_rect(as, sq);
        TexKey    tk = texkey_for_piece(pc);
        if (as->texturesLoaded && as->textures[tk])
            draw_piece_texture(r, as->textures[tk], rc);
        else {
            Color     c  = (pc >= B_PAWN ? BLACK : WHITE);
            PieceType pt = PieceType(pc & 0x7);  // types are aligned 1..5 per your enum
            draw_piece_fallback(r, c, pt, rc);
        }
    }
}

static void draw_selection(AppState* as) {
    SDL_Renderer* r = as->renderer;

    // Show selected square
    if (as->selectedSq) {
        draw_outline(r, square_rect(as, *as->selectedSq), Colors::SelectionOutline, 5.0f);

        // Show legal targets from that square
        auto ms = filter_moves_from(as->pos, *as->selectedSq);
        for (auto m : ms) {
            SDL_FRect rc = square_rect(as, m.to_sq());

            SDL_FPoint center{rc.x + rc.w * 0.5f, rc.y + rc.h * 0.5f};

            bool isCapture = as->pos.piece_on(m.to_sq()) != NO_PIECE;

            // unified color; adjust if you keep separate palette for captures
            const DrawColor hintColor = Colors::MoveHint;  // same color for both as requested

            if (!isCapture) {
                // small filled dot
                float r = rc.h * 0.15f;  // tweak to taste
                draw_filled_circle(as->renderer, center, r, hintColor);
            } else {
                // larger ring of same color
                float rOuter    = rc.h * 0.35f;
                float thickness = rc.h * 0.07f;
                draw_ring(as->renderer, center, rOuter, thickness, hintColor);
            }
        }
    }

    // Show selected drop piece (pocket), highlight legal drop squares
    if (as->selectedDropPiece) {
        auto drops = filter_drop_moves(as->pos, *as->selectedDropPiece);
        for (auto m : drops) {
            SDL_FRect rc = square_rect(as, m.to_sq());
            draw_filled_circle(r, SDL_FPoint{rc.x + rc.w / 2, rc.y + rc.h / 2}, rc.w * 0.15f,
                               Colors::MoveHint);
        }
    }
}

static void draw_pockets(AppState* as) {
    SDL_Renderer*    r = as->renderer;
    const SDL_FRect& U = as->ui.leftUIRect;

    draw_rect(r, U, Colors::PocketBackground);
    draw_outline(r, U, Colors::PocketBorder, 2.0f);

    const bool  flipped     = as->boardFlipped;
    const Color bottomColor = flipped ? ~as->humanSide : as->humanSide;
    const Color topColor    = ~bottomColor;

    const int slotsPerSide = 4;  // PAWN..WAZIR

    // Correct halves
    SDL_FRect topArea{U.x, U.y, U.w, U.h / 2};
    SDL_FRect bottomArea{U.x, U.y + U.h / 2, U.w, U.h / 2};

    auto draw_side_stack = [&](Color side, const SDL_FRect& area) {
        const float slotH = area.h / slotsPerSide;

        for (PieceType pt = PAWN; pt <= WAZIR; ++pt) {
            int count = as->pos.pocket(side).count(pt);
            if (count <= 0) continue;

            // Map enum to row index (assumes contiguous PAWN..WAZIR)
            int row = static_cast<int>(pt) - static_cast<int>(PAWN);

            SDL_FRect slotRect{area.x, area.y + row * slotH, area.w, slotH};

            TexKey tk = texkey_for_piece(side, pt);
            if (as->texturesLoaded && as->textures[tk]) {
                draw_piece_texture(r, as->textures[tk], slotRect);
            } else {
                draw_piece_fallback(r, side, pt, slotRect);
            }

            if (count > 1) {
                float countX = slotRect.x + slotRect.w - 15.0f;
                float countY = slotRect.y + slotRect.h - 15.0f;

                char countStr[4];
                snprintf(countStr, sizeof(countStr), "%d", count);
                draw_text(r, countStr, countX, countY, Colors::Bright);
            }
            // Show selected square
            if (as->selectedDropPiece == pt && side == as->humanSide) {
                draw_outline(r, slotRect, Colors::SelectionOutline, 2.5f);
            }
        }
    };

    draw_side_stack(topColor, topArea);
    draw_side_stack(bottomColor, bottomArea);
}

static void draw_promotion_overlay(AppState* as) {
    if (!as->promo.visible) return;
    SDL_Renderer* r = as->renderer;

    // Full square rect
    SDL_FRect sqRect = square_rect(as, as->promo.to);

    // Semi-transparent dark overlay
    SDL_SetRenderDrawColor(r, 0, 0, 0, 180);
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_BLEND);
    SDL_RenderFillRect(r, &sqRect);
    SDL_SetRenderDrawBlendMode(r, SDL_BLENDMODE_NONE);

    const float                    halfW = sqRect.w * 0.5f;
    const std::array<PieceType, 3> order = {HORSE, FERZ, WAZIR};

    Color who = as->pos.side_to_move();

    for (int i = 0; i < 3; ++i) {
        SDL_FRect cell{};
        if (i == 0) {
            cell = {sqRect.x, sqRect.y, halfW, halfW};
        }  // top-left
        else if (i == 1) {
            cell = {sqRect.x + halfW, sqRect.y, halfW, halfW};
        }  // top-right
        else {
            cell = {sqRect.x, sqRect.y + halfW, halfW, halfW};
        }  // bottom-left

        as->promo.rects[i] = cell;

        TexKey tk = texkey_for_piece(who, order[i]);
        if (as->texturesLoaded && as->textures[tk])
            draw_piece_texture(r, as->textures[tk], cell);
        else
            draw_piece_fallback(r, who, order[i], cell);
    }
}

static void draw_start_screen(AppState* as) {
    SDL_Renderer* r = as->renderer;

    // Create divided screen - white and black halves
    float halfWidth  = WINDOW_W / 2.0f;
    float fullHeight = WINDOW_H;

    // Left half - White side
    SDL_FRect whiteHalf{0, 0, halfWidth, fullHeight};
    draw_rect(r, whiteHalf, Colors::Bright);  // Light gray/white background

    // Right half - Black side
    SDL_FRect blackHalf{halfWidth, 0, halfWidth, fullHeight};
    draw_rect(r, blackHalf, Colors::Background);  // Dark gray/black background

    // Center divider line
    draw_rect(r, SDL_FRect{halfWidth - 2, 0, 4, fullHeight}, Colors::Border);

    // Calculate piece areas with proper aspect ratio
    float pieceAreaSize = std::min(halfWidth * 0.6f, fullHeight * 0.4f);
    float pieceSize     = pieceAreaSize * 0.6f;  // Keep 1:1 ratio for textures

    // White side piece area (left half)
    SDL_FRect whitePieceArea{halfWidth / 2 - pieceAreaSize / 2, fullHeight / 2 - pieceAreaSize / 2,
                             pieceAreaSize, pieceAreaSize};

    // Black side piece area (right half)
    SDL_FRect blackPieceArea{halfWidth + halfWidth / 2 - pieceAreaSize / 2,
                             fullHeight / 2 - pieceAreaSize / 2, pieceAreaSize, pieceAreaSize};

    // Draw background circles for pieces
    draw_rect(r, whitePieceArea, Colors::OnBright);
    draw_outline(r, whitePieceArea, Colors::Border, 3.0f);

    draw_rect(r, blackPieceArea, Colors::OnBackground);
    draw_outline(r, blackPieceArea, Colors::Border, 3.0f);

    // Draw piece textures with proper scaling (1:1 ratio)
    if (as->texturesLoaded) {
        // White king - center in white area
        SDL_FRect whiteKingRect{whitePieceArea.x + (whitePieceArea.w - pieceSize) / 2,
                                whitePieceArea.y + (whitePieceArea.h - pieceSize) / 2, pieceSize,
                                pieceSize};
        draw_piece_texture(r, as->textures[T_W_K], whiteKingRect);

        // Black king - center in black area
        SDL_FRect blackKingRect{blackPieceArea.x + (blackPieceArea.w - pieceSize) / 2,
                                blackPieceArea.y + (blackPieceArea.h - pieceSize) / 2, pieceSize,
                                pieceSize};
        draw_piece_texture(r, as->textures[T_B_K], blackKingRect);
    }
}

static void draw_right_panel(AppState* as) {
    SDL_Renderer*    r = as->renderer;
    const SDL_FRect& U = as->ui.rightUIRect;

    // Status section
    SDL_FRect statusRect = {U.x, U.y + as->ui.marginPx, U.w, 150};

    Colors::Bright.set_sdl_color(r);

    // Header line
    SDL_SetRenderScale(r, 3.0f, 3.0f);
    SDL_RenderDebugText(r, (statusRect.x + 10) / 3, (statusRect.y + 8) / 3, "Status");
    SDL_SetRenderScale(r, 1.0f, 1.0f);

    float tx = statusRect.x + 10;
    float ty = statusRect.y + 64;

    if (as->phase == Phase::GameOver) {
        // Result
        if (as->gameResult == GameResult::Draw) {
            SDL_SetRenderScale(r, 1.5f, 1.5f);
            SDL_RenderDebugText(r, tx / 1.5f, ty / 1.5f, "Draw by threefold repetition");
            SDL_SetRenderScale(r, 1.0f, 1.0f);
            ty += 22;

        } else {
            const char* res = (as->gameResult == GameResult::CheckMate) ? "Checkmate" : "Stalemate";
            const char* win = (as->winner == WHITE) ? "White" : "Black";
            SDL_SetRenderScale(r, 1.5f, 1.5f);
            SDL_RenderDebugTextFormat(r, tx / 1.5f, ty / 1.5f, "%s. Winner: %s", res, win);
            SDL_SetRenderScale(r, 1.0f, 1.0f);
            ty += 22;
        }
        // Small hints
        ty += 6;
        SDL_RenderDebugText(r, tx, ty, "[R] Restart");

    } else {
        // Side to move
        const char* stm = (as->pos.side_to_move() == WHITE) ? "White to move" : "Black to move";
        SDL_SetRenderScale(r, 1.5f, 1.5f);
        SDL_RenderDebugText(r, tx / 1.5f, ty / 1.5f, stm);
        SDL_SetRenderScale(r, 1.0f, 1.0f);
        ty += 22;

        // AI status / evaluation
        if (as->ai.thinking) {
            SDL_SetRenderScale(r, 1.5f, 1.5f);
            SDL_RenderDebugText(r, tx / 1.5f, ty / 1.5, "AI thinking…");
            SDL_SetRenderScale(r, 1.0f, 1.0f);
            ty += 22;
        } else {
            if (as->ai.lastEval) {
                // Note: score is from the AI's search perspective at the time it moved.
                SDL_SetRenderScale(r, 1.5f, 1.5f);
                SDL_RenderDebugTextFormat(r, tx / 1.5f, ty / 1.5f, "Eval (AI): %d",
                                          (int)*as->ai.lastEval);
                SDL_SetRenderScale(r, 1.0f, 1.0f);
                ty += 22;
            } else {
                SDL_SetRenderScale(r, 1.5f, 1.5f);
                SDL_RenderDebugText(r, tx / 1.5F, ty / 1.5F, "Eval: —");
                SDL_SetRenderScale(r, 1.0f, 1.0f);
                ty += 22;
            }
        }
        // Small hints
        ty += 6;
        SDL_RenderDebugText(r, tx, ty, "[R] Restart");
    }
}

// ---------- Game mechanics ----------
static void apply_move_and_advance(AppState* as, Move m) {
    as->states.emplace_back();
    as->pos.do_move(m, as->states.back());
    as->lastMove.from = m.from_sq();
    as->lastMove.to   = m.to_sq();
    as->selectedSq.reset();
    as->selectedDropPiece.reset();
    as->promo.visible = false;
    as->legalMoves    = legal_moves(as->pos);  // Update legal moves after the move
}

static void start_ai_thinking_if_needed(AppState* as) {
    if (as->phase != Phase::Playing) return;
    if (as->pos.side_to_move() == as->humanSide) return;
    if (as->ai.thinking) return;

    as->ai.thinking = true;

    // Snapshot everything the thread needs, _before_ launching it.
    Position  snapshot = as->pos;          // deep enough for your engine (as you tested)
    const int depth    = as->searchDepth;  // or as->searchDepth if you store it there

    as->ai.fut = std::async(std::launch::async, [snapshot, depth]() mutable {
        SearchResult res = search_best_move(snapshot, depth);
        return res;
    });
}

static void check_and_handle_terminal_state(AppState* as) {
    bool  isMate = false, isThree = false;
    Color win = WHITE;
    if (is_terminal(as->pos, isMate, win, isThree)) {
        as->phase      = Phase::GameOver;
        as->winner     = win;
        as->gameResult = isThree  ? GameResult::Draw
                         : isMate ? GameResult::CheckMate
                                  : GameResult::Stalemate;
    } else {
        start_ai_thinking_if_needed(as);
    }
}

static void maybe_finish_ai(AppState* as) {
    using namespace std::chrono_literals;
    if (as->ai.fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        SearchResult res = as->ai.fut.get();
        as->ai.thinking  = false;

        as->ai.lastEval = res.score;

        if (res.bestMove != MOVE_NONE) {
            apply_move_and_advance(as, res.bestMove);  // modifies as->pos on the main thread
            check_and_handle_terminal_state(as);
        }
    }
}

static void restart_to_side_select(AppState* as) {
    as->phase       = Phase::SideSelect;
    as->gameResult  = GameResult::None;
    as->ai.thinking = false;
    as->ai.lastEval.reset();
    as->selectedSq.reset();
    as->selectedDropPiece.reset();
    as->promo.visible = false;
    as->lastMove.from.reset();
    as->lastMove.to.reset();
}

// ---------- Event handling ----------
static bool click_in_pocket(AppState* as, float mx, float my, Color& pocketColorOut,
                            PieceType& ptOut) {
    const SDL_FRect& U = as->ui.leftUIRect;

    // Same color mapping as draw_pockets
    const bool  flipped     = as->boardFlipped;
    const Color bottomColor = flipped ? ~as->humanSide : as->humanSide;
    const Color topColor    = ~bottomColor;

    const int   slotsPerSide = 4;  // PAWN..WAZIR
    const float halfH        = U.h * 0.5f;

    // Same areas as draw_pockets (no padding, no borders)
    SDL_FRect topArea{U.x, U.y, U.w, halfH};
    SDL_FRect bottomArea{U.x, U.y + halfH, U.w, halfH};

    auto hit_stack = [&](Color side, const SDL_FRect& area) -> bool {
        const float slotH = area.h / slotsPerSide;

        for (PieceType pt = PAWN; pt <= WAZIR; ++pt) {
            int count = as->pos.pocket(side).count(pt);
            if (count <= 0) continue;  // only clickable if present

            int row = static_cast<int>(pt) - static_cast<int>(PAWN);

            SDL_FRect slotRect{area.x, area.y + row * slotH, area.w, slotH};

            if (point_in_rect(mx, my, slotRect)) {
                pocketColorOut = side;
                ptOut          = pt;
                return true;
            }
        }
        return false;
    };

    if (point_in_rect(mx, my, topArea) && hit_stack(topColor, topArea)) return true;
    if (point_in_rect(mx, my, bottomArea) && hit_stack(bottomColor, bottomArea)) return true;

    return false;
}

static void handle_board_click(AppState* as, float mx, float my) {
    if (as->pos.side_to_move() != as->humanSide) return;  // wait for AI

    // First, promotion overlay?
    if (as->promo.visible) {
        for (int i = 0; i < 3; ++i) {
            if (point_in_rect(mx, my, as->promo.rects[i])) {
                // Identify promotion piece type by rect index
                std::array<PieceType, 3> order{HORSE, FERZ, WAZIR};
                PieceType                want = order[i];
                for (auto m : as->promo.options) {
                    if (m.promotion_type() == want) {
                        apply_move_and_advance(as, m);
                        check_and_handle_terminal_state(as);
                        return;
                    }
                }
            }
        }
        // click outside: cancel
        as->promo.visible = false;
        return;
    }

    // Pocket click?
    Color     pc;
    PieceType pt;
    if (click_in_pocket(as, mx, my, pc, pt)) {
        if (pc == as->pos.side_to_move()) {
            as->selectedDropPiece = pt;
            as->selectedSq.reset();
        }
        return;
    }

    // Board click
    Square s = screen_to_square(as, mx, my);
    if (s == SQ_NONE) {
        as->selectedSq.reset();
        as->selectedDropPiece.reset();
        return;
    }

    // If a drop piece is selected, try to drop here
    if (as->selectedDropPiece) {
        auto m = find_drop_to(as->pos, *as->selectedDropPiece, s);
        if (m) {
            apply_move_and_advance(as, *m);
            check_and_handle_terminal_state(as);
            return;
        } else {
            // clicking elsewhere: clear drop selection or treat as select piece on board
            as->selectedDropPiece.reset();
            // continue to possibly select a board piece
        }
    }

    // If no source selected yet and clicked own piece -> select
    if (!as->selectedSq) {
        if (is_own_piece(as->pos, s)) {
            as->selectedSq = s;
            return;
        } else {
            // clicked empty or opponent; ignore
            return;
        }
    }

    // Have a source; attempt to move to clicked target
    auto candidates = filter_moves_from_to(as, *as->selectedSq, s);
    if (candidates.empty()) {
        // If clicked another own piece, switch selection
        if (is_own_piece(as->pos, s)) {
            as->selectedSq = s;
        } else {
            as->selectedSq.reset();
        }
        return;
    }

    if (candidates.size() == 1) {
        apply_move_and_advance(as, candidates[0]);
        check_and_handle_terminal_state(as);
    } else {
        // Multiple moves means promotion choices. Show chooser.
        as->promo.from    = *as->selectedSq;
        as->promo.to      = s;
        as->promo.options = candidates;  // all promotion variants
        as->promo.visible = true;
        return;
    }
}

// ---------- SDL Callbacks ----------
SDL_AppResult SDL_AppInit(void** appstate, int argc, char* argv[]) {
    if (!SDL_SetAppMetadata("TinyHouse variant of Chess game with AI", "1.0",
                            "com.example.tinyhouse")) {
        printf("Failed to set app metadata\n");
        return SDL_APP_FAILURE;
    }

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        printf("Failed to initialize SDL: %s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    AppState* as = new AppState();
    if (!as) {
        printf("Failed to allocate AppState\n");
        return SDL_APP_FAILURE;
    }
    *appstate = as;

    if (!SDL_CreateWindowAndRenderer("TinyHouse Chess", WINDOW_W, WINDOW_H,
                                     SDL_WINDOW_ALWAYS_ON_TOP, &as->window, &as->renderer)) {
        printf("Couldn't create window and renderer: %s\n", SDL_GetError());
        return SDL_APP_FAILURE;
    }

    SDL_SetRenderLogicalPresentation(as->renderer, WINDOW_W, WINDOW_H,
                                     SDL_LOGICAL_PRESENTATION_LETTERBOX);

    // Engine init
    Bitboards::init();
    Position::init();

    as->states.clear();
    as->states.emplace_back();

    as->pos.set(START_FEN, &as->states.back());
    as->legalMoves = legal_moves(as->pos);

    load_all_textures(as);

    as->phase        = Phase::SideSelect;
    as->boardFlipped = false;
    as->humanSide    = WHITE;
    as->lastTicks    = SDL_GetTicks();

    printf("SDL APP INIT PASSED\n");

    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppEvent(void* appstate, SDL_Event* event) {
    AppState* as = (AppState*)appstate;
    switch (event->type) {
        case SDL_EVENT_QUIT:
            return SDL_APP_SUCCESS;

        case SDL_EVENT_KEY_DOWN: {
            if (event->key.scancode == SDL_SCANCODE_R) {
                restart_to_side_select(as);
            };
            break;
        }

        case SDL_EVENT_MOUSE_BUTTON_DOWN: {
            float mx = (float)event->button.x;
            float my = (float)event->button.y;

            if (as->phase == Phase::SideSelect) {
                // But allow clicking left/right halves for convenience.
                if (mx < WINDOW_W / 2) {
                    as->humanSide = WHITE;
                } else {
                    as->humanSide = BLACK;
                }
                // Reset and start game with chosen side
                as->states.clear();
                as->states.emplace_back();
                as->pos.set(START_FEN, &as->states.back());
                as->legalMoves   = legal_moves(as->pos);  // Update legal moves for new game
                as->boardFlipped = false;
                as->selectedSq.reset();
                as->selectedDropPiece.reset();
                as->promo.visible = false;
                as->lastMove.from.reset();
                as->lastMove.to.reset();
                as->phase = Phase::Playing;

                // If AI to move first, start thinking
                if (as->pos.side_to_move() != as->humanSide) {
                    start_ai_thinking_if_needed(as);
                }
            } else {
                handle_board_click(as, mx, my);
                break;
            }
        }
        default:
            break;
    }
    return SDL_APP_CONTINUE;
}

SDL_AppResult SDL_AppIterate(void* appstate) {
    AppState* as = (AppState*)appstate;

    // AI progression
    if (as->ai.thinking) {
        maybe_finish_ai(as);
    }

    if (as->phase == Phase::SideSelect) {
        draw_start_screen(as);
    } else {
        Colors::Background.set_sdl_color(as->renderer);
        SDL_RenderClear(as->renderer);

        // Playing or picking the promotion type
        draw_board(as);
        draw_last_move(as);
        draw_check(as);
        draw_selection(as);
        draw_pieces(as);

        draw_promotion_overlay(as);

        draw_pockets(as);
        draw_right_panel(as);
    }

    SDL_RenderPresent(as->renderer);

    return SDL_APP_CONTINUE;
}

void SDL_AppQuit(void* appstate, SDL_AppResult result) {
    if (!appstate) return;
    AppState* as = (AppState*)appstate;

    for (auto& t : as->textures)
        if (t) SDL_DestroyTexture(t);
    if (as->renderer) SDL_DestroyRenderer(as->renderer);
    if (as->window) SDL_DestroyWindow(as->window);
    SDL_free(as);
}

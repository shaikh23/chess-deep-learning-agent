"""Static evaluation for chess positions.

Uses material balance + piece-square tables (PST) + simple mobility.
"""

import chess
import numpy as np

# Material values (centipawns)
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Piece-Square Tables (from Sunfish, flipped for white perspective)
# Format: [rank 0-7][file 0-7], from white's perspective
PST = {
    chess.PAWN: [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ],
}


def static_eval(board: chess.Board) -> float:
    """
    Static evaluation of position.

    Returns value from white's perspective in centipawns.
    Range: approximately [-10000, 10000]

    Components:
    - Material balance
    - Piece-square tables
    - Simple mobility (number of legal moves)
    """
    if board.is_checkmate():
        return -10000.0 if board.turn == chess.WHITE else 10000.0

    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    score = 0.0

    # Material and PST
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Material value
            value = PIECE_VALUES[piece.piece_type]

            # PST value (flip square for black pieces)
            pst_square = square if piece.color == chess.WHITE else chess.square_mirror(square)
            pst_value = PST[piece.piece_type][pst_square]

            piece_score = value + pst_value

            if piece.color == chess.WHITE:
                score += piece_score
            else:
                score -= piece_score

    # Mobility (simplified)
    mobility = len(list(board.legal_moves))
    if board.turn == chess.WHITE:
        score += mobility * 0.5
    else:
        score -= mobility * 0.5

    return score


def static_eval_normalized(board: chess.Board, scale: float = 800.0) -> float:
    """
    Static evaluation normalized to [-1, 1].

    Args:
        board: Chess board
        scale: Scaling factor (default: 800 centipawns)

    Returns:
        Evaluation in [-1, 1] from current player's perspective
    """
    cp_score = static_eval(board)

    # Convert to current player's perspective
    if board.turn == chess.BLACK:
        cp_score = -cp_score

    # Normalize
    return np.clip(cp_score / scale, -1.0, 1.0)

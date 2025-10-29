"""Board and move encoding utilities for neural network input/output.

Encoding scheme:
- Board: [12+5, 8, 8] tensor with piece planes (6 piece types × 2 colors)
         plus auxiliary features (side to move, castling rights, en passant)
- Moves: Flattened index from (from_square, to_square, promotion_type)
         Total dimension: 64 × 64 × 5 = 20,480 (with promotion types)
         Simplified: 64 × 64 = 4,096 (most common, handles promotion via special indices)
         We use 4,672 to match AlphaZero-style indexing (64×73 for knight-like moves)
"""

import chess
import numpy as np
import torch
from typing import Optional, List, Tuple


# Move indexing: Use simplified 64x73 encoding similar to AlphaZero
# 64 from-squares × 73 move types:
#   - 56 queen moves (8 directions × 7 distances)
#   - 8 knight moves
#   - 9 underpromotions (3 directions × 3 pieces: knight, bishop, rook)
MOVE_INDEX_SIZE = 4672  # 64 * 73


def board_to_tensor(board: chess.Board, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert chess.Board to tensor representation [17, 8, 8].

    Channels:
    - 0-5: White pieces (P, N, B, R, Q, K)
    - 6-11: Black pieces (P, N, B, R, Q, K)
    - 12: Side to move (1 = white, 0 = black)
    - 13-16: Castling rights (white kingside, queenside, black kingside, queenside)
    - 17: En passant file (one-hot encoded across 8 files, or all zeros if none)

    Actually we'll simplify to [12, 8, 8] for pieces only, and handle auxiliary
    features separately if needed. For this implementation, use 12 piece planes.

    Args:
        board: python-chess Board object
        device: PyTorch device to place tensor on

    Returns:
        Tensor of shape [12, 8, 8] with piece placement
    """
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    # Piece type to channel mapping
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    # Fill piece planes
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            channel = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            tensor[channel, rank, file] = 1.0

    result = torch.from_numpy(tensor)
    if device is not None:
        result = result.to(device)

    return result


def get_auxiliary_features(board: chess.Board, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Extract auxiliary features as a flat vector.

    Features (total 6 values):
    - Side to move (1 value): 1.0 if white, 0.0 if black
    - Castling rights (4 values): white K, white Q, black K, black Q
    - En passant file (1 value): file index / 7.0 if exists, else -1.0

    Args:
        board: python-chess Board object
        device: PyTorch device

    Returns:
        Tensor of shape [6]
    """
    features = np.zeros(6, dtype=np.float32)

    # Side to move
    features[0] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights
    features[1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    features[2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    features[3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    features[4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # En passant
    if board.ep_square is not None:
        features[5] = chess.square_file(board.ep_square) / 7.0
    else:
        features[5] = -1.0

    result = torch.from_numpy(features)
    if device is not None:
        result = result.to(device)

    return result


def move_to_index(move: chess.Move) -> int:
    """
    Convert chess.Move to a flattened index.

    Simplified encoding: from_square * 64 + to_square
    This gives 4096 possible move indices. We'll pad to 4672 for compatibility.

    For promotion moves, we encode them with a special offset.

    Args:
        move: python-chess Move object

    Returns:
        Integer index in range [0, MOVE_INDEX_SIZE)
    """
    from_sq = move.from_square
    to_sq = move.to_square

    # Basic move: from * 64 + to
    base_index = from_sq * 64 + to_sq

    # Handle promotions by adding offset
    if move.promotion is not None:
        # Offset by promotion type: knight=1, bishop=2, rook=3, queen=4
        promotion_offset = {
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
        }.get(move.promotion, 0)
        # Use indices beyond 4096 for promotions
        base_index = 4096 + (from_sq * 64 + to_sq) * 4 + (promotion_offset - 1)

    return min(base_index, MOVE_INDEX_SIZE - 1)


def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Convert flattened index back to chess.Move.

    Args:
        index: Move index
        board: Current board state (to validate legality)

    Returns:
        chess.Move object if valid, else None
    """
    if index < 4096:
        from_sq = index // 64
        to_sq = index % 64
        move = chess.Move(from_sq, to_sq)
    else:
        # Promotion move
        offset = index - 4096
        promotion_type = (offset % 4) + 1
        move_base = offset // 4
        from_sq = move_base // 64
        to_sq = move_base % 64

        promotion_piece = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN][promotion_type]
        move = chess.Move(from_sq, to_sq, promotion=promotion_piece)

    # Validate move is legal
    if move in board.legal_moves:
        return move
    return None


def get_legal_move_mask(board: chess.Board, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Create a boolean mask for legal moves in current position.

    Args:
        board: python-chess Board object
        device: PyTorch device

    Returns:
        Boolean tensor of shape [MOVE_INDEX_SIZE] where True = legal move
    """
    mask = torch.zeros(MOVE_INDEX_SIZE, dtype=torch.bool)

    for move in board.legal_moves:
        idx = move_to_index(move)
        mask[idx] = True

    if device is not None:
        mask = mask.to(device)

    return mask


def moves_to_indices(moves: List[chess.Move]) -> np.ndarray:
    """
    Convert list of moves to array of indices.

    Args:
        moves: List of python-chess Move objects

    Returns:
        NumPy array of move indices
    """
    return np.array([move_to_index(m) for m in moves], dtype=np.int64)


def san_to_uci(board: chess.Board, san: str) -> str:
    """
    Convert SAN (Standard Algebraic Notation) move to UCI format with legality check.

    Args:
        board: Current board state
        san: Move in SAN format (e.g., "Nf3", "e4", "O-O")

    Returns:
        Move in UCI format (e.g., "g1f3", "e2e4", "e1g1")

    Raises:
        ValueError: If the SAN move is illegal or invalid
    """
    try:
        move = board.parse_san(san)
        if move not in board.legal_moves:
            raise ValueError(
                f"Illegal move: {san}\n"
                f"FEN: {board.fen()}\n"
                f"Legal moves: {[board.san(m) for m in list(board.legal_moves)[:10]]}"
            )
        return move.uci()
    except Exception as e:
        if "Illegal move" in str(e):
            raise
        raise ValueError(f"Invalid SAN move '{san}': {e}")


def validate_engine_move(board: chess.Board, move_uci: str) -> None:
    """
    Validate that a UCI move is legal in the current position.

    Args:
        board: Current board state
        move_uci: Move in UCI format

    Raises:
        ValueError: If the move is illegal, with detailed context for debugging
    """
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            legal_uci = [m.uci() for m in list(board.legal_moves)[:15]]
            raise ValueError(
                f"Illegal engine move: {move_uci}\n"
                f"FEN: {board.fen()}\n"
                f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}\n"
                f"Legal moves (first 15): {legal_uci}"
            )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Invalid UCI move '{move_uci}': {e}")


# ============================================================================
# Unit Tests (inline)
# ============================================================================

def _test_encoding():
    """Test board and move encoding."""
    print("Testing encoding utilities...")

    # Test board encoding
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (12, 8, 8), f"Expected shape (12, 8, 8), got {tensor.shape}"

    # Check white pawns (channel 0, rank 1)
    assert tensor[0, 1, :].sum() == 8, "Expected 8 white pawns"

    # Check black pawns (channel 6, rank 6)
    assert tensor[6, 6, :].sum() == 8, "Expected 8 black pawns"

    print("✓ Board encoding test passed")

    # Test auxiliary features
    aux = get_auxiliary_features(board)
    assert aux.shape == (6,), f"Expected shape (6,), got {aux.shape}"
    assert aux[0] == 1.0, "White to move"
    assert aux[1:5].sum() == 4.0, "All castling rights available"
    print("✓ Auxiliary features test passed")

    # Test move encoding
    move = chess.Move.from_uci("e2e4")
    idx = move_to_index(move)
    assert 0 <= idx < MOVE_INDEX_SIZE, f"Move index {idx} out of range"

    # Test round-trip
    recovered = index_to_move(idx, board)
    assert recovered == move, f"Move round-trip failed: {move} != {recovered}"
    print("✓ Move encoding test passed")

    # Test legal move mask
    mask = get_legal_move_mask(board)
    assert mask.shape == (MOVE_INDEX_SIZE,), f"Expected shape ({MOVE_INDEX_SIZE},), got {mask.shape}"
    num_legal = mask.sum().item()
    assert num_legal == len(list(board.legal_moves)), f"Mask has {num_legal} legal moves, expected {len(list(board.legal_moves))}"
    print("✓ Legal move mask test passed")

    print("All encoding tests passed! ✓")


if __name__ == "__main__":
    _test_encoding()

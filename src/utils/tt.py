"""Zobrist hashing and transposition table for chess search.

Zobrist hashing provides a fast way to hash chess positions for lookup in a
transposition table, avoiding re-evaluation of previously seen positions.
"""

import chess
import random
from typing import Optional, Dict
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """Type of transposition table node."""
    EXACT = "exact"        # Exact value
    LOWERBOUND = "lower"   # Alpha cutoff (value >= stored_value)
    UPPERBOUND = "upper"   # Beta cutoff (value <= stored_value)


@dataclass
class TTEntry:
    """Transposition table entry."""
    zobrist_hash: int
    depth: int
    value: float
    node_type: NodeType
    best_move: Optional[chess.Move] = None


class ZobristHasher:
    """
    Zobrist hashing for chess positions.

    Generates a unique pseudo-random number for each (piece, square) pair
    and combines them with XOR to create position hashes.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize Zobrist hasher.

        Args:
            seed: Random seed for reproducibility
        """
        random.seed(seed)

        # Create Zobrist keys: [piece_type][color][square]
        # piece_type: 0-5 (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
        # color: 0-1 (WHITE, BLACK)
        # square: 0-63
        self.piece_keys = [
            [[random.getrandbits(64) for _ in range(64)] for _ in range(2)]
            for _ in range(6)
        ]

        # Additional keys for game state
        self.side_to_move_key = random.getrandbits(64)
        self.castling_keys = [random.getrandbits(64) for _ in range(4)]  # KQkq
        self.en_passant_keys = [random.getrandbits(64) for _ in range(8)]  # files a-h

    def hash(self, board: chess.Board) -> int:
        """
        Compute Zobrist hash for a chess position.

        Args:
            board: Chess board

        Returns:
            64-bit integer hash
        """
        h = 0

        # Hash pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_type_idx = piece.piece_type - 1  # 0-5
                color_idx = 0 if piece.color == chess.WHITE else 1
                h ^= self.piece_keys[piece_type_idx][color_idx][square]

        # Hash side to move
        if board.turn == chess.BLACK:
            h ^= self.side_to_move_key

        # Hash castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            h ^= self.castling_keys[0]
        if board.has_queenside_castling_rights(chess.WHITE):
            h ^= self.castling_keys[1]
        if board.has_kingside_castling_rights(chess.BLACK):
            h ^= self.castling_keys[2]
        if board.has_queenside_castling_rights(chess.BLACK):
            h ^= self.castling_keys[3]

        # Hash en passant
        if board.ep_square is not None:
            file_idx = chess.square_file(board.ep_square)
            h ^= self.en_passant_keys[file_idx]

        return h

    def incremental_hash(
        self,
        current_hash: int,
        board: chess.Board,
        move: chess.Move,
    ) -> int:
        """
        Update hash incrementally after a move (not implemented - use hash()).

        This would be more efficient but requires tracking all state changes.
        For simplicity, we'll just recompute the hash.

        Args:
            current_hash: Current position hash
            board: Board after the move
            move: Move that was made

        Returns:
            Updated hash
        """
        # For now, just recompute (simpler and safer)
        return self.hash(board)


class TranspositionTable:
    """
    Transposition table for storing evaluated positions.

    Uses Zobrist hashing for position keys. Implements replacement scheme
    where deeper searches override shallower ones.
    """

    def __init__(self, max_size: int = 100000):
        """
        Initialize transposition table.

        Args:
            max_size: Maximum number of entries (approximate)
        """
        self.max_size = max_size
        self.table: Dict[int, TTEntry] = {}
        self.hasher = ZobristHasher()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.collisions = 0

    def probe(self, board: chess.Board, depth: int) -> Optional[TTEntry]:
        """
        Probe transposition table for a position.

        Args:
            board: Chess board
            depth: Current search depth

        Returns:
            TTEntry if found with sufficient depth, None otherwise
        """
        zobrist = self.hasher.hash(board)

        if zobrist in self.table:
            entry = self.table[zobrist]

            # Only use entry if it was searched to at least the same depth
            if entry.depth >= depth:
                self.hits += 1
                return entry
            else:
                self.misses += 1
                return None
        else:
            self.misses += 1
            return None

    def store(
        self,
        board: chess.Board,
        depth: int,
        value: float,
        node_type: NodeType,
        best_move: Optional[chess.Move] = None,
    ):
        """
        Store position evaluation in transposition table.

        Args:
            board: Chess board
            depth: Search depth
            value: Evaluation value
            node_type: Type of node (exact, lower, upper bound)
            best_move: Best move found (optional)
        """
        zobrist = self.hasher.hash(board)

        # Check if we should store this entry
        if zobrist in self.table:
            existing = self.table[zobrist]

            # Always replace if new entry has greater depth
            if depth < existing.depth:
                return

            self.collisions += 1

        # Store entry
        entry = TTEntry(
            zobrist_hash=zobrist,
            depth=depth,
            value=value,
            node_type=node_type,
            best_move=best_move,
        )

        self.table[zobrist] = entry

        # Simple size management: clear if too large
        if len(self.table) > self.max_size:
            self.clear()

    def clear(self):
        """Clear the transposition table."""
        self.table.clear()
        self.hits = 0
        self.misses = 0
        self.collisions = 0

    def get_statistics(self) -> dict:
        """Get transposition table statistics."""
        total_probes = self.hits + self.misses
        hit_rate = self.hits / total_probes if total_probes > 0 else 0.0

        return {
            "size": len(self.table),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "collisions": self.collisions,
            "hit_rate": hit_rate,
        }

    def __len__(self) -> int:
        """Return number of entries in table."""
        return len(self.table)


# ============================================================================
# Unit Tests
# ============================================================================

def _test_zobrist():
    """Test Zobrist hashing."""
    print("Testing Zobrist hashing...")

    hasher = ZobristHasher(seed=42)

    # Test starting position
    board = chess.Board()
    hash1 = hasher.hash(board)
    print(f"  Starting position hash: {hash1}")
    assert hash1 != 0, "Hash should not be zero"

    # Test hash consistency
    hash2 = hasher.hash(board)
    assert hash1 == hash2, "Hash should be consistent"
    print("  ✓ Hash consistency")

    # Test different positions have different hashes
    board.push_san("e4")
    hash3 = hasher.hash(board)
    assert hash3 != hash1, "Different positions should have different hashes"
    print("  ✓ Different positions have different hashes")

    # Test hash after undo
    board.pop()
    hash4 = hasher.hash(board)
    assert hash4 == hash1, "Hash should match after undo"
    print("  ✓ Hash matches after undo")

    print("\nZobrist hashing tests passed! ✓")


def _test_transposition_table():
    """Test transposition table."""
    print("\nTesting TranspositionTable...")

    tt = TranspositionTable(max_size=1000)

    # Test storing and retrieving
    board = chess.Board()
    tt.store(board, depth=3, value=0.5, node_type=NodeType.EXACT)

    entry = tt.probe(board, depth=3)
    assert entry is not None, "Should find stored entry"
    assert entry.value == 0.5, "Value should match"
    assert entry.depth == 3, "Depth should match"
    print("  ✓ Store and retrieve")

    # Test depth filtering
    entry = tt.probe(board, depth=5)
    assert entry is None, "Should not return entry with insufficient depth"
    print("  ✓ Depth filtering")

    # Test replacement
    tt.store(board, depth=5, value=0.8, node_type=NodeType.EXACT)
    entry = tt.probe(board, depth=3)
    assert entry is not None, "Should find updated entry"
    assert entry.value == 0.8, "Should have new value"
    assert entry.depth == 5, "Should have new depth"
    print("  ✓ Replacement with deeper search")

    # Test statistics
    stats = tt.get_statistics()
    print(f"  Statistics: {stats['hits']} hits, {stats['misses']} misses")
    print(f"  Hit rate: {stats['hit_rate']:.2%}")
    print(f"  Table size: {len(tt)}")

    # Test clearing
    tt.clear()
    assert len(tt) == 0, "Table should be empty after clear"
    print("  ✓ Clear")

    print("\nTransposition table tests passed! ✓")


def _test_performance():
    """Test hashing performance."""
    print("\nTesting Zobrist hashing performance...")

    import time

    hasher = ZobristHasher()
    board = chess.Board()

    # Hash starting position many times
    n_iterations = 10000
    start = time.time()

    for _ in range(n_iterations):
        _ = hasher.hash(board)

    elapsed = time.time() - start
    rate = n_iterations / elapsed

    print(f"  Hashed {n_iterations:,} positions in {elapsed:.3f}s")
    print(f"  Rate: {rate:,.0f} hashes/sec")

    assert rate > 1000, "Should hash at least 1000 positions per second"
    print("  ✓ Performance acceptable")

    print("\nPerformance tests passed! ✓")


if __name__ == "__main__":
    _test_zobrist()
    _test_transposition_table()
    _test_performance()

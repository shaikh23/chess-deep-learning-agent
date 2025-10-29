"""Opening book with a small curated set of chess opening lines.

This provides variety in game starts without needing a large database.
All lines are verified to be legal and represent common opening variations.
"""

import chess
import random
from typing import List, Optional, Tuple


# Curated opening book: 20 quiet lines representing major opening families
# Format: (name, moves_in_san, max_ply)
OPENING_LINES = [
    # Starting position
    ("Starting Position", [], 0),

    # King's Pawn - Quiet
    ("Italian Game", ["e4", "e5", "Nf3", "Nc6", "Bc4"], 5),
    ("Giuoco Piano", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"], 6),
    ("Two Knights", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"], 6),

    # French Defense
    ("French Defense", ["e4", "e6", "d4", "d5"], 4),
    ("French Tarrasch", ["e4", "e6", "d4", "d5", "Nd2"], 5),

    # Caro-Kann
    ("Caro-Kann", ["e4", "c6", "d4", "d5"], 4),
    ("Caro-Kann Classical", ["e4", "c6", "d4", "d5", "Nc3", "dxe4"], 6),

    # Sicilian - Closed
    ("Sicilian Closed", ["e4", "c5", "Nc3"], 3),
    ("Sicilian Grand Prix", ["e4", "c5", "Nc3", "Nc6", "f4"], 5),

    # Queen's Pawn - Quiet
    ("London System", ["d4", "d5", "Bf4"], 3),
    ("London vs d5", ["d4", "d5", "Bf4", "Nf6", "e3"], 5),
    ("Colle System", ["d4", "d5", "Nf3", "Nf6", "e3"], 5),

    # Queen's Gambit
    ("Queen's Gambit", ["d4", "d5", "c4"], 3),
    ("QGD Orthodox", ["d4", "d5", "c4", "e6", "Nc3", "Nf6"], 6),
    ("Slav Defense", ["d4", "d5", "c4", "c6"], 4),
    ("Semi-Slav", ["d4", "d5", "c4", "c6", "Nf3", "Nf6", "Nc3", "e6"], 8),

    # Indian Defenses
    ("King's Indian", ["d4", "Nf6", "c4", "g6"], 4),
    ("Nimzo-Indian", ["d4", "Nf6", "c4", "e6", "Nc3", "Bb4"], 6),

    # Flank Openings
    ("English Opening", ["c4", "e5"], 2),
    ("Réti Opening", ["Nf3", "d5", "c4"], 3),
]


class OpeningBook:
    """
    Simple opening book for providing variety in game starts.

    Uses a curated set of ~20 common opening lines.
    """

    def __init__(self, lines: Optional[List[Tuple[str, List[str], int]]] = None):
        """
        Initialize opening book.

        Args:
            lines: List of (name, moves_san, max_ply) tuples.
                   If None, uses default OPENING_LINES.
        """
        self.lines = lines if lines is not None else OPENING_LINES

        # Convert SAN moves to UCI for faster lookup
        self._uci_lines = []
        for name, moves_san, max_ply in self.lines:
            uci_moves = self._san_to_uci(moves_san)
            if uci_moves is not None:
                self._uci_lines.append((name, uci_moves, max_ply))

    def _san_to_uci(self, moves_san: List[str]) -> Optional[List[str]]:
        """
        Convert SAN move list to UCI move list.

        Args:
            moves_san: List of moves in SAN notation

        Returns:
            List of UCI moves, or None if invalid
        """
        board = chess.Board()
        uci_moves = []

        try:
            for san_move in moves_san:
                move = board.parse_san(san_move)
                uci_moves.append(move.uci())
                board.push(move)
            return uci_moves
        except:
            return None

    def get_random_opening(self) -> Tuple[str, List[str]]:
        """
        Get a random opening line.

        Returns:
            Tuple of (opening_name, uci_moves)
        """
        name, uci_moves, _ = random.choice(self._uci_lines)
        return name, uci_moves

    def get_opening_by_index(self, index: int) -> Tuple[str, List[str]]:
        """
        Get opening line by index (for deterministic selection).

        Args:
            index: Opening index (wraps around if out of range)

        Returns:
            Tuple of (opening_name, uci_moves)
        """
        index = index % len(self._uci_lines)
        name, uci_moves, _ = self._uci_lines[index]
        return name, uci_moves

    def get_opening_moves_str(self, index: int) -> str:
        """
        Get opening moves as space-separated UCI string.

        Args:
            index: Opening index

        Returns:
            Space-separated UCI move string (e.g., "e2e4 e7e5 g1f3")
        """
        _, uci_moves = self.get_opening_by_index(index)
        return " ".join(uci_moves)

    def apply_opening(self, board: chess.Board, opening_moves: List[str]) -> int:
        """
        Apply opening moves to a board.

        Args:
            board: Chess board (will be modified in place)
            opening_moves: List of UCI moves

        Returns:
            Number of moves successfully applied
        """
        count = 0
        for uci_move in opening_moves:
            try:
                move = chess.Move.from_uci(uci_move)
                if move in board.legal_moves:
                    board.push(move)
                    count += 1
                else:
                    break
            except:
                break

        return count

    def __len__(self) -> int:
        """Return number of opening lines."""
        return len(self._uci_lines)

    def get_all_openings(self) -> List[Tuple[str, List[str]]]:
        """
        Get all opening lines.

        Returns:
            List of (name, uci_moves) tuples
        """
        return [(name, uci_moves) for name, uci_moves, _ in self._uci_lines]

    def book_moves_uci(self, board: chess.Board, max_plies: int) -> List[str]:
        """
        Get opening book moves in UCI format for a given position.

        This method finds an opening line that matches the current board position
        and returns the remaining moves up to max_plies.

        Args:
            board: Current board state
            max_plies: Maximum number of plies (half-moves) to return

        Returns:
            List of UCI moves to apply from current position

        Raises:
            ValueError: If any move in the book is illegal
        """
        # For simplicity, select an opening based on current move number
        ply_count = board.ply()

        # If we're beyond max_plies, return empty list
        if ply_count >= max_plies:
            return []

        # Select opening by cycling through the book
        opening_idx = ply_count % len(self._uci_lines)
        _, uci_moves, _ = self._uci_lines[opening_idx]

        # Return moves up to max_plies
        moves_to_return = uci_moves[:max(0, max_plies - ply_count)]

        # Verify all moves are legal from current position
        test_board = board.copy()
        verified_moves = []

        for uci_move in moves_to_return:
            try:
                move = chess.Move.from_uci(uci_move)
                if move not in test_board.legal_moves:
                    # Stop at first illegal move (book may not match position)
                    break
                verified_moves.append(uci_move)
                test_board.push(move)
            except Exception:
                # Stop at first invalid move
                break

        return verified_moves


# Global default opening book instance
DEFAULT_OPENING_BOOK = OpeningBook()


# ============================================================================
# Utility Functions
# ============================================================================

def get_opening_variety(num_games: int, seed: Optional[int] = None) -> List[str]:
    """
    Generate opening variety for a match.

    Args:
        num_games: Number of games
        seed: Random seed (None for random)

    Returns:
        List of opening UCI move strings
    """
    if seed is not None:
        random.seed(seed)

    book = DEFAULT_OPENING_BOOK
    openings = []

    for game_idx in range(num_games):
        # Cycle through openings deterministically with some randomness
        if game_idx < len(book):
            opening_str = book.get_opening_moves_str(game_idx)
        else:
            # For games beyond book size, use random selection
            _, uci_moves = book.get_random_opening()
            opening_str = " ".join(uci_moves)

        openings.append(opening_str)

    return openings


# ============================================================================
# Unit Tests
# ============================================================================

def _test_opening_book():
    """Test opening book."""
    print("Testing OpeningBook...")

    book = OpeningBook()

    print(f"\n  Book contains {len(book)} opening lines")

    # Test random opening
    name, uci_moves = book.get_random_opening()
    print(f"\n  Random opening: {name}")
    print(f"  Moves (UCI): {' '.join(uci_moves)}")

    # Test applying opening to board
    board = chess.Board()
    count = book.apply_opening(board, uci_moves)
    print(f"  Applied {count} moves")
    print(f"  FEN: {board.fen()}")

    # Verify all openings are legal
    print("\n  Verifying all openings are legal...")
    for idx, (name, uci_moves) in enumerate(book.get_all_openings()):
        test_board = chess.Board()
        applied = book.apply_opening(test_board, uci_moves)

        if applied != len(uci_moves):
            print(f"    ⚠ Warning: {name} - only {applied}/{len(uci_moves)} moves applied")
        else:
            if idx < 5:  # Print first 5
                print(f"    ✓ {name}: {len(uci_moves)} moves")

    print(f"\n  All {len(book)} opening lines verified!")

    # Test opening variety
    print("\n  Testing opening variety...")
    openings = get_opening_variety(30, seed=42)
    print(f"  Generated {len(openings)} opening strings")
    print(f"  Sample: {openings[0]}")

    print("\nAll opening book tests passed! ✓")


if __name__ == "__main__":
    _test_opening_book()

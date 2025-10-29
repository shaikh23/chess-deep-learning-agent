"""Extract training positions from PGN files.

This module parses chess games in PGN format and extracts positions along with
metadata (best move, outcome, game phase) for supervised learning.
"""

import chess
import chess.pgn
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import io


def classify_game_phase(board: chess.Board) -> str:
    """
    Classify the current position into opening, middlegame, or endgame.

    Simple heuristic based on piece count:
    - Opening: > 24 pieces (first ~10 moves)
    - Middlegame: 13-24 pieces
    - Endgame: <= 12 pieces

    Args:
        board: chess.Board object

    Returns:
        Phase name: "opening", "middlegame", or "endgame"
    """
    piece_count = len(board.piece_map())

    if piece_count > 24:
        return "opening"
    elif piece_count > 12:
        return "middlegame"
    else:
        return "endgame"


def extract_positions_from_game(
    game: chess.pgn.Game,
    min_elo: int = 1800,
    skip_first_n_moves: int = 3,
) -> List[Dict]:
    """
    Extract positions from a single game.

    Args:
        game: python-chess Game object
        min_elo: Minimum player Elo rating to include
        skip_first_n_moves: Skip opening book moves (default: 3)

    Returns:
        List of position dictionaries with keys:
        - fen: Position in FEN notation
        - move: Best move played (UCI format)
        - outcome: Game result from position player's perspective (1.0/0.5/0.0)
        - phase: Game phase (opening/middlegame/endgame)
        - move_number: Half-move number
    """
    positions = []

    # Check minimum Elo requirement
    try:
        white_elo = int(game.headers.get("WhiteElo", 0))
        black_elo = int(game.headers.get("BlackElo", 0))

        if white_elo < min_elo or black_elo < min_elo:
            return positions
    except (ValueError, TypeError):
        return positions

    # Get game outcome
    result = game.headers.get("Result", "*")
    if result not in ["1-0", "0-1", "1/2-1/2"]:
        return positions

    # Parse game moves
    board = game.board()
    move_number = 0

    for move in game.mainline_moves():
        move_number += 1

        # Skip opening book moves
        if move_number <= skip_first_n_moves:
            board.push(move)
            continue

        # Get position before move
        fen = board.fen()
        phase = classify_game_phase(board)

        # Determine outcome from current player's perspective
        if result == "1/2-1/2":
            outcome = 0.5
        elif (result == "1-0" and board.turn == chess.WHITE) or \
             (result == "0-1" and board.turn == chess.BLACK):
            outcome = 1.0
        else:
            outcome = 0.0

        positions.append({
            "fen": fen,
            "move": move.uci(),
            "outcome": outcome,
            "phase": phase,
            "move_number": move_number,
        })

        board.push(move)

    return positions


def extract_positions_from_pgn(
    pgn_path: Path,
    max_games: Optional[int] = None,
    min_elo: int = 1800,
    skip_first_n_moves: int = 3,
) -> pd.DataFrame:
    """
    Extract positions from PGN file.

    Args:
        pgn_path: Path to PGN file
        max_games: Maximum number of games to process (None = all)
        min_elo: Minimum player Elo rating
        skip_first_n_moves: Skip opening book moves

    Returns:
        DataFrame with columns: fen, move, outcome, phase, move_number
    """
    all_positions = []

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as pgn_file:
        game_count = 0
        pbar = tqdm(desc="Processing games", unit=" games")

        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break

            positions = extract_positions_from_game(
                game,
                min_elo=min_elo,
                skip_first_n_moves=skip_first_n_moves,
            )

            all_positions.extend(positions)
            game_count += 1
            pbar.update(1)

            if max_games is not None and game_count >= max_games:
                break

        pbar.close()

    print(f"\nExtracted {len(all_positions):,} positions from {game_count:,} games")

    return pd.DataFrame(all_positions)


def save_positions_to_csv(
    df: pd.DataFrame,
    output_path: Path,
    compression: str = "gzip",
) -> None:
    """
    Save positions DataFrame to CSV with optional compression.

    Args:
        df: Positions DataFrame
        output_path: Output file path
        compression: Compression type ('gzip', 'bz2', 'zip', or None)
    """
    df.to_csv(output_path, index=False, compression=compression)
    print(f"Saved {len(df):,} positions to {output_path}")


def load_positions_from_csv(
    csv_path: Path,
    compression: str = "gzip",
    sample_frac: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load positions from CSV file.

    Args:
        csv_path: Path to CSV file
        compression: Compression type ('gzip', 'bz2', 'zip', or None)
        sample_frac: Optional fraction of data to sample (0.0-1.0)

    Returns:
        DataFrame with positions
    """
    df = pd.read_csv(csv_path, compression=compression)

    if sample_frac is not None and 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
        print(f"Sampled {len(df):,} positions ({sample_frac:.1%})")

    print(f"Loaded {len(df):,} positions from {csv_path}")
    return df


def clean_positions(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
    balance_outcomes: bool = False,
) -> pd.DataFrame:
    """
    Clean and preprocess positions DataFrame.

    Args:
        df: Raw positions DataFrame
        remove_duplicates: Remove duplicate FENs
        balance_outcomes: Balance by outcome (1.0/0.5/0.0)

    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)

    # Remove duplicates
    if remove_duplicates:
        df = df.drop_duplicates(subset=["fen"], keep="first")
        print(f"Removed {initial_count - len(df):,} duplicate positions")

    # Balance outcomes
    if balance_outcomes:
        min_count = df.groupby("outcome").size().min()
        df = df.groupby("outcome").sample(n=min_count, random_state=42)
        print(f"Balanced to {min_count:,} positions per outcome")

    print(f"Final dataset: {len(df):,} positions")
    return df


def get_eda_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute EDA statistics for positions dataset.

    Args:
        df: Positions DataFrame

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_positions": len(df),
        "phase_distribution": df["phase"].value_counts().to_dict(),
        "outcome_distribution": df["outcome"].value_counts().to_dict(),
        "move_number_stats": {
            "mean": df["move_number"].mean(),
            "median": df["move_number"].median(),
            "min": df["move_number"].min(),
            "max": df["move_number"].max(),
        },
    }

    return stats


# ============================================================================
# Example Usage
# ============================================================================

def _demo_extraction():
    """Demonstrate position extraction from a sample PGN."""
    print("Demo: Position extraction from PGN")

    # Create a sample PGN in memory
    sample_pgn = """
[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[Date "2024.01.15"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "1950"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3 Na5 10. Bc2 c5 11. d4 1-0
"""

    # Parse game
    pgn = io.StringIO(sample_pgn)
    game = chess.pgn.read_game(pgn)

    # Extract positions
    positions = extract_positions_from_game(game, min_elo=1800, skip_first_n_moves=3)

    print(f"\nExtracted {len(positions)} positions:")
    for i, pos in enumerate(positions[:3], 1):
        print(f"\nPosition {i}:")
        print(f"  FEN: {pos['fen']}")
        print(f"  Move: {pos['move']}")
        print(f"  Phase: {pos['phase']}")
        print(f"  Outcome: {pos['outcome']}")


if __name__ == "__main__":
    _demo_extraction()

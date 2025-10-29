"""Match runner for playing games between engines with PGN logging and statistics.

This module provides a robust match runner with:
- Single authoritative Board (python-chess) as source of truth
- UCI-only I/O for all engines
- External opening book only (engines must disable internal books)
- Desync guard: validates every engine move before applying
- Per-ply JSON logging and PGN output
"""

import chess
import chess.pgn
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm


# Opening book (short lines for variety) - UCI format
OPENING_BOOK = [
    "",  # Starting position
    "e2e4",
    "e2e4 e7e5",
    "e2e4 c7c5",
    "e2e4 e7e6",
    "d2d4",
    "d2d4 d7d5",
    "d2d4 g8f6",
    "g1f3",
    "c2c4",
    "e2e4 e7e5 g1f3",
    "e2e4 c7c5 g1f3",
    "d2d4 d7d5 c2c4",
    "d2d4 g8f6 c2c4",
    "e2e4 e7e5 g1f3 b8c6",
    "d2d4 d7d5 c2c4 e7e6",
    "g1f3 d7d5 d2d4",
    "c2c4 g8f6",
    "g1f3 g8f6 c2c4",
    "e2e4 e7e5 g1f3 b8c6 f1b5",  # Ruy Lopez
]


@dataclass
class MatchResult:
    """Result of a single game."""
    game_number: int
    white_engine: str
    black_engine: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    num_moves: int
    duration_seconds: float
    opening: str
    termination: str  # "checkmate", "stalemate", etc.
    pgn: str


@dataclass
class MatchStatistics:
    """Statistics for a match (multiple games)."""
    total_games: int
    wins: int
    draws: int
    losses: int
    score: float  # wins + 0.5 * draws
    avg_moves: float
    avg_duration: float


class MatchRunner:
    """
    Run matches between chess engines with PGN logging and robust desync guards.

    Architecture:
    - Maintains single authoritative python-chess Board
    - Recomputes FEN before each engine call
    - Always sends full UCI move history to engines
    - Validates every engine move before applying
    - Logs per-ply JSON and PGN per game
    """

    def __init__(
        self,
        player1,
        player2,
        output_dir: Path,
        opening_book: Optional[List[str]] = None,
        enable_desync_guard: bool = True,
        log_per_ply: bool = False,
        enable_adjudication: bool = False,
        adjudication_threshold: float = 5.0,
        adjudication_plies: int = 3,
    ):
        """
        Initialize match runner.

        Args:
            player1: First engine (must have get_move() and get_name() methods)
            player2: Second engine
            output_dir: Directory for PGN and JSON outputs
            opening_book: List of opening move sequences (UCI format)
            enable_desync_guard: Enable move validation (default: True)
            log_per_ply: Enable per-ply JSON logging (default: False)
            enable_adjudication: Enable adjudication by evaluation (default: False)
            adjudication_threshold: Adjudication threshold in pawns (default: 5.0)
            adjudication_plies: Number of consecutive plies for adjudication (default: 3)
        """
        self.player1 = player1
        self.player2 = player2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.opening_book = opening_book or OPENING_BOOK
        self.enable_desync_guard = enable_desync_guard
        self.log_per_ply = log_per_ply
        self.enable_adjudication = enable_adjudication
        self.adjudication_threshold = adjudication_threshold
        self.adjudication_plies = adjudication_plies

        self.results: List[MatchResult] = []
        self.ply_logs: List[Dict] = []

    def play_game(
        self,
        white,
        black,
        game_number: int,
        opening_moves: str = "",
        max_moves: int = 200,
    ) -> MatchResult:
        """
        Play a single game with robust move validation.

        Uses single authoritative Board:
        1. Apply opening from external book
        2. For each ply:
           - Recompute FEN
           - Call engine with FEN + full move history
           - Validate returned move (desync guard)
           - Apply to authoritative Board
           - Log (optional)

        Args:
            white: White player engine
            black: Black player engine
            game_number: Game number
            opening_moves: Opening moves in UCI format (space-separated)
            max_moves: Maximum number of moves before draw

        Returns:
            MatchResult with game data
        """
        # Single authoritative Board
        board = chess.Board()
        start_time = time.time()

        # UCI move history (authoritative)
        uci_history: List[str] = []

        # Apply opening moves from external book
        opening_moves_list = opening_moves.split() if opening_moves else []
        for move_uci in opening_moves_list:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                    uci_history.append(move_uci)
                else:
                    print(f"Warning: Illegal opening move {move_uci}, stopping book.")
                    break
            except Exception as e:
                print(f"Warning: Invalid opening move {move_uci}: {e}")
                break

        # Play game
        move_count = 0
        adjudication_scores = []  # Track scores for adjudication

        while not board.is_game_over() and move_count < max_moves:
            try:
                current_player = white if board.turn == chess.WHITE else black
                ply_number = len(uci_history) + 1

                # Get move from engine
                move = current_player.get_move(board)

                # Desync guard: validate move before applying
                if self.enable_desync_guard:
                    if move not in board.legal_moves:
                        # Illegal move - log context and abort game
                        recent_moves = uci_history[-8:] if len(uci_history) > 8 else uci_history
                        print(f"\n✗ DESYNC DETECTED at game {game_number}, ply {ply_number}:")
                        print(f"  Engine: {current_player.get_name()}")
                        print(f"  Illegal move: {move.uci()}")
                        print(f"  FEN: {board.fen()}")
                        print(f"  Last 8 moves: {recent_moves}")
                        print(f"  Legal moves: {[m.uci() for m in list(board.legal_moves)[:10]]}")
                        break

                # Apply move to authoritative Board
                move_uci = move.uci()
                board.push(move)
                uci_history.append(move_uci)
                move_count += 1

                # Check adjudication if enabled
                if self.enable_adjudication and hasattr(white, 'get_evaluation'):
                    try:
                        # Get evaluation from white player's perspective
                        eval_score = white.get_evaluation(board) if board.turn == chess.WHITE else black.get_evaluation(board)
                        adjudication_scores.append(eval_score)

                        # Check if we should adjudicate
                        if len(adjudication_scores) >= self.adjudication_plies:
                            recent_scores = adjudication_scores[-self.adjudication_plies:]
                            # Check if all recent scores exceed threshold (same sign)
                            if all(score >= self.adjudication_threshold for score in recent_scores):
                                result = "1-0"
                                termination = "adjudication (white winning)"
                                break
                            elif all(score <= -self.adjudication_threshold for score in recent_scores):
                                result = "0-1"
                                termination = "adjudication (black winning)"
                                break
                    except:
                        pass  # Adjudication failed, continue playing

                # Optional per-ply logging
                if self.log_per_ply:
                    ply_log = {
                        "game": game_number,
                        "ply": ply_number,
                        "side": "white" if board.turn == chess.BLACK else "black",  # Side that just moved
                        "fen_before": board.copy(stack=False).fen(),
                        "move_uci": move_uci,
                        "source": current_player.get_name(),
                    }
                    self.ply_logs.append(ply_log)

            except Exception as e:
                print(f"Error during game {game_number}, ply {len(uci_history) + 1}: {e}")
                break

        duration = time.time() - start_time

        # Determine result and termination reason (if not already set by adjudication)
        if 'result' not in locals():
            if board.is_checkmate():
                result = "1-0" if board.turn == chess.BLACK else "0-1"
                termination = "checkmate"
            elif board.is_stalemate():
                result = "1/2-1/2"
                termination = "stalemate"
            elif board.is_insufficient_material():
                result = "1/2-1/2"
                termination = "insufficient material"
            elif board.is_seventyfive_moves():
                result = "1/2-1/2"
                termination = "75-move rule"
            elif board.is_fivefold_repetition():
                result = "1/2-1/2"
                termination = "repetition"
            elif move_count >= max_moves:
                result = "1/2-1/2"
                termination = "max moves reached"
            else:
                result = "1/2-1/2"
                termination = "unknown"

        # Create PGN
        game = chess.pgn.Game()
        game.headers["Event"] = "Neural Agent Match"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["Round"] = str(game_number)
        game.headers["White"] = white.get_name()
        game.headers["Black"] = black.get_name()
        game.headers["Result"] = result
        game.headers["Termination"] = termination

        # Add moves to PGN (from authoritative Board)
        node = game
        temp_board = chess.Board()

        for move_uci in uci_history:
            try:
                move = chess.Move.from_uci(move_uci)
                node = node.add_variation(move)
                temp_board.push(move)
            except:
                break

        pgn_string = str(game)

        return MatchResult(
            game_number=game_number,
            white_engine=white.get_name(),
            black_engine=black.get_name(),
            result=result,
            num_moves=len(uci_history),
            duration_seconds=duration,
            opening=opening_moves,
            termination=termination,
            pgn=pgn_string,
        )

    def run_match(
        self,
        num_games: int,
        alternate_colors: bool = True,
        max_moves_per_game: int = 200,
    ) -> MatchStatistics:
        """
        Run a match of multiple games.

        Args:
            num_games: Number of games to play
            alternate_colors: Alternate colors each game
            max_moves_per_game: Maximum moves per game

        Returns:
            MatchStatistics
        """
        self.results = []
        self.ply_logs = []

        # Determine player names
        player1_name = self.player1.get_name()
        player2_name = self.player2.get_name()

        print(f"\nStarting match: {player1_name} vs {player2_name}")
        print(f"Games: {num_games}, Alternate colors: {alternate_colors}")
        print(f"Desync guard: {'enabled' if self.enable_desync_guard else 'disabled'}")
        print(f"Per-ply logging: {'enabled' if self.log_per_ply else 'disabled'}\n")

        for game_num in tqdm(range(1, num_games + 1), desc="Playing games"):
            # Determine colors
            if alternate_colors:
                if game_num % 2 == 1:
                    white, black = self.player1, self.player2
                else:
                    white, black = self.player2, self.player1
            else:
                white, black = self.player1, self.player2

            # Select opening
            opening = self.opening_book[game_num % len(self.opening_book)]

            # Play game
            result = self.play_game(
                white,
                black,
                game_num,
                opening,
                max_moves_per_game,
            )

            self.results.append(result)

        # Compute statistics (from player1's perspective)
        stats = self._compute_statistics(player1_name)

        # Save results
        self._save_results()

        return stats

    def _compute_statistics(self, player_name: str) -> MatchStatistics:
        """
        Compute match statistics from player's perspective.

        Args:
            player_name: Name of player to compute stats for

        Returns:
            MatchStatistics
        """
        wins = 0
        draws = 0
        losses = 0
        total_moves = 0
        total_duration = 0.0

        for result in self.results:
            total_moves += result.num_moves
            total_duration += result.duration_seconds

            # Determine outcome from player's perspective
            if result.white_engine == player_name:
                if result.result == "1-0":
                    wins += 1
                elif result.result == "1/2-1/2":
                    draws += 1
                else:
                    losses += 1
            else:  # Player was black
                if result.result == "0-1":
                    wins += 1
                elif result.result == "1/2-1/2":
                    draws += 1
                else:
                    losses += 1

        total_games = len(self.results)
        score = wins + 0.5 * draws

        return MatchStatistics(
            total_games=total_games,
            wins=wins,
            draws=draws,
            losses=losses,
            score=score,
            avg_moves=total_moves / total_games if total_games > 0 else 0,
            avg_duration=total_duration / total_games if total_games > 0 else 0,
        )

    def _save_results(self):
        """Save match results to PGN and JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        player1_name = self.player1.get_name().replace(" ", "_")
        player2_name = self.player2.get_name().replace(" ", "_")

        # Save PGNs
        pgn_path = self.output_dir / f"match_{player1_name}_vs_{player2_name}_{timestamp}.pgn"
        with open(pgn_path, "w") as f:
            for result in self.results:
                f.write(result.pgn + "\n\n")

        print(f"\nSaved PGNs to: {pgn_path}")

        # Save JSON statistics
        json_data = {
            "player1": player1_name,
            "player2": player2_name,
            "timestamp": timestamp,
            "num_games": len(self.results),
            "results": [asdict(r) for r in self.results],
            "statistics_player1": asdict(self._compute_statistics(player1_name)),
            "statistics_player2": asdict(self._compute_statistics(player2_name)),
        }

        # Add per-ply logs if enabled
        if self.log_per_ply:
            json_data["ply_logs"] = self.ply_logs

        json_path = self.output_dir / f"match_{player1_name}_vs_{player2_name}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"Saved statistics to: {json_path}")


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Command-line interface for running matches."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run chess engine matches with PGN logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # vs Sunfish
  python -m src.play.match_runner --opponent sunfish --games 50 --depth 2

  # vs Stockfish
  python -m src.play.match_runner --opponent stockfish --games 50 --stockfish-skill 5 --movetime 300

  # vs Maia
  python -m src.play.match_runner --opponent maia --games 50 --movetime 300 \\
    --maia-weights weights/maia1500.pb.gz --lc0-path /usr/local/bin/lc0
        """,
    )

    parser.add_argument(
        "--opponent",
        type=str,
        required=True,
        choices=["sunfish", "stockfish", "maia"],
        help="Opponent engine (sunfish, stockfish, or maia)",
    )

    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of games to play (default: 50)",
    )

    parser.add_argument(
        "--movetime",
        type=int,
        default=None,
        help="Time per move in milliseconds (e.g., 300)",
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Search depth (alternative to movetime)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/matches",
        help="Output directory for PGNs and logs (default: artifacts/matches)",
    )

    # Stockfish options
    parser.add_argument(
        "--stockfish-path",
        type=str,
        default="/opt/homebrew/bin/stockfish",
        help="Path to Stockfish binary",
    )

    parser.add_argument(
        "--stockfish-skill",
        type=int,
        default=5,
        help="Stockfish skill level 0-20 (default: 5)",
    )

    # Maia options
    parser.add_argument(
        "--lc0-path",
        type=str,
        default="/usr/local/bin/lc0",
        help="Path to Lc0 binary (for Maia)",
    )

    parser.add_argument(
        "--maia-weights",
        type=str,
        default=None,
        help="Path to Maia weights file (required for --opponent maia)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads for Maia/Lc0 (default: 1)",
    )

    # Sunfish options
    parser.add_argument(
        "--sunfish-depth",
        type=int,
        default=2,
        help="Sunfish search depth (default: 2)",
    )

    # Match options
    parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Maximum moves per game (default: 200)",
    )

    parser.add_argument(
        "--no-desync-guard",
        action="store_true",
        help="Disable desync guard (not recommended)",
    )

    parser.add_argument(
        "--log-per-ply",
        action="store_true",
        help="Enable per-ply JSON logging",
    )

    args = parser.parse_args()

    # Import engines
    from src.play.sunfish_wrapper import SunfishWrapper
    from src.play.stockfish_wrapper import StockfishWrapper

    # Create opponent
    if args.opponent == "sunfish":
        opponent = SunfishWrapper(depth=args.sunfish_depth)
        print(f"Opponent: Sunfish (depth={args.sunfish_depth})")

    elif args.opponent == "stockfish":
        opponent = StockfishWrapper(
            stockfish_path=args.stockfish_path,
            skill_level=args.stockfish_skill,
            time_limit=args.movetime / 1000.0 if args.movetime else 0.2,
            depth_limit=args.depth,
        )
        opponent.__enter__()  # Start engine
        print(f"Opponent: Stockfish (skill={args.stockfish_skill})")

    elif args.opponent == "maia":
        if not args.maia_weights:
            print("Error: --maia-weights is required when using --opponent maia")
            return 1

        from src.play.maia_lc0_wrapper import MaiaLc0Engine

        opponent = MaiaLc0Engine(
            lc0_path=args.lc0_path,
            weights_path=args.maia_weights,
            movetime_ms=args.movetime or 300,
            depth=args.depth,
            threads=args.threads,
        )
        opponent.__enter__()  # Start engine
        print(f"Opponent: Maia (weights={args.maia_weights}, threads={args.threads})")

    else:
        print(f"Error: Unknown opponent {args.opponent}")
        return 1

    # Create dummy player1 (placeholder for neural agent)
    # For now, use Sunfish as player1
    player1 = SunfishWrapper(depth=args.sunfish_depth)
    print(f"Player1: Sunfish (depth={args.sunfish_depth})")

    # Create match runner
    runner = MatchRunner(
        player1=player1,
        player2=opponent,
        output_dir=Path(args.output_dir),
        enable_desync_guard=not args.no_desync_guard,
        log_per_ply=args.log_per_ply,
    )

    # Run match
    try:
        stats = runner.run_match(
            num_games=args.games,
            alternate_colors=True,
            max_moves_per_game=args.max_moves,
        )

        # Print results
        print("\n" + "=" * 70)
        print("MATCH RESULTS")
        print("=" * 70)
        print(f"Total games: {stats.total_games}")
        print(f"Wins: {stats.wins}, Draws: {stats.draws}, Losses: {stats.losses}")
        print(f"Score: {stats.score}/{stats.total_games}")
        print(f"Avg moves: {stats.avg_moves:.1f}")
        print(f"Avg duration: {stats.avg_duration:.2f}s")
        print("=" * 70)

    finally:
        # Cleanup
        if args.opponent in ["stockfish", "maia"]:
            opponent.__exit__(None, None, None)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

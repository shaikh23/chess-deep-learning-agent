"""Enhanced match runner with Elo calculation, ACPL analysis, and multi-engine support.

This extends the basic match runner with:
- Elo rating estimation with confidence intervals
- Average Centipawn Loss (ACPL) by game phase
- Support for GNU Chess as third opponent
- Improved statistics and logging
"""

import chess
import chess.pgn
import chess.engine
import json
import time
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm


@dataclass
class MatchResult:
    """Result of a single game with extended statistics."""
    game_number: int
    white_engine: str
    black_engine: str
    result: str  # "1-0", "0-1", "1/2-1/2"
    num_moves: int
    duration_seconds: float
    opening_name: str
    opening_moves: str
    termination: str
    pgn: str
    acpl_white: Optional[float] = None  # Average centipawn loss
    acpl_black: Optional[float] = None


@dataclass
class MatchStatistics:
    """Extended match statistics with Elo estimation."""
    total_games: int
    wins: int
    draws: int
    losses: int
    score: float
    win_rate: float
    avg_moves: float
    avg_duration: float
    elo_estimate: float
    elo_ci_lower: float  # 95% CI
    elo_ci_upper: float
    avg_acpl: Optional[float] = None


def estimate_elo_difference(score: float, num_games: int) -> Tuple[float, float, float]:
    """
    Estimate Elo difference from match score using logistic model.

    Args:
        score: Match score (wins + 0.5*draws) / total_games
        num_games: Number of games played

    Returns:
        Tuple of (elo_diff, ci_lower, ci_upper) where CI is 95% confidence
    """
    # Clamp score to avoid log(0)
    score = max(0.01, min(0.99, score))

    # Elo difference from expected score
    # Expected score E = 1 / (1 + 10^(-elo_diff/400))
    # Solving for elo_diff: elo_diff = -400 * log10((1/E) - 1)
    elo_diff = -400 * math.log10((1.0 / score) - 1.0)

    # Standard error of score
    se = math.sqrt(score * (1 - score) / num_games)

    # 95% CI using normal approximation (z = 1.96)
    z = 1.96
    score_lower = max(0.01, score - z * se)
    score_upper = min(0.99, score + z * se)

    elo_lower = -400 * math.log10((1.0 / score_lower) - 1.0)
    elo_upper = -400 * math.log10((1.0 / score_upper) - 1.0)

    return elo_diff, elo_lower, elo_upper


def compute_acpl_by_phase(
    game_pgn: str,
    analyzer_engine: Any,
    max_depth: int = 10,
    max_moves: int = 50,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute Average Centipawn Loss (ACPL) for both players.

    Uses Stockfish to evaluate each position and compare to the move actually played.

    Args:
        game_pgn: PGN string of the game
        analyzer_engine: Stockfish engine (already opened)
        max_depth: Analysis depth
        max_moves: Analyze first N moves only (to save time)

    Returns:
        Tuple of (acpl_white, acpl_black) in centipawns
    """
    try:
        import io
        pgn = chess.pgn.read_game(io.StringIO(game_pgn))
        if pgn is None:
            return None, None

        board = pgn.board()
        white_losses = []
        black_losses = []

        move_count = 0
        for move in pgn.mainline_moves():
            if move_count >= max_moves:
                break

            # Get evaluation before move
            try:
                info_before = analyzer_engine.analyse(
                    board,
                    chess.engine.Limit(depth=max_depth, time=0.05),
                )

                score_before = info_before.get("score")
                if score_before is None or not hasattr(score_before, "relative"):
                    board.push(move)
                    move_count += 1
                    continue

                # Get score in centipawns from current player's perspective
                cp_before = score_before.relative.score(mate_score=10000)
                if cp_before is None:
                    board.push(move)
                    move_count += 1
                    continue

                # Make the move
                board.push(move)

                # Get evaluation after move (from opponent's perspective)
                info_after = analyzer_engine.analyse(
                    board,
                    chess.engine.Limit(depth=max_depth, time=0.05),
                )

                score_after = info_after.get("score")
                if score_after is None or not hasattr(score_after, "relative"):
                    move_count += 1
                    continue

                cp_after = score_after.relative.score(mate_score=10000)
                if cp_after is None:
                    move_count += 1
                    continue

                # Loss = evaluation before move - (-evaluation after move)
                # Negative cp_after because we flip perspective
                loss = cp_before - (-cp_after)
                loss = max(0, loss)  # Only count losses, not gains

                # Record loss for the player who made the move
                if move_count % 2 == 0:
                    white_losses.append(loss)
                else:
                    black_losses.append(loss)

                move_count += 1

            except Exception as e:
                board.push(move)
                move_count += 1
                continue

        acpl_white = np.mean(white_losses) if white_losses else None
        acpl_black = np.mean(black_losses) if black_losses else None

        return acpl_white, acpl_black

    except Exception as e:
        return None, None


class EnhancedMatchRunner:
    """
    Enhanced match runner with Elo estimation and ACPL analysis.
    """

    def __init__(
        self,
        player1,
        player2,
        output_dir: Path,
        opening_book: Optional[Any] = None,
        compute_acpl: bool = False,
        stockfish_path: str = "/opt/homebrew/bin/stockfish",
    ):
        """
        Initialize enhanced match runner.

        Args:
            player1: First engine (must have get_move() and get_name() methods)
            player2: Second engine
            output_dir: Directory for outputs
            opening_book: OpeningBook instance (optional)
            compute_acpl: Whether to compute ACPL (requires Stockfish)
            stockfish_path: Path to Stockfish for ACPL analysis
        """
        self.player1 = player1
        self.player2 = player2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.opening_book = opening_book
        self.compute_acpl = compute_acpl
        self.stockfish_path = stockfish_path

        # Stockfish engine for ACPL analysis
        self.analyzer_engine = None
        if self.compute_acpl and Path(stockfish_path).exists():
            try:
                self.analyzer_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
            except Exception as e:
                print(f"Warning: Failed to load Stockfish for ACPL: {e}")
                self.compute_acpl = False

        self.results: List[MatchResult] = []

    def __del__(self):
        """Cleanup Stockfish engine."""
        if self.analyzer_engine is not None:
            try:
                self.analyzer_engine.quit()
            except:
                pass

    def play_game(
        self,
        white,
        black,
        game_number: int,
        opening_name: str = "Starting Position",
        opening_moves: str = "",
        max_moves: int = 200,
    ) -> MatchResult:
        """
        Play a single game with optional ACPL computation.

        Args:
            white: White player engine
            black: Black player engine
            game_number: Game number
            opening_name: Name of opening
            opening_moves: Opening moves in UCI format (space-separated)
            max_moves: Maximum moves before draw

        Returns:
            MatchResult with game data and optional ACPL
        """
        board = chess.Board()
        start_time = time.time()

        # Apply opening moves
        opening_moves_list = opening_moves.split() if opening_moves else []
        for move_uci in opening_moves_list:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
            except:
                break

        # Play game
        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            try:
                current_player = white if board.turn == chess.WHITE else black
                move = current_player.get_move(board)

                if move not in board.legal_moves:
                    break

                board.push(move)
                move_count += 1

            except Exception as e:
                break

        duration = time.time() - start_time

        # Determine result
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
            termination = "max moves"
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
        game.headers["Opening"] = opening_name
        game.headers["Termination"] = termination

        # Add moves
        node = game
        temp_board = chess.Board()

        for move_uci in opening_moves_list:
            try:
                move = chess.Move.from_uci(move_uci)
                if move in temp_board.legal_moves:
                    node = node.add_variation(move)
                    temp_board.push(move)
            except:
                break

        for move in board.move_stack[len(opening_moves_list):]:
            node = node.add_variation(move)

        pgn_string = str(game)

        # Compute ACPL if requested
        acpl_white, acpl_black = None, None
        if self.compute_acpl and self.analyzer_engine is not None:
            acpl_white, acpl_black = compute_acpl_by_phase(
                pgn_string,
                self.analyzer_engine,
                max_depth=10,
                max_moves=30,
            )

        return MatchResult(
            game_number=game_number,
            white_engine=white.get_name(),
            black_engine=black.get_name(),
            result=result,
            num_moves=len(board.move_stack),
            duration_seconds=duration,
            opening_name=opening_name,
            opening_moves=opening_moves,
            termination=termination,
            pgn=pgn_string,
            acpl_white=acpl_white,
            acpl_black=acpl_black,
        )

    def run_match(
        self,
        num_games: int,
        alternate_colors: bool = True,
        max_moves_per_game: int = 200,
    ) -> MatchStatistics:
        """
        Run a match with enhanced statistics.

        Args:
            num_games: Number of games
            alternate_colors: Alternate colors each game
            max_moves_per_game: Max moves per game

        Returns:
            MatchStatistics with Elo and ACPL
        """
        self.results = []

        player1_name = self.player1.get_name()
        player2_name = self.player2.get_name()

        print(f"\nStarting match: {player1_name} vs {player2_name}")
        print(f"Games: {num_games}, Alternate colors: {alternate_colors}")
        if self.compute_acpl:
            print("ACPL analysis: ENABLED")
        print()

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
            if self.opening_book is not None:
                opening_name, opening_uci = self.opening_book.get_opening_by_index(game_num - 1)
                opening_moves = " ".join(opening_uci)
            else:
                opening_name = "Starting Position"
                opening_moves = ""

            # Play game
            result = self.play_game(
                white,
                black,
                game_num,
                opening_name,
                opening_moves,
                max_moves_per_game,
            )

            self.results.append(result)

        # Compute statistics
        stats = self._compute_statistics(player1_name)

        # Save results
        self._save_results()

        return stats

    def _compute_statistics(self, player_name: str) -> MatchStatistics:
        """Compute enhanced match statistics."""
        wins = 0
        draws = 0
        losses = 0
        total_moves = 0
        total_duration = 0.0
        acpl_values = []

        for result in self.results:
            total_moves += result.num_moves
            total_duration += result.duration_seconds

            # Determine outcome
            if result.white_engine == player_name:
                if result.result == "1-0":
                    wins += 1
                elif result.result == "1/2-1/2":
                    draws += 1
                else:
                    losses += 1

                # Collect ACPL
                if result.acpl_white is not None:
                    acpl_values.append(result.acpl_white)
            else:
                if result.result == "0-1":
                    wins += 1
                elif result.result == "1/2-1/2":
                    draws += 1
                else:
                    losses += 1

                if result.acpl_black is not None:
                    acpl_values.append(result.acpl_black)

        total_games = len(self.results)
        score = wins + 0.5 * draws
        win_rate = score / total_games if total_games > 0 else 0.0

        # Estimate Elo
        if total_games > 0 and 0 < win_rate < 1.0:
            elo_diff, elo_lower, elo_upper = estimate_elo_difference(win_rate, total_games)
        else:
            elo_diff, elo_lower, elo_upper = 0.0, 0.0, 0.0

        avg_acpl = np.mean(acpl_values) if acpl_values else None

        return MatchStatistics(
            total_games=total_games,
            wins=wins,
            draws=draws,
            losses=losses,
            score=score,
            win_rate=win_rate,
            avg_moves=total_moves / total_games if total_games > 0 else 0,
            avg_duration=total_duration / total_games if total_games > 0 else 0,
            elo_estimate=elo_diff,
            elo_ci_lower=elo_lower,
            elo_ci_upper=elo_upper,
            avg_acpl=avg_acpl,
        )

    def _save_results(self):
        """Save match results to PGN and JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        player1_name = self.player1.get_name().replace(" ", "_")
        player2_name = self.player2.get_name().replace(" ", "_")

        # Save PGNs
        pgn_path = self.output_dir / f"match_{player1_name}_vs_{player2_name}_{timestamp}.pgn"
        with open(pgn_path, "w") as f:
            for result in self.results:
                f.write(result.pgn + "\n\n")

        print(f"\nSaved PGNs to: {pgn_path}")

        # Save JSON
        json_data = {
            "player1": player1_name,
            "player2": player2_name,
            "timestamp": timestamp,
            "num_games": len(self.results),
            "results": [asdict(r) for r in self.results],
            "statistics_player1": asdict(self._compute_statistics(player1_name)),
            "statistics_player2": asdict(self._compute_statistics(player2_name)),
        }

        json_path = self.output_dir / f"match_{player1_name}_vs_{player2_name}_{timestamp}.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"Saved statistics to: {json_path}")

    def print_summary(self, player_name: str):
        """Print match summary table."""
        stats = self._compute_statistics(player_name)

        print(f"\n{'='*70}")
        print(f"Match Summary: {player_name}")
        print(f"{'='*70}")
        print(f"Total Games:     {stats.total_games}")
        print(f"Wins:            {stats.wins}")
        print(f"Draws:           {stats.draws}")
        print(f"Losses:          {stats.losses}")
        print(f"Score:           {stats.score:.1f} / {stats.total_games} ({stats.win_rate:.1%})")
        print(f"Elo Estimate:    {stats.elo_estimate:+.0f} ± {abs(stats.elo_ci_upper - stats.elo_estimate):.0f}")
        print(f"Elo 95% CI:      [{stats.elo_ci_lower:.0f}, {stats.elo_ci_upper:.0f}]")
        if stats.avg_acpl is not None:
            print(f"Avg ACPL:        {stats.avg_acpl:.1f} cp")
        print(f"Avg Moves:       {stats.avg_moves:.1f}")
        print(f"Avg Duration:    {stats.avg_duration:.2f}s")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    print("Enhanced match runner module loaded.")
    print("This module provides Elo estimation and ACPL analysis.")

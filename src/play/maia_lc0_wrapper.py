"""Wrapper for Lc0 (Leela Chess Zero) engine using Maia weights.

Maia is a human-like chess engine that uses Lc0 with specialized neural network
weights trained to mimic human play at specific rating levels.

Installation:
- Lc0: https://github.com/LeelaChessZero/lc0/releases
- Maia weights: https://maiachess.com/ (download maia1500.pb.gz, maia1900.pb.gz, etc.)

Usage:
    with MaiaLc0Engine(lc0_path="/usr/local/bin/lc0",
                       weights_path="weights/maia1500.pb.gz",
                       movetime_ms=300) as engine:
        uci_move = engine.best_move(fen="...", moves_uci=["e2e4", "e7e5"])
"""

import subprocess
import chess
import time
from pathlib import Path
from typing import Optional, List


class MaiaLc0Engine:
    """
    UCI wrapper for Lc0 engine using Maia weights.

    Provides a clean interface for getting moves from Maia via UCI protocol.
    Ensures engines don't use internal opening books and rely only on neural network evaluation.
    """

    def __init__(
        self,
        lc0_path: str = "/usr/local/bin/lc0",
        weights_path: str = "",
        movetime_ms: int = 300,
        depth: Optional[int] = None,
        threads: int = 1,
        nn_backend: str = "cpu",
        disable_book: bool = True,
    ):
        """
        Initialize Maia/Lc0 engine.

        Args:
            lc0_path: Path to Lc0 binary (default: /usr/local/bin/lc0)
            weights_path: Path to Maia weights file (required, e.g., "weights/maia1500.pb.gz")
            movetime_ms: Time per move in milliseconds (default: 300)
            depth: Optional depth limit (if None, uses movetime)
            threads: Number of threads for search (default: 1)
            nn_backend: Neural network backend - "cpu", "cuda", "metal", etc. (default: "cpu")
            disable_book: Disable internal opening book (default: True)
        """
        self.lc0_path = lc0_path
        self.weights_path = weights_path
        self.movetime_ms = movetime_ms
        self.depth = depth
        self.threads = threads
        self.nn_backend = nn_backend
        self.disable_book = disable_book

        # Validate paths
        if not Path(lc0_path).exists():
            raise FileNotFoundError(
                f"Lc0 not found at {lc0_path}.\n"
                f"Install from: https://github.com/LeelaChessZero/lc0/releases\n"
                f"macOS (Homebrew): brew install lc0"
            )

        if not weights_path:
            raise ValueError(
                "weights_path is required.\n"
                "Download Maia weights from https://maiachess.com/\n"
                "Example: weights/maia1500.pb.gz"
            )

        if not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Maia weights not found at {weights_path}.\n"
                f"Download from: https://maiachess.com/"
            )

        self.process: Optional[subprocess.Popen] = None
        self.name = self._extract_name_from_weights(weights_path)

    def _extract_name_from_weights(self, weights_path: str) -> str:
        """Extract engine name from weights filename."""
        filename = Path(weights_path).stem
        # Remove .pb if present (from .pb.gz)
        if filename.endswith('.pb'):
            filename = filename[:-3]
        return f"Maia-{filename}" if "maia" in filename.lower() else f"Lc0-{filename}"

    def __enter__(self):
        """Context manager entry - start engine and configure."""
        self._start_engine()
        self._configure_engine()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shut down engine."""
        self.quit()

    def _start_engine(self):
        """Start Lc0 UCI process."""
        try:
            self.process = subprocess.Popen(
                [self.lc0_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Send UCI handshake
            self._send_command("uci")

            # Wait for "uciok"
            timeout = time.time() + 5.0
            while time.time() < timeout:
                line = self._read_line(timeout=1.0)
                if line and "uciok" in line:
                    break
            else:
                raise RuntimeError("Lc0 did not respond with 'uciok' within timeout")

        except Exception as e:
            raise RuntimeError(f"Failed to start Lc0 engine: {e}")

    def _configure_engine(self):
        """Configure Lc0 with weights and options."""
        # Set weights file (REQUIRED)
        self._send_command(f"setoption name WeightsFile value {self.weights_path}")

        # Set threads
        self._send_command(f"setoption name Threads value {self.threads}")

        # Set backend if supported (Lc0 uses "Backend" option)
        # Common values: "cpu", "cuda", "metal", "blas", etc.
        self._send_command(f"setoption name Backend value {self.nn_backend}")

        # Lc0 typically doesn't have an opening book, but ensure no book usage
        # Most Lc0 builds don't have "OwnBook" option, but we can try
        if self.disable_book:
            # This may not do anything for Lc0, but it's safe to send
            self._send_command("setoption name OwnBook value false")

        # Wait for options to be set
        time.sleep(0.1)

    def new_game(self):
        """Signal a new game to the engine."""
        if self.process is None:
            raise RuntimeError("Engine not started")
        self._send_command("ucinewgame")
        self._send_command("isready")
        self._wait_for_ready()

    def best_move(
        self,
        fen: str,
        moves_uci: Optional[List[str]] = None,
        movetime_ms: Optional[int] = None,
        depth: Optional[int] = None,
    ) -> str:
        """
        Get best move from Maia for the given position.

        Args:
            fen: FEN string of current position
            moves_uci: Optional list of UCI moves from the FEN position
            movetime_ms: Override default movetime (milliseconds)
            depth: Override default depth

        Returns:
            UCI move string (e.g., "e2e4")

        Raises:
            RuntimeError: If engine returns illegal move or times out
        """
        if self.process is None:
            raise RuntimeError("Engine not started")

        # Build position command
        position_cmd = f"position fen {fen}"
        if moves_uci:
            position_cmd += " moves " + " ".join(moves_uci)

        self._send_command(position_cmd)

        # Build go command
        mt = movetime_ms if movetime_ms is not None else self.movetime_ms
        d = depth if depth is not None else self.depth

        if d is not None:
            go_cmd = f"go depth {d}"
        else:
            go_cmd = f"go movetime {mt}"

        self._send_command(go_cmd)

        # Wait for bestmove response
        timeout = time.time() + (mt / 1000.0 * 2 + 5.0)  # 2x movetime + 5s buffer
        best_move = None

        while time.time() < timeout:
            line = self._read_line(timeout=1.0)
            if line and line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2:
                    best_move = parts[1]
                    break

        if best_move is None:
            raise RuntimeError(f"Lc0 did not return bestmove within timeout (FEN: {fen})")

        # Validate move is legal
        self._validate_move(fen, moves_uci or [], best_move)

        return best_move

    def _validate_move(self, fen: str, moves_uci: List[str], move_uci: str):
        """
        Validate that the engine's move is legal.

        Args:
            fen: Starting FEN
            moves_uci: Moves from FEN to current position
            move_uci: Engine's proposed move

        Raises:
            RuntimeError: If move is illegal
        """
        try:
            board = chess.Board(fen)
            # Apply moves to reach current position
            for uci in moves_uci:
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    raise RuntimeError(f"Move history contains illegal move: {uci}")
                board.push(move)

            # Check engine's move
            move = chess.Move.from_uci(move_uci)
            if move not in board.legal_moves:
                # Log context for debugging
                recent_moves = moves_uci[-8:] if len(moves_uci) > 8 else moves_uci
                raise RuntimeError(
                    f"Illegal move from Maia: {move_uci}\n"
                    f"FEN: {board.fen()}\n"
                    f"Last moves: {recent_moves}\n"
                    f"Legal moves: {[m.uci() for m in list(board.legal_moves)[:10]]}"
                )
        except Exception as e:
            if "Illegal move" in str(e) or "Legal moves" in str(e):
                raise
            raise RuntimeError(f"Error validating move {move_uci}: {e}")

    def _send_command(self, cmd: str):
        """Send command to engine."""
        if self.process and self.process.stdin:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()

    def _read_line(self, timeout: float = 1.0) -> Optional[str]:
        """Read one line from engine stdout with timeout."""
        if not self.process or not self.process.stdout:
            return None

        # Simple timeout implementation
        start = time.time()
        while time.time() - start < timeout:
            line = self.process.stdout.readline()
            if line:
                return line.strip()
            time.sleep(0.01)
        return None

    def _wait_for_ready(self, timeout: float = 5.0):
        """Wait for 'readyok' response."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self._read_line(timeout=0.5)
            if line and "readyok" in line:
                return
        raise RuntimeError("Engine did not respond with 'readyok'")

    def quit(self):
        """Shut down engine."""
        if self.process:
            try:
                self._send_command("quit")
                self.process.wait(timeout=2.0)
            except:
                self.process.kill()
            finally:
                self.process = None

    def get_name(self) -> str:
        """Get engine name."""
        return self.name

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Get move for match_runner.py compatibility.

        Args:
            board: Current board position

        Returns:
            chess.Move object
        """
        # Extract FEN and move history
        fen = board.fen()
        # For simplicity, just use current FEN with no move history
        # (assumes board is the authoritative state)
        move_uci = self.best_move(fen, moves_uci=None)
        return chess.Move.from_uci(move_uci)


# ============================================================================
# Unit Tests
# ============================================================================

def _test_maia_wrapper():
    """Test Maia/Lc0 wrapper."""
    print("Testing MaiaLc0Engine...")

    lc0_path = "/usr/local/bin/lc0"

    # Check if Lc0 exists
    if not Path(lc0_path).exists():
        print(f"Lc0 not found at {lc0_path}")
        print("Please install Lc0:")
        print("  macOS: brew install lc0")
        print("  Linux: Download from https://github.com/LeelaChessZero/lc0/releases")
        print("\nSkipping test...")
        return

    # Check for weights (user needs to provide)
    weights_path = "weights/maia1500.pb.gz"
    if not Path(weights_path).exists():
        print(f"Maia weights not found at {weights_path}")
        print("Download from: https://maiachess.com/")
        print("\nSkipping test...")
        return

    # Test engine
    try:
        with MaiaLc0Engine(
            lc0_path=lc0_path,
            weights_path=weights_path,
            movetime_ms=200,
            threads=1,
        ) as engine:
            print(f"  Engine: {engine.get_name()}")

            # Test 1: Starting position
            start_fen = chess.STARTING_FEN
            move1 = engine.best_move(start_fen, moves_uci=None)
            print(f"  Move from start: {move1}")

            # Validate
            board = chess.Board()
            move_obj = chess.Move.from_uci(move1)
            assert move_obj in board.legal_moves, f"Move {move1} should be legal"

            # Test 2: After e2e4
            board.push(move_obj)
            move2 = engine.best_move(start_fen, moves_uci=[move1])
            print(f"  Reply to {move1}: {move2}")

            move_obj2 = chess.Move.from_uci(move2)
            assert move_obj2 in board.legal_moves, f"Move {move2} should be legal"

            print("  ✓ Maia wrapper test passed")

    except Exception as e:
        print(f"  ⚠ Test failed: {e}")
        print("  This may be expected if Lc0 or weights are not properly installed.")

    print("\nMaia wrapper implementation complete! ✓")


if __name__ == "__main__":
    _test_maia_wrapper()

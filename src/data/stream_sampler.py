"""Stream-based PGN sampler with sharding for large-scale data processing.

Filters:
- Rated standard/rapid games
- Min moves >= 20
- Min Elo >= 2000
- Side-to-move balance (50/50)
- Phase tagging (opening/middle/end)

Output: Sharded .pt files to artifacts/data/shards/
"""

import chess
import chess.pgn
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import argparse
import json
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.encoding import board_to_tensor, move_to_index


def classify_phase(board: chess.Board) -> int:
    """Classify game phase by piece count.

    Returns:
        0 = opening (28+ pieces)
        1 = middlegame (15-27 pieces)
        2 = endgame (6-14 pieces)
    """
    piece_count = len(board.piece_map())
    if piece_count >= 28:
        return 0  # opening
    elif piece_count >= 15:
        return 1  # middlegame
    else:
        return 2  # endgame


def parse_time_control(tc_str: str) -> Optional[int]:
    """Parse time control string to seconds.

    Args:
        tc_str: Time control string (e.g., "600+0", "1800+10")

    Returns:
        Base time in seconds or None if invalid
    """
    if not tc_str or tc_str == "-":
        return None

    try:
        # Format: "base+increment"
        parts = tc_str.split("+")
        return int(parts[0])
    except:
        return None


def filter_game(game: chess.pgn.Game, min_elo: int = 2000, min_moves: int = 20,
                min_time_control: int = 300) -> bool:
    """Check if game passes filters.

    Args:
        game: Chess game object
        min_elo: Minimum Elo rating for both players
        min_moves: Minimum number of moves
        min_time_control: Minimum time control in seconds

    Returns:
        True if game passes all filters
    """
    headers = game.headers

    # Check rated
    if headers.get("Event", "").lower().find("rated") == -1:
        return False

    # Check time control
    tc = parse_time_control(headers.get("TimeControl", ""))
    if tc is None or tc < min_time_control:
        return False

    # Check Elo ratings
    try:
        white_elo = int(headers.get("WhiteElo", 0))
        black_elo = int(headers.get("BlackElo", 0))
        if white_elo < min_elo or black_elo < min_elo:
            return False
    except (ValueError, TypeError):
        return False

    # Count moves
    node = game
    move_count = 0
    while node.variations:
        node = node.variation(0)
        move_count += 1

    if move_count < min_moves:
        return False

    return True


def extract_positions_from_game(game: chess.pgn.Game,
                                  skip_first_n: int = 8) -> List[Dict]:
    """Extract positions from a single game.

    Args:
        game: Chess game object
        skip_first_n: Skip first N plies (opening book moves)

    Returns:
        List of position dicts with keys: board_tensor, move_index, value, phase, side
    """
    positions = []

    # Parse outcome
    result = game.headers.get("Result", "*")
    if result == "1-0":
        outcome = 1.0
    elif result == "0-1":
        outcome = 0.0
    elif result == "1/2-1/2":
        outcome = 0.5
    else:
        return positions  # Skip unfinished games

    board = game.board()
    node = game
    ply = 0

    while node.variations:
        node = node.variation(0)
        ply += 1

        if ply <= skip_first_n:
            board.push(node.move)
            continue

        # Encode position
        try:
            board_tensor = board_to_tensor(board)
            move_index = move_to_index(node.move)
            phase = classify_phase(board)

            # Value from side-to-move perspective
            if board.turn == chess.WHITE:
                value = outcome
            else:
                value = 1.0 - outcome

            positions.append({
                'board': board_tensor,
                'move': move_index,
                'value': value,
                'phase': phase,
                'side': int(board.turn),  # 1=white, 0=black
            })

        except Exception as e:
            # Skip invalid positions
            pass

        board.push(node.move)

    return positions


def augment_position(pos: Dict, flip_file: bool = False, flip_rank: bool = False,
                      flip_color: bool = False) -> Dict:
    """Augment position with geometric and color flips.

    Args:
        pos: Position dict with 'board' tensor [12, 8, 8]
        flip_file: Flip horizontally (a-h -> h-a)
        flip_rank: Flip vertically (1-8 -> 8-1)
        flip_color: Swap white/black

    Returns:
        Augmented position dict
    """
    board = pos['board'].clone()
    move = pos['move']  # TODO: Implement move remapping for augmentation

    if flip_file:
        board = board.flip(2)  # Flip along file dimension

    if flip_rank:
        board = board.flip(1)  # Flip along rank dimension

    if flip_color:
        # Swap white and black pieces (first 6 channels with last 6)
        white_pieces = board[:6].clone()
        black_pieces = board[6:].clone()
        board[:6] = black_pieces
        board[6:] = white_pieces
        # Invert value
        pos['value'] = 1.0 - pos['value']
        pos['side'] = 1 - pos['side']

    return {
        **pos,
        'board': board,
        # Note: move remapping not implemented for simplicity
    }


def balance_sides(positions: List[Dict]) -> List[Dict]:
    """Balance white/black side-to-move to 50/50.

    Args:
        positions: List of position dicts

    Returns:
        Balanced list
    """
    white_positions = [p for p in positions if p['side'] == 1]
    black_positions = [p for p in positions if p['side'] == 0]

    target_count = min(len(white_positions), len(black_positions))

    # Randomly sample to balance
    if len(white_positions) > target_count:
        white_positions = np.random.choice(white_positions, target_count, replace=False).tolist()
    if len(black_positions) > target_count:
        black_positions = np.random.choice(black_positions, target_count, replace=False).tolist()

    balanced = white_positions + black_positions
    np.random.shuffle(balanced)

    return balanced


def save_shard(positions: List[Dict], shard_path: Path):
    """Save positions as PyTorch tensor shard.

    Args:
        positions: List of position dicts
        shard_path: Output path
    """
    shard_data = {
        'boards': torch.stack([p['board'] for p in positions]),
        'moves': torch.tensor([p['move'] for p in positions], dtype=torch.long),
        'values': torch.tensor([p['value'] for p in positions], dtype=torch.float32),
        'phases': torch.tensor([p['phase'] for p in positions], dtype=torch.long),
        'sides': torch.tensor([p['side'] for p in positions], dtype=torch.long),
    }
    torch.save(shard_data, shard_path)


def sample_positions_stream(pgn_paths: List[Path],
                              target_size: int = 1_000_000,
                              output_dir: Path = Path("artifacts/data/shards"),
                              shard_size: int = 50_000,
                              min_elo: int = 2000,
                              min_moves: int = 20,
                              min_time_control: int = 300,
                              skip_first_n: int = 8,
                              balance_sides_flag: bool = True,
                              seed: int = 42) -> Dict:
    """Stream-sample positions from PGN files with sharding.

    Args:
        pgn_paths: List of PGN file paths
        target_size: Target number of positions
        output_dir: Output directory for shards
        shard_size: Positions per shard
        min_elo: Minimum Elo rating
        min_moves: Minimum game length
        min_time_control: Minimum time control (seconds)
        skip_first_n: Skip first N plies
        balance_sides_flag: Balance white/black 50/50
        seed: Random seed

    Returns:
        Summary statistics dict
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STREAM SAMPLER CONFIGURATION")
    print("=" * 70)
    print(f"Target positions: {target_size:,}")
    print(f"Shard size: {shard_size:,}")
    print(f"Min Elo: {min_elo}")
    print(f"Min moves: {min_moves}")
    print(f"Min time control: {min_time_control}s")
    print(f"Skip first: {skip_first_n} plies")
    print(f"Balance sides: {balance_sides_flag}")
    print(f"Output dir: {output_dir}")
    print(f"PGN files: {len(pgn_paths)}")
    print("=" * 70)

    all_positions = []
    shard_idx = 0
    games_processed = 0
    games_filtered = 0

    phase_counts = {0: 0, 1: 0, 2: 0}  # opening, middle, end
    side_counts = {0: 0, 1: 0}  # black, white

    for pgn_path in pgn_paths:
        print(f"\nProcessing: {pgn_path.name}")

        with open(pgn_path) as pgn_file:
            while len(all_positions) < target_size:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break

                games_processed += 1

                # Filter game
                if not filter_game(game, min_elo, min_moves, min_time_control):
                    games_filtered += 1
                    continue

                # Extract positions
                positions = extract_positions_from_game(game, skip_first_n)
                all_positions.extend(positions)

                # Update stats
                for p in positions:
                    phase_counts[p['phase']] += 1
                    side_counts[p['side']] += 1

                # Save shard when full
                if len(all_positions) >= shard_size:
                    shard_positions = all_positions[:shard_size]
                    all_positions = all_positions[shard_size:]

                    if balance_sides_flag:
                        shard_positions = balance_sides(shard_positions)

                    shard_path = output_dir / f"shard_{shard_idx:04d}.pt"
                    save_shard(shard_positions, shard_path)
                    print(f"  Saved shard {shard_idx}: {len(shard_positions):,} positions")
                    shard_idx += 1

                if games_processed % 100 == 0:
                    print(f"  Games: {games_processed:,} | Positions: {sum(phase_counts.values()):,}")

                if len(all_positions) >= target_size:
                    break

        if len(all_positions) >= target_size:
            break

    # Save remaining positions as final shard
    if len(all_positions) > 0:
        if balance_sides_flag:
            all_positions = balance_sides(all_positions)
        shard_path = output_dir / f"shard_{shard_idx:04d}.pt"
        save_shard(all_positions, shard_path)
        print(f"  Saved final shard {shard_idx}: {len(all_positions):,} positions")
        shard_idx += 1

    # Calculate final stats
    total_positions = sum(phase_counts.values())

    summary = {
        'total_positions': total_positions,
        'num_shards': shard_idx,
        'games_processed': games_processed,
        'games_filtered': games_filtered,
        'phase_distribution': phase_counts,
        'side_distribution': side_counts,
        'config': {
            'target_size': target_size,
            'min_elo': min_elo,
            'min_moves': min_moves,
            'min_time_control': min_time_control,
            'skip_first_n': skip_first_n,
            'balance_sides': balance_sides_flag,
        }
    }

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SAMPLING COMPLETE")
    print("=" * 70)
    print(f"Total positions: {total_positions:,}")
    print(f"Shards created: {shard_idx}")
    print(f"Games processed: {games_processed:,}")
    print(f"Games filtered: {games_filtered:,} ({games_filtered/games_processed*100:.1f}%)")
    print(f"\nPhase distribution:")
    print(f"  Opening:    {phase_counts[0]:8,} ({phase_counts[0]/total_positions*100:5.1f}%)")
    print(f"  Middlegame: {phase_counts[1]:8,} ({phase_counts[1]/total_positions*100:5.1f}%)")
    print(f"  Endgame:    {phase_counts[2]:8,} ({phase_counts[2]/total_positions*100:5.1f}%)")
    print(f"\nSide distribution:")
    print(f"  White: {side_counts[1]:8,} ({side_counts[1]/total_positions*100:5.1f}%)")
    print(f"  Black: {side_counts[0]:8,} ({side_counts[0]/total_positions*100:5.1f}%)")
    print(f"\nSaved to: {output_dir}")
    print("=" * 70)

    return summary


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Stream-sample positions from PGN files")
    parser.add_argument("--pgn-dir", type=str, required=True, help="Directory containing PGN files")
    parser.add_argument("--target", type=int, default=1_000_000, help="Target number of positions")
    parser.add_argument("--output", type=str, default="artifacts/data/shards", help="Output directory")
    parser.add_argument("--shard-size", type=int, default=50_000, help="Positions per shard")
    parser.add_argument("--min-elo", type=int, default=2000, help="Minimum Elo rating")
    parser.add_argument("--min-moves", type=int, default=20, help="Minimum game length")
    parser.add_argument("--min-time", type=int, default=300, help="Minimum time control (seconds)")
    parser.add_argument("--skip-first", type=int, default=8, help="Skip first N plies")
    parser.add_argument("--no-balance-sides", action="store_true", help="Don't balance white/black")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Find PGN files
    pgn_dir = Path(args.pgn_dir)
    if not pgn_dir.exists():
        print(f"Error: Directory not found: {pgn_dir}")
        return

    pgn_paths = list(pgn_dir.glob("*.pgn"))
    if not pgn_paths:
        print(f"Error: No .pgn files found in {pgn_dir}")
        return

    # Run sampler
    sample_positions_stream(
        pgn_paths=pgn_paths,
        target_size=args.target,
        output_dir=Path(args.output),
        shard_size=args.shard_size,
        min_elo=args.min_elo,
        min_moves=args.min_moves,
        min_time_control=args.min_time,
        skip_first_n=args.skip_first,
        balance_sides_flag=not args.no_balance_sides,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

#!/bin/bash
# Quick benchmark script for Maia-1500
# Usage: ./scripts/run_maia_benchmark.sh [num_games]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Configuration
LC0_PATH="/usr/local/bin/lc0"
MAIA_WEIGHTS="weights/maia-1500.pb.gz"
NUM_GAMES="${1:-20}"  # Default 20 games, or use first argument
MOVETIME="${2:-300}"   # Default 300ms, or use second argument

echo "========================================================================"
echo "MAIA-1500 BENCHMARK"
echo "========================================================================"
echo "Games:     $NUM_GAMES"
echo "Movetime:  ${MOVETIME}ms per move"
echo "Weights:   $MAIA_WEIGHTS"
echo "Lc0:       $LC0_PATH"
echo "========================================================================"
echo ""

# Check prerequisites
if [ ! -f "$LC0_PATH" ]; then
    echo "Error: Lc0 not found at $LC0_PATH"
    echo "Install: brew install lc0"
    exit 1
fi

if [ ! -f "$MAIA_WEIGHTS" ]; then
    echo "Error: Maia weights not found at $MAIA_WEIGHTS"
    echo "Download from: https://maiachess.com/"
    exit 1
fi

# Run sanity check first
echo "Step 1: Running sanity check..."
python -m src.play.sanity \
    --lc0-path "$LC0_PATH" \
    --maia-weights "$MAIA_WEIGHTS" \
    --movetime 200

echo ""
echo "Step 2: Running benchmark..."
echo ""

# Run benchmark
python -m src.play.match_runner \
    --opponent maia \
    --games "$NUM_GAMES" \
    --movetime "$MOVETIME" \
    --maia-weights "$MAIA_WEIGHTS" \
    --lc0-path "$LC0_PATH" \
    --threads 1 \
    --output-dir artifacts/matches

echo ""
echo "========================================================================"
echo "✓ Benchmark complete!"
echo "Results saved to: artifacts/matches/"
echo "========================================================================"

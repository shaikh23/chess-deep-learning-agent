"""Visualization utilities for EDA, training curves, and results."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10


def plot_phase_distribution(
    phase_counts: Dict[str, int],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot distribution of positions across game phases.

    Args:
        phase_counts: Dictionary mapping phase names to counts
        save_path: Optional path to save figure
    """
    phases = list(phase_counts.keys())
    counts = list(phase_counts.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(phases, counts, color=sns.color_palette("viridis", len(phases)))

    ax.set_xlabel("Game Phase")
    ax.set_ylabel("Number of Positions")
    ax.set_title("Distribution of Positions by Game Phase")

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(count):,}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved phase distribution plot to {save_path}")

    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot training and validation curves.

    Args:
        train_losses: Training loss per epoch
        val_losses: Validation loss per epoch
        train_accs: Optional training accuracy per epoch
        val_accs: Optional validation accuracy per epoch
        save_path: Optional path to save figure
    """
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 2 if train_accs else 1, figsize=(15 if train_accs else 10, 5))

    if not train_accs:
        axes = [axes]

    # Loss plot
    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o", linewidth=2)
    axes[0].plot(epochs, val_losses, label="Val Loss", marker="s", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot (if provided)
    if train_accs:
        axes[1].plot(epochs, train_accs, label="Train Accuracy", marker="o", linewidth=2)
        axes[1].plot(epochs, val_accs, label="Val Accuracy", marker="s", linewidth=2)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training and Validation Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")

    plt.show()


def plot_match_results(
    results: Dict[str, Dict[str, int]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot match results as grouped bar chart.

    Args:
        results: Dictionary mapping opponent names to result counts
                 e.g., {"Sunfish": {"wins": 70, "draws": 10, "losses": 20}}
        save_path: Optional path to save figure
    """
    opponents = list(results.keys())
    wins = [results[opp]["wins"] for opp in opponents]
    draws = [results[opp]["draws"] for opp in opponents]
    losses = [results[opp]["losses"] for opp in opponents]

    x = np.arange(len(opponents))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width, wins, width, label="Wins", color="green", alpha=0.8)
    bars2 = ax.bar(x, draws, width, label="Draws", color="gray", alpha=0.8)
    bars3 = ax.bar(x + width, losses, width, label="Losses", color="red", alpha=0.8)

    ax.set_xlabel("Opponent")
    ax.set_ylabel("Number of Games")
    ax.set_title("Match Results Against Different Opponents")
    ax.set_xticks(x)
    ax.set_xticklabels(opponents)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved match results to {save_path}")

    plt.show()


def plot_acpl_by_phase(
    acpl_data: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot Average Centipawn Loss by game phase for different agents.

    Args:
        acpl_data: Dictionary mapping agent names to phase ACPL
                   e.g., {"Our Agent": {"opening": 50, "middlegame": 80, "endgame": 60}}
        save_path: Optional path to save figure
    """
    agents = list(acpl_data.keys())
    phases = ["opening", "middlegame", "endgame"]

    # Prepare data
    data = []
    for agent in agents:
        for phase in phases:
            acpl = acpl_data[agent].get(phase, 0)
            data.append({"Agent": agent, "Phase": phase.capitalize(), "ACPL": acpl})

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Grouped bar plot
    x = np.arange(len(phases))
    width = 0.8 / len(agents)

    for i, agent in enumerate(agents):
        agent_data = df[df["Agent"] == agent]
        acpl_values = [agent_data[agent_data["Phase"] == p.capitalize()]["ACPL"].values[0] if len(agent_data[agent_data["Phase"] == p.capitalize()]) > 0 else 0 for p in phases]
        offset = (i - len(agents) / 2 + 0.5) * width
        ax.bar(x + offset, acpl_values, width, label=agent, alpha=0.8)

    ax.set_xlabel("Game Phase")
    ax.set_ylabel("Average Centipawn Loss")
    ax.set_title("ACPL by Game Phase (Lower is Better)")
    ax.set_xticks(x)
    ax.set_xticklabels([p.capitalize() for p in phases])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved ACPL by phase plot to {save_path}")

    plt.show()


def plot_elo_progression(
    elo_history: List[Tuple[int, float, float, float]],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot Elo rating progression with confidence intervals.

    Args:
        elo_history: List of (game_number, elo, lower_ci, upper_ci) tuples
        save_path: Optional path to save figure
    """
    if not elo_history:
        print("No Elo history to plot")
        return

    game_nums, elos, lower_cis, upper_cis = zip(*elo_history)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(game_nums, elos, label="Elo Rating", linewidth=2, color="blue")
    ax.fill_between(
        game_nums,
        lower_cis,
        upper_cis,
        alpha=0.3,
        color="blue",
        label="95% Confidence Interval",
    )

    ax.set_xlabel("Number of Games")
    ax.set_ylabel("Elo Rating Difference")
    ax.set_title("Elo Rating Progression")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved Elo progression plot to {save_path}")

    plt.show()


def plot_piece_count_distribution(
    piece_counts: List[int],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot histogram of piece counts across positions.

    Args:
        piece_counts: List of total piece counts per position
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(piece_counts, bins=range(2, 34), edgecolor="black", alpha=0.7)
    ax.set_xlabel("Total Piece Count")
    ax.set_ylabel("Number of Positions")
    ax.set_title("Distribution of Piece Counts")
    ax.grid(True, alpha=0.3, axis="y")

    # Add vertical lines for phase boundaries (rough estimates)
    ax.axvline(x=24, color="orange", linestyle="--", alpha=0.5, label="Opening/Mid")
    ax.axvline(x=12, color="red", linestyle="--", alpha=0.5, label="Mid/Endgame")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved piece count distribution to {save_path}")

    plt.show()


def plot_outcome_distribution(
    outcome_counts: Dict[str, int],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot pie chart of game outcomes.

    Args:
        outcome_counts: Dictionary mapping outcomes to counts
                        e.g., {"1-0": 450, "0-1": 400, "1/2-1/2": 150}
        save_path: Optional path to save figure
    """
    labels = list(outcome_counts.keys())
    sizes = list(outcome_counts.values())
    colors = ["green", "red", "gray"]

    fig, ax = plt.subplots(figsize=(8, 8))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
    )

    ax.set_title("Game Outcome Distribution")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved outcome distribution to {save_path}")

    plt.show()


# ============================================================================
# Example Usage
# ============================================================================

def _demo_plots():
    """Demonstrate plotting functions with synthetic data."""
    print("Generating demo plots...")

    # Phase distribution
    phase_counts = {"Opening": 150000, "Middlegame": 200000, "Endgame": 150000}
    plot_phase_distribution(phase_counts)

    # Training curves
    train_losses = [2.5, 2.0, 1.7, 1.5, 1.3, 1.2, 1.1, 1.0]
    val_losses = [2.6, 2.1, 1.8, 1.6, 1.5, 1.4, 1.3, 1.3]
    train_accs = [0.30, 0.38, 0.43, 0.46, 0.49, 0.51, 0.52, 0.53]
    val_accs = [0.29, 0.37, 0.42, 0.45, 0.47, 0.49, 0.50, 0.50]
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    # Match results
    results = {
        "Sunfish": {"wins": 75, "draws": 12, "losses": 13},
        "Stockfish-Lv1": {"wins": 25, "draws": 20, "losses": 55},
    }
    plot_match_results(results)

    # ACPL by phase
    acpl_data = {
        "Our Agent": {"opening": 45, "middlegame": 85, "endgame": 65},
        "Sunfish": {"opening": 60, "middlegame": 100, "endgame": 80},
    }
    plot_acpl_by_phase(acpl_data)

    print("Demo plots complete!")


if __name__ == "__main__":
    _demo_plots()

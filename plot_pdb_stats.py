"""
Plot PDB statistics from the annual per-method CSV file.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_pdb_stats(csv_path: str | Path | None = None, save_path: str | Path | None = None) -> None:
    """
    Load the PDB annual statistics CSV and plot total entries and annual releases
    for both X-ray and cryoEM methods over time.

    Args:
        csv_path: Path to the CSV file. Defaults to the Sheet1 file in the same directory.
        save_path: Optional path to save the figure. If None, displays the plot interactively.
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / "PDB Annual per method - Sheet1.csv"

    df = pd.read_csv(csv_path)

    # Remove commas from numeric columns and convert to int
    numeric_cols = [
        "Total Number of Entries Available X-ray",
        "Number of Structures Released Annually X-ray",
        "Total Number of Entries Available cryoEM",
        "Number of Structures Released Annually cryoEM",
    ]
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "").astype(int)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot 1: Total entries available over time
    ax1 = axes[0]
    ax1.plot(df["Year"], df["Total Number of Entries Available X-ray"], label="X-ray", marker="o", markersize=3)
    ax1.plot(df["Year"], df["Total Number of Entries Available cryoEM"], label="cryoEM", marker="s", markersize=3)
    ax1.set_ylabel("Total Entries")
    ax1.set_title("Total PDB Entries Available by Method")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Structures released annually over time
    ax2 = axes[1]
    ax2.plot(df["Year"], df["Number of Structures Released Annually X-ray"], label="X-ray", marker="o", markersize=3)
    ax2.plot(df["Year"], df["Number of Structures Released Annually cryoEM"], label="cryoEM", marker="s", markersize=3)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Structures Released Annually")
    ax2.set_title("PDB Structures Released Annually by Method")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    plot_pdb_stats()
    # Or save to file:
    # plot_pdb_stats(save_path="pdb_stats_plot.png")

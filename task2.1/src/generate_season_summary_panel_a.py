import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    analysis_csv = repo_root / "task1" / "table" / "fan_vote_shares_analysis.csv"
    save_path = repo_root / "task2.1" / "figure" / "Task2_1_season_summary_panel_a.png"

    analysis_df = pd.read_csv(analysis_csv)
    season_stats = (
        analysis_df.groupby("season")
        .agg(
            {
                "ffi_rank": "mean",
                "ffi_percent": "mean",
            }
        )
        .reset_index()
    )

    plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(
        season_stats["season"],
        season_stats["ffi_rank"],
        "o-",
        color="#9cb0ce",
        label="RANK",
        linewidth=2,
        markersize=6,
    )
    ax.plot(
        season_stats["season"],
        season_stats["ffi_percent"],
        "s-",
        color="#45618a",
        label="PERCENT",
        linewidth=2,
        markersize=6,
    )
    ax.axhline(y=0, color="#999999", linestyle="--", alpha=0.6, linewidth=1.2)

    ax.set_xlabel("Season", fontsize=11, fontweight="normal")
    ax.set_ylabel("Mean FFI", fontsize=11, fontweight="normal")
    ax.set_title("(a) Average FFI by Season", fontsize=12, fontweight="normal")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

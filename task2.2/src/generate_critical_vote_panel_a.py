from pathlib import Path
import matplotlib.pyplot as plt

from controversy_analysis import load_all_data, critical_vote_analysis, PALETTE


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_csv = repo_root / "2026_MCM_Problem_C_Data.csv"
    fan_csv = repo_root / "task1" / "table" / "fan_vote_shares.csv"
    out_path = repo_root / "task2.2" / "figure" / "Step_Critical_Vote_panel_a.png"

    fan_df, judge_df, _ = load_all_data(str(repo_root))

    critical_data = critical_vote_analysis(fan_df, judge_df, str(repo_root / "task2.2" / "table"))
    name = "Jerry Rice"
    data = critical_data.get(name)
    if not data or not data.get("weekly"):
        raise RuntimeError("Critical vote data missing for Jerry Rice.")

    weekly = data["weekly"]
    weeks = [w["week"] for w in weekly]
    actual = [w["actual"] for w in weekly]
    crit_rank = [w["critical_rank"] if w["critical_rank"] else 0 for w in weekly]
    crit_percent = [w["critical_percent"] if w["critical_percent"] else 0 for w in weekly]

    plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="white")
    ax.set_facecolor("white")

    ax.plot(weeks, actual, "o-", label="Actual", linewidth=3, markersize=9,
            color=PALETTE["jerry"], zorder=3)
    ax.plot(weeks, crit_rank, "s--", label="Critical (RANK)", linewidth=2,
            markersize=7, color=PALETTE["rank"], alpha=0.85, zorder=2)
    ax.plot(weeks, crit_percent, "d--", label="Critical (PERCENT)", linewidth=2,
            markersize=7, color=PALETTE["percent"], alpha=0.85, zorder=2)

    ax.fill_between(
        weeks, crit_rank, actual,
        where=[a >= c for a, c in zip(actual, crit_rank)],
        alpha=0.15, color=PALETTE["negative"], label="Safe Zone"
    )

    ax.set_xlabel("Week", fontsize=14, fontweight="normal")
    ax.set_ylabel("Fan Vote Share (%)", fontsize=14, fontweight="normal")
    ax.set_title("Jerry Rice: Actual vs Critical", fontsize=15, fontweight="normal")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="best", fontsize=10, frameon=False)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor="white")


if __name__ == "__main__":
    main()

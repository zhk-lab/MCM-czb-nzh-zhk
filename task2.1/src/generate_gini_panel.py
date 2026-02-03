import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_judge_scores(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        season = row["season"]
        name = row["celebrity_name"]
        for week in range(1, 12):
            scores = []
            for judge in range(1, 5):
                col = f"week{week}_judge{judge}_score"
                if col in row.index and pd.notna(row[col]) and row[col] != 0:
                    scores.append(row[col])
            if scores:
                rows.append(
                    {
                        "season": season,
                        "week": week,
                        "celebrity_name": name,
                        "judge_total_score": sum(scores),
                    }
                )
    return pd.DataFrame(rows)


def gini_coefficient(values: np.ndarray) -> float:
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (
        n * np.sum(sorted_values)
    ) - (n + 1) / n
    return gini


def compute_weekly_gini(df_fan: pd.DataFrame, df_judge: pd.DataFrame) -> tuple[list, list]:
    fan_list = []
    judge_list = []
    for (season, week), group_fan in df_fan.groupby(["season", "week"]):
        group_judge = df_judge[
            (df_judge["season"] == season) & (df_judge["week"] == week)
        ]
        if len(group_fan) < 3 or len(group_judge) < 3:
            continue
        fan_shares = group_fan["fan_vote_share"].values
        judge_scores = group_judge["judge_total_score"].values
        judge_total = judge_scores.sum()
        judge_percent = (
            judge_scores / judge_total if judge_total > 0 else judge_scores / len(judge_scores)
        )
        fan_list.append(gini_coefficient(fan_shares))
        judge_list.append(gini_coefficient(judge_percent))
    return fan_list, judge_list


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    data_csv = str(repo_root / "2026_MCM_Problem_C_Data.csv")
    fan_csv = str(repo_root / "task1" / "table" / "fan_vote_shares.csv")

    df_fan = pd.read_csv(fan_csv)
    df_judge = load_judge_scores(data_csv)
    fan_gini, judge_gini = compute_weekly_gini(df_fan, df_judge)

    # Style to match image.png
    plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    color_fan = "#a7b9d4"
    color_judge = "#2f4a73"

    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor="white")
    ax.hist(
        fan_gini,
        bins=30,
        alpha=0.8,
        color=color_fan,
        edgecolor="white",
        linewidth=1.0,
        label="Fan Vote",
    )
    ax.hist(
        judge_gini,
        bins=30,
        alpha=0.8,
        color=color_judge,
        edgecolor="white",
        linewidth=1.0,
        label="Judge Percent",
    )
    ax.axvline(np.mean(fan_gini), color=color_fan, linestyle="--", linewidth=2.0, alpha=0.9)
    ax.axvline(np.mean(judge_gini), color=color_judge, linestyle="--", linewidth=2.0, alpha=0.9)

    ax.set_xlabel("Gini Coefficient", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(frameon=False, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    out_path = repo_root / "task2.1" / "figure" / "Task2_1_distribution_histograms_panel_c.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

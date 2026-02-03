import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    save_dir = repo_root / "task3" / "figure"
    save_dir.mkdir(parents=True, exist_ok=True)

    weeks = np.arange(1, 11)

    # Physicality (Sigmoid)
    L_phys, k_phys, t0_phys, b_phys = -1.607, 3.02, 4.82, 0.887
    S_t = 1 / (1 + np.exp(-k_phys * (weeks - t0_phys)))
    beta_physicality = L_phys * S_t + b_phys

    # Age (Quadratic)
    a_age, b_age, c_age = 0.00183, -0.01521, -0.08461
    beta_age = a_age * weeks**2 + b_age * weeks + c_age
    t_star = -b_age / (2 * a_age)
    beta_min = beta_age[np.argmin(beta_age)]

    plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), facecolor="white")

    # (a) Physicality
    ax1 = axes[0]
    ax1.plot(
        weeks,
        beta_physicality,
        linewidth=2.2,
        color="#5d7fa6",
        marker="o",
        markersize=6,
        markerfacecolor="#9cb0ce",
        markeredgecolor="#5d7fa6",
        markeredgewidth=1.2,
    )
    ax1.fill_between(
        weeks,
        0,
        beta_physicality,
        where=(beta_physicality > 0),
        alpha=0.12,
        color="#9cb0ce",
    )
    ax1.fill_between(
        weeks,
        0,
        beta_physicality,
        where=(beta_physicality < 0),
        alpha=0.12,
        color="#caa5a5",
    )
    ax1.axhline(0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.6)
    ax1.axvline(t0_phys, color="#a66b6b", linestyle=":", linewidth=1.2, alpha=0.7)
    ax1.text(t0_phys + 0.2, beta_physicality[4], f"Inflection: Week {t0_phys:.1f}", fontsize=14)
    ax1.set_title("(a) Physicality: Sigmoid Transition", fontsize=14, fontweight="normal", pad=8)
    ax1.set_xlabel("Week", fontsize=13, fontweight="normal")
    ax1.set_ylabel("β(t) - Physicality Effect", fontsize=13, fontweight="normal")
    ax1.grid(False)
    ax1.set_facecolor("white")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # (b) Age
    ax2 = axes[1]
    ax2.plot(
        weeks,
        beta_age,
        linewidth=2.2,
        color="#a66b6b",
        marker="s",
        markersize=6,
        markerfacecolor="#cba4a4",
        markeredgecolor="#a66b6b",
        markeredgewidth=1.2,
    )
    ax2.fill_between(
        weeks,
        beta_age.min(),
        beta_age,
        alpha=0.12,
        color="#cba4a4",
    )
    ax2.axhline(0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.6)
    ax2.axvline(t_star, color="#a66b6b", linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.scatter([t_star], [beta_min], s=80, color="#a66b6b", zorder=5, edgecolor="white", linewidths=1.0)
    ax2.text(t_star + 0.2, beta_min, f"Min: Week {t_star:.1f}", fontsize=14, va="bottom")
    ax2.set_title("(b) Age: U-Shaped Trajectory", fontsize=14, fontweight="normal", pad=8)
    ax2.set_xlabel("Week", fontsize=13, fontweight="normal")
    ax2.set_ylabel("β(t) - Age Effect", fontsize=13, fontweight="normal")
    ax2.grid(False)
    ax2.set_facecolor("white")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    for ax in axes:
        ax.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    out_path = save_dir / "Task3_0_time_varying_beta_top2.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

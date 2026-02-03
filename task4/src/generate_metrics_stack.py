import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def main():
    df = pd.read_csv(r"task4/table/four_methods_metrics.csv", index_col=0)
    metrics = ["legitimacy", "engagement", "robustness", "transparency"]
    methods = ["RANK", "PERCENT", "SAVE", "TWO_KEY"]
    colors = ["#cad3e0", "#9cb0ce", "#45618a", "#b4b4b6"]

    plt.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    y = np.arange(len(metrics))
    left = np.zeros(len(metrics))
    bars = []

    for i, method in enumerate(methods):
        values = df.loc[method, metrics].values
        b = ax.barh(
            y,
            values,
            left=left,
            height=0.55,
            color=colors[i],
            edgecolor="none",
        )
        for patch in b:
            patch.set_path_effects(
                [
                    pe.SimplePatchShadow(offset=(2, -2), alpha=0.25, shadow_rgbFace=(0, 0, 0)),
                    pe.Normal(),
                ]
            )
        bars.append(b)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([s.replace("_", " ").title() for s in metrics], fontsize=12)
    ax.set_xlim(0, 4)
    ax.set_xticks([])
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.grid(False)

    ax.legend(
        [b[0] for b in bars],
        methods,
        frameon=False,
        fontsize=11,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )

    fig.tight_layout()
    fig.savefig(
        r"task4/figure/Task4_metrics_stack.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )


if __name__ == "__main__":
    main()

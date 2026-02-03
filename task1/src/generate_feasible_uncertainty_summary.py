from pathlib import Path
from PIL import Image


def crop_boxplot_panel(img: Image.Image) -> Image.Image:
    """Crop bottom-left panel from a 2x2 season figure."""
    w, h = img.size
    left = 0
    top = int(h * 0.5)
    right = int(w * 0.5)
    bottom = h
    return img.crop((left, top, right, bottom))


def crop_ci_panel(img: Image.Image) -> Image.Image:
    """Crop right half (CI width panel) from certainty heatmap."""
    w, h = img.size
    left = int(w * 0.5)
    top = 0
    right = w
    bottom = h
    return img.crop((left, top, right, bottom))


def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    if h == target_h:
        return img
    new_w = int(w * (target_h / h))
    return img.resize((new_w, target_h), Image.LANCZOS)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    fig_dir = repo_root / "task1" / "figure"

    season5 = Image.open(fig_dir / "fan_vote_v2_season5.png").convert("RGB")
    season33 = Image.open(fig_dir / "fan_vote_v2_season33.png").convert("RGB")
    heat5 = Image.open(fig_dir / "fan_vote_v2_certainty_heatmap_season5.png").convert("RGB")
    heat33 = Image.open(fig_dir / "fan_vote_v2_certainty_heatmap_season33.png").convert("RGB")

    box5 = crop_boxplot_panel(season5)
    box33 = crop_boxplot_panel(season33)
    ci5 = crop_ci_panel(heat5)
    ci33 = crop_ci_panel(heat33)

    target_h = 520
    box5 = resize_to_height(box5, target_h)
    box33 = resize_to_height(box33, target_h)
    ci5 = resize_to_height(ci5, target_h)
    ci33 = resize_to_height(ci33, target_h)

    row1_w = box5.width + ci5.width
    row2_w = box33.width + ci33.width
    out_w = max(row1_w, row2_w)
    out_h = target_h * 2

    canvas = Image.new("RGB", (out_w, out_h), "white")
    canvas.paste(box5, (0, 0))
    canvas.paste(ci5, (box5.width, 0))
    canvas.paste(box33, (0, target_h))
    canvas.paste(ci33, (box33.width, target_h))

    out_path = fig_dir / "fan_vote_v2_feasible_uncertainty_summary.png"
    canvas.save(out_path, dpi=(300, 300))


if __name__ == "__main__":
    main()

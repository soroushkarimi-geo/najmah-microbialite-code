"""Fig. 7: first two principal components of the standardized 404-feature set
for all images. Class 1 (MF-B) and Class 2 (MF-E) in colour, the other
microfacies and the unlabeled images in grey.

    python plot_pca_scatter.py --input-csv data/image_metadata_auto_label.csv --figures-dir out/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from train_xgboost_cv import get_feature_columns

UNLABELED = "unlabeled"
CLASS_1_COLOR = "#c9772e"
CLASS_2_COLOR = "#4b5d9c"
OTHER_COLOR = "#9aa3ab"
UNLABELED_COLOR = "#d9d9d9"

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["font.size"] = 11


def run(input_csv, figures_dir):
    """Draw Fig. 7: images on the first two principal components.

    All feature columns are standardized (zero mean, unit variance) and
    projected with PCA. Unlabeled images, other microfacies, Class 1 (MF-B)
    and Class 2 (MF-E) are drawn in that order, so Class 2 is on top. The
    legend gives the number of images per group.

    Args:
        input_csv: Feature table with a ``label`` column (all images).
        figures_dir: Output folder (``pathlib.Path``); writes
            ``fig6_pca_scatter.png``.
    """
    df = pd.read_csv(input_csv)
    feature_cols = get_feature_columns(df)
    X = StandardScaler().fit_transform(df[feature_cols].values)
    pca = PCA(n_components=2).fit(X)
    coords = pca.transform(X)
    var1, var2 = pca.explained_variance_ratio_ * 100
    print(f"PC1 {var1:.1f}%, PC2 {var2:.1f}% of variance")

    # drawn in this order, Class 2 on top
    groups = [
        ("unlabeled", df["label"] == UNLABELED, UNLABELED_COLOR, 10),
        ("other microfacies", ~df["label"].isin(["MF-B", "MF-E", UNLABELED]), OTHER_COLOR, 14),
        ("Class 1 (laminated)", df["label"] == "MF-B", CLASS_1_COLOR, 20),
        ("Class 2 (clotted)", df["label"] == "MF-E", CLASS_2_COLOR, 20),
    ]

    fig, ax = plt.subplots(figsize=(7.4, 5.7))
    for name, mask, color, size in groups:
        ax.scatter(coords[mask.values, 0], coords[mask.values, 1], s=size, c=color, alpha=0.75,
                   edgecolors="none", label=f"{name} ({int(mask.sum())})")
        print(f"  {name}: {int(mask.sum())} images")
    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)")
    ax.set_title("Principal components of the 404-feature set (images)", fontsize=14)
    ax.legend(loc="upper left", fontsize=10, frameon=False, markerscale=1.2)
    fig.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "fig6_pca_scatter.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``figures_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, figures_dir=Path(args.figures_dir))


if __name__ == "__main__":
    main()

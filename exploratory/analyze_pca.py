"""Principal component analysis of the full feature set, for diagnosis.

Two questions:
  1. Effective dimensionality: how many principal components are needed for
     90% and 95% of the variance of the standardized features? Few
     components mean strong redundancy between the GLCM, DINOv2 and
     lamination features, which matters when there are many more features
     than specimens.
  2. Class separation: do the labeled classes form separate groups on the
     first two components, or do they overlap? Overlap suggests a textural
     continuum between classes; separate groups that a classifier still
     confuses suggest too few training specimens instead.

The 10 features with the largest absolute loadings on PC1 and PC2 are printed.

Outputs (--figures-dir):
  pca_scree.png             cumulative explained variance, first 50 components
  pca_scatter_by_class.png  first two components, coloured by label

Run from the repository root:
    PYTHONPATH=. python exploratory/analyze_pca.py --input-csv data/image_metadata_auto_label.csv
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

UNLABELED = "unlabeled"
NON_FEATURE_COLS = {"image_id", "sample_code", "section", "image_type", "source_path", "label", "cluster"}

CLASS_COLORS = {
    "MF-A": "#1f77b4",
    "MF-B": "#ff7f0e",
    "MF-C": "#2ca02c",
    "MF-D": "#d62728",
    "MF-E": "#9467bd",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """List the feature columns of a feature table.

    Args:
        df: Feature table.

    Returns:
        list of str: Every column that is not metadata
        (``NON_FEATURE_COLS``).
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def plot_scree(cumulative_variance: np.ndarray, output_path: Path) -> None:
    """Plot cumulative explained variance for the first 50 components.

    Reference lines mark 90% and 95% of the variance.

    Args:
        cumulative_variance: Cumulative explained-variance ratios, one per
            component.
        output_path: Path of the PNG file to write.
    """
    n_show = min(50, len(cumulative_variance))
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_show + 1), cumulative_variance[:n_show], marker="o", markersize=3)
    plt.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="90% variance")
    plt.axhline(0.95, color="gray", linestyle=":", linewidth=1, label="95% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title(f"PCA cumulative explained variance (first {n_show} components)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_class_scatter(df: pd.DataFrame, pca_2d: PCA, output_path: Path) -> None:
    """Plot the images on the first two principal components, coloured by label.

    Unlabeled images are drawn in light grey underneath; the classes in
    ``CLASS_COLORS`` are drawn on top.

    Args:
        df: Image table with ``pc1``, ``pc2`` and ``label`` columns.
        pca_2d: Fitted two-component PCA, used for the axis labels.
        output_path: Path of the PNG file to write.
    """
    unlabeled_df = df[df["label"] == UNLABELED]
    labeled_df = df[df["label"] != UNLABELED]

    plt.figure(figsize=(9, 7))
    plt.scatter(
        unlabeled_df["pc1"], unlabeled_df["pc2"],
        c="lightgray", s=10, alpha=0.3, label="unlabeled", zorder=1,
    )
    for cls, color in CLASS_COLORS.items():
        subset = labeled_df[labeled_df["label"] == cls]
        if len(subset) == 0:
            continue
        plt.scatter(
            subset["pc1"], subset["pc2"],
            c=color, s=25, alpha=0.85, label=cls,
            edgecolors="black", linewidths=0.3, zorder=2,
        )
    plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0] * 100:.1f}% variance)")
    plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1] * 100:.1f}% variance)")
    plt.title("PCA of combined features, colored by facies label")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run(input_csv: str, figures_dir: Path) -> None:
    """Run the PCA diagnostic on all images, labeled and unlabeled.

    Args:
        input_csv: Labeled feature table (output of ``auto_label.py``).
        figures_dir: Output folder for the two figures (``pathlib.Path``).
    """
    df = pd.read_csv(input_csv)
    feature_cols = get_feature_columns(df)
    print(f"Total features: {len(feature_cols)}")
    print(f"Total images (labeled + unlabeled): {len(df)}")

    X = StandardScaler().fit_transform(df[feature_cols].values)

    pca_full = PCA().fit(X)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_90 = int(np.argmax(cumulative_variance >= 0.90) + 1)
    n_for_95 = int(np.argmax(cumulative_variance >= 0.95) + 1)

    print(f"\nComponents needed for 90% of total variance: {n_for_90} (of {len(feature_cols)} features)")
    print(f"Components needed for 95% of total variance: {n_for_95}")
    print(f"First 2 components alone explain: {cumulative_variance[1] * 100:.1f}% of variance")
    print(f"First 5 components alone explain: {cumulative_variance[4] * 100:.1f}% of variance")

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_scree(cumulative_variance, figures_dir / "pca_scree.png")
    print(f"\nSaved scree plot -> {figures_dir / 'pca_scree.png'}")

    pca_2d = PCA(n_components=2).fit(X)
    coords = pca_2d.transform(X)
    df = df.assign(pc1=coords[:, 0], pc2=coords[:, 1])

    plot_class_scatter(df, pca_2d, figures_dir / "pca_scatter_by_class.png")
    print(f"Saved class scatter plot -> {figures_dir / 'pca_scatter_by_class.png'}")

    loadings = pd.DataFrame(pca_2d.components_.T, index=feature_cols, columns=["PC1", "PC2"])
    print("\nTop 10 features driving PC1 (by absolute loading):")
    print(loadings["PC1"].abs().sort_values(ascending=False).head(10).to_string())
    print("\nTop 10 features driving PC2 (by absolute loading):")
    print(loadings["PC2"].abs().sort_values(ascending=False).head(10).to_string())

    print(
        "\nLook at pca_scatter_by_class.png: if MF-A points are visually mixed in "
        "among MF-B points, that supports a genuine textural overlap. If MF-A forms "
        "its own loose cluster but the model still struggles, that instead points to "
        "MF-A simply needing more training examples (4 is very few)."
    )


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``figures_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, figures_dir=Path(args.figures_dir))


if __name__ == "__main__":
    main()
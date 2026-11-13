"""Fig. 4: cumulative explained variance against number of principal
components, standardized 404-feature set, all images, first 50 components.

    python plot_pca_explained_variance.py --input-csv data/image_metadata_auto_label.csv --figures-dir out/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from train_xgboost_cv import get_feature_columns

N_SHOW = 50


def run(input_csv, figures_dir):
    """Draw Fig. 4: cumulative explained variance of the principal components.

    The standardized feature set is decomposed with PCA; the cumulative
    explained variance of the first 50 components is plotted with reference
    lines at 90% and 95%. The variance explained by 2, 30 and 50 components
    is printed.

    Args:
        input_csv: Feature table (all images).
        figures_dir: Output folder (``pathlib.Path``); writes
            ``fig4_pca_explained_variance.png``.
    """
    df = pd.read_csv(input_csv)
    feature_cols = get_feature_columns(df)
    print(f"{len(df)} images, {len(feature_cols)} features")

    X = StandardScaler().fit_transform(df[feature_cols].values)
    cumulative = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    for n in (2, 30, 50):
        print(f"  {n} components: {cumulative[n - 1] * 100:.1f}% of variance")

    n_show = min(N_SHOW, len(cumulative))
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_show + 1), cumulative[:n_show], marker="o", markersize=3)
    plt.axhline(0.90, color="gray", linestyle="--", linewidth=1, label="90% variance")
    plt.axhline(0.95, color="gray", linestyle=":", linewidth=1, label="95% variance")
    plt.xlabel("Number of principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title(f"PCA cumulative explained variance (first {n_show} components)")
    plt.legend()
    plt.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "fig4_pca_explained_variance.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
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

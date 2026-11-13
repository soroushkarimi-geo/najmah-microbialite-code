"""Fig. 5: two-class image confusion matrices of the four models (Table 2),
percentages per true class. Reads the compare_*_confusion.csv tables written
by compare_models.py, so run that first.

    python plot_model_confusion_matrices.py --tables-dir out/tables --figures-dir out/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# compare_models.py file name -> panel title
MODELS = {
    "XGBoost": "Gradient-boosted trees",
    "Logistic_Regression": "L1 logistic regression, PCA-30",
    "Random_Forest": "Random forest",
    "kNN": "$k$-nearest neighbours, PCA-30",
}
CLASS_NAMES = {"MF-B": "Class 1", "MF-E": "Class 2"}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["font.size"] = 11


def run(tables_dir, figures_dir):
    """Draw Fig. 5: one 2 x 2 confusion matrix per model.

    Each cell shows the image count and its share of the true class (row
    percentage), and is shaded by that share. Panel titles give the
    image-level accuracy.

    Args:
        tables_dir: Folder (``pathlib.Path``) with the
            ``compare_<model>_confusion.csv`` files from ``compare_models.py``.
        figures_dir: Output folder (``pathlib.Path``); writes
            ``fig5_model_confusion_matrices.png``.
    """
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 8.2))

    for ax, (file_name, title) in zip(axes.flat, MODELS.items()):
        cm = pd.read_csv(tables_dir / f"compare_{file_name}_confusion.csv", index_col=0)
        cm = cm.loc[list(CLASS_NAMES), list(CLASS_NAMES)]
        counts = cm.to_numpy()
        row_frac = counts / counts.sum(axis=1, keepdims=True)
        accuracy = counts.trace() / counts.sum()
        print(f"{title.replace('$', '')}: accuracy {accuracy * 100:.1f}%")

        # colour = share of the true class
        ax.imshow(row_frac, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                color = "white" if row_frac[i, j] > 0.5 else "black"
                ax.text(j, i, f"{counts[i, j]}\n({row_frac[i, j] * 100:.0f}%)",
                        ha="center", va="center", color=color, fontsize=12)

        labels = [CLASS_NAMES[c] for c in cm.index]
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{title}\naccuracy {accuracy * 100:.1f}% (images)", fontsize=11)

    fig.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "fig5_model_confusion_matrices.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``tables_dir`` and ``figures_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=str, default="results/tables")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(tables_dir=Path(args.tables_dir), figures_dir=Path(args.figures_dir))


if __name__ == "__main__":
    main()

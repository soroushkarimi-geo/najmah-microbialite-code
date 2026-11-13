"""Comparison plots of the four classifiers of compare_models.py (XGBoost,
L1 logistic regression, random forest and kNN).

Reads the per-model classification reports and confusion matrices that
compare_models.py writes (compare_<model>_report.csv and
compare_<model>_confusion.csv) and draws:
  model_comparison_overall.png         accuracy and macro F1 per model
  model_comparison_per_class_f1.png    F1 per class, grouped by model
  model_comparison_confusion_grid.png  the four confusion matrices
  model_comparison_mfa_row.png         predicted classes of the true MF-A
                                       images (only if MF-A is in the tables)

Run from the repository root, after compare_models.py:
    PYTHONPATH=. python exploratory/visualize_model_comparison.py --tables-dir out/tables --figures-dir out/figures
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save to file only, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_NAMES = ["XGBoost", "Logistic_Regression", "Random_Forest", "kNN"]
MODEL_DISPLAY = {
    "XGBoost": "XGBoost",
    "Logistic_Regression": "Logistic Regression\n(L1, PCA-30)",
    "Random_Forest": "Random Forest",
    "kNN": "kNN k=5\n(PCA-30)",
}
MODEL_COLORS = {
    "XGBoost": "#1f77b4",
    "Logistic_Regression": "#ff7f0e",
    "Random_Forest": "#2ca02c",
    "kNN": "#9467bd",
}


def load_reports(tables_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the classification report of every model in ``MODEL_NAMES``.

    Args:
        tables_dir: Folder with the ``compare_<model>_report.csv`` files.

    Returns:
        dict: Model name -> report DataFrame (rows: classes and averages).

    Raises:
        FileNotFoundError: If a report is missing.
    """
    reports = {}
    for name in MODEL_NAMES:
        path = tables_dir / f"compare_{name}_report.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} - run compare_models.py first.")
        reports[name] = pd.read_csv(path, index_col=0)
    return reports


def load_confusions(tables_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the confusion matrix of every model in ``MODEL_NAMES``.

    Args:
        tables_dir: Folder with the ``compare_<model>_confusion.csv`` files.

    Returns:
        dict: Model name -> confusion matrix (rows true, columns predicted).

    Raises:
        FileNotFoundError: If a matrix is missing.
    """
    confusions = {}
    for name in MODEL_NAMES:
        path = tables_dir / f"compare_{name}_confusion.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path} - run compare_models.py first.")
        confusions[name] = pd.read_csv(path, index_col=0)
    return confusions


def plot_overall_comparison(reports: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Bar chart of accuracy and macro F1 for each model.

    A large gap between the two shows that the smaller classes are
    predicted worse than the overall accuracy suggests.

    Args:
        reports: Output of ``load_reports``.
        output_path: Path of the PNG file to write.
    """
    accuracies = [reports[m].loc["accuracy", "precision"] for m in MODEL_NAMES]
    macro_f1s = [reports[m].loc["macro avg", "f1-score"] for m in MODEL_NAMES]

    x = np.arange(len(MODEL_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, accuracies, width, label="Accuracy", color="#4C72B0")
    ax.bar(x + width / 2, macro_f1s, width, label="Macro F1", color="#DD8452")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_NAMES])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Overall Accuracy vs. Macro F1 by Model\n(large gap = imbalance/minority-class failure, not just overall skill)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for i, (acc, f1) in enumerate(zip(accuracies, macro_f1s)):
        ax.text(i - width / 2, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=9)
        ax.text(i + width / 2, f1 + 0.02, f"{f1:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_per_class_f1(reports: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Grouped bar chart of the F1 score of every class, one bar per model.

    Args:
        reports: Output of ``load_reports``.
        output_path: Path of the PNG file to write.
    """
    classes = [c for c in reports[MODEL_NAMES[0]].index if c.startswith("MF-")]
    x = np.arange(len(classes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, model in enumerate(MODEL_NAMES):
        f1_scores = [reports[model].loc[c, "f1-score"] for c in classes]
        offset = (i - 1.5) * width
        ax.bar(x + offset, f1_scores, width, label=MODEL_DISPLAY[model].split("\n")[0], color=MODEL_COLORS[model])

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title("Per-Class F1 Score, All Four Models\n(MF-D/MF-F show 0 for all models - single-sample classes, skipped in CV)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_grid(confusions: dict[str, pd.DataFrame], output_path: Path) -> None:
    """2 x 2 grid of confusion-matrix heatmaps with counts, one per model.

    Args:
        confusions: Output of ``load_confusions``.
        output_path: Path of the PNG file to write.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    axes = axes.flatten()

    for ax, model in zip(axes, MODEL_NAMES):
        cm = confusions[model]
        im = ax.imshow(cm.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(cm.columns)))
        ax.set_yticks(range(len(cm.index)))
        ax.set_xticklabels(cm.columns, rotation=45, ha="right")
        ax.set_yticklabels(cm.index)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(MODEL_DISPLAY[model].split("\n")[0])

        max_val = cm.values.max()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm.values[i, j]
                color = "white" if val > max_val * 0.5 else "black"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    fig.suptitle("Confusion Matrices, All Four Models (Same Labels, Same LOSO CV Protocol)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_mfa_row(confusions: dict[str, pd.DataFrame], output_path: Path) -> None:
    """Bar chart of the classes predicted for the true MF-A images, per model.

    If all four models send MF-A images to the same wrong class, the
    confusion comes from the data rather than from one model. Skipped, with
    a message, when MF-A is not in the confusion matrices.

    Args:
        confusions: Output of ``load_confusions``.
        output_path: Path of the PNG file to write.
    """
    if "MF-A" not in confusions[MODEL_NAMES[0]].index:
        print("MF-A not present in confusion matrices - skipping this plot.")
        return

    classes = list(confusions[MODEL_NAMES[0]].columns)
    x = np.arange(len(classes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, model in enumerate(MODEL_NAMES):
        row = confusions[model].loc["MF-A"]
        offset = (i - 1.5) * width
        ax.bar(x + offset, row.values, width, label=MODEL_DISPLAY[model].split("\n")[0], color=MODEL_COLORS[model])

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Number of true MF-A images predicted as this class")
    ax.set_title("Where True MF-A Images Get Predicted, All Four Models\n(consistent MF-E spike across all models = data issue, not model-specific)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run(tables_dir: Path, figures_dir: Path) -> None:
    """Read the comparison tables and draw all four figures.

    Args:
        tables_dir: Folder (``pathlib.Path``) with the tables of
            ``compare_models.py``.
        figures_dir: Output folder (``pathlib.Path``); created if needed.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)

    reports = load_reports(tables_dir)
    confusions = load_confusions(tables_dir)

    plot_overall_comparison(reports, figures_dir / "model_comparison_overall.png")
    print(f"Saved -> {figures_dir / 'model_comparison_overall.png'}")

    plot_per_class_f1(reports, figures_dir / "model_comparison_per_class_f1.png")
    print(f"Saved -> {figures_dir / 'model_comparison_per_class_f1.png'}")

    plot_confusion_grid(confusions, figures_dir / "model_comparison_confusion_grid.png")
    print(f"Saved -> {figures_dir / 'model_comparison_confusion_grid.png'}")

    plot_mfa_row(confusions, figures_dir / "model_comparison_mfa_row.png")
    print(f"Saved -> {figures_dir / 'model_comparison_mfa_row.png'}")

    print(f"\nAll figures saved to {figures_dir}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``tables_dir`` and ``figures_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=str, default="results/tables")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(tables_dir=Path(args.tables_dir), figures_dir=Path(args.figures_dir))


if __name__ == "__main__":
    main()
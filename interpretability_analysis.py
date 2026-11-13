"""Feature importance by two independent methods (Figs. 5 and 6):
SHAP (TreeExplainer on XGBoost) and the ANOVA F-value (no model).

An XGBoost model is trained on all labeled images, then:
  1. global and per-class mean |SHAP| per feature
  2. F-value per feature
  3. overlap of the top 30 features of the two methods
  4. feature family (GLCM / DINOv2 / lamination) of the top features

Tables: shap_importance.csv, shap_per_class_importance.csv, f_value_importance.csv,
        shap_per_class_top10.csv, feature_family_breakdown.csv
Figures: interp_top_features_comparison.png (Fig. 5), interp_family_breakdown.png (Fig. 6),
         interp_shap_summary.png

This is a full-dataset fit; see foldwise_interpretability_check.py for the fold-wise check.

    python interpretability_analysis.py --input-csv data/image_metadata_auto_label.csv --tables-dir out/tables --figures-dir out/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from config import Config
from train_xgboost_cv import get_feature_columns

UNLABELED = "unlabeled"
TOP_N = 20
TOP_N_FAMILY = 30


def feature_family(feature_name):
    """Name the feature family of a feature column.

    Args:
        feature_name: Column name.

    Returns:
        str: ``'DINOv2'`` for ``dino_`` columns, ``'Lamination'`` for
        ``lam_`` columns and ``'GLCM'`` for everything else.
    """
    if feature_name.startswith("dino_"):
        return "DINOv2"
    if feature_name.startswith("lam_"):
        return "Lamination"
    return "GLCM"


def train_final_model(labeled_df, feature_cols, label_encoder):
    """Fit one XGBoost model on all labeled images.

    Uses ``Config.XGBOOST_PARAMS`` and balanced sample weights (inverse class
    frequency). This full-data model is used only to rank features; it is
    not a cross-validated estimate of accuracy.

    Args:
        labeled_df: Labeled images with a ``label`` column and the features.
        feature_cols: Names of the feature columns to use.
        label_encoder: ``LabelEncoder`` fitted on the class names.

    Returns:
        xgboost.XGBClassifier: The fitted model.
    """
    X = labeled_df[feature_cols]
    y = label_encoder.transform(labeled_df["label"])
    sample_weight = compute_sample_weight("balanced", y)

    params = dict(Config.XGBOOST_PARAMS)
    params["num_class"] = len(label_encoder.classes_)
    model = xgb.XGBClassifier(**params)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def compute_shap_importance(model, X, label_encoder):
    """SHAP feature importance of a fitted tree model.

    SHAP values are computed with ``shap.TreeExplainer`` on ``X``. Per-class
    importance is the mean absolute SHAP value of each feature for that
    class's output; global importance is the mean over the classes. Both
    the older (list of arrays per class) and newer (one 3-D array) output
    formats of the ``shap`` package are handled.

    Args:
        model: Fitted XGBoost classifier.
        X: Feature table (DataFrame) to explain.
        label_encoder: ``LabelEncoder`` whose classes name the columns.

    Returns:
        tuple: ``(global_importance, per_class_df, shap_values)``.
        ``global_importance`` is a Series sorted from most to least
        important; ``per_class_df`` has one row per feature and one column
        per class; ``shap_values`` is the raw explainer output.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # older shap: list of arrays, one per class; newer: one 3D array
    if isinstance(shap_values, list):
        per_class = np.stack([np.abs(sv).mean(axis=0) for sv in shap_values], axis=1)
    else:
        per_class = np.abs(shap_values).mean(axis=0)
        if per_class.ndim == 1:
            per_class = per_class[:, np.newaxis]

    per_class_df = pd.DataFrame(per_class, index=X.columns, columns=label_encoder.classes_)
    global_importance = per_class_df.mean(axis=1).sort_values(ascending=False)
    return global_importance, per_class_df, shap_values


def compute_f_value_importance(X, y):
    """Rank features by the one-way ANOVA F-value between classes.

    The F-value does not depend on any model, so it is an independent check
    of the SHAP ranking.

    Args:
        X: Feature table (DataFrame).
        y: Encoded class labels.

    Returns:
        pandas.Series: F-value per feature, sorted from highest to lowest.
    """
    f_values, _ = f_classif(X, y)
    return pd.Series(f_values, index=X.columns).sort_values(ascending=False)


def compare_top_features(shap_ranking, f_ranking, top_n):
    """Overlap between the top features of two rankings.

    Args:
        shap_ranking: Series sorted by SHAP importance.
        f_ranking: Series sorted by F-value.
        top_n: Number of top features taken from each ranking.

    Returns:
        dict: ``top_n``, ``overlap_count``, ``overlap_fraction``
        (overlap_count / top_n) and ``overlapping_features`` (sorted list).
    """
    shap_top = set(shap_ranking.head(top_n).index)
    f_top = set(f_ranking.head(top_n).index)
    overlap = shap_top & f_top
    return {
        "top_n": top_n,
        "overlap_count": len(overlap),
        "overlap_fraction": len(overlap) / top_n,
        "overlapping_features": sorted(overlap),
    }


def plot_top_features_comparison(shap_ranking, f_ranking, output_path):
    """Draw Fig. 5: the top 20 features by SHAP and by F-value, side by side.

    Bars are coloured by feature family (GLCM blue, lamination green,
    DINOv2 grey).

    Args:
        shap_ranking: Series sorted by SHAP importance.
        f_ranking: Series sorted by F-value.
        output_path: Path of the PNG file to write.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    shap_top = shap_ranking.head(TOP_N)
    colors_shap = ["#1f77b4" if feature_family(f) == "GLCM" else
                   "#2ca02c" if feature_family(f) == "Lamination" else "#cccccc"
                   for f in shap_top.index]
    axes[0].barh(range(len(shap_top)), shap_top.values[::-1], color=colors_shap[::-1])
    axes[0].set_yticks(range(len(shap_top)))
    axes[0].set_yticklabels(shap_top.index[::-1], fontsize=8)
    axes[0].set_xlabel("Mean |SHAP value|")
    axes[0].set_title(f"Top {TOP_N} Features — SHAP")

    f_top = f_ranking.head(TOP_N)
    colors_f = ["#1f77b4" if feature_family(f) == "GLCM" else
                "#2ca02c" if feature_family(f) == "Lamination" else "#cccccc"
                for f in f_top.index]
    axes[1].barh(range(len(f_top)), f_top.values[::-1], color=colors_f[::-1])
    axes[1].set_yticks(range(len(f_top)))
    axes[1].set_yticklabels(f_top.index[::-1], fontsize=8)
    axes[1].set_xlabel("ANOVA F-value")
    axes[1].set_title(f"Top {TOP_N} Features — F-value (model-independent)")

    fig.suptitle("SHAP (model-based) vs. F-value (statistical) — GLCM=blue, Lamination=green, DINOv2=gray", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_family_breakdown(shap_ranking, f_ranking, output_path):
    """Draw Fig. 6: how many of the top 30 features belong to each family.

    Args:
        shap_ranking: Series sorted by SHAP importance.
        f_ranking: Series sorted by F-value.
        output_path: Path of the PNG file to write.

    Returns:
        pandas.DataFrame: Counts per family (rows GLCM, Lamination, DINOv2)
        for the columns ``SHAP`` and ``F_value``.
    """
    families = ["GLCM", "Lamination", "DINOv2"]

    shap_families = [feature_family(f) for f in shap_ranking.head(TOP_N_FAMILY).index]
    f_families = [feature_family(f) for f in f_ranking.head(TOP_N_FAMILY).index]

    shap_counts = pd.Series(shap_families).value_counts().reindex(families, fill_value=0)
    f_counts = pd.Series(f_families).value_counts().reindex(families, fill_value=0)

    x = np.arange(len(families))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, shap_counts.values, width, label="SHAP", color="#4C72B0")
    ax.bar(x + width / 2, f_counts.values, width, label="F-value", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel(f"Count in top {TOP_N_FAMILY} features")
    ax.set_title(f"Feature Family Representation in Top {TOP_N_FAMILY}, Both Methods")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    return pd.DataFrame({"SHAP": shap_counts, "F_value": f_counts})


def save_shap_summary_plot(shap_values, X, label_encoder, output_path):
    """Save the standard SHAP summary plot (top 20 features).

    If the ``shap`` package cannot draw the plot, a message is printed and
    the figure is skipped; the rest of the analysis is not affected.

    Args:
        shap_values: Raw output of ``compute_shap_importance``.
        X: Feature table that was explained.
        label_encoder: ``LabelEncoder`` with the class names.
        output_path: Path of the PNG file to write.
    """
    plt.figure(figsize=(10, 8))
    try:
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values, X, class_names=list(label_encoder.classes_), show=False, max_display=20)
        else:
            shap.summary_plot(shap_values, X, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
    except Exception as e:
        print(f"Could not make the SHAP summary plot ({e}), skipped")
    finally:
        plt.close()


def run(input_csv, tables_dir, figures_dir):
    """Rank the features by SHAP and by F-value and save tables and figures.

    Trains ``train_final_model`` on all labeled images (every labeled
    class, not only MF-B / MF-E), computes both rankings, their top-30
    overlap, the top 10 SHAP features per class and the feature-family
    breakdown, and draws Figs. 5 and 6 and the SHAP summary plot.

    Args:
        input_csv: Labeled feature table. Rows labeled ``'unlabeled'`` are
            ignored.
        tables_dir: Output folder for the CSV tables (``pathlib.Path``).
        figures_dir: Output folder for the PNG figures (``pathlib.Path``).
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()
    feature_cols = get_feature_columns(df)

    present_classes = sorted(labeled_df["label"].unique())
    label_encoder = LabelEncoder().fit(present_classes)
    print(f"Training on {len(labeled_df)} labeled images, {len(present_classes)} classes...")

    model = train_final_model(labeled_df, feature_cols, label_encoder)
    X = labeled_df[feature_cols]
    y = label_encoder.transform(labeled_df["label"])

    # -----------------------------
    # SHAP and F-value
    print("SHAP...")
    shap_global, shap_per_class, shap_values = compute_shap_importance(model, X, label_encoder)
    shap_global.to_csv(tables_dir / "shap_importance.csv", header=["mean_abs_shap"])
    shap_per_class.to_csv(tables_dir / "shap_per_class_importance.csv")

    print("F-value...")
    f_ranking = compute_f_value_importance(X, y)
    f_ranking.to_csv(tables_dir / "f_value_importance.csv", header=["f_value"])

    print(f"\nTop {TOP_N} features by SHAP:")
    print(shap_global.head(TOP_N).to_string())
    print(f"\nTop {TOP_N} features by F-value:")
    print(f_ranking.head(TOP_N).to_string())

    comparison = compare_top_features(shap_global, f_ranking, TOP_N_FAMILY)
    print(f"\n=== Overlap of the top {TOP_N_FAMILY} features ===")
    print(f"{comparison['overlap_count']} of {TOP_N_FAMILY} "
          f"({comparison['overlap_fraction']*100:.1f}%)")
    print("In the top list of both methods:")
    print(comparison["overlapping_features"])

    print("\nTop 10 SHAP features per class:")
    per_class_top = {}
    for cls in shap_per_class.columns:
        top10 = shap_per_class[cls].sort_values(ascending=False).head(10)
        per_class_top[cls] = top10.index.tolist()
        print(f"\n  {cls}:")
        print(top10.to_string())
    pd.DataFrame(per_class_top).to_csv(tables_dir / "shap_per_class_top10.csv", index=False)

    # -----------------------------
    # Figures
    plot_top_features_comparison(shap_global, f_ranking, figures_dir / "interp_top_features_comparison.png")
    print(f"\nSaved {figures_dir / 'interp_top_features_comparison.png'}")

    family_df = plot_family_breakdown(shap_global, f_ranking, figures_dir / "interp_family_breakdown.png")
    family_df.to_csv(tables_dir / "feature_family_breakdown.csv")
    print(f"Saved {figures_dir / 'interp_family_breakdown.png'}")
    print("\nFeature families in the top 30:")
    print(family_df.to_string())

    save_shap_summary_plot(shap_values, X, label_encoder, figures_dir / "interp_shap_summary.png")
    print(f"Saved {figures_dir / 'interp_shap_summary.png'}")

    print(f"\nTables in {tables_dir}")
    print(f"Figures in {figures_dir}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv``, ``tables_dir`` and ``figures_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--tables-dir", type=str, default="results/tables")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, tables_dir=Path(args.tables_dir), figures_dir=Path(args.figures_dir))


if __name__ == "__main__":
    main()

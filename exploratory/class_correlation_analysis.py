"""Two correlation diagnostics between the microfacies classes.

1. Feature space: Pearson correlation between the mean feature vectors
   (centroids) of every pair of classes. This needs no trained model and shows
   whether classes that a classifier confuses are also close in feature space.

2. Stratigraphy: lagged cross-correlation between the occurrence sequences of
   two classes in the Dudan section, ordered by sample number (the field
   numbering follows the measured section). A peak at a non-zero lag would
   indicate a regular spacing, or one class systematically preceding the
   other, as expected from cyclic (parasequence-scale) deposition. Lags are
   counted in sample positions, not metres.

Outputs (--output-dir):
  class_feature_correlation.csv            class x class correlation matrix
  stratigraphic_lag_cross_correlation.csv  lag, correlation and overlap per pair

Run from the repository root:
    PYTHONPATH=. python exploratory/class_correlation_analysis.py --input-csv data/image_metadata_auto_label.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from train_xgboost_cv import get_feature_columns

UNLABELED = "unlabeled"


def class_feature_correlation(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Pearson correlation between the mean feature vectors of the classes.

    The centroid of each class is the mean of its images (unlabeled images
    excluded); centroids are correlated across the raw, unstandardized
    features.

    Args:
        df: Image-level table with ``label`` and the feature columns.
        feature_cols: Features to use.

    Returns:
        pandas.DataFrame: Square class x class correlation matrix.
    """
    labeled = df[df["label"] != UNLABELED]
    centroids = labeled.groupby("label")[feature_cols].mean()
    return centroids.T.corr()


def parse_sample_number(sample_code: str) -> float:
    """Extract the field sample number from a sample code.

    Args:
        sample_code: Code such as ``'SD_105'``, ``'SK_87'`` or ``'SD_102B'``.

    Returns:
        float: The first run of digits, or NaN if there is none. A letter
        suffix is ignored.
    """
    m = re.search(r"(\d+)", sample_code)
    return float(m.group(1)) if m else np.nan


def lag_cross_correlation(
    df: pd.DataFrame, section_prefix: str, class_a: str, class_b: str, max_lag: int = 5
) -> pd.DataFrame:
    """Lagged cross-correlation of two classes' occurrences up-section.

    The specimens of one section are ordered by sample number. Each class
    gives a 0/1 indicator sequence over all these specimens (1 where the
    specimen belongs to the class). The sequence of ``class_a`` is
    correlated with that of ``class_b`` shifted by every lag from
    ``-max_lag`` to ``+max_lag``; a positive lag pairs each specimen of
    ``class_a`` with the specimen ``lag`` positions above it.

    Args:
        df: Image-level table with ``sample_code`` and ``label``; reduced to
            one row per specimen.
        section_prefix: ``'SD'`` or ``'SK'``.
        class_a: First class.
        class_b: Second class.
        max_lag: Largest shift, in specimen positions, in either direction.

    Returns:
        pandas.DataFrame: ``lag``, ``correlation`` and ``n_overlapping``
        (length of the overlapping sequences). The correlation is NaN when
        fewer than 3 positions overlap or either sequence is constant. Empty
        if neither class occurs in the section.
    """
    per_sample = df.drop_duplicates("sample_code").copy()
    per_sample["section_prefix"] = per_sample["sample_code"].str.split("_").str[0]
    section_df = per_sample[per_sample["section_prefix"] == section_prefix].copy()
    section_df["sample_number"] = section_df["sample_code"].apply(parse_sample_number)
    section_df = section_df.dropna(subset=["sample_number"]).sort_values("sample_number")

    relevant = section_df[section_df["label"].isin([class_a, class_b])]
    if relevant.empty:
        return pd.DataFrame(columns=["lag", "correlation", "n_overlapping"])

    full_index = section_df["sample_number"].values
    ind_a = section_df["label"].eq(class_a).astype(float).values
    ind_b = section_df["label"].eq(class_b).astype(float).values

    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = ind_a[-lag:], ind_b[: len(ind_b) + lag]
        elif lag > 0:
            a, b = ind_a[: len(ind_a) - lag], ind_b[lag:]
        else:
            a, b = ind_a, ind_b
        n = len(a)
        if n < 3 or np.std(a) == 0 or np.std(b) == 0:
            rows.append({"lag": lag, "correlation": np.nan, "n_overlapping": n})
            continue
        corr = np.corrcoef(a, b)[0, 1]
        rows.append({"lag": lag, "correlation": corr, "n_overlapping": n})

    return pd.DataFrame(rows)


def run(input_csv: str, output_dir: Path) -> None:
    """Run both diagnostics, print the results and save the two tables.

    The lag analysis covers every pair of labeled classes (each class with
    itself included) in the Dudan section, with lags up to 5 positions,
    and prints the lag with the strongest absolute correlation per pair.

    Args:
        input_csv: Labeled feature table (output of ``auto_label.py``).
        output_dir: Output folder (``pathlib.Path``); created if needed.
    """
    df = pd.read_csv(input_csv)
    feature_cols = get_feature_columns(df)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== (1) Feature-space cross-correlation between class mean profiles ===")
    corr = class_feature_correlation(df, feature_cols)
    print(corr.round(3))
    corr.to_csv(output_dir / "class_feature_correlation.csv")

    print("\n=== (2) Stratigraphic lag cross-correlation, Dudan section (SD) ===")
    print("(sample number used as a stratigraphic-position proxy - NOT measured depth)")
    labels_present = [c for c in df["label"].unique() if c != UNLABELED]
    all_lag_results = []
    for i, class_a in enumerate(labels_present):
        for class_b in labels_present[i:]:
            lag_df = lag_cross_correlation(df, "SD", class_a, class_b, max_lag=5)
            if lag_df.empty or lag_df["correlation"].isna().all():
                continue
            lag_df["class_a"] = class_a
            lag_df["class_b"] = class_b
            all_lag_results.append(lag_df)
            best_row = lag_df.loc[lag_df["correlation"].abs().idxmax()]
            print(f"  {class_a} vs {class_b}: strongest |corr|={best_row['correlation']:.3f} at lag={int(best_row['lag'])} (n={int(best_row['n_overlapping'])})")

    if all_lag_results:
        combined = pd.concat(all_lag_results, ignore_index=True)
        combined.to_csv(output_dir / "stratigraphic_lag_cross_correlation.csv", index=False)
        print(f"\nFull lag table saved -> {output_dir / 'stratigraphic_lag_cross_correlation.csv'}")
    else:
        print("\nNo class pair had enough overlapping, non-constant sequence to compute lag correlation.")

    print(f"Class feature-correlation matrix saved -> {output_dir / 'class_feature_correlation.csv'}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``output_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/tables/class_correlation")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()

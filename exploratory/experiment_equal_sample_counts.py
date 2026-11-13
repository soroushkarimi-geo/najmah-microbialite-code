"""Equal-specimen-count experiment: is a weak class weak only because it has
few specimens?

Class weighting changes how much each image counts in the loss, but it cannot
add independent information about a class that has few specimens. This
experiment removes the difference in specimen numbers instead: every class is
randomly reduced to the specimen count of the smallest class (whole specimens,
with all their images), and the reduced set is evaluated with the same
leave-one-specimen-out XGBoost as train_xgboost_cv.py.

- If the smallest class now performs like the others, its lower scores in the
  full data set are explained by the number of specimens.
- If it still lags, the class is harder to separate (textural overlap with
  another class, or high variability within the class).

Subsampling uses a fixed seed (--seed), and the kept and dropped specimens are
printed so that a run can be repeated exactly. Running several seeds shows
whether the outcome depends on which specimens were drawn.

Run from the repository root:
    PYTHONPATH=. python exploratory/experiment_equal_sample_counts.py --input-csv data/image_metadata_auto_label.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from config import Config
from train_xgboost_cv import get_feature_columns, run_loso_cv

UNLABELED = "unlabeled"


def equalize_sample_counts(labeled_df: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    """Reduce every class to the specimen count of the smallest class.

    Specimens are drawn without replacement; all images of a kept specimen
    are kept. The kept and dropped specimens of each class are printed.

    Args:
        labeled_df: Labeled images with ``sample_code`` and ``label``.
        random_seed: Seed of the random draw.

    Returns:
        pandas.DataFrame: The images of the kept specimens.
    """
    rng = np.random.default_rng(random_seed)

    sample_labels = labeled_df.groupby("sample_code")["label"].first()
    class_counts = sample_labels.value_counts()
    min_count = int(class_counts.min())
    min_class = class_counts.idxmin()

    print(f"Smallest class: {min_class} with {min_count} samples - equalizing all classes to this count.\n")

    kept_samples = []
    for cls in sorted(class_counts.index):
        cls_samples = sample_labels[sample_labels == cls].index.to_numpy()
        if len(cls_samples) > min_count:
            chosen = rng.choice(cls_samples, size=min_count, replace=False)
            dropped = sorted(set(cls_samples) - set(chosen))
            print(f"{cls}: kept {sorted(chosen)}, dropped {dropped}")
        else:
            chosen = cls_samples
            print(f"{cls}: kept all {sorted(chosen)} (already at or below minimum)")
        kept_samples.extend(chosen)

    return labeled_df[labeled_df["sample_code"].isin(kept_samples)].copy()


def run(input_csv: str, random_seed: int) -> None:
    """Run the experiment and print the per-class report and confusion matrix.

    Args:
        input_csv: Labeled feature table (output of ``auto_label.py``). Rows
            labeled ``'unlabeled'`` are ignored.
        random_seed: Seed of the specimen draw; repeat with several seeds to
            check that the result does not depend on one particular draw.
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()
    feature_cols = get_feature_columns(df)

    equalized_df = equalize_sample_counts(labeled_df, random_seed)

    present_classes = sorted(equalized_df["label"].unique())
    print(f"\nClasses in equalized dataset: {present_classes}")
    print(f"Total samples: {equalized_df['sample_code'].nunique()}, "
          f"total images: {len(equalized_df)}")
    print("\nImages per class (sample count is now equal, but image count "
          "still varies with how many photos each sample happened to have):")
    print(equalized_df["label"].value_counts())

    label_encoder = LabelEncoder().fit(present_classes)
    params = dict(Config.XGBOOST_PARAMS)
    params["num_class"] = len(present_classes)

    print(f"\nRunning leave-one-sample-out CV over {equalized_df['sample_code'].nunique()} "
          "equalized samples...")
    predictions_df = run_loso_cv(equalized_df, feature_cols, label_encoder, params)

    y_true = label_encoder.transform(predictions_df["label"])
    y_pred = label_encoder.transform(predictions_df["predicted_label"])

    all_label_ids = list(range(len(label_encoder.classes_)))
    report = classification_report(
        y_true, y_pred, labels=all_label_ids, target_names=label_encoder.classes_,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    cm = confusion_matrix(y_true, y_pred, labels=all_label_ids)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

    print(f"\n=== Equal-sample-count CV results ({len(predictions_df)} images, "
          f"{predictions_df['sample_code'].nunique()} samples) ===")
    print(report_df.round(3))
    print("\n=== Confusion matrix ===")
    print(cm_df)

    print(
        "\nInterpretation: if MF-A's precision/recall here are now roughly in line "
        "with the other classes, sample count alone explains its earlier weakness. "
        "If MF-A still clearly underperforms even here, that points to a harder "
        "problem - likely genuine overlap with another class - not fixable by "
        "labeling count alone."
    )


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``seed``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42,
                         help="Random seed for subsampling - try several to check result stability")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, random_seed=args.seed)


if __name__ == "__main__":
    main()
"""Compare XGBoost tree depths (max_depth) under leave-one-specimen-out CV.

With few specimens per class, deep trees can fit fold-specific noise instead
of fabric differences that generalize. The script runs the same LOSO
cross-validation for max_depth = 2, 3, 4, 5, 6 and 8, with every other
hyperparameter taken from ``config.py``, and reports pooled accuracy and macro
F1 for each depth. Results are ranked by macro F1, because accuracy alone is
dominated by the largest class and can hide changes in the smaller ones.

Models are fitted without class weights here, unlike in train_xgboost_cv.py.

Output (--output-csv): max_depth, accuracy and macro_f1, best first.

Run from the repository root:
    PYTHONPATH=. python exploratory/tune_max_depth.py --input-csv data/image_metadata_auto_label.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from config import Config

UNLABELED = "unlabeled"
NON_FEATURE_COLS = {"image_id", "sample_code", "section", "image_type", "source_path", "label", "cluster"}

CANDIDATE_DEPTHS = [2, 3, 4, 5, 6, 8]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """List the feature columns of a feature table.

    Args:
        df: Feature table.

    Returns:
        list of str: Every column that is not metadata
        (``NON_FEATURE_COLS``).
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def run_loso_cv_for_depth(
    labeled_df: pd.DataFrame,
    feature_cols: list[str],
    label_encoder: LabelEncoder,
    base_params: dict,
    max_depth: int,
) -> tuple[float, float]:
    """Leave-one-specimen-out CV of XGBoost at one tree depth.

    Each specimen is held out in turn; a specimen is skipped if no other
    specimen of its class remains for training. No sample weights are used.

    Args:
        labeled_df: Labeled images with ``sample_code``, ``label`` and the
            feature columns.
        feature_cols: Names of the feature columns to use.
        label_encoder: ``LabelEncoder`` fitted on the class names.
        base_params: XGBoost hyperparameters; ``max_depth`` and
            ``num_class`` are overridden.
        max_depth: Tree depth to evaluate.

    Returns:
        tuple of float: ``(accuracy, macro_f1)`` over the pooled
        out-of-fold predictions.
    """
    params = dict(base_params)
    params["max_depth"] = max_depth
    params["num_class"] = len(label_encoder.classes_)

    unique_samples = labeled_df["sample_code"].unique()
    all_true, all_pred = [], []

    for held_out_sample in unique_samples:
        train_fold = labeled_df[labeled_df["sample_code"] != held_out_sample]
        test_fold = labeled_df[labeled_df["sample_code"] == held_out_sample]

        held_out_label = test_fold["label"].iloc[0]
        if held_out_label not in train_fold["label"].values:
            continue  # same skip rule as train_xgboost_cv.py

        X_train = train_fold[feature_cols]
        y_train = label_encoder.transform(train_fold["label"])
        X_test = test_fold[feature_cols]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        all_true.extend(label_encoder.transform(test_fold["label"]))
        all_pred.extend(y_pred)

    accuracy = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    return accuracy, macro_f1


def run(input_csv: str, output_csv: Path) -> None:
    """Evaluate every depth in ``CANDIDATE_DEPTHS`` and save the comparison.

    Args:
        input_csv: Labeled feature table (output of ``auto_label.py``). Rows
            labeled ``'unlabeled'`` are ignored.
        output_csv: Path of the results table (``pathlib.Path``).
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()
    feature_cols = get_feature_columns(df)

    present_classes = sorted(labeled_df["label"].unique())
    label_encoder = LabelEncoder().fit(present_classes)

    print(f"Sweeping max_depth over {CANDIDATE_DEPTHS} using {labeled_df['sample_code'].nunique()} samples...\n")

    results = []
    for depth in CANDIDATE_DEPTHS:
        print(f"Testing max_depth={depth}...")
        accuracy, macro_f1 = run_loso_cv_for_depth(
            labeled_df, feature_cols, label_encoder, Config.XGBOOST_PARAMS, depth
        )
        print(f"  accuracy={accuracy:.3f}  macro_f1={macro_f1:.3f}")
        results.append({"max_depth": depth, "accuracy": accuracy, "macro_f1": macro_f1})

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    print("\n=== Summary, sorted by macro F1 (fairer than accuracy given class imbalance) ===")
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    print(f"\nBest max_depth by macro F1: {int(best['max_depth'])} "
          f"(macro_f1={best['macro_f1']:.3f}, accuracy={best['accuracy']:.3f})")
    print("If this differs from config.py's current value, update XGBOOST_PARAMS['max_depth'].")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved comparison table -> {output_csv}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``output_csv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="results/tables/max_depth_sweep.csv")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, output_csv=Path(args.output_csv))


if __name__ == "__main__":
    main()
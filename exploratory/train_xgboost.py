"""Train an XGBoost microfacies classifier with a single specimen-level hold-out
split.

The features are the combined GLCM, DINOv2 and lamination columns, and the
labels come from auto_label.py; unlabeled images are ignored. 20% of the
specimens (stratified by class) are held out for testing. The split is made by
specimen, not by image: every thin section yields many PPL/XPL images, and
splitting by image would put images of the same specimen in both sets and
overstate the accuracy on new specimens.

Classes with a single specimen cannot be split; their specimen goes to the
training set, and their test scores are not an estimate of generalization.
The results in the paper use leave-one-specimen-out CV instead
(train_xgboost_cv.py).

Outputs:
  models_dir/xgb_microfacies.json, models_dir/label_encoder.pkl
  tables_dir/classification_report.csv, tables_dir/confusion_matrix.csv

Run from the repository root:
    PYTHONPATH=. python exploratory/train_xgboost.py --input-csv data/image_metadata_auto_label.csv
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import Config

UNLABELED = "unlabeled"
NON_FEATURE_COLS = {"image_id", "sample_code", "section", "image_type", "source_path", "label", "cluster"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """List the feature columns of a feature table.

    Args:
        df: Feature table.

    Returns:
        list of str: Every column that is not metadata
        (``NON_FEATURE_COLS``).
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def sample_level_split(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split labeled images into training and test sets by specimen.

    All images of a specimen go to the same set. Specimens are split with a
    class-stratified ``train_test_split``. Classes with a single specimen
    cannot be stratified; that specimen goes to the training set, so the
    class is learned but not tested.

    Args:
        df: Labeled images (no ``'unlabeled'`` rows) with ``sample_code``
            and ``label``.
        test_size: Share of specimens (not images) to hold out.
        random_state: Seed of the split.

    Returns:
        tuple: ``(train_df, test_df, single_sample_classes)``; the last is
        the sorted list of classes with only one specimen.
    """
    sample_labels = df.groupby("sample_code")["label"].first()
    class_sample_counts = sample_labels.value_counts()

    evaluable_classes = class_sample_counts[class_sample_counts >= 2].index
    single_sample_classes = sorted(class_sample_counts[class_sample_counts < 2].index)

    evaluable_mask = sample_labels.isin(evaluable_classes)
    evaluable_samples = sample_labels[evaluable_mask]
    singleton_samples = sample_labels[~evaluable_mask]

    if len(evaluable_samples) > 0:
        train_samples, test_samples = train_test_split(
            evaluable_samples.index,
            test_size=test_size,
            stratify=evaluable_samples.values,
            random_state=random_state,
        )
    else:
        train_samples, test_samples = [], []

    # Singleton-class samples go entirely to train - the model learns from
    # them, but they're never evaluated on held-out data.
    all_train_samples = list(train_samples) + list(singleton_samples.index)

    train_df = df[df["sample_code"].isin(all_train_samples)]
    test_df = df[df["sample_code"].isin(test_samples)]
    return train_df, test_df, single_sample_classes


def train(input_csv: str, models_dir: Path, tables_dir: Path) -> None:
    """Train on the training specimens, evaluate on the held-out ones, save.

    Uses ``Config.XGBOOST_PARAMS`` without class weights, a 20% specimen
    hold-out and seed 42. Prints the class balance, the held-out specimens
    per class, the classification report and the confusion matrix.

    Args:
        input_csv: Labeled feature table (output of ``auto_label.py``).
        models_dir: Folder (``pathlib.Path``) for the model and label
            encoder.
        tables_dir: Folder (``pathlib.Path``) for the classification report
            and confusion matrix.
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()

    present_classes = sorted(labeled_df["label"].unique())
    missing_classes = sorted(set(Config.MICROFACIES) - set(present_classes))
    print(f"Classes present in labeled data: {present_classes}")
    if missing_classes:
        print(
            f"WARNING: {missing_classes} have ZERO labeled examples and will "
            "NOT be predictable by this model until labeled examples exist."
        )

    print(f"\nTotal labeled images: {len(labeled_df)}")
    print(f"Total labeled samples (physical thin sections): {labeled_df['sample_code'].nunique()}")

    sample_labels = labeled_df.groupby("sample_code")["label"].first()
    print("\nSamples (not images) per class - the real signal of data sufficiency:")
    print(sample_labels.value_counts())

    train_df, test_df, single_sample_classes = sample_level_split(
        labeled_df, test_size=0.2, random_state=42
    )
    if single_sample_classes:
        print(
            f"\nWARNING: {single_sample_classes} have only ONE physical sample each. "
            "The model trains on them, but their test-set performance below is "
            "NOT a valid estimate of generalization - do not report these as validated."
        )
    print(f"\nTrain: {len(train_df)} images from {train_df['sample_code'].nunique()} samples")
    print(f"Test:  {len(test_df)} images from {test_df['sample_code'].nunique()} samples")

    print("\nTest samples by class (check these individually if a class scores 0%):")
    test_by_class = test_df.groupby("label")["sample_code"].unique()
    for label, codes in test_by_class.items():
        print(f"  {label}: {list(codes)}")

    feature_cols = get_feature_columns(df)
    X_train, X_test = train_df[feature_cols], test_df[feature_cols]

    label_encoder = LabelEncoder()
    label_encoder.fit(present_classes)
    y_train = label_encoder.transform(train_df["label"])
    y_test = label_encoder.transform(test_df["label"])

    params = dict(Config.XGBOOST_PARAMS)
    params["num_class"] = len(present_classes)  # actual classes seen, not config's full 6

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    all_label_ids = list(range(len(label_encoder.classes_)))
    report = classification_report(
        y_test, y_pred, labels=all_label_ids, target_names=label_encoder.classes_,
        output_dict=True, zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    cm = confusion_matrix(y_test, y_pred, labels=all_label_ids)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)

    print("\n=== Classification report (test set, sample-level split) ===")
    print(report_df.round(3))
    print("\n=== Confusion matrix ===")
    print(cm_df)

    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    model.get_booster().save_model(str(models_dir / "xgb_microfacies.json"))
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    report_df.to_csv(tables_dir / "classification_report.csv")
    cm_df.to_csv(tables_dir / "confusion_matrix.csv")

    print(f"\nSaved model -> {models_dir / 'xgb_microfacies.json'}")
    print(f"Saved label encoder -> {models_dir / 'label_encoder.pkl'}")
    print(f"Saved reports -> {tables_dir}")


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv``, ``models_dir`` and ``tables_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--tables-dir", type=str, default="results/tables")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: run ``train`` with the given options."""
    args = parse_args()
    train(
        input_csv=args.input_csv,
        models_dir=Path(args.models_dir),
        tables_dir=Path(args.tables_dir),
    )


if __name__ == "__main__":
    main()
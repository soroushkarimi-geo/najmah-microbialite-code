"""XGBoost classifier evaluated with leave-one-specimen-out (LOSO) cross-validation.

Each specimen is held out once and all its images are predicted by a model
trained on the other specimens. Out-of-fold predictions are pooled into one
report. Class weights (inverse class frequency) are computed from each fold's
training images only.

The model saved at the end is trained on all labeled images. The CV numbers,
not this model, are what the paper reports.

Outputs:
  models_dir/xgb_microfacies.json, models_dir/label_encoder.pkl
  tables_dir/cv_classification_report.csv
  tables_dir/cv_confusion_matrix.csv
  tables_dir/cv_per_sample_predictions.csv

    python train_xgboost_cv.py --input-csv data/image_metadata_reliable_classes.csv --models-dir out/models --tables-dir out/tables
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from config import Config

UNLABELED = "unlabeled"
NON_FEATURE_COLS = {"image_id", "sample_code", "section", "image_type", "source_path", "label", "cluster"}


def get_feature_columns(df):
    """List the feature columns of a feature table.

    Args:
        df: Feature table.

    Returns:
        list of str: Every column that is not metadata (``NON_FEATURE_COLS``),
        i.e. the ``glcm_``, ``dino_`` and ``lam_`` columns, in table order.
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def run_loso_cv(labeled_df, feature_cols, label_encoder, params):
    """Leave-one-specimen-out cross-validation of an XGBoost classifier.

    Each specimen is held out in turn; a model trained on the images of all
    other specimens predicts every image of the held-out specimen. Sample
    weights (balanced, inverse class frequency) are computed from the
    training images of each fold only. A specimen is skipped if no other
    specimen of its class remains for training.

    Args:
        labeled_df: Labeled images, with ``image_id``, ``sample_code``,
            ``label`` and the feature columns.
        feature_cols: Names of the feature columns to use.
        label_encoder: ``LabelEncoder`` fitted on the class names.
        params: Keyword arguments for ``xgboost.XGBClassifier``.

    Returns:
        pandas.DataFrame: One row per predicted image, with ``image_id``,
        ``sample_code``, ``label`` (true) and ``predicted_label``.
    """
    unique_samples = labeled_df["sample_code"].unique()
    all_predictions = []

    for i, held_out_sample in enumerate(unique_samples, start=1):
        train_fold = labeled_df[labeled_df["sample_code"] != held_out_sample]
        test_fold = labeled_df[labeled_df["sample_code"] == held_out_sample]

        # skip the fold if no other specimen of this class is left for training
        held_out_label = test_fold["label"].iloc[0]
        if held_out_label not in train_fold["label"].values:
            print(f"  [{i}/{len(unique_samples)}] Skipping {held_out_sample} ({held_out_label}), "
                  "no other specimen of this class")
            continue

        X_train = train_fold[feature_cols]
        y_train = label_encoder.transform(train_fold["label"])
        X_test = test_fold[feature_cols]
        sample_weight = compute_sample_weight("balanced", y_train)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weight)
        y_pred = model.predict(X_test)

        fold_result = test_fold[["image_id", "sample_code", "label"]].copy()
        fold_result["predicted_label"] = label_encoder.inverse_transform(y_pred)
        all_predictions.append(fold_result)

        print(f"  [{i}/{len(unique_samples)}] Held out {held_out_sample} ({held_out_label}): "
              f"{len(test_fold)} images evaluated")

    return pd.concat(all_predictions, ignore_index=True)


def train(input_csv, models_dir, tables_dir):
    """Run LOSO cross-validation, save the CV tables and a final model.

    Prints the class balance, runs ``run_loso_cv`` with the hyperparameters
    of ``Config.XGBOOST_PARAMS``, and reports the pooled out-of-fold
    classification report and confusion matrix. A final model is then
    trained on all labeled images and saved; it is used only to score new
    images (``score_against_reliable_model.py``), not for the CV numbers.

    Args:
        input_csv: Labeled feature table. Rows labeled ``'unlabeled'`` are
            ignored.
        models_dir: Folder (``pathlib.Path``) for ``xgb_microfacies.json``
            and ``label_encoder.pkl``.
        tables_dir: Folder (``pathlib.Path``) for
            ``cv_classification_report.csv``, ``cv_confusion_matrix.csv`` and
            ``cv_per_sample_predictions.csv``.
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()

    present_classes = sorted(labeled_df["label"].unique())
    missing_classes = sorted(set(Config.MICROFACIES) - set(present_classes))
    print(f"Classes: {present_classes}")
    if missing_classes:
        print(f"Not in this table: {missing_classes}")

    print(f"\nLabeled images: {len(labeled_df)}")
    print(f"Labeled specimens: {labeled_df['sample_code'].nunique()}")

    sample_labels = labeled_df.groupby("sample_code")["label"].first()
    print("\nSpecimens per class:")
    print(sample_labels.value_counts())

    print("\nImages per class (%):")
    print((labeled_df["label"].value_counts(normalize=True) * 100).round(1).astype(str) + "%")

    feature_cols = get_feature_columns(df)
    label_encoder = LabelEncoder()
    label_encoder.fit(present_classes)

    params = dict(Config.XGBOOST_PARAMS)
    params["num_class"] = len(present_classes)

    print(f"\nLOSO CV over {labeled_df['sample_code'].nunique()} specimens...")
    predictions_df = run_loso_cv(labeled_df, feature_cols, label_encoder, params)

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

    print(f"\n=== Pooled CV results ({len(predictions_df)} images, "
          f"{predictions_df['sample_code'].nunique()} specimens) ===")
    print(report_df.round(3))
    print("\n=== Confusion matrix ===")
    print(cm_df)

    # -----------------------------
    # Final model on all labeled images (not used for the CV numbers above)
    X_all = labeled_df[feature_cols]
    y_all = label_encoder.transform(labeled_df["label"])
    final_sample_weight = compute_sample_weight("balanced", y_all)
    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X_all, y_all, sample_weight=final_sample_weight)

    models_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    final_model.get_booster().save_model(str(models_dir / "xgb_microfacies.json"))
    joblib.dump(label_encoder, models_dir / "label_encoder.pkl")
    report_df.to_csv(tables_dir / "cv_classification_report.csv")
    cm_df.to_csv(tables_dir / "cv_confusion_matrix.csv")
    predictions_df.to_csv(tables_dir / "cv_per_sample_predictions.csv", index=False)

    print(f"\nModel saved to {models_dir / 'xgb_microfacies.json'}")
    print(f"CV tables saved to {tables_dir}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv``, ``models_dir`` and
        ``tables_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--tables-dir", type=str, default="results/tables")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``train`` with the given options."""
    args = parse_args()
    train(
        input_csv=args.input_csv,
        models_dir=Path(args.models_dir),
        tables_dir=Path(args.tables_dir),
    )


if __name__ == "__main__":
    main()

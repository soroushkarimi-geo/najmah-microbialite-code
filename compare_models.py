"""Compare four classifiers under the same leave-one-specimen-out protocol:
XGBoost, L1 logistic regression (PCA-30), random forest and kNN (PCA-30).

Logistic regression and kNN get PCA-reduced, standardized features; the
tree models use the raw features. Image-level reports and confusion matrices
for Table 2 / Fig. 5 are written per model.

    python compare_models.py --input-csv data/image_metadata_reliable_classes.csv --tables-dir out/tables
"""

import argparse
from pathlib import Path

import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import Config
from train_xgboost_cv import get_feature_columns

UNLABELED = "unlabeled"
PCA_COMPONENTS = 30


def build_models():
    """Build the four classifiers compared in Table 2, unfitted.

    - XGBoost on the raw features, with ``Config.XGBOOST_PARAMS``.
    - L1-penalized logistic regression (saga solver) after standardization
      and PCA to 30 components.
    - Random forest on the raw features (300 trees, max depth 8).
    - k-nearest neighbours (k = 5) after standardization and PCA to 30
      components.

    All random seeds are fixed at 42.

    Returns:
        dict: Model name -> unfitted scikit-learn compatible estimator. The
        names are used as table labels; the part before " (" names the
        output files.
    """
    xgb_params = dict(Config.XGBOOST_PARAMS)
    return {
        "XGBoost (raw features, current default)": xgb.XGBClassifier(**xgb_params),
        "Logistic Regression (L1, PCA-30)": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=PCA_COMPONENTS, random_state=42)),
            ("clf", LogisticRegression(penalty="l1", solver="saga", max_iter=5000, random_state=42)),
        ]),
        "Random Forest (raw features)": RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=42
        ),
        "kNN (PCA-30)": Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=PCA_COMPONENTS, random_state=42)),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
    }


def run_loso_cv_generic(labeled_df, feature_cols, label_encoder, model_factory):
    """Leave-one-specimen-out cross-validation for any scikit-learn model.

    Same folds as ``train_xgboost_cv.run_loso_cv``: each specimen is held
    out in turn, and a specimen is skipped if no other specimen of its class
    remains for training. Balanced sample weights are computed from each
    fold's training images and passed to the model (to the ``clf`` step for
    a ``Pipeline``); models whose ``fit`` takes no sample weights (kNN) are
    fitted without them.

    Args:
        labeled_df: Labeled images, with ``image_id``, ``sample_code``,
            ``label`` and the feature columns.
        feature_cols: Names of the feature columns to use.
        label_encoder: ``LabelEncoder`` fitted on the class names.
        model_factory: Callable with no arguments that returns a new,
            unfitted model for each fold.

    Returns:
        pandas.DataFrame: One row per predicted image, with ``image_id``,
        ``sample_code``, ``label`` (true) and ``predicted_label``.
    """
    unique_samples = labeled_df["sample_code"].unique()
    all_predictions = []

    for held_out_sample in unique_samples:
        train_fold = labeled_df[labeled_df["sample_code"] != held_out_sample]
        test_fold = labeled_df[labeled_df["sample_code"] == held_out_sample]

        held_out_label = test_fold["label"].iloc[0]
        if held_out_label not in train_fold["label"].values:
            continue

        X_train = train_fold[feature_cols]
        y_train = label_encoder.transform(train_fold["label"])
        X_test = test_fold[feature_cols]

        model = model_factory()
        try:
            sample_weight = compute_sample_weight("balanced", y_train)
            model.fit(X_train, y_train, clf__sample_weight=sample_weight) \
                if hasattr(model, "named_steps") else model.fit(X_train, y_train, sample_weight=sample_weight)
        except TypeError:
            # kNN does not take sample_weight
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fold_result = test_fold[["image_id", "sample_code", "label"]].copy()
        fold_result["predicted_label"] = label_encoder.inverse_transform(y_pred)
        all_predictions.append(fold_result)

    return pd.concat(all_predictions, ignore_index=True)


def run(input_csv, tables_dir):
    """Cross-validate the four models of ``build_models`` and save the results.

    For each model, the pooled out-of-fold classification report and
    confusion matrix are printed and written as
    ``compare_<model>_report.csv`` and ``compare_<model>_confusion.csv``
    (read by ``plot_model_confusion_matrices.py`` for Fig. 5).

    Args:
        input_csv: Labeled feature table. Rows labeled ``'unlabeled'`` are
            ignored.
        tables_dir: Output folder (``pathlib.Path``); created if needed.
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()
    feature_cols = get_feature_columns(df)

    present_classes = sorted(labeled_df["label"].unique())
    label_encoder = LabelEncoder().fit(present_classes)

    models = build_models()
    tables_dir.mkdir(parents=True, exist_ok=True)

    for name, model_template in models.items():
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        safe_name = name.split(" (")[0].replace(" ", "_")

        def factory(template=model_template):
            """Return a fresh, unfitted copy of the current model."""
            from sklearn.base import clone
            return clone(template)

        predictions_df = run_loso_cv_generic(labeled_df, feature_cols, label_encoder, factory)

        y_true = label_encoder.transform(predictions_df["label"])
        y_pred = label_encoder.transform(predictions_df["predicted_label"])
        all_label_ids = list(range(len(label_encoder.classes_)))

        report = classification_report(
            y_true, y_pred, labels=all_label_ids, target_names=label_encoder.classes_,
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose()
        print(report_df.round(3))
        report_df.to_csv(tables_dir / f"compare_{safe_name}_report.csv")

        cm = confusion_matrix(y_true, y_pred, labels=all_label_ids)
        cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
        cm_df.to_csv(tables_dir / f"compare_{safe_name}_confusion.csv")

        # only when MF-A is in the table
        if "MF-A" in label_encoder.classes_:
            print("\nMF-A predicted as:")
            print(cm_df.loc["MF-A"])

    print(f"\nReports and confusion matrices saved to {tables_dir}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``tables_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--tables-dir", type=str, default="results/tables")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, tables_dir=Path(args.tables_dir))


if __name__ == "__main__":
    main()

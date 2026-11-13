"""Blind check of the classifier on specimens that played no part in building it.

Leave-one-specimen-out cross-validation measures how well the model
generalizes to specimens from the labeled set, which also shaped the labels
themselves. Unlabeled specimens had no role in training or labeling, so
comparing the model with an independent petrographic call on them is a
separate test, analogous to blind-well validation of lithofacies classifiers.

The check runs in two steps, and their order matters: the petrographer's calls
must be made before the model's predictions are seen, otherwise they are no
longer independent.

  Step 1 draws a random set of unlabeled specimens and writes a blank template
         (sample_code, your_independent_call, confidence_1_to_5, notes). No
         predictions are computed or shown.
  Step 2 checks that every call is filled in, predicts each specimen with the
         saved model (majority vote of its images) and reports the agreement.

Nothing is relabeled and nothing is fed back into the model; using the blind
specimens for training would remove what makes the check independent.

Files: results/tables/blind_validation_sample.csv (template, step 1) and
results/tables/blind_validation_revealed.csv (comparison, step 2).

Run from the repository root:
    PYTHONPATH=. python exploratory/blind_validation_check.py --step 1 --input-csv data/image_metadata_auto_label.csv --n-sample 10
    (examine the thin sections of the listed specimens and fill in the template)
    PYTHONPATH=. python exploratory/blind_validation_check.py --step 2 --input-csv data/image_metadata_auto_label.csv --models-dir out/models
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from config import Config
from train_xgboost_cv import get_feature_columns

UNLABELED = "unlabeled"
BLIND_SAMPLE_PATH = Path("results/tables/blind_validation_sample.csv")
REVEALED_PATH = Path("results/tables/blind_validation_revealed.csv")


def step1_generate_sample(input_csv: str, n_sample: int, seed: int) -> None:
    """Step 1: draw unlabeled specimens and write the blank template.

    No model predictions are made in this step, so the calls entered in the
    template are independent of the model.

    Args:
        input_csv: Labeled feature table (output of ``auto_label.py``).
        n_sample: Number of unlabeled specimens to draw.
        seed: Seed of the random draw.

    Raises:
        ValueError: If there are fewer than ``n_sample`` unlabeled specimens.
    """
    df = pd.read_csv(input_csv)
    unlabeled_specimens = (
        df[df["label"] == UNLABELED]
        .drop_duplicates("sample_code")["sample_code"]
        .tolist()
    )
    if len(unlabeled_specimens) < n_sample:
        raise ValueError(
            f"Only {len(unlabeled_specimens)} unlabeled specimens available, "
            f"cannot sample {n_sample}."
        )

    rng = np.random.default_rng(seed)
    sampled = sorted(rng.choice(unlabeled_specimens, size=n_sample, replace=False).tolist())

    template = pd.DataFrame({
        "sample_code": sampled,
        "your_independent_call": ["" for _ in sampled],  # fill this in BEFORE step 2
        "confidence_1_to_5": ["" for _ in sampled],
        "notes": ["" for _ in sampled],
    })
    BLIND_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(BLIND_SAMPLE_PATH, index=False)

    print(f"Sampled {n_sample} unlabeled specimens -> {BLIND_SAMPLE_PATH}")
    print("\nSpecimens selected for blind check:")
    for s in sampled:
        print(f"  {s}")
    print(
        "\nNext: open each specimen's thin-section images directly, make your own "
        "independent facies call for each one, and fill in 'your_independent_call' "
        f"in {BLIND_SAMPLE_PATH} - do NOT run Step 2 until this is done. "
        "Do not look at the model's output for these specimens in the meantime."
    )


def step2_reveal_and_compare(input_csv: str, models_dir: Path) -> None:
    """Step 2: predict the drawn specimens and compare with the blind calls.

    Each specimen's prediction is the most frequent predicted class of its
    images. The result table has the call, the model's prediction, whether
    they agree and the stated confidence.

    Args:
        input_csv: Labeled feature table (output of ``auto_label.py``).
        models_dir: Folder (``pathlib.Path``) with ``xgb_microfacies.json``
            and ``label_encoder.pkl`` from ``train_xgboost_cv.py``.

    Raises:
        FileNotFoundError: If step 1 has not been run.
        ValueError: If any specimen has no independent call yet.
    """
    if not BLIND_SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Run Step 1 first - {BLIND_SAMPLE_PATH} not found.")

    blind_df = pd.read_csv(BLIND_SAMPLE_PATH)
    if blind_df["your_independent_call"].isna().any() or (blind_df["your_independent_call"] == "").any():
        raise ValueError(
            "Some rows in your_independent_call are still empty. "
            "Fill in an independent call for every specimen before revealing model predictions."
        )

    model = xgb.XGBClassifier()
    model.load_model(str(models_dir / "xgb_microfacies.json"))
    label_encoder = joblib.load(models_dir / "label_encoder.pkl")

    df = pd.read_csv(input_csv)
    feature_cols = get_feature_columns(df)

    results = []
    for _, row in blind_df.iterrows():
        specimen_images = df[df["sample_code"] == row["sample_code"]]
        X = specimen_images[feature_cols]
        preds = model.predict(X)
        pred_labels = label_encoder.inverse_transform(preds)
        # specimen prediction = majority vote of its images
        model_call = pd.Series(pred_labels).mode().iloc[0]
        agree = (model_call == row["your_independent_call"])
        results.append({
            "sample_code": row["sample_code"],
            "your_independent_call": row["your_independent_call"],
            "model_prediction": model_call,
            "agree": agree,
            "your_confidence": row["confidence_1_to_5"],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(REVEALED_PATH, index=False)

    n_agree = results_df["agree"].sum()
    n_total = len(results_df)
    print(f"\n=== Blind validation result: {n_agree} of {n_total} agree ({100*n_agree/n_total:.0f}%) ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved -> {REVEALED_PATH}")
    print(
        "\nInterpretation: this is a small sample (by design - it draws from a limited "
        "unlabeled pool) so treat the percentage as informative, not a precise estimate. "
        "Disagreements are worth a closer look each, individually - they may reveal a "
        "genuine model weakness, a genuinely ambiguous specimen, or an error in the "
        "independent call itself. Do NOT use this exercise to relabel these specimens "
        "or retrain the model - its value is as an honest, disclosed check, not as new "
        "training data (using it as training data would break the 'never influenced "
        "the model' property that makes this check meaningful in the first place)."
    )


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``step``, ``input_csv``, ``n_sample``, ``seed`` and ``models_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, choices=[1, 2], required=True)
    parser.add_argument("--input-csv", type=str, default="results/features/image_metadata_auto_label.csv")
    parser.add_argument("--n-sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=str, default="models/reliable_classes")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point: run step 1 or step 2."""
    args = parse_args()
    if args.step == 1:
        step1_generate_sample(args.input_csv, args.n_sample, args.seed)
    else:
        step2_reveal_and_compare(args.input_csv, Path(args.models_dir))


if __name__ == "__main__":
    main()
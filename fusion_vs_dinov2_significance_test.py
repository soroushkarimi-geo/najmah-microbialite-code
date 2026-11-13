"""Is the fused feature set (GLCM + DINOv2 + lamination) better than DINOv2 alone?

Both sets are run with the same LOSO folds (run_loso_cv from train_xgboost_cv.py),
so their per-specimen outcomes are paired. Each specimen gets the majority vote of
its images. Tests:
  - exact McNemar test (binomial on the discordant specimens)
  - paired sign-flip permutation test, 10000 permutations, seed 42

    python fusion_vs_dinov2_significance_test.py --input-csv data/image_metadata_reliable_classes.csv --output-csv out/tables/fusion_vs_dinov2_significance_test.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import LabelEncoder

from config import Config
from train_xgboost_cv import run_loso_cv

UNLABELED = "unlabeled"
N_PERMUTATIONS = 10000
RANDOM_SEED = 42

FEATURE_SETS = {
    "DINOv2 only": lambda cols: [c for c in cols if c.startswith("dino_")],
    "Fused, all three": lambda cols: [c for c in cols if c.startswith(("glcm_", "dino_", "lam_"))],
}


def specimen_level_correctness(predictions_df):
    """Whether each specimen is classified correctly by majority vote.

    The specimen prediction is the most frequent predicted class of its
    images. With an even split, ``pandas.Series.mode`` returns the classes
    in sorted order and the first one is used.

    Args:
        predictions_df: Image-level predictions with ``sample_code``,
            ``label`` and ``predicted_label``.

    Returns:
        pandas.Series: Boolean, indexed by ``sample_code``.
    """
    def majority_correct(group):
        """True if the most frequent prediction of the group is its label."""
        true_label = group["label"].iloc[0]
        majority_pred = group["predicted_label"].mode().iloc[0]
        return majority_pred == true_label

    return predictions_df.groupby("sample_code").apply(majority_correct, include_groups=False)


def paired_permutation_test(correct_a, correct_b, n_permutations, rng):
    """Paired sign-flip permutation test for a difference in accuracy.

    Under the null hypothesis the two methods are exchangeable for every
    specimen. Each permutation swaps the pair of outcomes of each specimen
    with probability 0.5 and recomputes the difference in accuracy. The
    two-sided p-value is (number of permutations with |difference| >=
    |observed| + 1) / (n_permutations + 1).

    Args:
        correct_a: Boolean array, specimen correct with method A.
        correct_b: Boolean array, specimen correct with method B, same
            specimen order as ``correct_a``.
        n_permutations: Number of random sign flips.
        rng: ``numpy.random.Generator``.

    Returns:
        dict: ``observed_accuracy_diff`` (A - B), ``null_mean``,
        ``null_std`` and ``p_value_two_sided``.
    """
    a = correct_a.astype(float)
    b = correct_b.astype(float)
    observed_diff = a.mean() - b.mean()

    n = len(a)
    null_diffs = np.empty(n_permutations)
    for i in range(n_permutations):
        flip = rng.random(n) < 0.5
        pa = np.where(flip, b, a)
        pb = np.where(flip, a, b)
        null_diffs[i] = pa.mean() - pb.mean()

    p_ge = (float((np.abs(null_diffs) >= abs(observed_diff)).sum()) + 1) / (n_permutations + 1)

    return {
        "observed_accuracy_diff": float(observed_diff),
        "null_mean": float(null_diffs.mean()),
        "null_std": float(null_diffs.std()),
        "p_value_two_sided": p_ge,
    }


def run(input_csv, output_csv):
    """Compare fused features with DINOv2 alone at the specimen level.

    Runs LOSO XGBoost on both feature sets with identical folds, takes the
    majority vote per specimen, and tests the paired outcomes with an exact
    McNemar test (two-sided binomial test, p = 0.5, on the discordant
    specimens) and with ``paired_permutation_test`` (10000 permutations,
    seed 42). The McNemar p-value is NaN when there are no discordant
    specimens.

    Args:
        input_csv: Labeled feature table. Rows labeled ``'unlabeled'`` are
            ignored.
        output_csv: Path of the one-row results table (accuracies, 2 x 2
            counts and both p-values).
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()

    present_classes = sorted(labeled_df["label"].unique())
    label_encoder = LabelEncoder().fit(present_classes)

    params = dict(Config.XGBOOST_PARAMS)
    params["num_class"] = len(present_classes)

    specimen_correct = {}
    accuracy_by_set = {}
    for set_name, selector in FEATURE_SETS.items():
        feature_cols = selector(list(df.columns))
        print(f"\n=== {set_name}: {len(feature_cols)} features ===")
        predictions_df = run_loso_cv(labeled_df, feature_cols, label_encoder, params)
        img_acc = (predictions_df["label"] == predictions_df["predicted_label"]).mean()
        print(f"  image-level accuracy: {img_acc:.4f}")
        specimen_correct[set_name] = specimen_level_correctness(predictions_df)

    common_specimens = specimen_correct["DINOv2 only"].index.intersection(
        specimen_correct["Fused, all three"].index
    )
    correct_dino = specimen_correct["DINOv2 only"].loc[common_specimens].to_numpy(dtype=bool)
    correct_fused = specimen_correct["Fused, all three"].loc[common_specimens].to_numpy(dtype=bool)
    n_specimens = len(common_specimens)

    print(f"\n=== Specimen level, n={n_specimens} ===")
    print(f"  DINOv2 only: {correct_dino.mean():.4f} ({correct_dino.sum()}/{n_specimens} correct)")
    print(f"  Fused, all three: {correct_fused.mean():.4f} ({correct_fused.sum()}/{n_specimens} correct)")

    # 2x2 table
    both_correct = int((correct_dino & correct_fused).sum())
    both_wrong = int((~correct_dino & ~correct_fused).sum())
    dino_only_correct = int((correct_dino & ~correct_fused).sum())
    fused_only_correct = int((~correct_dino & correct_fused).sum())

    print("\n  2x2 table:")
    print(f"    both correct: {both_correct}")
    print(f"    both wrong: {both_wrong}")
    print(f"    DINOv2 correct, fused wrong: {dino_only_correct}")
    print(f"    fused correct, DINOv2 wrong: {fused_only_correct}")

    n_discordant = dino_only_correct + fused_only_correct
    if n_discordant == 0:
        print("\n  No discordant specimens, McNemar test is undefined.")
        mcnemar_p = float("nan")
    else:
        # exact McNemar = two-sided binomial test on the discordant pairs, p = 0.5
        k = min(dino_only_correct, fused_only_correct)
        mcnemar_result = binomtest(k, n_discordant, p=0.5, alternative="two-sided")
        mcnemar_p = mcnemar_result.pvalue
        print(f"\n  McNemar exact test ({n_discordant} discordant): p = {mcnemar_p:.4f}")

    rng = np.random.default_rng(RANDOM_SEED)
    perm_result = paired_permutation_test(correct_fused, correct_dino, N_PERMUTATIONS, rng)
    print(f"\n  Permutation test ({N_PERMUTATIONS} sign flips, seed={RANDOM_SEED}):")
    print(f"    observed difference (fused - DINOv2): {perm_result['observed_accuracy_diff']:+.4f}")
    print(f"    null mean: {perm_result['null_mean']:+.4f} (sd={perm_result['null_std']:.4f})")
    print(f"    p (two-sided): {perm_result['p_value_two_sided']:.4f}")

    results_df = pd.DataFrame([{
        "n_specimens": n_specimens,
        "dino_only_accuracy": correct_dino.mean(),
        "fused_accuracy": correct_fused.mean(),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "dino_correct_fused_wrong": dino_only_correct,
        "fused_correct_dino_wrong": fused_only_correct,
        "mcnemar_exact_p": mcnemar_p,
        "permutation_p": perm_result["p_value_two_sided"],
        "permutation_observed_diff": perm_result["observed_accuracy_diff"],
    }])
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``output_csv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str,
                         default="results/tables/fusion_vs_dinov2_significance_test.csv")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, output_csv=args.output_csv)


if __name__ == "__main__":
    main()

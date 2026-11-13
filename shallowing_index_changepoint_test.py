"""Mean-shift test of the lamination index S at the two candidate sequence
boundaries SB5 and SB6, Dudan (= Dawan) section only (Table 5, last two rows).

Boundary positions come from the measured section, not from S:
  SB4 ~ SD-73   (83.75 m, between last Sargelu SD-68 and first Najmah SD-77)
  SB5 ~ SD-92   (113.75 m = SB4 + TR4 ~30 m)
  SB6 ~ SD-122  (163.75 m = SB5 + TR5 50 m)
  SB7 ~ SD-136  (188.75 m = SB6 + TR6 ~25 m, extrapolated past the table)
Thicknesses are converted to sample numbers with the 39-point
thickness-sample calibration table of the Dudan section (data underlying Bayet-Goll et al. 2022).

SB5 test = TR4 vs TR5 mean S, SB6 test = TR5 vs TR6 mean S.
Two-sided permutation test, 10000 permutations, seed 42. Only these two
positions are tested (no search for a best changepoint).

    python shallowing_index_changepoint_test.py --input-csv data/image_metadata_auto_label.csv --output-dir out/tables/cyclicity
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shallowing_index_cyclicity_analysis import (
    LAM_FEATURES,
    build_section_df,
    build_shallowing_index,
    verify_feature_direction,
)

UNLABELED = "unlabeled"
N_PERMUTATIONS = 10000
RANDOM_SEED = 42

# SD sample number ranges, [start, end)
SEGMENT_BOUNDS = {
    "TR4": (73, 92),
    "TR5": (92, 122),
    "TR6": (122, 137),  # includes SD-136
}


def assign_segment(sample_number):
    """Name the T-R sequence that contains a Dudan sample number.

    Args:
        sample_number: SD sample number.

    Returns:
        str or None: ``'TR4'``, ``'TR5'`` or ``'TR6'`` from
        ``SEGMENT_BOUNDS`` (lower bound included, upper bound excluded), or
        None outside these ranges.
    """
    for segment, (lo, hi) in SEGMENT_BOUNDS.items():
        if lo <= sample_number < hi:
            return segment
    return None


def mean_shift_permutation_test(group_a, group_b, n_permutations, rng):
    """Two-sided permutation test for a difference in means.

    The two groups are pooled and randomly re-split with the original group
    sizes. The p-value is (number of permutations with |difference| >=
    |observed| + 1) / (n_permutations + 1).

    Args:
        group_a: Values below the boundary.
        group_b: Values above the boundary.
        n_permutations: Number of random re-splits.
        rng: ``numpy.random.Generator``.

    Returns:
        dict: ``n_a``, ``n_b``, ``mean_a``, ``mean_b``, ``observed_diff``
        (mean_b - mean_a), ``null_diff_mean``, ``null_diff_std`` and
        ``p_value_two_sided``.
    """
    observed_diff = float(group_b.mean() - group_a.mean())
    pooled = np.concatenate([group_a, group_b])
    n_a = len(group_a)

    null_diffs = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        shuffled = rng.permutation(pooled)
        null_diffs[i] = shuffled[n_a:].mean() - shuffled[:n_a].mean()

    p_value = (float((np.abs(null_diffs) >= abs(observed_diff)).sum()) + 1) / (n_permutations + 1)

    return {
        "n_a": n_a,
        "n_b": len(group_b),
        "mean_a": float(group_a.mean()),
        "mean_b": float(group_b.mean()),
        "observed_diff": observed_diff,
        "null_diff_mean": float(null_diffs.mean()),
        "null_diff_std": float(null_diffs.std()),
        "p_value_two_sided": p_value,
    }


def run(input_csv, output_dir):
    """Test for a shift in mean S at SB5 and at SB6 in the Dudan section.

    MF-B / MF-E specimens are assigned to TR4, TR5 or TR6 by sample number,
    S is computed over these specimens, and TR4 is compared with TR5 (SB5)
    and TR5 with TR6 (SB6). A boundary is skipped if either side has fewer
    than two specimens.

    Args:
        input_csv: Labeled feature table. Rows labeled ``'unlabeled'`` are
            ignored.
        output_dir: Output folder (``pathlib.Path``); writes
            ``shallowing_index_changepoint_test_results.csv``.
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    signs = verify_feature_direction(labeled_df)
    rng = np.random.default_rng(RANDOM_SEED)

    section_df = build_section_df(labeled_df, "SD")
    be_df = section_df[section_df["label"].isin(["MF-B", "MF-E"])].copy()
    be_df["segment"] = be_df["sample_number"].apply(assign_segment)

    print(f"=== Dudan: {len(be_df)} MF-B/MF-E specimens ===")
    print(be_df[["sample_code", "label", "sample_number", "segment"]].to_string(index=False))
    print("\nSpecimens per segment:", be_df["segment"].value_counts().to_dict())

    all_results = []
    weights = {feat: 1.0 for feat in LAM_FEATURES}
    be_df["shallowing_index"] = build_shallowing_index(be_df, signs, weights)

    for boundary_name, (seg_a, seg_b) in [("SB5 (~SD-92)", ("TR4", "TR5")), ("SB6 (~SD-122)", ("TR5", "TR6"))]:
        vals_a = be_df.loc[be_df["segment"] == seg_a, "shallowing_index"].to_numpy()
        vals_b = be_df.loc[be_df["segment"] == seg_b, "shallowing_index"].to_numpy()
        if len(vals_a) < 2 or len(vals_b) < 2:
            print(f"  {boundary_name} ({seg_a} vs {seg_b}): too few specimens, skipped")
            continue
        result = mean_shift_permutation_test(vals_a, vals_b, N_PERMUTATIONS, rng)
        result.update({"boundary": boundary_name, "segment_a": seg_a, "segment_b": seg_b})
        all_results.append(result)
        print(f"  {boundary_name} ({seg_a}, n={result['n_a']} vs {seg_b}, n={result['n_b']}): "
              f"mean {seg_a}={result['mean_a']:.4f}, mean {seg_b}={result['mean_b']:.4f}, "
              f"diff={result['observed_diff']:.4f}, "
              f"null mean={result['null_diff_mean']:.4f} (sd={result['null_diff_std']:.4f}), "
              f"p={result['p_value_two_sided']:.4f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "shallowing_index_changepoint_test_results.csv", index=False)
    print(f"\nSaved to {output_dir / 'shallowing_index_changepoint_test_results.csv'}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``output_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/tables/cyclicity")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()

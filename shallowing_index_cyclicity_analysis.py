"""Turning-point permutation test for stratigraphic order (Table 5, turning point rows).

Two resolutions, run separately for Dudan (SD) and Kezi (SK):

1. Coarse: ordinal rank of the microbialite types, MF-D = 1, MF-B = MF-E = 2
   (stromatolite F6b and thrombolite F6c sit in the same facies association F6
   of Bayet-Goll et al. 2022). No specimen is dropped from the ordered sequence;
   classes without a rank (MF-C, MF-F, MF-G) stay in the printed sequence and
   are only left out of the numeric test.

2. Fine: lamination index S of the MF-B / MF-E specimens. S is the specimen
   mean of lam_gabor_anisotropy, sign-corrected so MF-B is high, then z-scored.
   This is the only lamination feature that passed the SHAP + F-value check in
   the fold-wise analysis (foldwise_interpretability_check.py, 42/49 folds).

Specimens are ordered by sample number (field numbering follows the measured
section). Two-sided test, 10000 permutations, seed 42.

    python shallowing_index_cyclicity_analysis.py --input-csv data/image_metadata_auto_label.csv --output-dir out/tables/cyclicity
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

UNLABELED = "unlabeled"
N_PERMUTATIONS = 10000
RANDOM_SEED = 42

LAM_FEATURES = ["lam_gabor_anisotropy"]

# Coarse rank, after Bayet-Goll et al. (2022) Table 2
COARSE_RANK = {"MF-D": 1, "MF-B": 2, "MF-E": 2}


def parse_sample_number(sample_code):
    """Extract the field sample number from a sample code.

    Args:
        sample_code: Code such as ``'SD_105'`` or ``'SD_102B'``.

    Returns:
        float: The first run of digits (``105.0``, ``102.0``), or NaN if
        the code has no digits. A letter suffix is ignored here; see
        ``build_section_df`` for how SD_102 and SD_102B are ordered.
    """
    m = re.search(r"(\d+)", sample_code)
    return float(m.group(1)) if m else np.nan


def build_section_df(df, section_prefix):
    """One row per specimen of one section, in stratigraphic order.

    Image-level features are averaged per specimen. Specimens are ordered
    by sample number (field numbering follows the measured section, from
    base to top) and then by sample code, with a stable sort, so that
    SD_102 always comes before SD_102B (which was taken above it).

    Args:
        df: Image-level table with ``sample_code``, ``label`` and the
            ``LAM_FEATURES`` columns.
        section_prefix: ``'SD'`` (Dudan) or ``'SK'`` (Kezi).

    Returns:
        pandas.DataFrame: ``sample_code``, ``label``, the specimen-mean
        lamination features, ``section_prefix`` and ``sample_number``.
        Specimens without a sample number are dropped.
    """
    per_sample = (
        df.groupby("sample_code")
        .agg(label=("label", "first"), **{feat: (feat, "mean") for feat in LAM_FEATURES})
        .reset_index()
    )
    per_sample["section_prefix"] = per_sample["sample_code"].str.split("_").str[0].str.split("-").str[0]
    section_df = per_sample[per_sample["section_prefix"] == section_prefix].copy()
    section_df["sample_number"] = section_df["sample_code"].apply(parse_sample_number)
    # SD_102 and SD_102B have the same number; sort by code too (stable sort)
    # so SD_102 always comes first, the B sample was taken above it
    return (section_df.dropna(subset=["sample_number"])
            .sort_values(["sample_number", "sample_code"], kind="mergesort").reset_index(drop=True))


def verify_feature_direction(df):
    """Find the sign that makes each lamination feature higher in MF-B.

    The class means of MF-B and MF-E are printed for each feature.

    Args:
        df: Labeled table with ``label`` and the ``LAM_FEATURES`` columns.

    Returns:
        dict: Feature name -> +1.0 if its MF-B mean is above its MF-E mean,
        otherwise -1.0.
    """
    signs = {}
    print("=== Feature direction ===")
    for feat in LAM_FEATURES:
        b_mean = df.loc[df["label"] == "MF-B", feat].mean()
        e_mean = df.loc[df["label"] == "MF-E", feat].mean()
        sign = 1.0 if b_mean > e_mean else -1.0
        signs[feat] = sign
        flag = "" if sign == 1.0 else "  (reversed, sign flipped)"
        print(f"  {feat}: mean(MF-B)={b_mean:.4f}, mean(MF-E)={e_mean:.4f}{flag}")
    return signs


def build_shallowing_index(specimen_df, signs, weights):
    """Compute the lamination index S of each specimen.

    Each feature is multiplied by its sign (so that higher means more
    laminated, i.e. more MF-B like), z-scored over the given specimens
    (population standard deviation), and the weighted z-scores are summed.
    With the single feature ``lam_gabor_anisotropy``, S is its sign-
    corrected z-score.

    Args:
        specimen_df: Specimen-level table with the ``LAM_FEATURES`` columns.
        signs: Output of ``verify_feature_direction``.
        weights: Feature name -> weight.

    Returns:
        pandas.Series: S for each row of ``specimen_df``.
    """
    z_scores = {}
    for feat in LAM_FEATURES:
        raw = specimen_df[feat] * signs[feat]
        z_scores[feat] = (raw - raw.mean()) / raw.std(ddof=0)
    index = sum(weights[feat] * z_scores[feat] for feat in LAM_FEATURES)
    return index


def count_turning_points(sequence):
    """Count the local maxima and minima of an ordered sequence.

    Steps of zero (equal neighbours) are removed first, so a plateau is not
    a change of direction.

    Args:
        sequence: 1-D array of values in stratigraphic order.

    Returns:
        int: Number of sign changes between successive non-zero steps; 0 if
        there are fewer than two such steps.
    """
    if len(sequence) < 3:
        return 0
    diffs = np.diff(sequence)
    diffs = diffs[diffs != 0]
    if len(diffs) < 2:
        return 0
    signs = np.sign(diffs)
    return int((signs[:-1] != signs[1:]).sum())


def turning_point_permutation_test(values, n_permutations, rng):
    """Permutation test of stratigraphic order based on turning points.

    The observed number of turning points is compared with its distribution
    over random reorderings of the same values. The test is two-sided
    because ordered cycles can give either more turning points than random
    (short-period alternation) or fewer (long, smooth trends). One-sided
    p-values use the (count + 1) / (n_permutations + 1) estimate; the
    two-sided p-value is twice the smaller one, capped at 1.

    Args:
        values: 1-D array in stratigraphic order.
        n_permutations: Number of random reorderings.
        rng: ``numpy.random.Generator``.

    Returns:
        dict: ``observed_turning_points``, ``n_specimens``, ``null_mean``,
        ``null_std`` and ``p_value_two_sided``.
    """
    observed = count_turning_points(values)
    null_counts = np.empty(n_permutations, dtype=int)
    for i in range(n_permutations):
        shuffled = rng.permutation(values)
        null_counts[i] = count_turning_points(shuffled)

    p_ge = (float((null_counts >= observed).sum()) + 1) / (n_permutations + 1)
    p_le = (float((null_counts <= observed).sum()) + 1) / (n_permutations + 1)
    p_two_sided = min(1.0, 2 * min(p_ge, p_le))

    return {
        "observed_turning_points": observed,
        "n_specimens": len(values),
        "null_mean": float(null_counts.mean()),
        "null_std": float(null_counts.std()),
        "p_value_two_sided": p_two_sided,
    }


def run_coarse_resolution(section_df, section_name, rng):
    """Turning-point test on the coarse ordinal rank of the microbialite types.

    Ranks come from ``COARSE_RANK`` (MF-D = 1, MF-B = MF-E = 2). The whole
    ordered sequence is printed; specimens without a rank are left out of
    the test only.

    Args:
        section_df: Output of ``build_section_df`` for one section.
        section_name: Label used in the output.
        rng: ``numpy.random.Generator``.

    Returns:
        dict or None: Result of ``turning_point_permutation_test`` plus
        ``section`` and ``resolution``; None if fewer than 3 ranked
        specimens or only one distinct rank.
    """
    print(f"\n--- Coarse ordinal rank, {section_name} ---")
    ranked = section_df.copy()
    ranked["coarse_rank"] = ranked["label"].map(COARSE_RANK)
    print("Ordered sequence (NaN rank = not in the test):")
    print(ranked[["sample_code", "label", "coarse_rank"]].to_string(index=False))

    rank_only = ranked.dropna(subset=["coarse_rank"])
    if len(rank_only) < 3 or rank_only["coarse_rank"].nunique() < 2:
        print("  Not enough ranked specimens with different ranks, skipped")
        return None

    result = turning_point_permutation_test(rank_only["coarse_rank"].to_numpy(dtype=float), N_PERMUTATIONS, rng)
    result["section"] = section_name
    result["resolution"] = "coarse_ordinal_rank"
    print(f"  Turning points: observed={result['observed_turning_points']}, "
          f"random-shuffle mean={result['null_mean']:.2f} (sd={result['null_std']:.2f}), "
          f"p(two-sided)={result['p_value_two_sided']:.4f}, n={result['n_specimens']}")
    return result


def run_fine_resolution(section_df, section_name, signs, rng):
    """Turning-point test on the lamination index S of MF-B / MF-E specimens.

    S is computed with ``build_shallowing_index`` over the MF-B and MF-E
    specimens of the section, in stratigraphic order.

    Args:
        section_df: Output of ``build_section_df`` for one section.
        section_name: Label used in the output.
        signs: Output of ``verify_feature_direction``.
        rng: ``numpy.random.Generator``.

    Returns:
        list of dict: One result (as in ``run_coarse_resolution``), or an
        empty list if the section has fewer than 3 MF-B / MF-E specimens.
    """
    print(f"\n--- Fine, lamination index S, {section_name} ---")
    be_df = section_df[section_df["label"].isin(["MF-B", "MF-E"])].copy()
    if len(be_df) < 3:
        print("  Fewer than 3 MF-B/MF-E specimens, skipped")
        return []

    # one feature, so the weight does not change the ranking
    weights = {feat: 1.0 for feat in LAM_FEATURES}
    be_df["shallowing_index"] = build_shallowing_index(be_df, signs, weights)
    print("\n  Ordered sequence:")
    print(be_df[["sample_code", "label", "shallowing_index"]].round(4).to_string(index=False))
    result = turning_point_permutation_test(be_df["shallowing_index"].to_numpy(), N_PERMUTATIONS, rng)
    result["section"] = section_name
    result["resolution"] = "fine"
    print(f"  Turning points: observed={result['observed_turning_points']}, "
          f"random-shuffle mean={result['null_mean']:.2f} (sd={result['null_std']:.2f}), "
          f"p(two-sided)={result['p_value_two_sided']:.4f}, n={result['n_specimens']}")
    return [result]


def run(input_csv, output_dir):
    """Run the coarse and fine turning-point tests for Dudan and Kezi.

    One random generator (seed 42) is shared by all tests, in a fixed order,
    so the p-values repeat exactly.

    Args:
        input_csv: Labeled feature table. Rows labeled ``'unlabeled'`` are
            ignored.
        output_dir: Output folder (``pathlib.Path``); writes
            ``shallowing_index_permutation_test_results.csv``.
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    signs = verify_feature_direction(labeled_df)

    rng = np.random.default_rng(RANDOM_SEED)
    all_results = []
    for section_prefix, section_name in [
        ("SD", "Dudan (= Dawan; TR4-TR6 all present)"),
        ("SK", "Kezi (TR4 only, per Bayet-Goll et al. 2022)"),
    ]:
        section_df = build_section_df(labeled_df, section_prefix)
        print(f"\n=== {section_name}: {len(section_df)} specimens ===")

        coarse_result = run_coarse_resolution(section_df, section_name, rng)
        if coarse_result:
            all_results.append(coarse_result)

        fine_results = run_fine_resolution(section_df, section_name, signs, rng)
        all_results.extend(fine_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / "shallowing_index_permutation_test_results.csv", index=False)
    print(f"\n{results_df.to_string(index=False)}")
    print(f"\nSaved to {output_dir / 'shallowing_index_permutation_test_results.csv'}")


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

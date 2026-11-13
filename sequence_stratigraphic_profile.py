"""Dudan (SD) section profile on a real thickness axis: facies of every specimen
and the lamination index S of the MF-B / MF-E specimens, with SB4-SB7.

Thickness comes from the 39-point thickness-sample calibration table (same as
shallowing_index_changepoint_test.py). S is built with build_shallowing_index()
from shallowing_index_cyclicity_analysis.py, so the values are the same as in the paper.

Within each T-R sequence (TR4, TR5, TR6) the Pearson correlation between S and
thickness is tested with a two-sided permutation test (10000, seed 42).
Note: the lowest classified specimen is SD-87, so there is no data between
SB4 (SD-73) and SD-87.

    python sequence_stratigraphic_profile.py --input-csv data/image_metadata_auto_label.csv --output-dir out/tables/cyclicity --figures-dir out/figures
"""

import argparse
import bisect
from pathlib import Path

import matplotlib.pyplot as plt
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

# (thickness in m, SD sample number), Dudan section
CALIB = [
    (12, 4), (14, 7), (17, 10), (20.5, 14), (21.5, 16), (27, 21), (28, 22),
    (32, 25), (34, 27), (36, 30), (37, 32), (42, 35), (44, 36), (50, 39),
    (53, 42), (55.9, 46), (58, 49), (59, 50), (66, 55), (72, 61), (80, 68),
    (87.5, 77), (89.5, 78), (96.5, 81), (101, 84), (105.5, 86), (112.5, 91),
    (117.5, 94), (131.5, 101), (134.5, 104), (143, 110), (144.8, 113),
    (146.8, 115), (159, 120), (165, 122), (173.5, 126), (175.5, 129),
    (176.5, 130), (184.9, 134),
]
_calib_sorted = sorted(CALIB, key=lambda t: t[1])
_samples = [s for _, s in _calib_sorted]
_thicknesses = [t for t, _ in _calib_sorted]

# Sequence boundaries as SD sample numbers (see shallowing_index_changepoint_test.py)
SB_POSITIONS = {"SB4": 73, "SB5": 92, "SB6": 122, "SB7": 136}
TR_INTERVALS = {"TR4": (73, 92), "TR5": (92, 122), "TR6": (122, 137)}

FACIES_COLORS = {
    "MF-A": "#c9a66b", "MF-B": "#3b7dd8", "MF-C": "#7fae7f", "MF-D": "#a8452e",
    "MF-E": "#e08214", "MF-F": "#8c8c8c", "MF-G": "#b0b0b0", UNLABELED: "#e0e0e0",
}


def sample_to_thickness(sample_number):
    """Convert a Dudan sample number to thickness with the calibration table.

    Linear interpolation between the neighbouring points of ``CALIB``.
    Outside the table (below SD-4 or above SD-134) the first or last
    segment is extended linearly, and the result is flagged as
    extrapolated.

    Args:
        sample_number: SD sample number.

    Returns:
        tuple: ``(thickness_m, extrapolated)``, height above the base of the
        measured section in metres and True if outside the table.
    """
    i = bisect.bisect_left(_samples, sample_number)
    if i == 0:
        i = 1
    if i >= len(_samples):
        i = len(_samples) - 1
    s0, t0 = _samples[i - 1], _thicknesses[i - 1]
    s1, t1 = _samples[i], _thicknesses[i]
    extrapolated = sample_number < _samples[0] or sample_number > _samples[-1]
    frac = (sample_number - s0) / (s1 - s0)
    return t0 + frac * (t1 - t0), extrapolated


def correlation_permutation_test(x, y, n_permutations, rng):
    """Two-sided permutation test for a Pearson correlation.

    ``y`` is randomly reordered against ``x``; the p-value is (number of
    permutations with |r| >= |observed r| + 1) / (n_permutations + 1).

    Args:
        x: First variable (here thickness, m).
        y: Second variable (here the lamination index S).
        n_permutations: Number of random reorderings.
        rng: ``numpy.random.Generator``.

    Returns:
        dict: ``r`` (observed Pearson r), ``p_value_two_sided`` and ``n``.
    """
    observed_r = float(np.corrcoef(x, y)[0, 1])
    null_r = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        y_shuffled = rng.permutation(y)
        null_r[i] = np.corrcoef(x, y_shuffled)[0, 1]
    p_value = (float((np.abs(null_r) >= abs(observed_r)).sum()) + 1) / (n_permutations + 1)
    return {"r": observed_r, "p_value_two_sided": p_value, "n": len(x)}


def run(input_csv, output_dir, figures_dir):
    """Test the trend of S within each T-R sequence and draw the Dudan profile.

    For TR4, TR5 and TR6, S of the MF-B / MF-E specimens is correlated
    with thickness (``correlation_permutation_test``, 10000 permutations,
    seed 42); a positive r is read as shallowing-upward and a negative r
    as deepening-upward. Sequences with fewer than 3 specimens are skipped,
    and fewer than 8 are flagged as low power. The figure shows the facies
    of every Dudan specimen and S against thickness, with SB4-SB7.

    Args:
        input_csv: Labeled feature table (all images).
        output_dir: Output folder for ``sequence_trend_test_results.csv``
            (``pathlib.Path``).
        figures_dir: Output folder for ``sequence_stratigraphic_profile.png``
            (``pathlib.Path``).
    """
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # All classes, for the facies column
    all_df = df[df["section"] == "Dudan"].drop_duplicates("sample_code").copy()
    all_df["sample_number"] = all_df["sample_code"].str.extract(r"(\d+)").astype(float)
    all_df = all_df.dropna(subset=["sample_number"]).sort_values(["sample_number", "sample_code"], kind="mergesort")
    all_df[["thickness_m", "extrapolated"]] = all_df["sample_number"].apply(
        lambda n: pd.Series(sample_to_thickness(n))
    )

    # -----------------------------
    # Lamination index S
    signs = verify_feature_direction(labeled_df)
    section_df = build_section_df(labeled_df, "SD")
    be_df = section_df[section_df["label"].isin(["MF-B", "MF-E"])].copy()
    weights = {feat: 1.0 for feat in LAM_FEATURES}
    be_df["shallowing_index"] = build_shallowing_index(be_df, signs, weights)
    be_df[["thickness_m", "extrapolated"]] = be_df["sample_number"].apply(
        lambda n: pd.Series(sample_to_thickness(n))
    )

    # -----------------------------
    # Trend test per T-R sequence
    rng = np.random.default_rng(RANDOM_SEED)
    trend_rows = []
    print("=== Trend of S with thickness, per T-R sequence ===")
    for tr_name, (lo, hi) in TR_INTERVALS.items():
        seg = be_df[(be_df["sample_number"] >= lo) & (be_df["sample_number"] < hi)]
        if len(seg) < 3:
            print(f"  {tr_name}: n={len(seg)}, too few specimens")
            continue
        result = correlation_permutation_test(
            seg["thickness_m"].to_numpy(), seg["shallowing_index"].to_numpy(), N_PERMUTATIONS, rng
        )
        direction = "shallowing-upward (S increases with height)" if result["r"] > 0 else \
            "deepening-upward (S decreases with height)"
        power_flag = "  [low power, n<8]" if len(seg) < 8 else ""
        print(f"  {tr_name} (n={result['n']}): r={result['r']:.3f}, "
              f"p(two-sided)={result['p_value_two_sided']:.4f} -> {direction}{power_flag}")
        trend_rows.append({"sequence": tr_name, **result, "direction": direction})

    pd.DataFrame(trend_rows).to_csv(output_dir / "sequence_trend_test_results.csv", index=False)
    print(f"\nSaved to {output_dir / 'sequence_trend_test_results.csv'}")

    # -----------------------------
    # Figure
    fig, (ax_facies, ax_S) = plt.subplots(
        1, 2, figsize=(9, 12), sharey=True, gridspec_kw={"width_ratios": [1.1, 1.6]}
    )

    bar_height = 0.9  # m
    for _, row in all_df.iterrows():
        color = FACIES_COLORS.get(row["label"], "#dddddd")
        ax_facies.barh(row["thickness_m"], 1.0, height=bar_height, color=color,
                        edgecolor="none", left=0)
    ax_facies.set_xlim(0, 1)
    ax_facies.set_xticks([])
    ax_facies.set_ylabel("Real thickness above base of measured section (m)")
    ax_facies.set_title("Facies\n(Dudan, all classes)", fontsize=10)

    ax_S.plot(be_df["shallowing_index"], be_df["thickness_m"], "-", color="#555555", linewidth=1, zorder=1)
    ax_S.scatter(be_df["shallowing_index"], be_df["thickness_m"],
                 c=be_df["label"].map(FACIES_COLORS), s=28, zorder=2, edgecolor="black", linewidth=0.4)
    ax_S.axvline(0, color="gray", linewidth=0.7, linestyle=":")
    ax_S.set_xlabel("Lamination index $S$\n(higher = more laminated/MF-B-like)")
    ax_S.set_title("Fabric signal\n(MF-B/MF-E only)", fontsize=10)

    for ax in (ax_facies, ax_S):
        for name, sb in SB_POSITIONS.items():
            t, _ = sample_to_thickness(sb)
            ax.axhline(t, color="#a8452e", linewidth=1.1, linestyle="--", zorder=0)
        ax.axhline(sample_to_thickness(87)[0], color="#888888", linewidth=0.8, linestyle=":", zorder=0)

    for name, sb in SB_POSITIONS.items():
        t, _ = sample_to_thickness(sb)
        ax_S.text(ax_S.get_xlim()[1], t, f"  {name}", va="center", fontsize=8, color="#a8452e")

    y_lowest_classified = sample_to_thickness(87)[0]
    y_axis_bottom, _ = sample_to_thickness(TR_INTERVALS["TR4"][0])
    ax_facies.text(0.5, (y_axis_bottom + y_lowest_classified) / 2,
                    "no classified\nspecimens\nbelow here", fontsize=8,
                    color="#888888", va="center", ha="center", style="italic")

    lowest_classified_num = all_df["sample_number"].min()
    for tr_name, (lo, hi) in TR_INTERVALS.items():
        t_lo, _ = sample_to_thickness(max(lo, lowest_classified_num))
        t_hi, _ = sample_to_thickness(hi)
        mid = (t_lo + t_hi) / 2
        ax_facies.text(-0.35, mid, tr_name, rotation=90, va="center", ha="center",
                        fontsize=9, fontweight="bold", clip_on=False)

    fig.suptitle("Dudan (SD) section: sequence-stratigraphic profile on a real-thickness axis",
                  fontsize=13, y=0.995)
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FACIES_COLORS.values()]
    fig.legend(legend_handles, list(FACIES_COLORS.keys()), loc="lower center", ncol=8, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.03, 0.02, 1, 0.97))
    out_path = figures_dir / "sequence_stratigraphic_profile.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved to {out_path}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv``, ``output_dir`` and ``figures_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/tables/cyclicity")
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, output_dir=Path(args.output_dir), figures_dir=Path(args.figures_dir))


if __name__ == "__main__":
    main()

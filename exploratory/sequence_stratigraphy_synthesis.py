"""Sequence-stratigraphic synthesis of the classified microfacies log.

Every labeled specimen is placed by sample number within its section (a proxy
for stratigraphic position, assuming the field numbering follows the measured
section, base to top), and two kinds of evidence are combined:

  1. Facies belt (Deep Subtidal, Shallow Subtidal-Intertidal,
     Intertidal-Supratidal), from the specimen's microfacies through
     ``Config.ENVIRONMENTAL_ZONE``. A change of belt between successive
     specimens is a candidate parasequence boundary or flooding surface: by
     Walther's Law, an upward shift to a deeper belt records relative
     deepening, and the reverse records shallowing (progradation).

  2. The lamination index S of the MF-B / MF-E specimens: sign-corrected,
     z-scored ``lam_gabor_anisotropy``, the lamination feature that passed the
     fold-wise SHAP + F-value check (foldwise_interpretability_check.py). It is
     computed with the functions of shallowing_index_cyclicity_analysis.py, so
     the values are those tested in the paper. S adds resolution within a
     belt: a specimen close to a belt change can show a transitional S even
     where its class does not change, as continuous image-derived lamina
     transects do in other microbialite studies.

Specimens without a belt:
  - MF-F and MF-G have no facies belt in config.py; they are logged as
    'UNASSIGNED' rather than placed in one of the three belts.
  - SD_100, SD_128 and SD_136 (MF-A) are not used by the classifier and are
    unlabeled in auto_label.py. They are shown here as 'MF-A (unconfirmed)'
    in the MF-A belt; their labels elsewhere are not changed.

Input:   results/features/image_metadata_auto_label.csv
Outputs: results/tables/sequence_stratigraphy/stratigraphic_log.csv
         results/figures/stratigraphic_column_<SD|SK>.png

Run from the repository root:
    PYTHONPATH=. python exploratory/sequence_stratigraphy_synthesis.py
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import Config
from shallowing_index_cyclicity_analysis import (
    LAM_FEATURES,
    build_section_df,
    build_shallowing_index,
    verify_feature_direction,
)

INPUT_CSV = "results/features/image_metadata_auto_label.csv"
OUTPUT_DIR = Path("results/tables/sequence_stratigraphy")
FIGURE_DIR = Path("results/figures")
UNLABELED = "unlabeled"

# S is computed with the functions (and LAM_FEATURES) of
# shallowing_index_cyclicity_analysis.py, so it is the same S as in the order
# tests of the paper. It exists only for MF-B / MF-E specimens; other classes
# appear in the facies-belt panel without an S value.

# MF-A specimens: too few for the classifier, but shown in the facies-belt
# column as unconfirmed MF-A.
MF_A_SPECIMENS = {"SD_100", "SD_128", "SD_136"}


def parse_sample_number(sample_code: str) -> float:
    """Extract the field sample number from a sample code.

    Args:
        sample_code: Code such as ``'SD_105'`` or ``'SD_102B'``.

    Returns:
        float: The first run of digits, or NaN if there is none.
    """
    m = re.search(r"(\d+)", sample_code)
    return float(m.group(1)) if m else np.nan


def facies_belt_for(label: str) -> str:
    """Facies belt of a microfacies.

    Args:
        label: Microfacies code, e.g. ``'MF-B'``.

    Returns:
        str: The belt from ``Config.ENVIRONMENTAL_ZONE``, or
        ``'UNASSIGNED'`` if the microfacies has none.
    """
    return Config.ENVIRONMENTAL_ZONE.get(label, "UNASSIGNED")


def build_lamination_index_by_specimen(df: pd.DataFrame) -> pd.Series:
    """Lamination index S of every MF-B / MF-E specimen.

    Uses the functions of shallowing_index_cyclicity_analysis.py, so the
    values equal those of the order tests (Table 5). The sign of the feature
    is set from both sections together; z-scoring is done within each
    section, over its MF-B / MF-E specimens.

    Args:
        df: Labeled image-level table (no ``'unlabeled'`` rows).

    Returns:
        pandas.Series: S, indexed by ``sample_code``, named
        ``lamination_index_S``.
    """
    signs = verify_feature_direction(df)
    weights = {feat: 1.0 for feat in LAM_FEATURES}  # one feature, weight 1

    section_prefixes = df["sample_code"].str.split("_").str[0].str.split("-").str[0].unique()
    s_by_specimen = {}
    for section_prefix in section_prefixes:
        section_df = build_section_df(df, section_prefix)
        be_df = section_df[section_df["label"].isin(["MF-B", "MF-E"])].copy()
        if be_df.empty:
            continue
        be_df["S"] = build_shallowing_index(be_df, signs, weights)
        s_by_specimen.update(dict(zip(be_df["sample_code"], be_df["S"])))

    return pd.Series(s_by_specimen, name="lamination_index_S")


def build_stratigraphic_log() -> pd.DataFrame:
    """Build the per-specimen stratigraphic log of both sections.

    Unlabeled specimens are dropped, except the MF-A specimens of
    ``MF_A_SPECIMENS``, which are kept as ``'MF-A (unconfirmed)'``.

    Returns:
        pandas.DataFrame: One row per specimen, sorted by section and sample
        number, with ``sample_code``, ``label``, ``n_images``,
        ``lamination_index_S`` (MF-B / MF-E only), ``section_prefix``,
        ``sample_number`` and ``facies_belt``.
    """
    df = pd.read_csv(INPUT_CSV)
    labeled_df = df[df["label"] != UNLABELED].copy()
    s_series = build_lamination_index_by_specimen(labeled_df)

    per_specimen = df.groupby("sample_code").agg(
        label=("label", "first"),
        n_images=("label", "size"),
    ).reset_index()
    per_specimen["lamination_index_S"] = per_specimen["sample_code"].map(s_series)

    # MF-A is set for this log only; auto_label.py and the classifier are
    # not affected.
    per_specimen.loc[per_specimen["sample_code"].isin(MF_A_SPECIMENS), "label"] = "MF-A (unconfirmed)"

    per_specimen = per_specimen[per_specimen["label"] != "unlabeled"].copy()
    per_specimen["section_prefix"] = per_specimen["sample_code"].str.split("_").str[0]
    per_specimen["sample_number"] = per_specimen["sample_code"].apply(parse_sample_number)
    per_specimen["facies_belt"] = per_specimen["label"].apply(
        lambda lab: facies_belt_for(lab.replace(" (unconfirmed)", ""))
    )

    per_specimen = per_specimen.dropna(subset=["sample_number"]).sort_values(
        ["section_prefix", "sample_number"]
    )
    return per_specimen.reset_index(drop=True)


def find_belt_transitions(section_log: pd.DataFrame) -> pd.DataFrame:
    """Mark the specimens whose facies belt differs from the one below.

    Args:
        section_log: Log of one section, ordered from base to top.

    Returns:
        pandas.DataFrame: A copy with a boolean ``belt_transition`` column;
        the lowest specimen is never a transition.
    """
    section_log = section_log.copy()
    section_log["belt_transition"] = section_log["facies_belt"].ne(
        section_log["facies_belt"].shift(1)
    )
    section_log.loc[section_log.index[0], "belt_transition"] = False
    return section_log


def plot_stratigraphic_column(log: pd.DataFrame, section_prefix: str, out_path: Path) -> None:
    """Draw the stratigraphic column of one section.

    Left panel: facies belt of every specimen by sample number, with
    candidate belt transitions as dashed red lines. Right panel: S of the
    MF-B / MF-E specimens. Nothing is drawn if the section has no specimens.

    Args:
        log: Output of ``build_stratigraphic_log``.
        section_prefix: ``'SD'`` or ``'SK'``.
        out_path: Path of the PNG file to write.
    """
    section_log = log[log["section_prefix"] == section_prefix].copy()
    if section_log.empty:
        return
    section_log = find_belt_transitions(section_log)

    belt_colors = {
        "Intertidal-Supratidal": "#d9a441",
        "Shallow_Subtidal-Intertidal": "#4477aa",
        "Deep_Subtidal": "#222266",
        "UNASSIGNED": "#999999",
    }

    fig, (ax_belt, ax_lam) = plt.subplots(
        1, 2, figsize=(7, max(6, 0.28 * len(section_log))), sharey=True,
        gridspec_kw={"width_ratios": [1, 1.3]},
    )

    y = section_log["sample_number"].values
    for _, row in section_log.iterrows():
        ax_belt.barh(
            row["sample_number"], 1, height=0.9,
            color=belt_colors.get(row["facies_belt"], "#cccccc"), edgecolor="none",
        )
        ax_belt.text(
            1.05, row["sample_number"], f"{row['sample_code']} ({row['label']})",
            va="center", fontsize=6,
        )

    for _, row in section_log[section_log["belt_transition"]].iterrows():
        ax_belt.axhline(row["sample_number"] - 0.5, color="red", linewidth=0.8, linestyle="--")
        ax_lam.axhline(row["sample_number"] - 0.5, color="red", linewidth=0.8, linestyle="--")

    ax_belt.set_xlim(0, 4)
    ax_belt.set_xticks([])
    ax_belt.set_ylabel("Sample number (stratigraphic-position proxy)")
    ax_belt.set_title(f"{section_prefix} - Facies belt\n(dashed red = candidate belt transition)")

    s_log = section_log.dropna(subset=["lamination_index_S"])
    ax_lam.plot(
        s_log["lamination_index_S"], s_log["sample_number"],
        marker="o", markersize=4, linewidth=1.2, color="#333333",
    )
    ax_lam.axvline(0, color="#bbbbbb", linewidth=0.6, linestyle=":")
    ax_lam.set_xlabel("Lamination index $S$ (sign-corrected, $z$-scored Gabor anisotropy;\nMF-B/MF-E specimens only)")
    ax_lam.set_title("Lamination index\n(higher = more laminated/MF-B-like)")

    fig.suptitle(f"Sequence-stratigraphic column, {section_prefix} section", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build the log, save it, and print and plot each section."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    log = build_stratigraphic_log()
    log.to_csv(OUTPUT_DIR / "stratigraphic_log.csv", index=False)

    print(f"Built stratigraphic log: {len(log)} specimens across {log['section_prefix'].nunique()} sections\n")

    for section_prefix in sorted(log["section_prefix"].unique()):
        section_log = log[log["section_prefix"] == section_prefix].copy()
        section_log = find_belt_transitions(section_log)
        n_transitions = section_log["belt_transition"].sum()
        print(f"=== {section_prefix} section: {len(section_log)} specimens, "
              f"{n_transitions} candidate belt transitions ===")
        print(section_log[["sample_code", "sample_number", "label", "facies_belt",
                            "lamination_index_S", "belt_transition"]]
              .to_string(index=False))
        print()

        fig_path = FIGURE_DIR / f"stratigraphic_column_{section_prefix}.png"
        plot_stratigraphic_column(log, section_prefix, fig_path)
        print(f"Saved column figure -> {fig_path}\n")


if __name__ == "__main__":
    main()

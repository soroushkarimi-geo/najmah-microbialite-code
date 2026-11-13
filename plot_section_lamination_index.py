"""Fig. 10 (Kezi) and the matching Dudan plot: microbialite class of every
specimen of one section by sample number (other microfacies in grey), next to
the lamination index S of the Class 1 / Class 2 specimens. Dudan also shows the candidate boundaries SB5
(~SD-92) and SB6 (~SD-122).

S comes from build_section_df() and build_shallowing_index() in
shallowing_index_cyclicity_analysis.py, so the values are the ones tested in
Table 5 (z-scored within each section).

    python plot_section_lamination_index.py --input-csv data/image_metadata_auto_label.csv --section SD --figures-dir out/figures
    python plot_section_lamination_index.py --input-csv data/image_metadata_auto_label.csv --section SK --figures-dir out/figures
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd

from sequence_stratigraphic_profile import SB_POSITIONS
from shallowing_index_cyclicity_analysis import (
    LAM_FEATURES,
    build_section_df,
    build_shallowing_index,
    verify_feature_direction,
)

UNLABELED = "unlabeled"

SECTIONS = {
    "SD": {"name": "Dudan", "boundaries": ["SB5", "SB6"], "figsize": (7.6, 12.4),
           "out_name": "dudan_section_lamination_index.png"},
    "SK": {"name": "Kezi", "boundaries": [], "figsize": (7.5, 3.7),
           "out_name": "fig10_kezi_section.png"},
}

CLASS_NAMES = {"MF-B": "Class 1", "MF-E": "Class 2"}
OTHER_NAMES = {
    "MF-C": "bioclast-calcisphere",
    "MF-D": "peloid-oncoid boundstone",
    "MF-F": "mollusc wackestone",
    "MF-G": "radiolarian wackestone",
    # SD-100, SD-128, SD-136 (MF-A, not confirmed, see auto_label.py)
    UNLABELED: "dolostone (unconfirmed)",
}
COLORS = {"MF-B": "#c9772e", "MF-E": "#4b5d9c"}
OTHER_COLOR = "#9aa3ab"
SB_COLOR = "#b04a3a"
LINE_COLOR = "#1f2329"
ZERO_COLOR = "#c3cad2"

BAR_WIDTH = 0.31
BAR_HEIGHT = 0.85

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]
plt.rcParams["font.size"] = 11


def plot_positions(section_df):
    """Vertical plotting position and bar height of each specimen.

    The position is the sample number. Specimens that share a sample
    number (SD_102 and SD_102B) split that slot evenly, in table order
    (lower specimen first), with bars of proportionally smaller height.

    Args:
        section_df: Output of ``build_section_df``, in stratigraphic order.

    Returns:
        tuple of pandas.Series: ``(y, height)``, indexed like
        ``section_df``.
    """
    y = section_df["sample_number"].astype(float).copy()
    height = pd.Series(BAR_HEIGHT, index=section_df.index)
    for _, group in section_df.groupby("sample_number"):
        n = len(group)
        if n == 1:
            continue
        step = 1.0 / n
        for k, idx in enumerate(group.index):
            y[idx] = group.loc[idx, "sample_number"] + (k - (n - 1) / 2) * step
            height[idx] = BAR_HEIGHT / n
    return y, height


def run(input_csv, section_prefix, figures_dir):
    """Draw the class and lamination-index plot of one section (Fig. 10 for Kezi).

    Left panel: the class of every specimen of the section by sample
    number, unlabeled specimens included. Right panel: the lamination index
    S of the Class 1 / Class 2 specimens, computed with
    ``build_shallowing_index`` (the same values as in Table 5). For Dudan,
    the candidate boundaries SB5 and SB6 are drawn.

    Args:
        input_csv: Labeled feature table (all images).
        section_prefix: ``'SD'`` (Dudan) or ``'SK'`` (Kezi).
        figures_dir: Output folder (``pathlib.Path``); writes
            ``dudan_section_lamination_index.png`` or ``fig10_kezi_section.png``.
    """
    info = SECTIONS[section_prefix]
    df = pd.read_csv(input_csv)
    labeled_df = df[df["label"] != UNLABELED].copy()

    # every specimen of the section, unlabeled ones included
    section_df = build_section_df(df, section_prefix)
    section_df["y"], section_df["bar_height"] = plot_positions(section_df)

    # lamination index S of the Class 1 / Class 2 specimens
    signs = verify_feature_direction(labeled_df)
    be_df = section_df[section_df["label"].isin(CLASS_NAMES)].copy()
    weights = {feat: 1.0 for feat in LAM_FEATURES}
    be_df["S"] = build_shallowing_index(be_df, signs, weights)
    print(f"\n{info['name']}: {len(section_df)} specimens, {len(be_df)} Class 1/2")
    print(be_df[["sample_code", "label", "S"]].round(3).to_string(index=False))

    fig, (ax_class, ax_S) = plt.subplots(
        1, 2, figsize=info["figsize"], sharey=True, gridspec_kw={"width_ratios": [0.8, 1]}
    )

    # left: class column
    for _, row in section_df.iterrows():
        code = row["sample_code"].replace("_", "-")
        name = CLASS_NAMES.get(row["label"], OTHER_NAMES.get(row["label"], row["label"]))
        small = row["bar_height"] < BAR_HEIGHT
        ax_class.barh(row["y"], BAR_WIDTH, height=row["bar_height"], left=0,
                      color=COLORS.get(row["label"], OTHER_COLOR), edgecolor="none")
        ax_class.text(BAR_WIDTH + 0.02, row["y"], f"{code}  {name}", va="center",
                      fontsize=6.5 if small else 7.5)
    ax_class.set_xlim(0, 1)
    ax_class.set_xticks([])
    ax_class.set_ylabel("Sample number (stratigraphic position proxy)")
    ax_class.set_title(f"{info['name']}: microbialite class", fontsize=11)

    # right: lamination index
    ax_S.axvline(0, color=ZERO_COLOR, linewidth=1, zorder=0)
    ax_S.plot(be_df["S"], be_df["y"], "-", color=LINE_COLOR, linewidth=1, zorder=1)
    for label, class_name in CLASS_NAMES.items():
        sub = be_df[be_df["label"] == label]
        if len(sub):
            fabric = "laminated" if label == "MF-B" else "clotted"
            ax_S.scatter(sub["S"], sub["y"], s=28, color=COLORS[label], zorder=2,
                         label=f"{class_name} ({fabric})")
    ax_S.set_xlabel("Lamination index $S$\n(higher = more laminated, Class 1-like)")
    ax_S.set_title("Lamination index", fontsize=11)
    ax_S.legend(loc="upper right", fontsize=9, frameon=False)

    # candidate sequence boundaries
    trans = mtransforms.blended_transform_factory(ax_S.transAxes, ax_S.transData)
    for sb in info["boundaries"]:
        y_sb = SB_POSITIONS[sb]
        ax_class.plot([0, BAR_WIDTH], [y_sb, y_sb], color=SB_COLOR, linestyle="--", linewidth=1, zorder=3)
        ax_S.axhline(y_sb, color=SB_COLOR, linestyle="--", linewidth=1.1, zorder=0)
        ax_S.text(0.995, y_sb + 0.25, f"{sb} ({section_prefix}-{y_sb})", transform=trans,
                  ha="right", va="bottom", color=SB_COLOR, fontsize=10)

    fig.tight_layout(w_pad=5)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / info["out_name"]
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {out_path}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv``, ``section`` and ``figures_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--section", type=str, choices=list(SECTIONS), required=True)
    parser.add_argument("--figures-dir", type=str, default="results/figures")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, section_prefix=args.section, figures_dir=Path(args.figures_dir))


if __name__ == "__main__":
    main()

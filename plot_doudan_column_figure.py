"""Fig. 7: Dudan measured log with specimen classes, model output and lamination index.

The measured log (lithology, sample ticks, formation contacts) is cropped without modification
from a vector PDF of the Dudan log, passed with --log-pdf (not distributed in this repository). That drawing is at true thickness: a
linear fit of its sample-number labels against the 39-point thickness calibration table gives
y_pt = 927.29 - 3.609 * thickness_m, with residuals below 0.8 m. All data panels are drawn on
the same metre scale so they line up with the log.

Specimen heights: calibration table up to SD-134 (as in the paper); SD-135 to SD-145 read from
the drawing's own sample ticks (one tick per sample).

Data panels:
  (b) specimen class: Class 1 / Class 2 / other microfacies, from results/features labels
  (c) model output: share of each specimen's images assigned to Class 2 in leave-one-specimen-
      out CV (paper's saved XGBoost predictions); other microfacies show the frozen model's
      mean P(Class 2) from the triage table
  (d) lamination index S (Gabor anisotropy, specimen mean, z-scored within Dudan)

How to run (from the repository root, after train_xgboost_cv.py and score_against_reliable_model.py):
    python plot_doudan_column_figure.py --log-pdf /path/to/doudan_log.pdf
Outputs: out/figures/fig_doudan_column.pdf and .png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymupdf
from matplotlib.lines import Line2D

LOG_PDF = None        # set from --log-pdf in main()
FEATURES = Path("data/image_metadata_reliable_classes.csv")
ALL_LABELS = Path("data/image_metadata_auto_label.csv")
PREDICTIONS = Path("out/tables/cv_per_sample_predictions.csv")
TRIAGE = Path("out/tables/small_class_triage.csv")
OUT = Path("out/figures/fig_doudan_column")

# Drawing geometry (PDF points), measured from the vector file.
Y0_PT, PT_PER_M = 927.29, 3.609
CROP_X = (45.6, 242.0)           # formation, sample, thickness and lithology columns
CROP_Y = (125.3, 640.0)          # header top to ~80 m
HEADER_BOTTOM_PT = 202.1         # bottom of the column header row
LEGEND_CROP = (14.3, 955.0, 628.0, 1140.0)

CALIB = [(12, 4), (14, 7), (17, 10), (20.5, 14), (21.5, 16), (27, 21), (28, 22), (32, 25),
         (34, 27), (36, 30), (37, 32), (42, 35), (44, 36), (50, 39), (53, 42), (55.9, 46),
         (58, 49), (59, 50), (66, 55), (72, 61), (80, 68), (87.5, 77), (89.5, 78), (96.5, 81),
         (101, 84), (105.5, 86), (112.5, 91), (117.5, 94), (131.5, 101), (134.5, 104),
         (143, 110), (144.8, 113), (146.8, 115), (159, 120), (165, 122), (173.5, 126),
         (175.5, 129), (176.5, 130), (184.9, 134)]
# Sample ticks read from the drawing above the calibration table (y in PDF points).
TICKS_ABOVE_134 = {135: 252.4, 136: 249.5, 137: 244.9, 138: 240.4, 139: 234.1, 140: 228.85,
                   141: 224.8, 142: 219.7, 143: 216.4, 144: 213.1, 145: 206.1}

# Contacts drawn in the measured log, and the candidate boundaries used in the order tests.
CONTACTS_PT = {"Sargelu / Najmah": 606.9, "Najmah / Garau": 226.1}
CANDIDATE_SB_M = {"SB5": 114.17, "SB6": 165.0}
CONSENSUS_MISSES = {"SD_88", "SD_89", "SD_96", "SD_102", "SD_109", "SK_85"}

# Categorical slots 1-2 of the validated default palette; shape is the secondary encoding.
C1, C2, OTHER, INK, MUTED = "#2a78d6", "#eb6834", "#8c8c86", "#222220", "#6b6a64"
# Short codes for the non-evaluated microfacies (the drawing's own A-J letters mean other things).
OTHER_CODE = {"MF-C": "Bc", "MF-D": "Po", "MF-F": "Mw", "MF-G": "Rw", "unlabeled": "Do?"}


def pt_to_m(y_pt: float) -> float:
    """Convert a vertical position in the log drawing to thickness.

    Args:
        y_pt: y coordinate in the PDF, in points.

    Returns:
        float: Thickness in metres, from y_pt = 927.29 - 3.609 * thickness_m.
    """
    return (Y0_PT - y_pt) / PT_PER_M


def sample_height(code: str) -> float:
    """Thickness of a Dudan specimen in metres.

    Up to SD-134 the height is interpolated in the calibration table
    ``CALIB``; above it, it is read from the sample ticks of the drawing
    (``TICKS_ABOVE_134``). A ``B`` suffix adds 0.5 to the sample number,
    because the B sample lies above the sample of the same number.

    Args:
        code: Sample code such as ``'SD_102'`` or ``'SD_102B'``.

    Returns:
        float: Thickness above the base of the section, in metres.
    """
    n = float("".join(ch for ch in code.split("_")[1] if ch.isdigit()))
    if code.endswith("B"):
        n += 0.5
    if n > 134:
        return pt_to_m(TICKS_ABOVE_134[int(n)])
    cs, ct = zip(*[(s, t) for t, s in CALIB])
    return float(np.interp(n, cs, ct))


def render_crop(rect, dpi=600) -> np.ndarray:
    """Render a rectangle of the first page of the log PDF as an RGB image.

    Args:
        rect: ``(x0, y0, x1, y1)`` in PDF points.
        dpi: Rendering resolution.

    Returns:
        numpy.ndarray: uint8 array of shape (height, width, 3).
    """
    page = pymupdf.open(LOG_PDF)[0]
    pix = page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(*rect))
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)[..., :3]


def build_specimen_table() -> pd.DataFrame:
    """Collect the data panels' values for every Dudan specimen.

    Returns:
        pandas.DataFrame: Indexed by ``sample_code`` and sorted by height,
        with ``section``, ``label``, ``m`` (thickness, m), ``S`` (specimen-
        mean Gabor anisotropy z-scored over the Dudan MF-B / MF-E
        specimens), ``share_c2`` (share of the specimen's images assigned to
        Class 2 in LOSO CV), ``triage_c2`` (mean P(Class 2) from the triage
        table, for the other microfacies) and ``correct`` (majority vote
        matches the label).
    """
    labels = (pd.read_csv(ALL_LABELS, usecols=["sample_code", "section", "label"])
              .groupby("sample_code").first())
    sp = labels[labels.section == "Dudan"].copy()
    sp["m"] = [sample_height(c) for c in sp.index]

    feats = pd.read_csv(FEATURES, usecols=["sample_code", "section", "lam_gabor_anisotropy"])
    an = feats[feats.section == "Dudan"].groupby("sample_code").lam_gabor_anisotropy.mean()
    sp["S"] = (an - an.mean()) / an.std()

    pred = pd.read_csv(PREDICTIONS)
    sp["share_c2"] = pred.groupby("sample_code").predicted_label.apply(lambda s: (s == "MF-E").mean())
    tri = pd.read_csv(TRIAGE).set_index("sample_code")["prob_MF-E"]
    sp["triage_c2"] = tri
    sp["correct"] = np.where(sp.label == "MF-E", sp.share_c2 > 0.5, sp.share_c2 < 0.5)
    return sp.sort_values("m")


def main() -> None:
    """Command-line entry point: draw the Dudan column figure.

    Crops the measured log and its legend from ``--log-pdf``, draws the
    three data panels on the same metre scale, and saves the figure as PDF
    (600 dpi) and PNG (300 dpi) at ``--output``.
    """
    global LOG_PDF, PREDICTIONS, TRIAGE, OUT
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--log-pdf", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--triage", type=Path, default=TRIAGE)
    parser.add_argument("--output", type=Path, default=OUT, help="output path without extension")
    args = parser.parse_args()
    LOG_PDF, PREDICTIONS, TRIAGE, OUT = args.log_pdf, args.predictions, args.triage, args.output

    sp = build_specimen_table()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5,
                         "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})

    W, H = 7.2, 9.6
    h_main = 7.25                                        # inches for the cropped log
    scale = h_main / (CROP_Y[1] - CROP_Y[0])             # inches per PDF point
    w_log = (CROP_X[1] - CROP_X[0]) * scale
    bottom = 1.95
    m_lo, m_top_img = pt_to_m(CROP_Y[1]), pt_to_m(CROP_Y[0])
    m_hi = pt_to_m(HEADER_BOTTOM_PT)
    h_data = (CROP_Y[1] - HEADER_BOTTOM_PT) * scale

    fig = plt.figure(figsize=(W, H))
    ax_log = fig.add_axes([0.08 / W, bottom / H, w_log / W, h_main / H])
    ax_log.imshow(render_crop((CROP_X[0], CROP_Y[0], CROP_X[1], CROP_Y[1])),
                  extent=(CROP_X[0], CROP_X[1], m_lo, m_top_img), aspect="auto", interpolation="lanczos")
    ax_log.set_axis_off()

    x = 0.08 + w_log + 0.62
    widths = [0.72, 1.3, 1.3]
    titles = ["Specimen\nclass", "Share of images\nassigned to Class 2", "Lamination\nindex S"]
    axes = []
    for i, (w, title) in enumerate(zip(widths, titles)):
        ax = fig.add_axes([x / W, bottom / H, w / W, h_data / H], sharey=axes[0] if axes else None)
        ax.set_ylim(m_lo, m_hi)
        ax.text(0.5, 1.012, title, transform=ax.transAxes, ha="center", va="bottom", color=INK, fontsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        axes.append(ax)
        x += w + 0.14
    ax_b, ax_c, ax_d = axes

    ax_b.set_ylabel("Thickness (m)", color=INK)
    ax_b.set_yticks(np.arange(80, 201, 10))
    for ax in (ax_c, ax_d):
        plt.setp(ax.get_yticklabels(), visible=False)

    # reference lines: contacts from the measured log (solid), candidate boundaries (dashed)
    for name, y_pt in CONTACTS_PT.items():
        for ax in axes:
            ax.axhline(pt_to_m(y_pt), color=INK, lw=0.7, zorder=1)
        ax_d.text(0.98, pt_to_m(y_pt) + 0.6, name, transform=ax_d.get_yaxis_transform(),
                  va="bottom", ha="right", fontsize=6.3, color=INK)
    for name, m in CANDIDATE_SB_M.items():
        for ax in axes:
            ax.axhline(m, color=MUTED, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax_d.text(0.98, m + 0.6, name, transform=ax_d.get_yaxis_transform(),
                  va="bottom", ha="right", fontsize=6.3, color=MUTED)

    c1, c2 = sp[sp.label == "MF-B"], sp[sp.label == "MF-E"]
    oth = sp[~sp.label.isin(["MF-B", "MF-E"])]

    # (b) specimen class
    ax_b.scatter(np.full(len(c1), 0.35), c1.m, s=22, marker="s", color=C1, edgecolor="white", lw=0.6, zorder=3)
    ax_b.scatter(np.full(len(c2), 0.35), c2.m, s=26, marker="o", color=C2, edgecolor="white", lw=0.6, zorder=3)
    ax_b.scatter(np.full(len(oth), 0.35), oth.m, s=20, marker="^", color=OTHER, edgecolor="white", lw=0.6, zorder=3)
    groups, prev = [], None                      # consecutive specimens with the same code -> one label
    for code, r in oth.iterrows():
        c = OTHER_CODE.get(r.label, "?")
        if prev is not None and groups[-1][0] == c and r.m - prev < 6:
            groups[-1][1].append(r.m)
        else:
            groups.append((c, [r.m]))
        prev = r.m
    last_m, col = -99, 0
    for c, ms in groups:
        m = float(np.mean(ms))
        col = 1 - col if m - last_m < 5 else 0    # stagger labels that would collide
        ax_b.text(0.56 + 0.24 * col, m, c, va="center", fontsize=6, color=MUTED)
        last_m = m
    ax_b.set_xlim(0, 1)
    ax_b.set_xticks([])

    # (c) model output
    ax_c.axvline(0.5, color=MUTED, lw=0.6, ls=":", zorder=1)
    for sub, col, mk, sz in [(c1, C1, "s", 22), (c2, C2, "o", 26)]:
        ok, bad = sub[sub.correct], sub[~sub.correct]
        ax_c.scatter(ok.share_c2, ok.m, s=sz, marker=mk, color=col, edgecolor="white", lw=0.6, zorder=3)
        ax_c.scatter(bad.share_c2, bad.m, s=sz, marker=mk, facecolor="white", edgecolor=col, lw=1.2, zorder=3)
    ax_c.scatter(oth.triage_c2, oth.m, s=20, marker="^", color=OTHER, edgecolor="white", lw=0.6, zorder=3)
    for code, r in sp[sp.index.isin(CONSENSUS_MISSES)].iterrows():
        ax_c.scatter(r.share_c2, r.m, s=110, marker="o", facecolor="none", edgecolor=INK, lw=0.7, zorder=4)
        ax_c.text(r.share_c2 + (0.07 if r.share_c2 < 0.75 else -0.07), r.m, code.replace("_", "-"),
                  va="center", ha="left" if r.share_c2 < 0.75 else "right", fontsize=6, color=INK)
    ax_c.set_xlim(-0.03, 1.03)
    ax_c.set_xticks([0, 0.5, 1])
    ax_c.set_xlabel("← Class 1     Class 2 →", color=INK)

    # (d) lamination index
    s = sp.dropna(subset=["S"])
    ax_d.plot(s.S, s.m, color=MUTED, lw=0.8, zorder=2)
    ax_d.axvline(0, color=MUTED, lw=0.6, ls=":", zorder=1)
    ax_d.scatter(s[s.label == "MF-B"].S, s[s.label == "MF-B"].m, s=22, marker="s", color=C1, edgecolor="white", lw=0.6, zorder=3)
    ax_d.scatter(s[s.label == "MF-E"].S, s[s.label == "MF-E"].m, s=26, marker="o", color=C2, edgecolor="white", lw=0.6, zorder=3)
    ax_d.set_xlim(-1.6, 4.9)
    ax_d.set_xticks([-1, 0, 1, 2, 3, 4])
    ax_d.set_xlabel("← clotted     laminated →", color=INK)

    for ax, letter in zip([ax_log] + axes, "abcd"):
        lx, ly = (-0.02, 1.003) if ax is ax_log else (-0.05, 1.07)
        ax.text(lx, ly, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.5, fontweight="bold", color=INK)

    # legends: lithology from the drawing, plus the data symbols
    leg_img = render_crop(LEGEND_CROP, dpi=500).copy()
    # blank the drawing's 'Depositional Textures' box, which sits in the crop's lower-right corner
    px_per_pt = leg_img.shape[1] / (LEGEND_CROP[2] - LEGEND_CROP[0])
    leg_img[int((1022 - LEGEND_CROP[1]) * px_per_pt):int((1136 - LEGEND_CROP[1]) * px_per_pt), int((575 - LEGEND_CROP[0]) * px_per_pt):] = 255
    lw_in = 4.95
    lh_in = lw_in * (LEGEND_CROP[3] - LEGEND_CROP[1]) / (LEGEND_CROP[2] - LEGEND_CROP[0])
    ax_leg = fig.add_axes([0.08 / W, 0.12 / H, lw_in / W, lh_in / H])
    ax_leg.imshow(leg_img, interpolation="lanczos")
    ax_leg.set_axis_off()

    handles = [
        Line2D([], [], ls="", marker="s", ms=6, color=C1, label="Class 1, laminated"),
        Line2D([], [], ls="", marker="o", ms=6.5, color=C2, label="Class 2, clotted"),
        Line2D([], [], ls="", marker="^", ms=6, color=OTHER, label="Other microfacies (code)"),
        Line2D([], [], ls="", marker="o", ms=6.5, mfc="white", mec=MUTED, mew=1.2, label="Misclassified (open)"),
        Line2D([], [], ls="", marker="o", ms=10, mfc="none", mec=INK, mew=0.7, label="Missed by all 4 models"),
        Line2D([], [], color=INK, lw=0.8, label="Formation contact (log)"),
        Line2D([], [], color=MUTED, lw=0.8, ls=(0, (4, 3)), label="Candidate SB (order tests)"),
    ]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=((0.08 + lw_in + 0.15) / W, 0.1 / H),
               frameon=False, fontsize=6.6, handletextpad=0.5, labelspacing=0.55, borderaxespad=0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), dpi=600)
    fig.savefig(OUT.with_suffix(".png"), dpi=300)
    print("saved", OUT.with_suffix(".pdf"), OUT.with_suffix(".png"))
    print(sp[["label", "m", "share_c2", "triage_c2", "S"]].round(2).to_string())


if __name__ == "__main__":
    main()

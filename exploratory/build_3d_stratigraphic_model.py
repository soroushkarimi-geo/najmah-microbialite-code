"""Schematic 3D view of the Dudan and Kezi sections, coloured by the lamination
index S.

Each section is drawn as a vertical column at its real map position; the
columns are coloured by S of the MF-B / MF-E specimens, so changes in fabric
up-section can be compared between the two sections. This is a schematic
placement of per-specimen data, not a reconstruction of the outcrop geometry:

- Horizontal: the section coordinates of Bayet-Goll et al. (2022), converted
  to a local east-north grid in km with a flat-Earth approximation (adequate
  over the ~20 km between the sections).
- Vertical: approximate heights from sample numbers. Dudan uses a linear
  scale from SB4 = 82 m at SD-87 to SB7 = 193 m at SD-145; Kezi spreads the
  reported ~13 m of Najmah Formation evenly over SK-83 to SK-87.
- Colour: S as written to ``stratigraphic_log.csv`` by
  ``sequence_stratigraphy_synthesis.py`` (specimens without S are left out),
  clipped to its 2nd-98th percentile range.

Output: results/figures/stratigraphic_3d_model.png

Run from the repository root, after sequence_stratigraphy_synthesis.py:
    PYTHONPATH=. python exploratory/build_3d_stratigraphic_model.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# Real coordinates (DMS converted to decimal degrees), Bayet-Goll et al. (2022)
SECTION_COORDS = {
    "Dudan": (35 + 1/60 + 1.6/3600, 46 + 11/60 + 14.6/3600),
    "Kezi": (34 + 51/60 + 50.6/3600, 46 + 8/60 + 25.1/3600),
}


def latlon_to_local_xy(lat, lon, lat0, lon0):
    """Convert latitude/longitude to local east/north offsets in metres.

    Uses a flat-Earth (equirectangular) approximation around a reference
    point, accurate enough over a few tens of kilometres.

    Args:
        lat, lon: Point to convert, in decimal degrees.
        lat0, lon0: Reference point, in decimal degrees.

    Returns:
        tuple of float: ``(x, y)``, east and north of the reference, in m.
    """
    R = 6371000.0
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    x = R * dlon * np.cos(np.radians(lat0))
    y = R * dlat
    return x, y


def dudan_meters(sample_number):
    """Approximate height of a Dudan sample, in metres.

    Linear between SD-87 at 82 m (SB4) and SD-145 at 193 m (SB7), read from
    Fig. 4 of Bayet-Goll et al. (2022).

    Args:
        sample_number: SD sample number.

    Returns:
        float: Approximate height in metres.
    """
    return 82 + (sample_number - 87) * (193 - 82) / (145 - 87)


def kezi_meters(sample_number):
    """Approximate height of a Kezi sample above SK-83, in metres.

    The reported ~13 m of Najmah Formation at Kezi is spread evenly over
    SK-83 to SK-87.

    Args:
        sample_number: SK sample number.

    Returns:
        float: Approximate height in metres.
    """
    return (sample_number - 83) * (13.0 / (87 - 83))


def main():
    """Draw and save the 3D figure (see the module docstring)."""
    # S is read, not recomputed, so it is identical to the value in
    # stratigraphic_log.csv. It exists only for MF-B / MF-E specimens.
    log = pd.read_csv("results/tables/sequence_stratigraphy/stratigraphic_log.csv")
    log = log.dropna(subset=["lamination_index_S"]).copy()
    log["lamination_index"] = log["lamination_index_S"]

    lat0 = np.mean([c[0] for c in SECTION_COORDS.values()])
    lon0 = np.mean([c[1] for c in SECTION_COORDS.values()])

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")

    section_meta = {"SD": ("Dudan", dudan_meters), "SK": ("Kezi", kezi_meters)}
    all_lam = log["lamination_index"].to_numpy()
    vmin, vmax = np.nanpercentile(all_lam, [2, 98])

    for prefix, (name, meters_fn) in section_meta.items():
        sec = log[log["section_prefix"] == prefix].sort_values("sample_number").copy()
        sec["meters"] = sec["sample_number"].apply(meters_fn)
        lat, lon = SECTION_COORDS[name]
        x0, y0 = latlon_to_local_xy(lat, lon, lat0, lon0)
        x0, y0 = x0 / 1000.0, y0 / 1000.0  # km

        z = sec["meters"].to_numpy()
        lam = sec["lamination_index"].to_numpy()
        x = np.full_like(z, x0, dtype=float)
        y = np.full_like(z, y0, dtype=float)

        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm_lam = (lam[:-1] + lam[1:]) / 2
        lc = Line3DCollection(segments, cmap="inferno", norm=plt.Normalize(vmin, vmax), linewidths=10)
        lc.set_array(norm_lam)
        ax.add_collection3d(lc)

        ax.scatter(x, y, z, c=lam, cmap="inferno", vmin=vmin, vmax=vmax, s=25, edgecolor="black", linewidth=0.3)
        ax.text(x0, y0, z.max() + 8, f"{name}\n({len(sec)} specimens)", fontsize=10, ha="center")

    ax.set_xlabel("Local east-west position (km)")
    ax.set_ylabel("Local north-south position (km)")
    ax.set_zlabel("Approximate stratigraphic height (m)")
    ax.set_title(
        "Schematic 3D stratigraphic model, Dudan and Kezi sections\n"
        "Column colour = lamination index (yellow: laminated/MF-B-like, dark: clotted/MF-E-like)\n"
        "Horizontal spacing from real coordinates; vertical position approximate (see script docstring)",
        fontsize=10,
    )

    mappable = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(vmin, vmax))
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Lamination index (relative, unitless)")

    ax.view_init(elev=18, azim=-60)
    fig.tight_layout()
    outpath = "results/figures/stratigraphic_3d_model.png"
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    print("Saved ->", outpath)


if __name__ == "__main__":
    main()

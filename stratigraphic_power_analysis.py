"""Runs test on the binary class log, Lomb-Scargle permutation test on the
lamination index, and power simulation for TR-scale cycles (Dudan section).

Produces the "Runs test" and "Lomb-Scargle" rows of Table 5 and all of
Table 4 (simulated detection power) in the paper.

Heights (m) from the 39-point thickness-sample calibration table of the Dudan
(= Dawan) section up to SD-134, and from the measured log's own sample ticks above it. S = specimen mean of
lam_gabor_anisotropy, z-scored over the Dudan specimens.
Seed 0; the 2000 permutations are drawn first, then the power simulation, so the
numbers repeat exactly.

    python stratigraphic_power_analysis.py data/image_metadata_reliable_classes.csv
"""
import argparse
import re

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import lombscargle

# (thickness in m, SD sample number), Dudan section.
CALIB = [(12, 4), (14, 7), (17, 10), (20.5, 14), (21.5, 16), (27, 21), (28, 22), (32, 25), (34, 27), (36, 30),
         (37, 32), (42, 35), (44, 36), (50, 39), (53, 42), (55.9, 46), (58, 49), (59, 50), (66, 55), (72, 61),
         (80, 68), (87.5, 77), (89.5, 78), (96.5, 81), (101, 84), (105.5, 86), (112.5, 91), (117.5, 94),
         (131.5, 101), (134.5, 104), (143, 110), (144.8, 113), (146.8, 115), (159, 120), (165, 122),
         (173.5, 126), (175.5, 129), (176.5, 130), (184.9, 134)]
CALIB_SAMPLE = np.array([c[1] for c in CALIB], float)
CALIB_THICKNESS = np.array([c[0] for c in CALIB], float)

# Measured log drawn at true thickness: y_pt = LOG_Y0_PT - LOG_PT_PER_M * thickness_m.
LOG_Y0_PT, LOG_PT_PER_M = 927.29, 3.609
LOG_TICKS_PT = {135: 252.4, 136: 249.5, 137: 244.9, 138: 240.4, 139: 234.1, 140: 228.85,
                141: 224.8, 142: 219.7, 143: 216.4, 144: 213.1, 145: 206.1}

LS_PERIODS = np.linspace(6, 90, 200)       # periods scanned by Lomb-Scargle, m
N_NULL_PERMUTATIONS = 2000
POWER_PERIODS = [5, 10, 30, 50]            # m
POWER_AMPLITUDES = [0.5, 0.75, 1.0, 1.5]   # in units of the noise sd
N_POWER_SIMULATIONS = 300
RANDOM_SEED = 0


def sample_to_thickness(n):
    """Convert a Dudan sample number to height in the section.

    Up to SD-134, heights are linearly interpolated in the 39-point
    thickness-sample calibration table ``CALIB``. Above it, they come from
    the sample ticks of the measured log (``LOG_TICKS_PT``), converted from
    drawing coordinates with y_pt = 927.29 - 3.609 * thickness_m.

    Args:
        n: Sample number (may be fractional, e.g. 102.5 for SD_102B).

    Returns:
        float: Height above the base of the section, in metres.
    """
    cs, ct = CALIB_SAMPLE, CALIB_THICKNESS
    if n <= cs[-1]:
        return float(np.interp(n, cs, ct))
    upper_n = [cs[-1]] + sorted(LOG_TICKS_PT)
    upper_t = [ct[-1]] + [(LOG_Y0_PT - LOG_TICKS_PT[k]) / LOG_PT_PER_M for k in sorted(LOG_TICKS_PT)]
    return float(np.interp(n, upper_n, upper_t))


def sample_number(code):
    """Numeric stratigraphic position of a sample code.

    Args:
        code: Code such as ``'SD_102'`` or ``'SD_102B'``.

    Returns:
        float: The digits of the code, plus 0.5 for a ``B`` suffix because
        the B sample lies above the sample of the same number
        (``'SD_102'`` -> 102.0, ``'SD_102B'`` -> 102.5).
    """
    return float(re.sub(r"[^0-9]", "", code)) + (0.5 if code.endswith("B") else 0)


def runs_test(binary):
    """Wald-Wolfowitz runs test on an ordered 0/1 sequence.

    Too few runs means the two classes are clustered in the sequence; too
    many means they alternate. Uses the normal approximation, two-sided.

    Args:
        binary: Sequence of 0 and 1 in stratigraphic order.

    Returns:
        dict: ``runs`` (observed), ``expected`` (under random order), ``z``,
        ``p_two_sided``, ``n1`` (number of 1s) and ``n0`` (number of 0s).
    """
    b = np.asarray(binary).astype(int)
    runs = 1 + int(np.sum(b[1:] != b[:-1]))
    n1 = int(b.sum()); n0 = len(b) - n1
    mu = 1 + 2 * n1 * n0 / (n1 + n0)
    var = 2 * n1 * n0 * (2 * n1 * n0 - n1 - n0) / ((n1 + n0) ** 2 * (n1 + n0 - 1))
    z = (runs - mu) / np.sqrt(var)
    return {"runs": runs, "expected": mu, "z": z, "p_two_sided": 2 * stats.norm.sf(abs(z)), "n1": n1, "n0": n0}


def ls_stat(z, x, periods):
    """Maximum Lomb-Scargle power of an unevenly sampled series.

    Lomb-Scargle handles the irregular spacing of the specimens in height.

    Args:
        z: Heights (m) of the observations.
        x: Values at those heights; the mean is removed first.
        periods: Periods (m) to scan.

    Returns:
        float: The largest normalized periodogram value over ``periods``.
    """
    w = 2 * np.pi / periods
    x = x - x.mean()
    return lombscargle(z, x, w, normalize=True).max()


def build_dudan_specimens(df):
    """One row per Dudan specimen, ordered by height.

    Args:
        df: Image-level table with ``section``, ``sample_code``, ``label``
            and ``lam_gabor_anisotropy``.

    Returns:
        pandas.DataFrame: ``sample_code``, ``lab`` (class), ``an``
        (specimen-mean Gabor anisotropy), ``num`` (from ``sample_number``),
        ``z`` (height, m, from ``sample_to_thickness``) and ``S`` (``an``
        z-scored over the Dudan specimens, sample standard deviation).
    """
    d = df[df.section == "Dudan"]
    sp = d.groupby("sample_code").agg(lab=("label", "first"), an=("lam_gabor_anisotropy", "mean")).reset_index()
    sp["num"] = [sample_number(s) for s in sp.sample_code]
    sp = sp.sort_values("num").reset_index(drop=True)
    sp["z"] = [sample_to_thickness(n) for n in sp.num]
    sp["S"] = (sp.an - sp.an.mean()) / sp.an.std()
    return sp


def power_table(z, threshold, rng, periods=POWER_PERIODS, amplitudes=POWER_AMPLITUDES,
                n_sim=N_POWER_SIMULATIONS):
    """Simulated power of the Lomb-Scargle test at the real sample heights.

    For every period P and amplitude A, ``n_sim`` series are simulated as
    A * sin(2 pi z / P + random phase) plus Gaussian noise with sd 1,
    standardized, and counted as detected when their maximum Lomb-Scargle
    power reaches ``threshold``.

    Args:
        z: Heights (m) of the real specimens.
        threshold: Detection threshold, the 95th percentile of the
            permutation null distribution (alpha = 0.05).
        rng: ``numpy.random.Generator``.
        periods: Cycle periods to test, in metres.
        amplitudes: Cycle amplitudes, in units of the noise sd.
        n_sim: Number of simulated series per cell.

    Returns:
        pandas.DataFrame: One row per period (``period_m``) and one column
        ``A=<amplitude>`` per amplitude, holding the detection rate (Table 4).
    """
    rows = []
    for P in periods:
        row = {"period_m": P}
        for A in amplitudes:
            hits = 0
            for _ in range(n_sim):
                x = A * np.sin(2 * np.pi * z / P + rng.uniform(0, 2 * np.pi)) + rng.normal(0, 1, len(z))
                x = (x - x.mean()) / x.std()
                hits += ls_stat(z, x, LS_PERIODS) >= threshold
            row[f"A={A}"] = hits / n_sim
        rows.append(row)
    return pd.DataFrame(rows)


def run(input_csv, output_csv=None):
    """Runs test, Lomb-Scargle permutation test and power table for Dudan.

    1. Runs test on the Class 1 (MF-B) / Class 2 sequence ordered by height.
    2. Lomb-Scargle maximum power of S, compared with 2000 random
       reorderings of S over the same heights.
    3. ``power_table`` with the 95th percentile of that null distribution
       as the threshold.

    One random generator (seed 0) is used for the permutations first and
    the simulations second, so the results repeat exactly.

    Args:
        input_csv: Feature table with ``section``, ``sample_code``, ``label``
            and ``lam_gabor_anisotropy``.
        output_csv: Optional path for the power table (Table 4).
    """
    sp = build_dudan_specimens(pd.read_csv(input_csv))
    print(sp[["sample_code", "lab", "z", "S"]].round(2).to_string())

    b = (sp.lab == "MF-B").values.astype(int)
    rt = runs_test(b)
    print(f"binary runs={rt['runs']} expected={rt['expected']:.1f} z={rt['z']:.2f} p2={rt['p_two_sided']:.2f}")
    print("C2->C1 transitions", np.sum((b[1:] == 1) & (b[:-1] == 0)), "C1->C2", np.sum((b[1:] == 0) & (b[:-1] == 1)),
          "(binary: these differ by at most 1, so a directional Markov test is degenerate)")

    z = sp.z.values; S = sp.S.values
    print("Dudan span %.1f-%.1f m, n=%d, median spacing %.2f m" % (z.min(), z.max(), len(z), np.median(np.diff(z))))

    rng = np.random.default_rng(RANDOM_SEED)
    obs = ls_stat(z, S, LS_PERIODS)
    null = np.array([ls_stat(z, rng.permutation(S), LS_PERIODS) for _ in range(N_NULL_PERMUTATIONS)])
    thr = np.quantile(null, 0.95)
    print("observed LS max power %.3f, perm p=%.3f" % (obs, (1 + (null >= obs).sum()) / (1 + len(null))))

    print("Power (LS permutation threshold, alpha=.05) for injected sinusoid, noise sd=1:")
    power = power_table(z, thr, rng)
    for _, r in power.iterrows():
        print(f"  period {int(r.period_m):>3} m: " + "  ".join(f"A={A}:{r[f'A={A}']:.2f}" for A in POWER_AMPLITUDES))
    if output_csv:
        power.to_csv(output_csv, index=False)
        print(f"Saved to {output_csv}")


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``output_csv``.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_csv", help="Feature CSV with section, sample_code, label, lam_gabor_anisotropy")
    parser.add_argument("--output-csv", default=None, help="Optional: save the power table (Table 4) here")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()

"""Specimen-level accuracy (majority vote of each specimen's images) for the four
models of Table 2 and the four feature families of Table 3, with 95% Wilson
intervals, for all 49 specimens and for Dudan only (44).

Uses the functions of compare_models.py and feature_family_ablation.py as they are.

    python specimen_level_rerun.py --input-csv data/image_metadata_reliable_classes.csv --output-dir out/specimen_level
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from config import Config
from compare_models import build_models, run_loso_cv_generic, get_feature_columns
from feature_family_ablation import FEATURE_FAMILIES, run_loso_cv


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion.

    Args:
        k: Number of successes (specimens correct).
        n: Number of trials (specimens).
        z: Standard normal quantile; 1.96 gives a 95% interval.

    Returns:
        tuple of float: ``(lower, upper)`` bounds of the interval for k / n.
    """
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def specimen_majority_vote(pred):
    """Specimen-level prediction by majority vote of its images.

    Args:
        pred: Image-level predictions with ``sample_code``, ``label`` and
            ``predicted_label`` (MF-B / MF-E).

    Returns:
        pandas.DataFrame: Indexed by ``sample_code``, with the true
        ``label``, ``frac_E`` (share of the specimen's images predicted
        MF-E) and ``spec_pred``: ``'MF-E'`` if frac_E > 0.5, ``'MF-B'`` if
        frac_E < 0.5 and ``'tie'`` if exactly 0.5. A tie counts as wrong.
    """
    g = pred.groupby("sample_code").agg(label=("label", "first"),
                                        frac_E=("predicted_label", lambda s: (s == "MF-E").mean()))
    g["spec_pred"] = np.where(g.frac_E > 0.5, "MF-E", np.where(g.frac_E < 0.5, "MF-B", "tie"))
    return g


def summarize(pred, tag, subset, out_dir):
    """Image and specimen accuracy of one cross-validated model.

    Saves the image-level predictions as ``pred_<subset>_<tag>.csv`` in
    ``out_dir`` and prints the summary row.

    Args:
        pred: Image-level out-of-fold predictions (``sample_code``,
            ``label``, ``predicted_label``).
        tag: Model name used in the output.
        subset: Name of the specimen subset (``'all49'`` or ``'dudan44'``).
        out_dir: Output folder (``pathlib.Path``).

    Returns:
        tuple: ``(row, ok)``. ``row`` is a dict with image accuracy,
        specimens correct, specimen accuracy, the 95% Wilson interval, the
        number of ties and the codes of the misclassified specimens. ``ok``
        is a boolean Series, indexed by specimen, True where the majority
        vote is correct.
    """
    img = (pred.label == pred.predicted_label).mean()
    g = specimen_majority_vote(pred)
    ok = g.spec_pred == g.label
    lo, hi = wilson(ok.sum(), len(g))
    pred.to_csv(out_dir / f"pred_{subset}_{tag}.csv", index=False)
    row = dict(subset=subset, model=tag, image_acc=round(img, 4), spec_correct=int(ok.sum()), n_spec=len(g),
               spec_acc=round(ok.mean(), 4), wilson_lo=round(lo, 3), wilson_hi=round(hi, 3),
               ties=int((g.spec_pred == "tie").sum()), wrong=" ".join(sorted(g.index[~ok])))
    print(row, flush=True)
    return row, ok


def run(input_csv, out_dir):
    """Specimen-level results for Tables 2 and 3, all specimens and Dudan only.

    For each subset (all 49 specimens; the 44 Dudan specimens), runs LOSO
    XGBoost on each feature family of Table 3 and the three non-XGBoost
    models of Table 2 (the fused XGBoost of Table 2 is the "Fused" family
    row). Writes one predictions file per run and ``summary.csv`` with one
    row per subset and model, including the run time in seconds. Finally
    prints how many specimens only the fused set or only DINOv2 classifies
    correctly.

    Args:
        input_csv: MF-B / MF-E feature table.
        out_dir: Output folder (``pathlib.Path``); created if needed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all = pd.read_csv(input_csv)
    rows, correct = [], {}
    for subset, df in [("all49", df_all), ("dudan44", df_all[df_all.section == "Dudan"])]:
        df = df.copy()
        le = LabelEncoder().fit(sorted(df.label.unique()))
        feats = get_feature_columns(df)
        params = dict(Config.XGBOOST_PARAMS); params["num_class"] = 2
        for fam, sel in FEATURE_FAMILIES.items():       # Table 3, XGBoost per feature family
            t = time.time(); pred = run_loso_cv(df, sel(list(df.columns)), le, params)
            r, ok = summarize(pred, "XGB_" + fam.split(" ")[0], subset, out_dir); r["sec"] = round(time.time() - t); rows.append(r)
            correct[(subset, r["model"])] = ok
        for name, tmpl in build_models().items():      # Table 2, other models
            if name.startswith("XGBoost"):
                continue                                # = XGB_Fused
            t = time.time(); pred = run_loso_cv_generic(df, feats, le, lambda tm=tmpl: clone(tm))
            r, ok = summarize(pred, name.split(" (")[0].replace(" ", "_"), subset, out_dir); r["sec"] = round(time.time() - t); rows.append(r)
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)

    # fused vs DINOv2 only, specimens right with one and wrong with the other
    for subset in ["all49", "dudan44"]:
        a, b = correct.get((subset, "XGB_Fused")), correct.get((subset, "XGB_DINOv2"))
        if a is not None and b is not None:
            print(f"{subset}: fused-only-correct={int((a & ~b).sum())} dino-only-correct={int((~a & b).sum())}", flush=True)
    print(f"Saved to {out_dir / 'summary.csv'}", flush=True)


def parse_args():
    """Read the command-line options.

    Returns:
        argparse.Namespace with ``input_csv`` and ``output_dir``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=str, default="data/image_metadata_reliable_classes.csv")
    parser.add_argument("--output-dir", type=str, default="results/specimen_level")
    return parser.parse_args()


def main():
    """Command-line entry point: run ``run`` with the given options."""
    args = parse_args()
    run(input_csv=args.input_csv, out_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()

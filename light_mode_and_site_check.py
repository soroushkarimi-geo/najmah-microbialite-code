"""Light-mode breakdown and site-confound check (Class 1 vs Class 2).

Same protocol as train_xgboost_cv.py: XGBoost with config.py hyperparameters, class-balanced
sample weights recomputed per fold, leave-one-specimen-out. Probabilities are kept (objective
multi:softprob gives the same model as multi:softmax, with probabilities as output).

  A. Light mode
     A1  the paper's pooled model (saved predictions), scored separately on PPL and XPL images
     A2  mode-specific models, trained and tested on PPL images only, or XPL images only
  B. Site confound: Class 1 specimens only (the one class present at both Dudan and Kezi),
     predict site instead of class; specimen-level AUC of the mean out-of-fold P(Kezi), on
     all images, PPL only and XPL only, with an exact Mann-Whitney p-value (5 vs 20 specimens).

How to run (from the repository root, after train_xgboost_cv.py):
    python light_mode_and_site_check.py --output-dir out/tables/light_mode_and_site_check
Outputs: light_mode.csv and site_auc.csv in the output directory.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from config import Config

FEATURE_PREFIXES = ("glcm_", "dino_", "lam_")


def loso_proba(df: pd.DataFrame, target: pd.Series) -> pd.Series:
    """Out-of-fold probability of the positive class, leave-one-specimen-out.

    For each specimen, an XGBoost model (``Config`` settings with
    ``multi:softprob``, balanced sample weights from the training images)
    is trained on all other specimens and predicts that specimen's images.

    Args:
        df: Image-level table with ``sample_code`` and the ``glcm_``,
            ``dino_`` and ``lam_`` feature columns.
        target: Binary (0/1) target per image, aligned with ``df``.

    Returns:
        pandas.Series: P(target == 1) per image, indexed like ``df``.
    """
    params = dict(Config.XGBOOST_PARAMS, num_class=2, objective="multi:softprob")
    cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    X, y, g = df[cols].to_numpy(), target.to_numpy(), df.sample_code.to_numpy()
    p = np.full(len(df), np.nan)
    for s in np.unique(g):
        tr = g != s
        model = xgb.XGBClassifier(**params)
        model.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
        p[~tr] = model.predict_proba(X[~tr])[:, 1]
    return pd.Series(p, index=df.index)


def specimen_scores(df: pd.DataFrame, pred_is_e: pd.Series) -> tuple[float, int, int]:
    """Image accuracy and specimen majority-vote accuracy, Class 1 vs Class 2.

    A specimen is correct when more than half of its images are assigned
    to its true class; an exact 50/50 split counts as wrong.

    Args:
        df: Image-level table with ``label`` (MF-B / MF-E) and
            ``sample_code``.
        pred_is_e: Boolean Series, True where an image is predicted MF-E,
            aligned with ``df``.

    Returns:
        tuple: ``(image_accuracy, specimens_correct, n_specimens)``.
    """
    truth_e = df.label == "MF-E"
    img = float((pred_is_e == truth_e).mean())
    vote = pred_is_e.groupby(df.sample_code).mean()
    lab = truth_e.groupby(df.sample_code).first()
    ok = ((vote > 0.5) & lab) | ((vote < 0.5) & ~lab)
    return img, int(ok.sum()), len(ok)


def main() -> None:
    """Command-line entry point: run the light-mode and site checks.

    Writes ``light_mode.csv`` (analyses A1 and A2) and ``site_auc.csv``
    (analysis B) to the output directory. The site AUC is computed on the
    specimen means of the out-of-fold P(Kezi), and its p-value is the exact
    one-sided Mann-Whitney test that Kezi specimens score higher.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--features", type=Path, default=Path("data/image_metadata_reliable_classes.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("out/tables/cv_per_sample_predictions.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/tables/light_mode_and_site_check"))
    args = parser.parse_args()
    OUT = args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.features)

    # A1: pooled model, scored per light mode
    saved = pd.read_csv(args.predictions).merge(d[["image_id", "image_type"]], on="image_id")
    rows = []
    for mode in ["all", "PPL", "XPL"]:
        s = saved if mode == "all" else saved[saved.image_type == mode]
        img, k, n = specimen_scores(s, s.predicted_label == "MF-E")
        rows.append(dict(analysis="A1 pooled model, scored per mode", mode=mode,
                         images=len(s), image_acc=round(img, 4), spec_correct=k, n_spec=n))
        print(rows[-1], flush=True)

    # A2: mode-specific models
    for mode in ["PPL", "XPL"]:
        s = d[d.image_type == mode].reset_index(drop=True)
        p = loso_proba(s, (s.label == "MF-E").astype(int))
        img, k, n = specimen_scores(s, p > 0.5)
        rows.append(dict(analysis="A2 trained and tested on one mode", mode=mode,
                         images=len(s), image_acc=round(img, 4), spec_correct=k, n_spec=n))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "light_mode.csv", index=False)

    # B: site confound within Class 1
    rows = []
    for mode in ["all", "PPL", "XPL"]:
        s = d[(d.label == "MF-B") & ((d.image_type == mode) | (mode == "all"))].reset_index(drop=True)
        is_kezi = (s.section == "Kezi").astype(int)
        p = loso_proba(s, is_kezi)
        sp = pd.DataFrame({"p": p, "kezi": is_kezi, "code": s.sample_code}).groupby("code").mean()
        auc = roc_auc_score(sp.kezi, sp.p)
        pval = mannwhitneyu(sp.p[sp.kezi == 1], sp.p[sp.kezi == 0], alternative="greater", method="exact").pvalue
        rows.append(dict(mode=mode, n_kezi=int(sp.kezi.sum()), n_dudan=int((1 - sp.kezi).sum()),
                         images=len(s), specimen_auc=round(auc, 3), exact_p_one_sided=round(pval, 4)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "site_auc.csv", index=False)
    print("saved", OUT)


if __name__ == "__main__":
    main()

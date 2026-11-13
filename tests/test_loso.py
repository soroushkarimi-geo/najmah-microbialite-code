"""Tests of the leave-one-specimen-out cross-validation and the specimen-level
summaries: no specimen may appear in both the training and the test set of a
fold, every image is predicted exactly once, and the Wilson interval and
majority vote behave as reported in the paper."""
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import compare_models
import train_xgboost_cv
from train_xgboost_cv import get_feature_columns, run_loso_cv


def make_synthetic(n_specimens=8, images_per_specimen=5, seed=0):
    """Build a small synthetic feature table with the real column layout.

    Specimens alternate between MF-B and MF-E; each feature is drawn from a
    normal distribution shifted by +1 (MF-B) or -1 (MF-E).

    Args:
        n_specimens: Number of specimens (``SD_100``, ``SD_101``, ...).
        images_per_specimen: Images per specimen.
        seed: Random seed.

    Returns:
        pandas.DataFrame: The metadata columns of the real table plus
        ``glcm_contrast_d1``, ``dino_000`` and ``lam_gabor_anisotropy``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_specimens):
        label = "MF-B" if s % 2 == 0 else "MF-E"
        shift = 1.0 if label == "MF-B" else -1.0
        for i in range(images_per_specimen):
            rows.append({
                "image_id": f"SD_{100 + s}_PPL_{i:02d}", "sample_code": f"SD_{100 + s}",
                "section": "Dudan", "image_type": "PPL", "source_path": "x.jpg", "label": label,
                "glcm_contrast_d1": rng.normal(shift, 1), "dino_000": rng.normal(shift, 1),
                "lam_gabor_anisotropy": rng.normal(shift, 1),
            })
    return pd.DataFrame(rows)


class RecordingXGB(xgb.XGBClassifier):
    """XGBoost classifier that records the row index of every training set."""

    fitted_indices: list = []

    def fit(self, X, y, **kwargs):
        """Store the index of ``X`` in ``fitted_indices``, then fit as usual."""
        RecordingXGB.fitted_indices.append(X.index.copy())
        return super().fit(X, y, **kwargs)


def test_feature_columns_exclude_metadata():
    """Only the feature columns are returned, in table order."""
    df = make_synthetic()
    assert get_feature_columns(df) == ["glcm_contrast_d1", "dino_000", "lam_gabor_anisotropy"]


def test_xgboost_loso_never_shares_a_specimen(monkeypatch):
    """Each XGBoost fold trains on exactly all other specimens, and every
    image is predicted once."""
    df = make_synthetic()
    RecordingXGB.fitted_indices = []
    monkeypatch.setattr(train_xgboost_cv.xgb, "XGBClassifier", RecordingXGB)
    le = LabelEncoder().fit(["MF-B", "MF-E"])
    params = {"n_estimators": 5, "max_depth": 2, "objective": "multi:softmax", "num_class": 2, "random_state": 0}

    pred = run_loso_cv(df, get_feature_columns(df), le, params)

    specimens = df["sample_code"].unique()
    assert len(RecordingXGB.fitted_indices) == len(specimens)
    for train_idx, held_out in zip(RecordingXGB.fitted_indices, specimens):
        train_specimens = set(df.loc[train_idx, "sample_code"])
        assert held_out not in train_specimens
        assert train_specimens == set(specimens) - {held_out}
    # every image predicted once
    assert sorted(pred["image_id"]) == sorted(df["image_id"])
    assert pred["image_id"].is_unique


def test_generic_loso_never_shares_a_specimen():
    """The generic LOSO loop never trains on the held-out specimen."""
    df = make_synthetic()
    seen = []

    class Recording(LogisticRegression):
        """Logistic regression that records the specimens it is trained on."""

        def fit(self, X, y, sample_weight=None):
            """Record the specimens in the training set, then fit."""
            seen.append(set(df.loc[X.index, "sample_code"]))
            return super().fit(X, y, sample_weight=sample_weight)

    le = LabelEncoder().fit(["MF-B", "MF-E"])
    pred = compare_models.run_loso_cv_generic(df, get_feature_columns(df), le, lambda: clone(Recording()))

    for train_specimens, held_out in zip(seen, df["sample_code"].unique()):
        assert held_out not in train_specimens
    assert pred["image_id"].is_unique and len(pred) == len(df)


def test_fold_skipped_when_class_absent_from_training():
    """A specimen whose class has no other specimen is not predicted."""
    df = make_synthetic(n_specimens=5)
    df.loc[df["sample_code"] == "SD_100", "label"] = "MF-G"  # the only MF-G specimen
    le = LabelEncoder().fit(sorted(df["label"].unique()))
    params = {"n_estimators": 3, "max_depth": 2, "objective": "multi:softmax", "num_class": 3, "random_state": 0}
    pred = run_loso_cv(df, get_feature_columns(df), le, params)
    assert "SD_100" not in set(pred["sample_code"])


def test_wilson_interval_matches_paper():
    """42 of 49 specimens gives the 95% interval 0.733-0.929."""
    from specimen_level_rerun import wilson
    lo, hi = wilson(42, 49)
    assert lo == pytest.approx(0.733, abs=5e-4)
    assert hi == pytest.approx(0.929, abs=5e-4)


def test_specimen_majority_vote():
    """Majority, minority and exact 50/50 splits give MF-B, MF-E and 'tie'."""
    from specimen_level_rerun import specimen_majority_vote
    pred = pd.DataFrame({
        "sample_code": ["A"] * 3 + ["B"] * 2 + ["C"] * 2,
        "label": ["MF-B"] * 3 + ["MF-E"] * 2 + ["MF-B"] * 2,
        "predicted_label": ["MF-B", "MF-B", "MF-E", "MF-E", "MF-E", "MF-B", "MF-E"],
    })
    g = specimen_majority_vote(pred)
    assert g.loc["A", "spec_pred"] == "MF-B"
    assert g.loc["B", "spec_pred"] == "MF-E"
    assert g.loc["C", "spec_pred"] == "tie"

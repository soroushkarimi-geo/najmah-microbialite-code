"""Tests of the stratigraphic order tests (Table 5), the height calibration
and the Lomb-Scargle statistic used in the power simulation (Table 4)."""
import numpy as np
import pandas as pd
import pytest

from shallowing_index_changepoint_test import assign_segment, mean_shift_permutation_test
from shallowing_index_cyclicity_analysis import build_section_df, count_turning_points
from stratigraphic_power_analysis import (CALIB, ls_stat, runs_test, sample_number,
                                          sample_to_thickness)


def test_runs_test_expected_runs_for_20_and_24():
    """20 and 24 specimens give 22.8 expected runs; two blocks are significant."""
    # 20 Class 1 and 24 Class 2 specimens, like Dudan
    seq = np.array([1] * 20 + [0] * 24)
    result = runs_test(seq)
    assert result["n1"] == 20 and result["n0"] == 24
    assert result["expected"] == pytest.approx(1 + 2 * 20 * 24 / 44)
    assert round(result["expected"], 1) == 22.8
    assert result["runs"] == 2   # all clustered
    assert result["p_two_sided"] < 1e-6


def test_runs_test_alternating_and_independent_of_order_labels():
    """Strict alternation gives z > 0, and swapping 0 and 1 changes nothing."""
    alt = np.array([0, 1] * 10)
    r = runs_test(alt)
    assert r["runs"] == 20
    assert r["z"] > 0   # more runs than expected
    assert runs_test(1 - alt)["expected"] == pytest.approx(r["expected"])


def test_count_turning_points():
    """Monotonic, zig-zag, plateau and two-point sequences."""
    assert count_turning_points(np.array([1, 2, 3, 4])) == 0
    assert count_turning_points(np.array([1, 3, 2, 4, 0])) == 3
    assert count_turning_points(np.array([1, 2, 2, 1])) == 1   # tie is not a turn
    assert count_turning_points(np.array([5.0, 1.0])) == 0


def test_mean_shift_permutation_test_detects_large_shift():
    """A shift of 5 sd is detected; identical groups are not."""
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20)
    res = mean_shift_permutation_test(a, a + 5, 2000, rng)
    assert res["p_value_two_sided"] < 0.01
    null = mean_shift_permutation_test(a, a.copy(), 2000, rng)
    assert null["p_value_two_sided"] > 0.5


def test_assign_segment_boundaries():
    """Lower bounds belong to the sequence above; numbers past TR6 get None."""
    assert assign_segment(91) == "TR4"
    assert assign_segment(92) == "TR5"
    assert assign_segment(122) == "TR6"
    assert assign_segment(200) is None


def test_sample_to_thickness_hits_calibration_points_and_is_monotonic():
    """Heights reproduce every calibration point and never decrease upward."""
    for t, n in CALIB:
        assert sample_to_thickness(n) == pytest.approx(t)
    heights = [sample_to_thickness(n) for n in np.arange(4, 146, 0.5)]
    assert np.all(np.diff(heights) >= 0)


def test_sample_number_b_suffix_sits_above():
    """SD_102B is placed half a sample above SD_102."""
    assert sample_number("SD_102") == 102
    assert sample_number("SD_102B") == 102.5


def test_build_section_df_orders_b_suffix_after_base_sample():
    """SD_102 comes before SD_102B, and other sections are excluded."""
    rows = []
    for code, label, val in [("SD_103", "MF-B", 1.0), ("SD_102B", "MF-E", 2.0),
                             ("SD_102", "MF-B", 3.0), ("SK_84", "MF-B", 4.0)]:
        rows += [{"sample_code": code, "label": label, "lam_gabor_anisotropy": val}] * 2
    out = build_section_df(pd.DataFrame(rows), "SD")
    assert list(out["sample_code"]) == ["SD_102", "SD_102B", "SD_103"]


def test_lomb_scargle_finds_injected_cycle():
    """A 30 m sine on top of noise has more Lomb-Scargle power than the noise."""
    z = np.linspace(0, 100, 60)
    periods = np.linspace(6, 90, 200)
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, len(z))
    strong = 3 * np.sin(2 * np.pi * z / 30) + noise
    assert ls_stat(z, strong, periods) > ls_stat(z, noise, periods)

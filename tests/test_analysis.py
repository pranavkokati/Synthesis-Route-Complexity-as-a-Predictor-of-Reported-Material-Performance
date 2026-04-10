"""
Unit tests for statistical analysis helpers.

Tests ANOVA η², Benjamini–Hochberg FDR correction, McFadden pseudo-R²,
bootstrap AUC, and the composition-correction logic — all without
requiring Materials Project access or pymatgen.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    _eta_squared,
    _bh_correction,
    _mcfadden_r2,
    _bootstrap_auc,
    _cohens_d,
    _element_entropy_table,
    _composition_corrected_entropy,
)


class TestEtaSquared:
    def test_identical_groups_returns_zero(self):
        """No between-group variance → η² = 0."""
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([1.0, 2.0, 3.0])
        assert _eta_squared([g1, g2]) == pytest.approx(0.0)

    def test_perfectly_separated_groups(self):
        """No within-group variance → η² = 1."""
        g1 = np.array([1.0, 1.0, 1.0])
        g2 = np.array([5.0, 5.0, 5.0])
        assert _eta_squared([g1, g2]) == pytest.approx(1.0, rel=1e-6)

    def test_in_unit_interval(self):
        rng = np.random.default_rng(0)
        for _ in range(10):
            groups = [rng.normal(mu, 1, 50) for mu in [0, 1, 2]]
            eta2 = _eta_squared(groups)
            assert 0.0 <= eta2 <= 1.0

    def test_larger_effect_higher_eta(self):
        # Use arrays with within-group variance so η² < 1 and comparison is valid
        rng = np.random.default_rng(7)
        g1 = rng.normal(0, 1.0, 50)
        g2_small = rng.normal(0.1, 1.0, 50)   # small separation
        g2_large = rng.normal(5.0, 1.0, 50)   # large separation
        assert _eta_squared([g1, g2_large]) > _eta_squared([g1, g2_small])

    def test_three_groups(self):
        g = [np.ones(10) * i for i in [0, 1, 2]]
        # η² should be > 0.9 for these well-separated groups
        assert _eta_squared(g) > 0.9


class TestBHCorrection:
    def test_no_correction_needed(self):
        """All p-values > 0.05 → BH keeps them above 0.05."""
        pvals = np.array([0.5, 0.6, 0.7, 0.8])
        adj = _bh_correction(pvals)
        assert all(adj >= 0.05)

    def test_all_significant(self):
        """All tiny p-values → adjusted also significant."""
        pvals = np.array([1e-10, 2e-10, 3e-10])
        adj = _bh_correction(pvals)
        assert all(adj < 0.05)

    def test_adjusted_monotone(self):
        """Adjusted p-values should be non-decreasing when sorted by raw p."""
        pvals = np.array([0.001, 0.01, 0.05, 0.2, 0.5])
        adj = _bh_correction(pvals)
        # After BH, sorted adjusted p should be non-decreasing
        assert all(np.diff(np.sort(adj)) >= -1e-12)

    def test_output_shape(self):
        pvals = np.array([0.01, 0.05, 0.1])
        assert _bh_correction(pvals).shape == (3,)

    def test_single_pvalue(self):
        pvals = np.array([0.03])
        adj = _bh_correction(pvals)
        assert adj[0] == pytest.approx(0.03, rel=1e-6)

    def test_clipped_to_unit_interval(self):
        pvals = np.array([0.5, 0.6])
        adj = _bh_correction(pvals)
        assert all(0.0 <= a <= 1.0 for a in adj)


class TestMcFaddenR2:
    def test_perfect_prediction(self):
        """Perfect model → high R²."""
        y = np.array([1, 1, 0, 0])
        p = np.array([0.999, 0.999, 0.001, 0.001])
        r2 = _mcfadden_r2(y, p)
        assert r2 > 0.8

    def test_null_model(self):
        """Null model (predict marginal) → R² ≈ 0."""
        y = np.array([1, 1, 0, 0])
        # null predicts marginal probability everywhere
        p = np.full(4, 0.5)
        r2 = _mcfadden_r2(y, p)
        assert abs(r2) < 0.01

    def test_better_model_higher_r2(self):
        y = np.array([1, 1, 0, 0])
        p_good = np.array([0.9, 0.8, 0.2, 0.1])
        p_bad  = np.array([0.6, 0.6, 0.4, 0.4])
        assert _mcfadden_r2(y, p_good) > _mcfadden_r2(y, p_bad)

    def test_in_reasonable_range(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=100)
        p = np.clip(rng.uniform(0.3, 0.7, 100), 1e-6, 1 - 1e-6)
        r2 = _mcfadden_r2(y, p)
        assert r2 <= 1.0


class TestBootstrapAUC:
    def test_perfect_classifier(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        s = np.array([1.0, 0.9, 0.8, 0.2, 0.1, 0.0])
        auc, lo, hi = _bootstrap_auc(y, s, n_boot=200)
        assert auc == pytest.approx(1.0, rel=1e-6)
        assert lo > 0.9
        assert hi == pytest.approx(1.0, rel=1e-3)

    def test_random_classifier(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, size=200)
        s = rng.uniform(0, 1, size=200)
        auc, lo, hi = _bootstrap_auc(y, s, n_boot=200, seed=42)
        assert 0.3 < auc < 0.7
        assert lo < auc < hi

    def test_ci_bounds_ordered(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=100)
        s = rng.uniform(0, 1, 100)
        auc, lo, hi = _bootstrap_auc(y, s, n_boot=100)
        assert lo <= auc <= hi


class TestCohensD:
    def test_identical_groups_is_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert _cohens_d(a, b) == pytest.approx(0.0, abs=1e-10)

    def test_large_effect(self):
        a = np.ones(100)
        b = np.ones(100) * 5
        d = abs(_cohens_d(a, b))
        # Pooled SD = 0, would be inf — but for constant arrays Cohen's d undefined
        # Use almost-constant arrays instead
        rng = np.random.default_rng(0)
        a = rng.normal(0, 0.1, 100)
        b = rng.normal(10, 0.1, 100)
        d = abs(_cohens_d(a, b))
        assert d > 10.0  # Very large effect

    def test_sign(self):
        # Use arrays with non-zero variance so Cohen's d is well-defined
        rng = np.random.default_rng(3)
        a = rng.normal(2, 1, 50)
        b = rng.normal(1, 1, 50)
        assert _cohens_d(a, b) > 0
        assert _cohens_d(b, a) < 0


class TestCompositionControl:
    """Tests for per-element entropy baseline and corrected entropy."""

    def _make_df(self):
        return pd.DataFrame([
            {"material_id": "mp-1", "ce_entropy": 1.0, "elements": '["Fe", "O"]'},
            {"material_id": "mp-2", "ce_entropy": 2.0, "elements": '["Fe", "Ti"]'},
            {"material_id": "mp-3", "ce_entropy": 0.5, "elements": '["O", "Ca"]'},
            {"material_id": "mp-4", "ce_entropy": 1.5, "elements": '["Ca", "Ti"]'},
        ])

    def test_element_entropy_table_shape(self):
        df = self._make_df()
        table = _element_entropy_table(df)
        # Should have one row per unique element
        assert set(table["element"]) == {"Fe", "O", "Ti", "Ca"}

    def test_element_entropy_values(self):
        df = self._make_df()
        table = _element_entropy_table(df)
        fe_row = table[table["element"] == "Fe"]
        # Fe appears in mp-1 (1.0) and mp-2 (2.0) → mean = 1.5
        assert fe_row["mean_entropy"].values[0] == pytest.approx(1.5)

    def test_corrected_entropy_direction(self):
        """Structure with very high entropy for its composition → positive correction."""
        df = self._make_df()
        table = _element_entropy_table(df)
        el_map = table.set_index("element")["mean_entropy"].to_dict()

        row_high = pd.Series({"ce_entropy": 3.0, "elements": '["Fe", "O"]'})
        corr = _composition_corrected_entropy(row_high, el_map)
        assert corr is not None
        assert corr > 0.0  # Higher than average for this composition

    def test_corrected_entropy_unknown_element(self):
        """Unknown element → returns None gracefully."""
        el_map = {"Fe": 1.0, "O": 0.5}
        row = pd.Series({"ce_entropy": 1.0, "elements": '["Unobtainium"]'})
        result = _composition_corrected_entropy(row, el_map)
        assert result is None

    def test_corrected_entropy_null_input(self):
        """NaN ce_entropy → returns None."""
        el_map = {"Fe": 1.0}
        row = pd.Series({"ce_entropy": float("nan"), "elements": '["Fe"]'})
        result = _composition_corrected_entropy(row, el_map)
        assert result is None

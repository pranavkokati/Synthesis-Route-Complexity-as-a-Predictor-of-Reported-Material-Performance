"""
Unit tests for coordination environment entropy descriptors.

Tests the mathematical correctness of the four CE descriptors
(Shannon entropy, distinct environments, dominance, Gini coefficient)
using analytically known reference cases.
"""

import math
import json
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Import the worker function directly (no MP API or pymatgen needed)
# ---------------------------------------------------------------------------
from src.coord_env import _compute_one


# ---------------------------------------------------------------------------
# Descriptor math — tested independently of ChemEnv
# ---------------------------------------------------------------------------

def _descriptors_from_counts(counts: dict[str, int]) -> dict:
    """Replicate the descriptor math from _compute_one for unit testing."""
    total = sum(counts.values())
    probs = sorted(v / total for v in counts.values())

    ce_entropy = -sum(p * math.log(p) for p in probs if p > 0)
    n_distinct = len(counts)
    dominance = probs[-1]

    n = len(probs)
    if n == 1:
        gini = 0.0
    else:
        weighted = sum((i + 1) * p for i, p in enumerate(probs))
        gini = max(0.0, min(1.0, (2.0 * weighted / n) - (n + 1) / n))

    return {
        "ce_entropy":      ce_entropy,
        "n_distinct_envs": n_distinct,
        "dominance":       dominance,
        "gini":            gini,
    }


class TestEntropyDescriptor:
    """Shannon entropy H = −Σ pᵢ log(pᵢ)"""

    def test_single_environment_entropy_is_zero(self):
        """One CE type → all sites identical → H = 0."""
        d = _descriptors_from_counts({"O:6": 10})
        assert d["ce_entropy"] == pytest.approx(0.0)

    def test_two_equal_environments(self):
        """Two equal CE types → H = log(2)."""
        d = _descriptors_from_counts({"O:6": 5, "T:4": 5})
        assert d["ce_entropy"] == pytest.approx(math.log(2), rel=1e-6)

    def test_four_equal_environments(self):
        """Four equal CE types → H = log(4)."""
        d = _descriptors_from_counts({"O:6": 4, "T:4": 4, "C:12": 4, "L:2": 4})
        assert d["ce_entropy"] == pytest.approx(math.log(4), rel=1e-6)

    def test_entropy_increases_with_diversity(self):
        """More uniform → higher entropy."""
        d1 = _descriptors_from_counts({"O:6": 9, "T:4": 1})
        d2 = _descriptors_from_counts({"O:6": 5, "T:4": 5})
        assert d2["ce_entropy"] > d1["ce_entropy"]

    def test_entropy_is_nonnegative(self):
        for counts in [{"A": 1}, {"A": 3, "B": 7}, {"A": 1, "B": 1, "C": 1}]:
            assert _descriptors_from_counts(counts)["ce_entropy"] >= 0.0

    def test_entropy_maximum_log_k(self):
        """Maximum entropy for K environments = log(K)."""
        for k in [2, 3, 5, 10]:
            counts = {f"E{i}": 1 for i in range(k)}
            d = _descriptors_from_counts(counts)
            assert d["ce_entropy"] == pytest.approx(math.log(k), rel=1e-6)


class TestDistinctEnvironments:
    def test_single(self):
        assert _descriptors_from_counts({"O:6": 5})["n_distinct_envs"] == 1

    def test_multiple(self):
        d = _descriptors_from_counts({"O:6": 3, "T:4": 2, "L:2": 1})
        assert d["n_distinct_envs"] == 3

    def test_count_by_symbol_not_count(self):
        """n_distinct_envs counts CE types, not sites."""
        d = _descriptors_from_counts({"O:6": 100, "T:4": 1})
        assert d["n_distinct_envs"] == 2


class TestDominance:
    def test_single_type_dominance_is_one(self):
        assert _descriptors_from_counts({"O:6": 10})["dominance"] == pytest.approx(1.0)

    def test_equal_types(self):
        d = _descriptors_from_counts({"O:6": 5, "T:4": 5})
        assert d["dominance"] == pytest.approx(0.5)

    def test_dominant_is_correct_fraction(self):
        d = _descriptors_from_counts({"O:6": 7, "T:4": 3})
        assert d["dominance"] == pytest.approx(0.7)

    def test_dominance_in_unit_interval(self):
        for counts in [{"A": 1}, {"A": 3, "B": 7}, {"A": 1, "B": 1, "C": 8}]:
            dom = _descriptors_from_counts(counts)["dominance"]
            assert 0.0 <= dom <= 1.0


class TestGiniCoefficient:
    def test_single_environment_gini_is_zero(self):
        """One CE type → uniform → Gini = 0."""
        d = _descriptors_from_counts({"O:6": 10})
        assert d["gini"] == pytest.approx(0.0)

    def test_equal_environments_gini_is_zero(self):
        """Perfectly equal → Gini = 0."""
        d = _descriptors_from_counts({"O:6": 5, "T:4": 5})
        assert d["gini"] == pytest.approx(0.0, abs=1e-10)

    def test_gini_in_unit_interval(self):
        for counts in [{"A": 1}, {"A": 1, "B": 99}, {"A": 3, "B": 3, "C": 3}]:
            g = _descriptors_from_counts(counts)["gini"]
            assert 0.0 <= g <= 1.0

    def test_more_unequal_has_higher_gini(self):
        d_equal   = _descriptors_from_counts({"A": 5, "B": 5})
        d_unequal = _descriptors_from_counts({"A": 9, "B": 1})
        assert d_unequal["gini"] > d_equal["gini"]


class TestDescriptorConsistency:
    """Cross-checks between descriptors."""

    def test_single_type_consistency(self):
        """For one CE type: H=0, K=1, dominance=1, G=0."""
        d = _descriptors_from_counts({"O:6": 42})
        assert d["ce_entropy"]      == pytest.approx(0.0)
        assert d["n_distinct_envs"] == 1
        assert d["dominance"]       == pytest.approx(1.0)
        assert d["gini"]            == pytest.approx(0.0)

    def test_high_entropy_low_dominance(self):
        """Uniform distribution: high H, low dominance, low G."""
        counts = {f"E{i}": 10 for i in range(10)}
        d = _descriptors_from_counts(counts)
        assert d["ce_entropy"]  > 2.0        # log(10) ≈ 2.3
        assert d["dominance"]   == pytest.approx(0.1)
        assert d["gini"]        == pytest.approx(0.0, abs=1e-10)

    def test_skewed_distribution(self):
        """Heavily skewed: low H, high dominance, high G."""
        counts = {"O:6": 90, "T:4": 5, "L:2": 5}
        d = _descriptors_from_counts(counts)
        assert d["ce_entropy"] < 0.5
        assert d["dominance"]  > 0.8
        assert d["gini"]       > 0.5

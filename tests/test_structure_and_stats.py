"""Tests for structure scoring, the Proposition 2 cell, statistics and gates."""

import numpy as np
import pytest

from topocircuit.eval.metrics import (
    check_gate,
    compute_minimum_detectable_effect,
    holm_bonferroni_correction,
    paired_wilcoxon_and_bootstrap,
)


def test_exact_parent_set_f1_counts_whole_sets_only():
    pytest.importorskip("torch")
    from topocircuit.eval.structure import exact_parent_set_f1

    true = {2: [0, 1], 3: [0], 4: []}
    assert exact_parent_set_f1(true, true)["parent_set_f1"] == 1.0
    # One of two parents found is not an exact match.
    res = exact_parent_set_f1({2: [0], 3: [0]}, true)
    assert res["exact_set_precision"] == 0.5
    assert res["exact_set_recall"] == 0.5
    assert exact_parent_set_f1({}, true)["parent_set_f1"] == 0.0


def test_hungarian_matching_recovers_a_permutation():
    pytest.importorskip("torch")
    from topocircuit.eval.structure import match_latents

    rng = np.random.default_rng(0)
    Z = (rng.random((400, 6)) > 0.7) * rng.random((400, 6))
    perm = np.array([3, 0, 5, 1, 4, 2])
    mapping, corr = match_latents(Z[:, perm], Z)
    assert all(perm[mapping[t]] == t for t in range(6))
    assert np.allclose(corr, 1.0)


def test_prop2_cell_runs_and_reports_both_parents():
    pytest.importorskip("torch")
    from topocircuit.eval.prop2 import run_cell

    cell = run_cell(0.01, 0.4, 0.2, seed=0, n_tokens=600, n_steps=150)
    for key in ("abs_p1", "abs_p2", "same_latent"):
        assert 0.0 <= cell[key] <= 1.0


def test_wilcoxon_detects_a_clear_reduction():
    rng = np.random.default_rng(0)
    b = rng.uniform(0.3, 0.5, 60)
    a = b * 0.5
    res = paired_wilcoxon_and_bootstrap(a, b, n_boot=500)
    assert res["p_value_raw"] < 1e-6
    assert res["ci_95_low"] > 0.4


def test_holm_is_step_down():
    out = holm_bonferroni_correction({"H1": 0.001, "H2": 0.04, "H3": 0.03})
    assert out["H1"]["significant"] == 1.0
    assert out["H3"]["significant"] == 0.0  # 0.03 > 0.05 / 2
    assert out["H2"]["significant"] == 0.0  # step-down stops after the first failure


def test_mde_shrinks_with_more_samples():
    spread = [0.1, 0.12, 0.09, 0.11]
    assert compute_minimum_detectable_effect(spread, 100) < compute_minimum_detectable_effect(spread, 10)


def test_gate_has_no_fallback_values():
    res = check_gate("G3", {"f1": 0.85, "hub": 0.01}, {"f1": (">=", 0.8), "hub": ("<=", 0.02)})
    assert res["passed"]
    assert not check_gate("G3", {"f1": 0.5}, {"f1": (">=", 0.8)})["passed"]
    with pytest.raises(KeyError):
        check_gate("G3", {}, {"f1": (">=", 0.8)})

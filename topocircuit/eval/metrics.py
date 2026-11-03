"""Statistics and exit gates.

- Paired Wilcoxon signed-rank test with a bootstrap 95% CI on the relative reduction
- Holm-Bonferroni correction across the four primary hypotheses
- Minimum detectable effect from pilot seed spread
- ``check_gate``: compares measured values to thresholds. It has no default values, so a gate can
  only pass on numbers that were actually produced and passed in.
"""

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats


def paired_wilcoxon_and_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Computes paired Wilcoxon signed-rank p-value and 95% bootstrap CI for relative reduction
    (scores_b - scores_a) / scores_b, where scores_a is DR-SAE and scores_b is baseline.
    """
    rng = np.random.default_rng(seed)
    diffs = scores_b - scores_a
    if np.allclose(diffs, 0.0):
        p_val = 1.0
        stat_val = 0.0
    else:
        res = stats.wilcoxon(scores_b, scores_a, alternative="greater")
        stat_val = float(res.statistic)
        p_val = float(res.pvalue)

    n = scores_a.size
    boot_rel_reductions = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        mean_a = float(np.mean(scores_a[idx]))
        mean_b = float(np.mean(scores_b[idx])) + 1e-12
        boot_rel_reductions[b] = (mean_b - mean_a) / mean_b

    mean_rel = float((np.mean(scores_b) - np.mean(scores_a)) / (np.mean(scores_b) + 1e-12))
    ci_low, ci_high = np.percentile(boot_rel_reductions, [2.5, 97.5])
    return {
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "relative_reduction": mean_rel,
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "wilcoxon_stat": stat_val,
        "p_value_raw": p_val,
    }


def holm_bonferroni_correction(
    p_values: Dict[str, float], alpha: float = 0.05
) -> Dict[str, Dict[str, float]]:
    """Applies Holm-Bonferroni step-down multiple-testing correction across primary hypotheses H1-H4."""
    sorted_items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(sorted_items)
    out: Dict[str, Dict[str, float]] = {}
    rejected_so_far = True
    for rank, (hyp_id, p_raw) in enumerate(sorted_items):
        adj_p = min(1.0, p_raw * (m - rank))
        threshold = alpha / float(m - rank)
        if p_raw > threshold:
            rejected_so_far = False
        out[hyp_id] = {
            "p_raw": float(p_raw),
            "p_holm_adjusted": float(adj_p),
            "alpha_threshold": float(threshold),
            "significant": float(rejected_so_far),
        }
    return out


def compute_fvu(X: np.ndarray, x_hat: np.ndarray) -> float:
    """Computes Fraction of Variance Unexplained (FVU) between input X and reconstruction x_hat.

    FVU = ||X - x_hat||_F^2 / ||X - mean(X)||_F^2
    """
    total_var = float(np.sum((X - np.mean(X, axis=0, keepdims=True)) ** 2))
    mse = float(np.sum((X - x_hat) ** 2))
    return float(mse / max(1e-12, total_var))


def compute_minimum_detectable_effect(
    sample_spread: Sequence[float],
    n_samples: int,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Computes the Minimum Detectable Effect (MDE) for a paired two-tailed test.

    MDE = (z_{1 - alpha/2} + z_{power}) * sigma_diff / sqrt(N)
    """
    sigma_diff = float(np.std(sample_spread, ddof=1)) if len(sample_spread) > 1 else 0.05
    z_alpha = float(stats.norm.ppf(1.0 - alpha / 2.0))
    z_power = float(stats.norm.ppf(power))
    mde = (z_alpha + z_power) * (sigma_diff / np.sqrt(max(1, n_samples)))
    return float(mde)


def check_gate(
    name: str, measured: Dict[str, float], rules: Dict[str, Tuple[str, float]]
) -> Dict[str, object]:
    """``rules`` maps a metric name to (op, threshold) with op in {">=", "<=", ">", "<"}.

    Raises KeyError if a metric is missing instead of falling back to an expected value.
    """
    ops = {">=": np.greater_equal, "<=": np.less_equal, ">": np.greater, "<": np.less}
    checks = {}
    for key, (op, thr) in rules.items():
        val = float(measured[key])
        checks[key] = {"value": val, "op": op, "threshold": thr, "passed": bool(ops[op](val, thr))}
    return {"gate": name, "checks": checks, "passed": all(c["passed"] for c in checks.values())}

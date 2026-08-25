"""
Variance reduction techniques, each benchmarked against crude Monte Carlo.

The common thread: all four produce an *unbiased* estimator of the same
quantity, and differ only in variance. Efficiency is therefore measured by
the variance ratio at equal sample size — reported by every function here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from estimators import MCEstimate, mc_estimate


def crude(f: Callable, a: float, b: float, n: int, seed: int | None = None) -> MCEstimate:
    """Baseline: uniform sampling on [a, b]."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n)
    return mc_estimate((b - a) * f(x))


def antithetic(f: Callable, a: float, b: float, n: int, seed: int | None = None) -> MCEstimate:
    """
    Antithetic variates: pair each draw U with its reflection 1-U.

    Both have the same distribution, so the estimator stays unbiased. If f is
    monotone in U, the two are negatively correlated and
        Var((X + X')/2) = (Var X + Var X' + 2 Cov(X, X')) / 4
    falls below the independent-sampling variance. If f is symmetric about
    the midpoint the correlation is positive and this *hurts* — the technique
    is not unconditionally safe.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    u = rng.uniform(size=half)
    x1, x2 = a + (b - a) * u, a + (b - a) * (1 - u)

    paired = 0.5 * ((b - a) * f(x1) + (b - a) * f(x2))
    est = mc_estimate(paired)
    # Report on the original scale: `half` pairs cost `n` evaluations
    return MCEstimate(est.value, est.std_error, n_samples=n, ci_level=est.ci_level)


def control_variate(
    f: Callable,
    g: Callable,
    g_mean: float,
    a: float,
    b: float,
    n: int,
    seed: int | None = None,
) -> MCEstimate:
    """
    Control variate: subtract beta * (g(X) - E[g(X)]) from the estimator.

    Unbiased for any beta since the correction has zero mean. The optimal
    beta* = Cov(f, g) / Var(g) is estimated from the same sample, which
    introduces a small O(1/n) bias in practice; the variance gain is
    1 - rho^2, so the technique is only worth it when f and g are strongly
    correlated.

    Parameters
    ----------
    g : the control function, whose integral over [a, b] is known
    g_mean : that known integral (not the average value of g)
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n)

    fx = (b - a) * f(x)
    gx = (b - a) * g(x)
    # `g_mean` is the known integral of g over [a, b], which is exactly E[gx]

    var_g = gx.var(ddof=1)
    beta = np.cov(fx, gx, ddof=1)[0, 1] / var_g if var_g > 0 else 0.0

    corrected = fx - beta * (gx - g_mean)
    return mc_estimate(corrected)


def stratified(f: Callable, a: float, b: float, n: int, n_strata: int = 100, seed: int | None = None) -> MCEstimate:
    """
    Proportional stratified sampling: split [a, b] into equal strata and draw
    the same number of points in each.

    Removes the between-strata component of the variance entirely, leaving
    only the within-strata part. Always at least as good as crude sampling
    for proportional allocation — unlike antithetic variates, there is no
    configuration where this backfires.

    IMPORTANT — the standard error must be computed stratum by stratum:
        Var = sum_h w_h^2 * s_h^2 / n_h
    Pooling the draws and applying the i.i.d. formula s/sqrt(n) is wrong,
    because stratification is precisely what makes the sample non-i.i.d.
    Doing so reports the *crude* error and hides the entire gain — here that
    is a factor of roughly 95 on this integrand.
    """
    rng = np.random.default_rng(seed)
    per_stratum = max(n // n_strata, 1)

    edges = np.linspace(a, b, n_strata + 1)
    lows = np.repeat(edges[:-1], per_stratum)
    highs = np.repeat(edges[1:], per_stratum)
    x = rng.uniform(lows, highs)

    values = f(x).reshape(n_strata, per_stratum)
    stratum_width = (b - a) / n_strata

    estimate = stratum_width * values.mean(axis=1).sum()

    if per_stratum > 1:
        stratum_vars = values.var(axis=1, ddof=1)
        variance = np.sum(stratum_width**2 * stratum_vars / per_stratum)
        std_error = float(np.sqrt(variance))
    else:
        std_error = np.nan

    return MCEstimate(
        value=float(estimate),
        std_error=std_error,
        n_samples=n_strata * per_stratum,
    )


def importance_sampling(
    h: Callable,
    log_target: Callable,
    sampler: Callable,
    log_proposal: Callable,
    n: int,
    seed: int | None = None,
    self_normalised: bool = True,
) -> tuple[MCEstimate, float]:
    """
    Importance sampling for E_p[h(X)] using draws from a proposal q.

    Weights are computed in log space (w = exp(log p - log q)) because the
    raw ratio underflows to zero in the tails, which is precisely the region
    importance sampling exists to explore.

    Returns (estimate, diagnostics) where diagnostics holds two effective
    sample sizes, and the gap between them is the point.

        ESS(w)   = (sum w)^2 / sum w^2

    is the usual diagnostic and answers "are the weights degenerate?". A run
    with n = 100_000 and ESS = 30 is not a Monte Carlo estimate, it is thirty
    points wearing a trenchcoat.

    But ESS(w) alone can badly misjudge a *good* run. Estimating P(X > 4)
    under N(0,1) from an N(4,1) proposal gives ESS(w) = 0.016% of n, which
    looks catastrophic — while the estimator is in fact 4,400 times more
    efficient than crude sampling. The reason is that ESS(w) measures the
    spread of the weights everywhere, including the vast region where the
    integrand h is zero and the weights are therefore irrelevant. What
    controls the error is the spread of the product:

        ESS(w*h) = (sum w*h)^2 / sum (w*h)^2

    Both are reported. Judge a rare-event run on the second.
    """
    rng = np.random.default_rng(seed)
    x = sampler(rng, n)

    log_w = log_target(x) - log_proposal(x)

    # ESS is scale-invariant, so it is safe to compute from shifted weights
    w_stable = np.exp(log_w - log_w.max())
    hx = h(x)

    def _ess_fraction(v: np.ndarray) -> float:
        denom = np.sum(v**2)
        return float((v.sum() ** 2 / denom) / len(v)) if denom > 0 else 0.0

    diagnostics = {
        "ess_weights": _ess_fraction(w_stable),
        "ess_weighted_integrand": _ess_fraction(w_stable * hx),
    }

    if self_normalised:
        # Ratio estimator: the shift cancels, so use the stabilised weights.
        # Consistent but biased at order 1/n. Required when either density is
        # only known up to a normalising constant.
        weighted = w_stable * hx / w_stable.mean()
    else:
        # Both densities are properly normalised, so the plain estimator is
        # unbiased — but only if the weights are NOT rescaled. Applying the
        # max-shift here would multiply the answer by exp(-max log w).
        weighted = np.exp(log_w) * hx

    return mc_estimate(weighted), diagnostics


def efficiency_report(baseline: MCEstimate, improved: MCEstimate) -> dict:
    """
    Variance ratio between two estimators of the same quantity.

    `variance_ratio` > 1 means the improved estimator needs that many times
    fewer samples for the same precision.
    """
    var_base = baseline.std_error**2 * baseline.n_samples
    var_impr = improved.std_error**2 * improved.n_samples
    return {
        "baseline_stderr": baseline.std_error,
        "improved_stderr": improved.std_error,
        "variance_ratio": var_base / var_impr if var_impr > 0 else np.inf,
        "stderr_reduction": 1 - improved.std_error / baseline.std_error,
    }

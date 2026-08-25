"""
Run with: pytest tests/ -v

The tests that carry the actual content are `test_ess_detects_correlation`
and `test_importance_sampling_beats_crude_in_the_tail` — they check that the
diagnostics catch the failure modes they exist for.
"""

import os
import sys

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimators import mc_estimate, mc_integrate, sample_size_for_precision
from variance_reduction import crude, antithetic, control_variate, stratified, importance_sampling, efficiency_report
from mcmc import (
    metropolis_hastings,
    gibbs_bivariate_normal,
    effective_sample_size,
    autocorrelation,
    gelman_rubin,
)
from annealing import random_tsp_instance, solve_tsp, tour_length, geometric_schedule


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------


def test_ci_covers_truth_at_nominal_rate():
    """Across many independent runs, ~95% of 95% CIs should contain the truth."""
    rng = np.random.default_rng(0)
    truth = 0.5
    covered = 0
    n_trials = 400

    for _ in range(n_trials):
        samples = rng.uniform(size=500)
        est = mc_estimate(samples)
        lo, hi = est.ci
        covered += lo <= truth <= hi

    coverage = covered / n_trials
    assert 0.90 < coverage < 0.99, f"coverage was {coverage:.2%}"


def test_error_decays_as_inverse_sqrt_n():
    """Multiplying n by 100 should divide the standard error by about 10."""
    rng = np.random.default_rng(1)
    small = mc_estimate(rng.standard_normal(1_000))
    large = mc_estimate(rng.standard_normal(100_000))
    ratio = small.std_error / large.std_error
    assert 7 < ratio < 14


def test_sample_size_formula_is_consistent():
    rng = np.random.default_rng(2)
    pilot = rng.standard_normal(1_000)
    n_needed = sample_size_for_precision(pilot, target_half_width=0.01)

    achieved = mc_estimate(rng.standard_normal(n_needed))
    lo, hi = achieved.ci
    assert (hi - lo) / 2 < 0.012


# --------------------------------------------------------------------------
# Variance reduction
# --------------------------------------------------------------------------


def test_all_estimators_agree_on_the_same_integral():
    """Every technique is unbiased: they must agree within their error bars."""
    f = lambda x: np.exp(x)
    truth = np.e - 1  # integral of exp over [0, 1]

    for est in [
        crude(f, 0, 1, 200_000, seed=3),
        antithetic(f, 0, 1, 200_000, seed=3),
        stratified(f, 0, 1, 200_000, n_strata=200, seed=3),
    ]:
        assert abs(est.value - truth) < 5 * est.std_error


def test_antithetic_helps_on_monotone_integrand():
    f = lambda x: np.exp(x)
    base = crude(f, 0, 1, 100_000, seed=4)
    anti = antithetic(f, 0, 1, 100_000, seed=4)
    assert anti.std_error < base.std_error


def test_antithetic_can_backfire_on_symmetric_integrand():
    """
    The technique is not unconditionally safe: for an integrand symmetric
    about the midpoint, f(U) and f(1-U) are positively correlated and the
    variance goes UP.
    """
    f = lambda x: (x - 0.5) ** 2
    base = crude(f, 0, 1, 100_000, seed=5)
    anti = antithetic(f, 0, 1, 100_000, seed=5)
    assert anti.std_error > base.std_error


def test_control_variate_reduces_variance():
    f = lambda x: np.exp(x)
    g = lambda x: 1 + x  # correlated with exp(x), integral over [0,1] is 1.5
    base = crude(f, 0, 1, 100_000, seed=6)
    cv = control_variate(f, g, g_mean=1.5, a=0, b=1, n=100_000, seed=6)

    report = efficiency_report(base, cv)
    assert report["variance_ratio"] > 5


def test_stratified_never_worse_than_crude():
    f = lambda x: np.exp(x)
    base = crude(f, 0, 1, 100_000, seed=7)
    strat = stratified(f, 0, 1, 100_000, n_strata=100, seed=7)
    assert strat.std_error < base.std_error


def test_stratified_variance_is_computed_per_stratum():
    """
    Regression test for a real bug in an earlier version of this module.

    The stratified estimator was reporting its standard error via the pooled
    i.i.d. formula s/sqrt(n), which ignores the stratification entirely and
    reproduces the crude error almost exactly. The correct figure is roughly
    two orders of magnitude smaller here.
    """
    f = lambda x: np.exp(x)
    base = crude(f, 0, 1, 100_000, seed=7)
    strat = stratified(f, 0, 1, 100_000, n_strata=100, seed=7)

    assert strat.std_error < base.std_error / 10

    # And the reported interval must still be honest about the true value
    truth = np.e - 1
    assert abs(strat.value - truth) < 5 * strat.std_error


def test_stratified_error_improves_with_more_strata():
    """More strata remove more of the between-stratum variance."""
    f = lambda x: np.exp(x)
    few = stratified(f, 0, 1, 100_000, n_strata=10, seed=7)
    many = stratified(f, 0, 1, 100_000, n_strata=500, seed=7)
    assert many.std_error < few.std_error


def test_importance_sampling_beats_crude_in_the_tail():
    """
    Estimate P(X > 4) for X ~ N(0,1), a rare event under the target.

    Crude sampling from N(0,1) sees almost no hits at n=100k; a proposal
    shifted to N(4,1) puts mass where the indicator is non-zero.
    """
    truth = stats.norm.sf(4.0)

    rng = np.random.default_rng(8)
    crude_samples = (rng.standard_normal(100_000) > 4.0).astype(float)
    crude_est = mc_estimate(crude_samples)

    est, diag = importance_sampling(
        h=lambda x: (x > 4.0).astype(float),
        log_target=lambda x: stats.norm.logpdf(x),
        sampler=lambda rng, n: rng.normal(4.0, 1.0, size=n),
        log_proposal=lambda x: stats.norm.logpdf(x, loc=4.0, scale=1.0),
        n=100_000,
        seed=8,
        self_normalised=False,
    )

    assert abs(est.value - truth) < 5 * est.std_error
    # Roughly three orders of magnitude less variance than crude sampling
    assert (crude_est.std_error / est.std_error) ** 2 > 1_000


def test_weight_ess_misjudges_a_good_rare_event_run():
    """
    The two ESS diagnostics disagree sharply here, and only one of them is
    right. ESS(w) collapses because most weights sit where the indicator is
    zero; ESS(w*h) stays healthy, matching the estimator's actual accuracy.
    """
    _, diag = importance_sampling(
        h=lambda x: (x > 4.0).astype(float),
        log_target=lambda x: stats.norm.logpdf(x),
        sampler=lambda rng, n: rng.normal(4.0, 1.0, size=n),
        log_proposal=lambda x: stats.norm.logpdf(x, loc=4.0, scale=1.0),
        n=100_000,
        seed=8,
        self_normalised=False,
    )

    assert diag["ess_weights"] < 0.001
    assert diag["ess_weighted_integrand"] > 0.10


# --------------------------------------------------------------------------
# MCMC
# --------------------------------------------------------------------------


def test_metropolis_recovers_standard_normal():
    chain = metropolis_hastings(
        log_target=lambda x: -0.5 * np.sum(x**2),
        x0=0.0,
        n_samples=40_000,
        step_size=2.4,
        seed=9,
    )
    assert abs(chain.samples.mean()) < 0.05
    assert abs(chain.samples.std() - 1.0) < 0.05
    assert 0.2 < chain.acceptance_rate < 0.7


def test_step_size_controls_acceptance_rate():
    """Tiny steps accept nearly everything; huge steps accept nearly nothing."""
    log_target = lambda x: -0.5 * np.sum(x**2)
    tiny = metropolis_hastings(log_target, 0.0, n_samples=5_000, step_size=0.01, seed=10)
    huge = metropolis_hastings(log_target, 0.0, n_samples=5_000, step_size=100.0, seed=10)

    assert tiny.acceptance_rate > 0.95
    assert huge.acceptance_rate < 0.05


def test_ess_detects_correlation():
    """
    ESS must be far below n for a correlated chain, and close to n for
    i.i.d. draws. This is the diagnostic that stops MCMC error bars from
    being fiction.
    """
    rng = np.random.default_rng(11)
    iid = rng.standard_normal(20_000)
    assert effective_sample_size(iid) > 0.7 * len(iid)

    chain = metropolis_hastings(
        log_target=lambda x: -0.5 * np.sum(x**2),
        x0=0.0,
        n_samples=20_000,
        step_size=0.1,  # deliberately too small -> sticky chain
        seed=11,
    )
    assert effective_sample_size(chain.samples) < 0.1 * len(chain.samples)


def test_gibbs_recovers_target_correlation():
    rho = 0.8
    chain = gibbs_bivariate_normal(rho=rho, n_samples=40_000, seed=12)
    empirical_rho = np.corrcoef(chain.samples.T)[0, 1]
    assert abs(empirical_rho - rho) < 0.03


def test_gibbs_mixing_degrades_with_correlation():
    """High correlation cripples axis-aligned Gibbs moves."""
    mild = gibbs_bivariate_normal(rho=0.1, n_samples=20_000, seed=13)
    severe = gibbs_bivariate_normal(rho=0.99, n_samples=20_000, seed=13)

    ess_mild = effective_sample_size(mild.samples[:, 0])
    ess_severe = effective_sample_size(severe.samples[:, 0])
    assert ess_severe < 0.1 * ess_mild


def test_autocorrelation_starts_at_one():
    rng = np.random.default_rng(14)
    acf = autocorrelation(rng.standard_normal(5_000), max_lag=20)
    assert abs(acf[0] - 1.0) < 1e-9


def test_gelman_rubin_near_one_for_converged_chains():
    log_target = lambda x: -0.5 * np.sum(x**2)
    chains = [
        metropolis_hastings(log_target, x0, n_samples=15_000, step_size=2.4, seed=s).samples
        for s, x0 in enumerate([-2.0, 0.0, 2.0], start=20)
    ]
    assert gelman_rubin(chains) < 1.05


def test_gelman_rubin_flags_unconverged_chains():
    """Sticky chains started far apart have not forgotten their origin."""
    log_target = lambda x: -0.5 * np.sum(x**2)
    chains = [
        metropolis_hastings(log_target, x0, n_samples=800, step_size=0.02, burn_in=0, seed=s).samples
        for s, x0 in enumerate([-8.0, 0.0, 8.0], start=30)
    ]
    assert gelman_rubin(chains) > 1.1


# --------------------------------------------------------------------------
# Simulated annealing
# --------------------------------------------------------------------------


def test_annealing_improves_on_random_tour():
    _, distances = random_tsp_instance(n_cities=30, seed=15)
    rng = np.random.default_rng(15)
    random_length = tour_length(rng.permutation(30), distances)

    result = solve_tsp(distances, n_steps=30_000, seed=15)
    assert result.best_energy < 0.5 * random_length


def test_annealing_beats_greedy_descent():
    """
    A zero-temperature schedule is pure greedy descent and freezes in the
    first local minimum. Real annealing, which accepts uphill moves early,
    should end up lower.
    """
    from annealing import simulated_annealing, two_opt_neighbour

    _, distances = random_tsp_instance(n_cities=50, seed=16)
    rng = np.random.default_rng(16)
    x0 = rng.permutation(50)
    energy = lambda t: tour_length(t, distances)

    greedy = simulated_annealing(energy, two_opt_neighbour, x0, np.full(40_000, 1e-12), seed=16)
    annealed = simulated_annealing(energy, two_opt_neighbour, x0, geometric_schedule(1.0, 0.9998, 40_000), seed=16)

    assert annealed.best_energy < greedy.best_energy


def test_annealing_acceptance_falls_as_temperature_drops():
    _, distances = random_tsp_instance(n_cities=40, seed=17)
    result = solve_tsp(distances, n_steps=40_000, T0=1.0, alpha=0.9998, seed=17)
    # Energy should be monotonically non-increasing in the long run
    early = result.energy_history[:2_000].mean()
    late = result.energy_history[-2_000:].mean()
    assert late < early

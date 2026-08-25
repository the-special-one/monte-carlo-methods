"""
End-to-end demo: variance reduction benchmark, MCMC diagnostics, annealing.
Produces the tables and figures used in the README.

Run with: python notebooks/demo_methods.py
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from estimators import mc_estimate, convergence_path
from variance_reduction import crude, antithetic, control_variate, stratified, importance_sampling, efficiency_report
from mcmc import metropolis_hastings, gibbs_bivariate_normal, autocorrelation, effective_sample_size, gelman_rubin
from annealing import random_tsp_instance, solve_tsp, tour_length, simulated_annealing, two_opt_neighbour, geometric_schedule

ROOT = os.path.join(os.path.dirname(__file__), "..")
N = 100_000

# ==========================================================================
# 1. Variance reduction benchmark on integral of exp(x) over [0, 1]
# ==========================================================================
print("=" * 74)
print("1. VARIANCE REDUCTION  —  integral of exp(x) over [0,1] = e - 1")
print("=" * 74)

f = lambda x: np.exp(x)
truth = np.e - 1

base = crude(f, 0, 1, N, seed=42)
methods = {
    "Crude": base,
    "Antithetic": antithetic(f, 0, 1, N, seed=42),
    "Control variate": control_variate(f, lambda x: 1 + x, g_mean=1.5, a=0, b=1, n=N, seed=42),
    "Stratified (100)": stratified(f, 0, 1, N, n_strata=100, seed=42),
}

print(f"\n{'Method':<20} {'Estimate':>12} {'Std error':>12} {'Var ratio':>12} {'Error':>12}")
print("-" * 74)
for name, est in methods.items():
    ratio = efficiency_report(base, est)["variance_ratio"]
    print(f"{name:<20} {est.value:>12.6f} {est.std_error:>12.2e} {ratio:>12.1f} {abs(est.value - truth):>12.2e}")
print(f"{'Truth':<20} {truth:>12.6f}")

# Antithetic backfiring
print("\nAntithetic variates are not unconditionally safe:")
for label, integrand in [("exp(x)  (monotone)", lambda x: np.exp(x)), ("(x-0.5)^2 (symmetric)", lambda x: (x - 0.5) ** 2)]:
    b = crude(integrand, 0, 1, N, seed=1)
    a = antithetic(integrand, 0, 1, N, seed=1)
    verdict = "helps" if a.std_error < b.std_error else "HURTS"
    print(f"  {label:<24} var ratio {(b.std_error/a.std_error)**2:>7.2f}   -> {verdict}")

# ==========================================================================
# 2. Importance sampling on a rare event
# ==========================================================================
print("\n" + "=" * 74)
print("2. IMPORTANCE SAMPLING  —  P(X > 4) for X ~ N(0,1)")
print("=" * 74)

truth_tail = stats.norm.sf(4.0)
rng = np.random.default_rng(8)
crude_tail = mc_estimate((rng.standard_normal(N) > 4.0).astype(float))

is_est, diag = importance_sampling(
    h=lambda x: (x > 4.0).astype(float),
    log_target=lambda x: stats.norm.logpdf(x),
    sampler=lambda rng, n: rng.normal(4.0, 1.0, size=n),
    log_proposal=lambda x: stats.norm.logpdf(x, loc=4.0, scale=1.0),
    n=N,
    seed=8,
    self_normalised=False,
)

print(f"\nTruth                 : {truth_tail:.6e}")
print(f"Crude MC              : {crude_tail.value:.6e} +/- {crude_tail.std_error:.2e}  (rel. error {crude_tail.rel_error:.0%})")
print(f"Importance sampling   : {is_est.value:.6e} +/- {is_est.std_error:.2e}  (rel. error {is_est.rel_error:.2%})")
print(f"Variance ratio        : {(crude_tail.std_error / is_est.std_error) ** 2:,.0f}x")
print(f"\nESS(weights)          : {diag['ess_weights']:.4%}   <- looks catastrophic")
print(f"ESS(weights * h)      : {diag['ess_weighted_integrand']:.2%}   <- the one that matters")

# ==========================================================================
# 3. MCMC diagnostics
# ==========================================================================
print("\n" + "=" * 74)
print("3. MCMC  —  step size, effective sample size, Gelman-Rubin")
print("=" * 74)

log_target = lambda x: -0.5 * np.sum(x**2)

print(f"\n{'Step size':>10} {'Accept rate':>13} {'ESS':>10} {'ESS / n':>10} {'Naive SE':>11} {'True SE':>11}")
print("-" * 74)
chains = {}
for step in [0.05, 0.5, 2.4, 10.0, 50.0]:
    ch = metropolis_hastings(log_target, 0.0, n_samples=20_000, step_size=step, seed=99)
    ess = effective_sample_size(ch.samples)
    naive_se = ch.samples.std(ddof=1) / np.sqrt(len(ch.samples))
    true_se = ch.samples.std(ddof=1) / np.sqrt(ess)
    chains[step] = ch
    print(f"{step:>10.2f} {ch.acceptance_rate:>12.1%} {ess:>10.0f} {ess/len(ch.samples):>9.1%} {naive_se:>11.2e} {true_se:>11.2e}")

print("\nThe naive standard error understates the true one by up to a factor of")
print(f"{np.sqrt(len(chains[0.05].samples) / effective_sample_size(chains[0.05].samples)):.0f} on the stickiest chain above.")

# Gibbs mixing vs correlation
print("\nGibbs sampler — mixing collapses as the target correlation rises:")
for rho in [0.0, 0.5, 0.9, 0.99]:
    g = gibbs_bivariate_normal(rho=rho, n_samples=20_000, seed=7)
    print(f"  rho = {rho:<5}  ESS/n = {effective_sample_size(g.samples[:, 0]) / 20_000:>7.2%}")

# Gelman-Rubin
converged = [metropolis_hastings(log_target, x0, n_samples=15_000, step_size=2.4, seed=s).samples
             for s, x0 in enumerate([-2.0, 0.0, 2.0], start=20)]
stuck = [metropolis_hastings(log_target, x0, n_samples=800, step_size=0.02, burn_in=0, seed=s).samples
         for s, x0 in enumerate([-8.0, 0.0, 8.0], start=30)]
print(f"\nGelman-Rubin R-hat, well-tuned chains : {gelman_rubin(converged):.4f}")
print(f"Gelman-Rubin R-hat, sticky chains     : {gelman_rubin(stuck):.4f}")

# ==========================================================================
# 4. Simulated annealing on TSP
# ==========================================================================
print("\n" + "=" * 74)
print("4. SIMULATED ANNEALING  —  50-city travelling salesman")
print("=" * 74)

coords, distances = random_tsp_instance(n_cities=50, seed=16)
rng = np.random.default_rng(16)
x0 = rng.permutation(50)
energy = lambda t: tour_length(t, distances)

random_len = energy(x0)
greedy = simulated_annealing(energy, two_opt_neighbour, x0, np.full(40_000, 1e-12), seed=16)
annealed = simulated_annealing(energy, two_opt_neighbour, x0, geometric_schedule(1.0, 0.9998, 40_000), seed=16)

print(f"\nRandom initial tour   : {random_len:.4f}")
print(f"Greedy descent (T=0)  : {greedy.best_energy:.4f}   ({greedy.acceptance_rate:.1%} accepted)")
print(f"Simulated annealing   : {annealed.best_energy:.4f}   ({annealed.acceptance_rate:.1%} accepted)")
print(f"\nAnnealing improves on greedy by {(greedy.best_energy - annealed.best_energy) / greedy.best_energy:.1%}")

# ==========================================================================
# Figures
# ==========================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

# (a) convergence
rng = np.random.default_rng(0)
samples = np.exp(rng.uniform(size=200_000))
n_grid, run_mean, run_se = convergence_path(samples)
ax = axes[0, 0]
ax.plot(n_grid, run_se, "o-", markersize=3, label="Observed std error")
ax.plot(n_grid, run_se[0] * np.sqrt(n_grid[0] / n_grid), "--", color="black", label=r"$O(n^{-1/2})$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of samples"); ax.set_ylabel("Standard error")
ax.set_title("(a) Monte Carlo error decays as $n^{-1/2}$")
ax.legend(frameon=False)

# (b) variance reduction comparison
ax = axes[0, 1]
names = list(methods.keys())
ratios = [efficiency_report(base, methods[k])["variance_ratio"] for k in names]
ax.barh(names, ratios, color=["#888", "#4c72b0", "#dd8452", "#55a868"])
ax.set_xscale("log")
ax.axvline(1, color="black", linewidth=0.8)
ax.set_xlabel("Variance ratio vs crude (log scale)")
ax.set_title("(b) Variance reduction efficiency")

# (c) MCMC autocorrelation
ax = axes[1, 0]
for step in [0.05, 0.5, 2.4, 50.0]:
    acf = autocorrelation(chains[step].samples, max_lag=60)
    ax.plot(acf, label=f"step={step} (acc {chains[step].acceptance_rate:.0%})")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrelation")
ax.set_title("(c) MCMC autocorrelation vs step size")
ax.legend(frameon=False, fontsize=8)

# (d) annealing
ax = axes[1, 1]
ax.plot(greedy.energy_history, label="Greedy descent (T=0)", linewidth=1.0, alpha=0.8)
ax.plot(annealed.energy_history, label="Simulated annealing", linewidth=1.0)
ax.set_xlabel("Iteration"); ax.set_ylabel("Tour length")
ax.set_title("(d) Annealing escapes local minima")
ax.legend(frameon=False)

fig.tight_layout()
fig.savefig(os.path.join(ROOT, "methods_overview.png"), dpi=150)

# TSP tour figure
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
for ax, tour, title in [
    (ax1, x0, f"Random tour — length {random_len:.2f}"),
    (ax2, annealed.best_state, f"After annealing — length {annealed.best_energy:.2f}"),
]:
    closed = np.append(tour, tour[0])
    ax.plot(coords[closed, 0], coords[closed, 1], "-o", markersize=4, linewidth=1)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
fig2.tight_layout()
fig2.savefig(os.path.join(ROOT, "tsp_tours.png"), dpi=150)

print(f"\nSaved figures to {os.path.abspath(ROOT)}")

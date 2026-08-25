"""
Markov Chain Monte Carlo: Metropolis-Hastings, Gibbs sampling, and the
diagnostics without which neither can be trusted.

The defining difference from plain Monte Carlo: the draws are *correlated*.
The chain is only asymptotically distributed as the target, and consecutive
states carry redundant information. Both facts are handled explicitly here —
burn-in for the first, effective sample size for the second — because an
MCMC standard error computed as s/sqrt(n) is simply wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ChainResult:
    samples: np.ndarray      # shape (n_kept,) or (n_kept, dim)
    acceptance_rate: float
    n_burn_in: int

    def __repr__(self) -> str:
        return f"ChainResult(n={len(self.samples)}, accept={self.acceptance_rate:.1%}, burn_in={self.n_burn_in})"


def metropolis_hastings(
    log_target: Callable[[np.ndarray], float],
    x0: np.ndarray | float,
    n_samples: int = 20_000,
    step_size: float = 1.0,
    burn_in: int = 2_000,
    seed: int | None = None,
) -> ChainResult:
    """
    Random-walk Metropolis-Hastings with a Gaussian proposal.

    Works in log space throughout: the acceptance ratio is
        log(alpha) = log p(x') - log p(x)
    which stays finite where the raw densities would underflow. The proposal
    is symmetric, so its density cancels from the ratio.

    On step_size: too small and the chain accepts everything while barely
    moving; too large and it rejects everything and stalls. The classical
    target is roughly 23% acceptance in high dimension, ~44% in one dimension
    (Roberts, Gelman & Gilks).
    """
    rng = np.random.default_rng(seed)

    x = np.atleast_1d(np.asarray(x0, dtype=float))
    dim = x.shape[0]
    total = n_samples + burn_in

    chain = np.empty((total, dim))
    log_p = log_target(x)
    n_accepted = 0

    for t in range(total):
        proposal = x + step_size * rng.standard_normal(dim)
        log_p_prop = log_target(proposal)

        # Accept with probability min(1, p(x')/p(x)); log(U) < log(alpha)
        if np.log(rng.uniform()) < log_p_prop - log_p:
            x, log_p = proposal, log_p_prop
            n_accepted += 1

        chain[t] = x

    kept = chain[burn_in:]
    return ChainResult(
        samples=kept.ravel() if dim == 1 else kept,
        acceptance_rate=n_accepted / total,
        n_burn_in=burn_in,
    )


def gibbs_bivariate_normal(
    rho: float,
    n_samples: int = 20_000,
    burn_in: int = 1_000,
    x0: tuple[float, float] = (0.0, 0.0),
    seed: int | None = None,
) -> ChainResult:
    """
    Gibbs sampler for a standard bivariate normal with correlation rho.

    Each coordinate is drawn from its full conditional, which here is known
    in closed form: X1 | X2 = x2  ~  N(rho * x2, 1 - rho^2).

    Every proposal is accepted by construction — that is the appeal of Gibbs.
    The cost appears elsewhere: as |rho| -> 1 the conditionals become tight,
    the chain moves in small axis-aligned steps, and mixing collapses. This
    is visible in the effective sample size, not in any acceptance rate.
    """
    rng = np.random.default_rng(seed)
    total = n_samples + burn_in

    x1, x2 = x0
    chain = np.empty((total, 2))
    cond_sd = np.sqrt(1 - rho**2)

    for t in range(total):
        x1 = rho * x2 + cond_sd * rng.standard_normal()
        x2 = rho * x1 + cond_sd * rng.standard_normal()
        chain[t] = (x1, x2)

    return ChainResult(samples=chain[burn_in:], acceptance_rate=1.0, n_burn_in=burn_in)


def autocorrelation(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Autocorrelation function of a scalar chain, lags 0..max_lag."""
    x = np.asarray(x, dtype=float).ravel()
    x = x - x.mean()
    n = len(x)
    max_lag = min(max_lag, n - 1)

    var = np.dot(x, x) / n
    if var == 0:
        return np.zeros(max_lag + 1)

    return np.array([np.dot(x[: n - k], x[k:]) / (n * var) for k in range(max_lag + 1)])


def effective_sample_size(x: np.ndarray, max_lag: int = 200) -> float:
    """
    ESS = n / (1 + 2 * sum_k rho_k), truncated at the first negative pair
    (Geyer's initial positive sequence).

    This is the number that should replace n in any MCMC standard error.
    A chain of 20,000 draws with ESS = 400 gives error bars fifty times wider
    than the naive s/sqrt(n) would suggest.
    """
    acf = autocorrelation(x, max_lag)

    # Truncate at the first lag where consecutive pairs sum to <= 0
    total = 0.0
    for k in range(1, len(acf) - 1, 2):
        pair = acf[k] + acf[k + 1]
        if pair <= 0:
            break
        total += pair

    return float(len(x) / (1 + 2 * total))


def mcmc_standard_error(x: np.ndarray) -> float:
    """Standard error of the chain mean, corrected for autocorrelation."""
    x = np.asarray(x, dtype=float).ravel()
    return float(x.std(ddof=1) / np.sqrt(effective_sample_size(x)))


def gelman_rubin(chains: list[np.ndarray]) -> float:
    """
    Gelman-Rubin R-hat from several independently-initialised chains.

    Compares between-chain and within-chain variance. R-hat near 1 is
    necessary for convergence, never sufficient: chains stuck in the same
    mode of a multimodal target agree perfectly with each other and give
    R-hat = 1 while missing most of the distribution.

    Rule of thumb: investigate anything above 1.01.
    """
    arr = np.array([np.asarray(c, dtype=float).ravel() for c in chains])
    m, n = arr.shape
    if m < 2:
        raise ValueError("Gelman-Rubin needs at least 2 chains")

    chain_means = arr.mean(axis=1)
    chain_vars = arr.var(axis=1, ddof=1)

    W = chain_vars.mean()                          # within-chain variance
    B = n * chain_means.var(ddof=1)                # between-chain variance
    var_hat = (n - 1) / n * W + B / n

    return float(np.sqrt(var_hat / W)) if W > 0 else np.inf

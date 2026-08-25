"""
Core Monte Carlo estimators.

Everything here returns an estimate *together with its uncertainty*. A Monte
Carlo number without a confidence interval is not a result, it is a guess:
the whole point of the method is that the error is itself estimable, at the
canonical O(1/sqrt(n)) rate given by the CLT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats


@dataclass
class MCEstimate:
    """A Monte Carlo estimate with its sampling uncertainty."""

    value: float
    std_error: float
    n_samples: int
    ci_level: float = 0.95

    @property
    def ci(self) -> tuple[float, float]:
        z = stats.norm.ppf(0.5 + self.ci_level / 2)
        return (self.value - z * self.std_error, self.value + z * self.std_error)

    @property
    def rel_error(self) -> float:
        """Relative standard error, the usual stopping criterion in practice."""
        return abs(self.std_error / self.value) if self.value != 0 else np.inf

    def __repr__(self) -> str:
        lo, hi = self.ci
        return f"MCEstimate({self.value:.6f} ± {self.std_error:.6f}, CI{self.ci_level:.0%}=[{lo:.6f}, {hi:.6f}], n={self.n_samples})"


def mc_estimate(samples: np.ndarray, ci_level: float = 0.95) -> MCEstimate:
    """
    Estimate E[X] from i.i.d. draws, with the CLT standard error.

    Uses the unbiased sample variance (ddof=1). The CLT approximation is what
    justifies the interval; it degrades for heavy-tailed integrands, which is
    exactly the case `importance_sampling` is meant to fix.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    n = len(samples)
    if n < 2:
        raise ValueError("need at least 2 samples to estimate a standard error")

    return MCEstimate(
        value=float(samples.mean()),
        std_error=float(samples.std(ddof=1) / np.sqrt(n)),
        n_samples=n,
        ci_level=ci_level,
    )


def mc_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int = 100_000,
    seed: int | None = None,
) -> MCEstimate:
    """
    Estimate the integral of f over [a, b] by uniform sampling.

    Deliberately naive: this is the baseline every variance reduction
    technique in `variance_reduction.py` is benchmarked against.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, size=n)
    return mc_estimate((b - a) * f(x))


def convergence_path(
    samples: np.ndarray, checkpoints: int = 50
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Running estimate and running standard error along the sample path.

    Returns (n_grid, running_mean, running_stderr) — the input for the
    classic log-log convergence plot whose slope should be -1/2.
    """
    samples = np.asarray(samples, dtype=float).ravel()
    n_grid = np.unique(np.logspace(1, np.log10(len(samples)), checkpoints).astype(int))

    running_mean = np.array([samples[:k].mean() for k in n_grid])
    running_se = np.array([samples[:k].std(ddof=1) / np.sqrt(k) for k in n_grid])
    return n_grid, running_mean, running_se


def sample_size_for_precision(pilot_samples: np.ndarray, target_half_width: float, ci_level: float = 0.95) -> int:
    """
    How many samples are needed to reach a target CI half-width?

    From a pilot run of variance s^2: n >= (z * s / half_width)^2. Because the
    error decays as 1/sqrt(n), halving the interval costs four times the work
    — the practical reason variance reduction matters more than raw compute.
    """
    s = np.asarray(pilot_samples, dtype=float).std(ddof=1)
    z = stats.norm.ppf(0.5 + ci_level / 2)
    return int(np.ceil((z * s / target_half_width) ** 2))

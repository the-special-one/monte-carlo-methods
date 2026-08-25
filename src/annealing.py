"""
Simulated annealing.

The link to MCMC is direct: annealing is Metropolis-Hastings on the Boltzmann
distribution p_T(x) ∝ exp(-E(x)/T), with T driven to zero. At high T the chain
explores freely; as T falls, the distribution concentrates on the minimisers of
E. The cooling schedule is the whole algorithm — cool too fast and it is just
a greedy descent that freezes into the first local minimum it meets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class AnnealingResult:
    best_state: np.ndarray
    best_energy: float
    energy_history: np.ndarray
    temperature_history: np.ndarray
    acceptance_rate: float


def geometric_schedule(T0: float, alpha: float, n_steps: int) -> np.ndarray:
    """T_k = T0 * alpha^k. The standard workhorse; alpha in [0.90, 0.999]."""
    return T0 * alpha ** np.arange(n_steps)


def logarithmic_schedule(T0: float, n_steps: int) -> np.ndarray:
    """
    T_k = T0 / log(k + e).

    Slow enough to carry a theoretical guarantee of convergence to the global
    optimum (Geman & Geman), and far too slow to be usable — included to make
    the gap between the theory and the practice explicit.
    """
    k = np.arange(n_steps)
    return T0 / np.log(k + np.e)


def simulated_annealing(
    energy: Callable[[np.ndarray], float],
    neighbour: Callable[[np.ndarray, np.random.Generator], np.ndarray],
    x0: np.ndarray,
    schedule: np.ndarray,
    seed: int | None = None,
) -> AnnealingResult:
    """
    Generic simulated annealing loop.

    The acceptance rule is Metropolis: always accept an improvement, accept a
    worsening move of size dE with probability exp(-dE/T). Accepting bad moves
    is not a flaw to be tuned away — it is the only mechanism that escapes
    local minima.
    """
    rng = np.random.default_rng(seed)

    x = np.array(x0, copy=True)
    e = energy(x)
    best_x, best_e = x.copy(), e

    energies = np.empty(len(schedule))
    n_accepted = 0

    for k, T in enumerate(schedule):
        candidate = neighbour(x, rng)
        e_new = energy(candidate)
        delta = e_new - e

        if delta <= 0 or rng.uniform() < np.exp(-delta / max(T, 1e-12)):
            x, e = candidate, e_new
            n_accepted += 1
            if e < best_e:
                best_x, best_e = x.copy(), e

        energies[k] = e

    return AnnealingResult(
        best_state=best_x,
        best_energy=best_e,
        energy_history=energies,
        temperature_history=schedule,
        acceptance_rate=n_accepted / len(schedule),
    )


# ---------------------------------------------------------------------------
# Travelling salesman instance
# ---------------------------------------------------------------------------


def tour_length(tour: np.ndarray, distances: np.ndarray) -> float:
    """Total length of a closed tour visiting each city once."""
    return float(distances[tour, np.roll(tour, -1)].sum())


def two_opt_neighbour(tour: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    2-opt move: reverse a random contiguous segment of the tour.

    Preferred over swapping two random cities because it changes only two
    edges, so the energy landscape stays smooth enough for the temperature
    to mean something.
    """
    new_tour = tour.copy()
    i, j = sorted(rng.choice(len(tour), size=2, replace=False))
    new_tour[i : j + 1] = new_tour[i : j + 1][::-1]
    return new_tour


def random_tsp_instance(n_cities: int = 40, seed: int | None = None):
    """Uniform cities in the unit square, with their Euclidean distance matrix."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(size=(n_cities, 2))
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    return coords, distances


def solve_tsp(
    distances: np.ndarray,
    n_steps: int = 60_000,
    T0: float = 1.0,
    alpha: float = 0.9999,
    seed: int | None = None,
) -> AnnealingResult:
    """Anneal a TSP instance from a random initial tour."""
    rng = np.random.default_rng(seed)
    n = distances.shape[0]
    x0 = rng.permutation(n)

    return simulated_annealing(
        energy=lambda t: tour_length(t, distances),
        neighbour=two_opt_neighbour,
        x0=x0,
        schedule=geometric_schedule(T0, alpha, n_steps),
        seed=seed,
    )

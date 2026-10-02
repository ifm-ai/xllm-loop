"""Recurrent-depth distributions and draws for `DepthControlledHuginn` training.

Fixed PLN-5 samples each training forward's depth from min(cap, 1 + Poisson(λ))
with log λ ~ N(μ, σ²), where μ is calibrated so that the capped mean equals the
target. The arithmetic follows the paper's training code operation by
operation; PLN-5 uses the PMF the paper runs computed, so its draws match those
runs exactly.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from functools import lru_cache

import torch


PLN_SEED_OFFSET = 514229

# The PLN-5 PMF (target mean 5, sigma 0.5, cap 64) the paper's training runs used,
# as computed in their environment with torch 2.8.0+cu128. Other CPUs and torch
# builds change `calibrated_capped_poisson_lognormal` in the last bits.
PAPER_PLN5_PMF = (
    0.05768747546482591,
    0.1271367387177668,
    0.1634086392890813,
    0.1619074983340164,
    0.13796282734225382,
    0.10698272319110169,
    0.07804764707860895,
    0.05470693907876852,
    0.037361700976777726,
    0.025099433949814245,
    0.016697717702268286,
    0.011052629620047851,
    0.007304055652903804,
    0.004830737772361538,
    0.003203151771039994,
    0.002132081017445029,
    0.0014258721987138567,
    0.0009586921076838706,
    0.0006483099746274615,
    0.0004410716241755722,
    0.000301944551073642,
    0.00020800269392146997,
    0.00014419125207641085,
    0.00010058218331207551,
    7.059667258419262e-05,
    4.9852412062464184e-05,
    3.5414136427653425e-05,
    2.5304622107571485e-05,
    1.8184285152127846e-05,
    1.3140259666362847e-05,
    9.546846563012203e-06,
    6.972695919262932e-06,
    5.118738343708846e-06,
    3.7764502416124336e-06,
    2.7996353510214844e-06,
    2.085238462098128e-06,
    1.5602277964955316e-06,
    1.1725776685897305e-06,
    8.850369680325107e-07,
    6.708005361757733e-07,
    5.104868279453078e-07,
    3.900165388933937e-07,
    2.9911563900765575e-07,
    2.3025345865338855e-07,
    1.778857064050211e-07,
    1.3791253597731962e-07,
    1.0728910765229482e-07,
    8.3744719506e-08,
    6.557941898380174e-08,
    5.151599353655518e-08,
    4.059163481228815e-08,
    3.207815872458309e-08,
    2.5422939995903076e-08,
    2.020502320279681e-08,
    1.6102459981160398e-08,
    1.286800205515741e-08,
    1.0311018925701726e-08,
    8.284031836490163e-09,
    6.672645135115369e-09,
    5.387960542342231e-09,
    4.3608027065798596e-09,
    3.5372756494056925e-09,
    2.8753142244955275e-09,
    1.3322268843651841e-08,
)


@lru_cache(maxsize=None)
def standard_normal_gauss_hermite(order: int = 64) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return Gauss-Hermite nodes and weights for expectations under N(0, 1)."""
    jacobi = torch.zeros((order, order), dtype=torch.float64, device="cpu")
    off_diagonal = torch.sqrt(torch.arange(1, order, dtype=torch.float64, device="cpu") / 2.0)
    jacobi.diagonal(1).copy_(off_diagonal)
    jacobi.diagonal(-1).copy_(off_diagonal)
    nodes, eigenvectors = torch.linalg.eigh(jacobi)
    normal_nodes = math.sqrt(2.0) * nodes
    normal_weights = eigenvectors[0].square()
    return tuple(normal_nodes.tolist()), tuple(normal_weights.tolist())


def capped_poisson_lognormal_pmf(mu: float, sigma: float, max_steps: int) -> tuple[float, ...]:
    """Return the PMF of min(max_steps, 1 + Poisson(LogNormal(mu, sigma)))."""
    nodes, weights = standard_normal_gauss_hermite()
    nodes_tensor = torch.tensor(nodes, dtype=torch.float64, device="cpu")
    weights_tensor = torch.tensor(weights, dtype=torch.float64, device="cpu")
    rates = torch.exp(mu + sigma * nodes_tensor)
    log_rates = torch.log(rates)  # not mu + sigma * nodes: the paper code's rounding
    probabilities = []
    for poisson_count in range(max_steps - 1):
        log_probability = -rates + poisson_count * log_rates - math.lgamma(poisson_count + 1)
        probabilities.append(float(torch.sum(weights_tensor * torch.exp(log_probability)).item()))
    probabilities.append(max(0.0, 1.0 - sum(probabilities)))
    normalization = sum(probabilities)
    return tuple(probability / normalization for probability in probabilities)


@lru_cache(maxsize=None)
def calibrated_capped_poisson_lognormal(
    target_mean: float, sigma: float, max_steps: int
) -> tuple[float, tuple[float, ...]]:
    """Bisect μ so the capped depth mean equals `target_mean`; return (μ, PMF).

    [-20, 20] brackets every target mean between 1 and the cap, and 80 halvings reach float64 resolution.
    """
    lower_mu, upper_mu = -20.0, 20.0
    for _ in range(80):
        mu = (lower_mu + upper_mu) / 2.0
        pmf = capped_poisson_lognormal_pmf(mu, sigma, max_steps)
        mean = sum(step * probability for step, probability in enumerate(pmf, start=1))
        if mean < target_mean:
            lower_mu = mu
        else:
            upper_mu = mu
    calibrated_mu = (lower_mu + upper_mu) / 2.0
    return calibrated_mu, capped_poisson_lognormal_pmf(calibrated_mu, sigma, max_steps)


def pln_pmf(target_mean: float, sigma: float, max_steps: int) -> tuple[float, ...]:
    """Return the capped PLN PMF, using the paper runs' exact values for PLN-5."""
    if (target_mean, sigma, max_steps) == (5.0, 0.5, 64):
        return PAPER_PLN5_PMF
    return calibrated_capped_poisson_lognormal(target_mean, sigma, max_steps)[1]


def cumulative_distribution(pmf: tuple[float, ...]) -> tuple[float, ...]:
    """Return the CDF used for inverse-transform draws; its last entry is 1."""
    cumulative_probability = 0.0
    cdf = []
    for probability in pmf:
        cumulative_probability += probability
        cdf.append(min(cumulative_probability, 1.0))
    cdf[-1] = 1.0
    return tuple(cdf)


def sample_depth(
    cdf: tuple[float, ...],
    iteration_step: int,
    *,
    antithetic: bool,
    seed_offset: int = PLN_SEED_OFFSET,
) -> int:
    """Draw one depth for the given training forward index.

    Every rank draws the same depth from a CPU generator seeded by the forward
    index. Antithetic draws pair consecutive forwards as u and 1 - u.
    """
    sampling_step = iteration_step // 2 if antithetic else iteration_step
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sampling_step + seed_offset)
    quantile = float(torch.rand((), generator=generator).item())
    if antithetic and iteration_step % 2 == 1:
        quantile = 1.0 - quantile
    return min(bisect_left(cdf, quantile) + 1, len(cdf))

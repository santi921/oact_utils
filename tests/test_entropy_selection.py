"""Tests for the shared entropy-selection core math.

Validates the rank-1 covariance update, the single-candidate delta-log-det score,
the whitening transform, and the closed-form feature gradient documented in
``entropy_downselect_math.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from oact_utilities.utils.entropy_selection import (
    compute_single_score,
    init_seed_from_external,
    update_state,
    whiten,
)


def _random_spd(D: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.standard_normal((D, D))
    return A @ A.T + D * np.eye(D)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_update_state_matches_bruteforce(rng):
    """C_new, C_inv_new, and delta_log_det match a direct rank-1 recomputation."""
    D, n = 6, 137
    C = _random_spd(D, rng)
    C_inv = np.linalg.inv(C)
    mu = rng.standard_normal(D)
    x = rng.standard_normal(D)

    scale = n / (n + 1)
    alpha = n / (n + 1) ** 2
    delta = x - mu
    C_new_ref = scale * C + alpha * np.outer(delta, delta)

    C_new, C_inv_new, mu_new, n_new, dldet = update_state(x, C, C_inv, mu, n)

    assert n_new == n + 1
    np.testing.assert_allclose(C_new, C_new_ref, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        C_inv_new, np.linalg.inv(C_new_ref), rtol=1e-8, atol=1e-8
    )
    np.testing.assert_allclose(mu_new, mu + delta / (n + 1), rtol=1e-12, atol=1e-12)

    dldet_ref = np.linalg.slogdet(C_new_ref)[1] - np.linalg.slogdet(C)[1]
    assert dldet == pytest.approx(dldet_ref, rel=1e-9, abs=1e-9)


def test_compute_single_score_equals_bruteforce_delta_logdet(rng):
    """compute_single_score equals slogdet(C') - slogdet(C) for the rank-1 update."""
    D, n = 5, 73
    C = _random_spd(D, rng)
    C_inv = np.linalg.inv(C)
    mu = rng.standard_normal(D)

    for _ in range(20):
        x = rng.standard_normal(D) * 2.0
        scale = n / (n + 1)
        alpha = n / (n + 1) ** 2
        delta = x - mu
        C_new = scale * C + alpha * np.outer(delta, delta)
        ref = np.linalg.slogdet(C_new)[1] - np.linalg.slogdet(C)[1]

        score = compute_single_score(x, C_inv, mu, n)
        assert score == pytest.approx(ref, rel=1e-9, abs=1e-9)
        # also consistent with update_state's reported delta
        assert score == pytest.approx(update_state(x, C, C_inv, mu, n)[4], rel=1e-9)


def test_whiten_yields_identity_covariance(rng):
    """Whitening with reference == data gives ~identity covariance (up to reg)."""
    N, D = 4000, 6
    base = rng.standard_normal((N, D)) @ _random_spd(D, rng)
    X = base.astype(np.float32)
    ref = X.copy()

    whiten(X, ref=ref, reg=1e-6)

    cov = np.cov(X.astype(np.float64), rowvar=False)
    np.testing.assert_allclose(cov, np.eye(D), atol=1e-3)


def test_feature_gradient_matches_finite_difference(rng):
    """Closed-form gradient 2 C^-1 (x-mu)/((n+1)+q) matches central differences."""
    D, n = 5, 211
    C = _random_spd(D, rng)
    C_inv = np.linalg.inv(C)
    mu = rng.standard_normal(D)
    x = rng.standard_normal(D)

    delta = x - mu
    q = float(delta @ C_inv @ delta)
    analytic = 2.0 * (C_inv @ delta) / ((n + 1) + q)

    eps = 1e-6
    fd = np.zeros(D)
    for i in range(D):
        xp = x.copy()
        xp[i] += eps
        xm = x.copy()
        xm[i] -= eps
        fd[i] = (
            compute_single_score(xp, C_inv, mu, n)
            - compute_single_score(xm, C_inv, mu, n)
        ) / (2 * eps)

    np.testing.assert_allclose(analytic, fd, rtol=1e-5, atol=1e-7)


def test_init_seed_log_det_consistent(rng):
    """init_seed_from_external returns C, C_inv, mu consistent with the data."""
    N, D = 500, 6
    X = (rng.standard_normal((N, D)) @ _random_spd(D, rng)).astype(np.float32)
    reg = 1e-6

    C, C_inv, mu, n = init_seed_from_external(X, reg)

    assert n == N
    np.testing.assert_allclose(mu, X.astype(np.float64).mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(C_inv @ C, np.eye(D), atol=1e-6)
    centered = X.astype(np.float64) - mu
    C_ref = centered.T @ centered / N + reg * np.eye(D)
    np.testing.assert_allclose(C, C_ref, rtol=1e-8, atol=1e-8)

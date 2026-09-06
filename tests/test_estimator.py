"""Synthetic recovery tests. The estimator must find subspaces we planted."""

import numpy as np
import pytest

from caliper.estimator import fit, select_k, subspace_alignment


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def _correlated_stimulus(n, d, seed=0, decay=1.5):
    """Anisotropic stimuli with a power-law spectrum, as residual streams have."""
    rng = np.random.default_rng(seed)
    scale = np.arange(1, d + 1) ** (-decay / 2)
    basis = np.linalg.qr(rng.standard_normal((d, d)))[0]
    return (rng.standard_normal((n, d)) * scale) @ basis.T


def test_recovers_planted_1d():
    d, n = 64, 20_000
    rng = np.random.default_rng(1)
    s = _correlated_stimulus(n, d, seed=1)
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    a = _gelu(s @ w)

    f = fit(s, a, k=1, n_restarts=4, steps=1200, seed=1)
    assert abs(subspace_alignment(f.subspace, w[:, None])) > 0.95
    assert f.test_r2 > 0.9


def test_recovers_planted_2d():
    d, n = 48, 30_000
    rng = np.random.default_rng(2)
    s = _correlated_stimulus(n, d, seed=2)
    v = np.linalg.qr(rng.standard_normal((d, 2)))[0]
    z = s @ v
    a = _gelu(z[:, 0]) * np.tanh(z[:, 1])  # genuinely two-dimensional

    f = fit(s, a, k=2, n_restarts=6, steps=1800, seed=2)
    assert abs(subspace_alignment(f.subspace, v)) > 0.9


def test_selects_k_one_for_one_dimensional_unit():
    d, n = 64, 20_000
    rng = np.random.default_rng(3)
    s = _correlated_stimulus(n, d, seed=3)
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    a = _gelu(s @ w)

    k, fits = select_k(s, a, k_max=3, tol=0.01, n_restarts=3, steps=900, seed=3)
    assert k == 1, f"selected k={k}, gains={[round(x.test_r2, 3) for x in fits]}"


def test_pure_noise_yields_no_structure():
    """The null: a response independent of the stimulus must not be explained."""
    d, n = 64, 20_000
    rng = np.random.default_rng(4)
    s = _correlated_stimulus(n, d, seed=4)
    a = rng.standard_normal(n)

    f = fit(s, a, k=1, n_restarts=4, steps=900, seed=4)
    assert f.test_r2 < 0.05, f"found structure in noise: R2={f.test_r2:.3f}"


@pytest.mark.parametrize("k", [1, 2])
def test_subspace_alignment_is_one_for_identical_subspaces(k):
    rng = np.random.default_rng(5)
    v = np.linalg.qr(rng.standard_normal((32, k)))[0]
    assert abs(subspace_alignment(v, v) - 1.0) < 1e-6


def test_subspace_is_invariant_under_monotone_response_transform():
    """If a = f(V^T s) then g(a) = (g.f)(V^T s), so the subspace must not move."""
    d, n = 48, 20_000
    rng = np.random.default_rng(6)
    s = _correlated_stimulus(n, d, seed=6)
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    a = _gelu(s @ w)

    plain = fit(s, a, k=1, n_restarts=3, steps=900, seed=6, response_transform="none")
    ranked = fit(s, a, k=1, n_restarts=3, steps=900, seed=6, response_transform="rank")
    assert abs(subspace_alignment(plain.subspace, ranked.subspace)) > 0.95


def test_rank_transform_does_not_break_a_heavy_tailed_response():
    """Both objectives must recover a sparse-firing unit.

    An earlier version asserted that ranking strictly beats squared error here. It does
    not, because this synthetic has no silent mass and raw already reaches 1.000 - there
    was no headroom to win. On real neurons ranking helps the worst units and badly
    breaks units with a large silent mass, so the honest requirement is that neither
    objective fails on a clean sparse unit.
    """
    d, n = 48, 30_000
    rng = np.random.default_rng(7)
    s = _correlated_stimulus(n, d, seed=7)
    w = rng.standard_normal(d)
    w /= np.linalg.norm(w)
    z = s @ w
    a = _gelu((z - np.quantile(z, 0.98)) * 12)  # fires on ~2% of inputs

    for transform in ("none", "rank"):
        f = fit(s, a, k=1, n_restarts=3, steps=1200, seed=7,
                response_transform=transform)
        got = abs(subspace_alignment(f.subspace, w[:, None]))
        assert got > 0.95, f"{transform}: {got:.3f}"

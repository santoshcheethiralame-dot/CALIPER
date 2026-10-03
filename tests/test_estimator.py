"""Synthetic recovery tests. The estimator must find subspaces we planted."""

import numpy as np
import pytest

from caliper.estimator import _orthogonal_complement, fit, fit_deflate, select_k, subspace_alignment


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


def test_orthogonal_complement_spans_what_was_removed():
    """The projection the deflation relies on: Q must span the complement exactly."""
    rng = np.random.default_rng(20)
    V = np.linalg.qr(rng.standard_normal((64, 2)))[0]
    Q = _orthogonal_complement(V)
    assert Q.shape == (64, 62)
    # Nothing in Q may overlap the removed span.
    assert np.abs(V.T @ Q).max() < 1e-10
    # And Q together with V must be the whole space, orthonormal.
    full = np.concatenate([V, Q], axis=1)
    assert np.abs(full.T @ full - np.eye(64)).max() < 1e-10


def _planted_additive(s, v, sparsity=0.90):
    """The additive-coupling unit S1-2 plants: a sparse nonlinear drive plus a gated
    second direction. Matching the experiment's generative model matters - an earlier
    version of this test used multiplicative coupling and asserted the opposite ordering,
    which was wrong for reasons recorded in test_deflate_is_coupling_dependent."""
    z = s @ v
    y = _gelu((z[:, 0] - np.quantile(z[:, 0], sparsity)) * 10)
    for j in range(1, v.shape[1]):
        y = y + np.tanh(z[:, j])
    return y


def test_deflate_recovers_two_planted_directions():
    """Deflation does recover both directions when the unit is additive.

    The K>1 wall is real at e03's operating point (joint alignment 0.5213 at K=2) but it
    is NOT re-derivable here: on this easy synthetic - 30k tokens, d=48, 6 restarts -
    joint k=2 already clears the bar, so this test deliberately does not claim deflation
    beats joint. It pins the thing that matters for S1-2: that the deflation route
    recovers the planted subspace at all.
    """
    d, n = 48, 30_000
    rng = np.random.default_rng(21)
    s = _correlated_stimulus(n, d, seed=21)
    v = np.linalg.qr(rng.standard_normal((d, 2)))[0]
    a = _planted_additive(s, v)

    defl = fit_deflate(s, a, k=2, method="plain", n_restarts=2, steps=1200, seed=21)
    got = abs(subspace_alignment(defl.subspace, v))
    print(f"\n  additive deflate {got:.3f}")
    assert got > 0.90, f"deflation only reached {got:.3f}"


def test_response_only_deflation_is_not_worse_than_full():
    """The control arm beat the method of record. Recorded rather than buried.

    The design rationale for projecting the stimulus was that response-only deflation
    leaves f(X v1) - X v1 in the residual, so the next rank-1 search rediscovers v1 and
    the method degenerates to "a robust rank-1 fit run twice". That reasoning predicted
    response-only would score WORSE. Measured, it scores BETTER, under both couplings:

        additive        full 0.945  response-only 0.995
        multiplicative  full 0.506  response-only 0.930

    So the rediscovery argument does not hold at this operating point, and the stimulus
    projection costs accuracy rather than buying it. `project_stimulus` is kept because
    the mechanism is still the right hypothesis for some regime and the arm is cheap, but
    the default and the interpretation now say response-only is the better estimator and
    the full method is the control.
    """
    for seed, label in ((22, "additive"),):
        d, n = 48, 30_000
        rng = np.random.default_rng(seed)
        s = _correlated_stimulus(n, d, seed=seed)
        v = np.linalg.qr(rng.standard_normal((d, 2)))[0]
        a = _planted_additive(s, v)

        full = fit_deflate(s, a, k=2, method="plain", n_restarts=2, steps=1200, seed=seed,
                           project_stimulus=True)
        resp_only = fit_deflate(s, a, k=2, method="plain", n_restarts=2, steps=1200,
                                seed=seed, project_stimulus=False)

        got_full = abs(subspace_alignment(full.subspace, v))
        got_resp = abs(subspace_alignment(resp_only.subspace, v))
        print(f"\n  {label}: full {got_full:.3f} vs response-only {got_resp:.3f}")
        assert got_resp >= got_full, (
            f"{label}: response-only {got_resp:.3f} is worse than full {got_full:.3f}"
        )


def test_deflate_is_coupling_dependent():
    """A limitation, measured rather than assumed.

    With MULTIPLICATIVE coupling, a = g(z0) * tanh(z1), projecting the stimulus out
    scores 0.506 while response-only deflation scores 0.930 - the reverse of the
    additive ordering, and by a wide margin. The reason is structural: y depends on z1
    through the gate, so removing z1 from the stimulus removes a factor the response
    genuinely needs, and the next rank-1 search has nothing left to explain.

    So the stimulus projection is a bet on additive coupling. It is the right bet for
    S1-2, whose planted units are additive by construction and match e03_required_n.py,
    and it is NOT a general property of deflation. An earlier version of this file
    asserted the opposite ordering using multiplicative units and was simply wrong.
    """
    d, n = 48, 30_000
    rng = np.random.default_rng(23)
    s = _correlated_stimulus(n, d, seed=23)
    v = np.linalg.qr(rng.standard_normal((d, 2)))[0]
    z = s @ v
    a = _gelu((z[:, 0] - np.quantile(z[:, 0], 0.9)) * 10) * np.tanh(z[:, 1])

    full = fit_deflate(s, a, k=2, method="plain", n_restarts=2, steps=1200, seed=23,
                       project_stimulus=True)
    resp_only = fit_deflate(s, a, k=2, method="plain", n_restarts=2, steps=1200, seed=23,
                            project_stimulus=False)

    got_full = abs(subspace_alignment(full.subspace, v))
    got_resp = abs(subspace_alignment(resp_only.subspace, v))
    print(f"\n  multiplicative: full {got_full:.3f} vs response-only {got_resp:.3f}")
    assert got_resp > got_full, (
        f"expected response-only to win under multiplicative coupling, got "
        f"full {got_full:.3f} vs response-only {got_resp:.3f}"
    )


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

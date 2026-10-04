"""The multiplicative arm of S1-2 rests on a claim about the generative model, so the model
is tested rather than trusted.

The claim: the stimulus projection is only expensive when the secondary planted directions
enter the response *multiplicatively*. Under additive coupling the projected method loses
0.05 (0.945 vs 0.995); under multiplicative it loses 0.42 (0.506 vs 0.930). If
`planted_units.planted` silently stopped gating, the multiplicative cells would quietly
become additive cells and the run would report a small effect while claiming a large one -
which is the exact failure this repo's gotchas keep recording.

These tests check the structure of the planted response, not the estimator, so they run in
milliseconds and need no model.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from planted_units import COUPLINGS, gelu, planted  # noqa: E402

D, N, UNITS = 48, 3000, 4


@pytest.fixture(scope="module")
def stim() -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.standard_normal((N, D)).astype(np.float32)


def test_additive_matches_the_archived_e03_formula(stim):
    """Regression guard: the joint baseline in e03_required_n.json is only comparable if
    the additive plant is still the same function."""
    V, Y = planted(stim, k=2, n_units=1, coupling="additive", seed=1000 * 2 + N)
    rng = np.random.default_rng(1000 * 2 + N)
    V_ref = np.stack([np.linalg.qr(rng.standard_normal((D, 2)))[0]])
    Z = np.einsum("sd,ndk->snk", stim, V_ref)
    lead = Z[:, :, 0]
    expected = gelu(lead - np.quantile(lead, 0.90, axis=0)) + np.tanh(Z[:, :, 1])

    assert np.allclose(V[0], V_ref[0])
    assert np.allclose(Y[:, 0], expected[:, 0].astype(np.float32), atol=1e-6)


def test_multiplicative_k2_is_base_times_gate(stim):
    """At K=2 the multiplicative plant must reduce to gelu(...)*tanh(z1) - the exact form
    under which the projected method was measured collapsing to 0.506."""
    V, Y = planted(stim, k=2, n_units=1, coupling="multiplicative", seed=7)
    Z = stim @ V[0]
    lead = Z[:, 0]
    expected = gelu(lead - np.quantile(lead, 0.90)) * np.tanh(Z[:, 1])

    assert np.allclose(Y[:, 0], expected.astype(np.float32), atol=1e-6)


def test_multiplicative_actually_differs_from_additive(stim):
    """If the two couplings produced the same response the arm would be a duplicate."""
    _, add = planted(stim, k=2, n_units=1, coupling="additive", seed=7)
    _, mul = planted(stim, k=2, n_units=1, coupling="multiplicative", seed=7)
    assert not np.allclose(add, mul)


def test_secondary_direction_is_gated_not_additive(stim):
    """The mechanism: under multiplicative coupling, the response's dependence on z1 must
    vanish wherever the gate closes. Under additive coupling it must not. This is what
    makes removing z1 from the stimulus destructive rather than harmless."""
    _, add = planted(stim, k=2, n_units=1, coupling="additive", seed=7)
    _, mul = planted(stim, k=2, n_units=1, coupling="multiplicative", seed=7)
    rng = np.random.default_rng(7)
    V = np.linalg.qr(rng.standard_normal((D, 2)))[0]
    Z = stim @ V
    gate = np.tanh(Z[:, 1])

    # where the gate is near zero, additive still carries z1 linearly, multiplicative does not
    quiet = np.abs(gate) < np.percentile(np.abs(gate), 10)
    loud = np.abs(gate) > np.percentile(np.abs(gate), 90)

    assert np.std(add[quiet, 0] - add[loud, 0]) > 0
    gated_spread = np.std(mul[quiet, 0] - mul[loud, 0])
    linear_spread = np.std(add[quiet, 0] - add[loud, 0])
    assert gated_spread < linear_spread


def test_multiplicative_is_narrower_in_scale(stim):
    """A product of gated terms has smaller dynamic range than a sum. A sanity check that
    the multiplicative cells are not simply the additive ones rescaled, which would make the
    two arms non-independent."""
    _, add = planted(stim, k=3, n_units=1, coupling="additive", seed=7)
    _, mul = planted(stim, k=3, n_units=1, coupling="multiplicative", seed=7)
    add_std = float(np.std(add))
    mul_std = float(np.std(mul))
    assert mul_std < add_std
    # and not degenerate either - a near-constant response would fail to be recoverable
    assert mul_std > 0.05 * add_std


def test_k3_chains_all_secondary_directions(stim):
    """The gate chain must include every j>0, not just the first, or K=3 would silently
    degenerate to K=2 plus a spare additive term."""
    V, Y = planted(stim, k=3, n_units=1, coupling="multiplicative", seed=7)
    Z = stim @ V[0]
    lead = Z[:, 0]
    expected = gelu(lead - np.quantile(lead, 0.90))
    for j in (1, 2):
        expected = expected * np.tanh(Z[:, j])
    assert np.allclose(Y[:, 0], expected.astype(np.float32), atol=1e-6)


def test_unknown_coupling_is_rejected(stim):
    with pytest.raises(ValueError, match="unknown coupling"):
        planted(stim, k=2, n_units=1, coupling="quadratic")


def test_seed_defaults_to_the_cell_so_cells_are_independent(stim):
    """Different cells must get different directions, or K=2 and K=3 would share a plant."""
    _, a = planted(stim, k=2, n_units=1, coupling="additive")
    _, b = planted(stim, k=3, n_units=1, coupling="additive")
    assert not np.allclose(a, b[:, :1].repeat(1, axis=1)[:, :1])


def test_couplings_are_the_documented_pair():
    assert COUPLINGS == ("additive", "multiplicative")
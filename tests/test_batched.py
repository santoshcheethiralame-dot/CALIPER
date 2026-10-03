"""Batched fitting must agree with the single-neuron path, and be faster."""

import time

import numpy as np
import torch

from caliper.batched import _BatchedBottleneck, fit_batch
from caliper.estimator import fit, subspace_alignment


def _gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def _setup(n_neurons, d=128, n=8000, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.arange(1, d + 1) ** -0.75
    basis = np.linalg.qr(rng.standard_normal((d, d)))[0]
    X = (rng.standard_normal((n, d)) * scale) @ basis.T
    W = rng.standard_normal((d, n_neurons))
    W /= np.linalg.norm(W, axis=0, keepdims=True)
    Y = _gelu(X @ W)
    return X, Y, W


def test_batched_recovers_planted_directions():
    X, Y, W = _setup(6, seed=1)
    fits = fit_batch(X, Y, k=1, n_restarts=2, steps=800, seed=1)
    aligned = [abs(subspace_alignment(f.subspace, W[:, i][:, None]))
               for i, f in enumerate(fits)]
    assert min(aligned) > 0.95, f"worst {min(aligned):.3f} of {len(aligned)}"


def test_batched_matches_single_neuron_path():
    """Same seed, same data: batched and unbatched must find the same subspace.

    They are not bitwise identical - batching sums losses across neurons, so Adam sees a
    different (but per-neuron equivalent) graph - but the recovered directions must
    agree, or batching is not a drop-in replacement.
    """
    X, Y, W = _setup(4, seed=2)
    batched = fit_batch(X, Y, k=1, n_restarts=2, steps=800, seed=2)
    for i in range(Y.shape[1]):
        single = fit(X, Y[:, i], k=1, n_restarts=2, steps=800, seed=2)
        agree = abs(subspace_alignment(batched[i].subspace, single.subspace))
        assert agree > 0.95, f"neuron {i}: batched vs single agreement {agree:.3f}"


def test_stacked_init_depends_on_batch_size():
    """The defect, pinned as a test so the fix cannot be quietly reverted.

    B-0 measured five of sixteen units changing pass/fail side purely from batch size.
    The cause is here: one stacked randn of shape (n, d, k) consumes the generator in
    n*d*k steps, so unit i's slice of the stream moves whenever n moves. Same units,
    same seed, same device, different answer.
    """
    big = _BatchedBottleneck(8, 128, 1, seed=0)
    small = _BatchedBottleneck(3, 128, 1, seed=0)
    assert not torch.equal(big.w1[:3], small.w1), (
        "expected the stacked draw to differ between n=8 and n=3; if it does not, "
        "B-0's batch-size result needs a different explanation"
    )


def test_per_neuron_seed_removes_the_batch_size_coupling():
    """Unit i's initialisation must be a function of (seed, i) alone.

    This is what makes a paired comparison across batch sizes legitimate: the same unit
    starts in the same place whether it is fitted alone or alongside seven others, so
    any remaining difference is arithmetic rather than a different basin.
    """
    d, k, width = 128, 1, 64
    big = _BatchedBottleneck(8, d, k, width=width, seed=0, per_neuron_seed=True)
    small = _BatchedBottleneck(3, d, k, width=width, seed=0, per_neuron_seed=True)
    for name in ("v", "w1", "w2", "w3"):
        full, part = getattr(big, name), getattr(small, name)
        assert torch.equal(full[:3], part), f"{name} still depends on batch size"
    # A different restart must still get a different start.
    other = _BatchedBottleneck(3, d, k, width=width, seed=1000, per_neuron_seed=True)
    assert not torch.equal(big.w1[:3], other.w1), "restart seed is being ignored"


def test_per_neuron_seed_is_opt_in():
    """Opt-in only. Turning it on by default would change every number already collected."""
    legacy = _BatchedBottleneck(3, 128, 1, seed=0)
    per_neuron = _BatchedBottleneck(3, 128, 1, seed=0, per_neuron_seed=True)
    assert not torch.equal(legacy.v, per_neuron.v), (
        "default and per-neuron init are identical; one of the two paths is not wired up"
    )


def test_batching_is_faster_per_neuron():
    """The whole point. Must beat the single-neuron path per unit of work."""
    X, Y, _ = _setup(8, seed=3)
    t0 = time.time()
    fit_batch(X, Y, k=1, n_restarts=1, steps=400, seed=3)
    batched_s = time.time() - t0

    t0 = time.time()
    for i in range(Y.shape[1]):
        fit(X, Y[:, i], k=1, n_restarts=1, steps=400, seed=3)
    single_s = time.time() - t0

    speedup = single_s / batched_s
    print(f"\n  batched {batched_s:.1f}s vs single {single_s:.1f}s -> {speedup:.1f}x")
    assert speedup > 1.5, f"only {speedup:.2f}x speedup"

"""Planted-direction generators for the Study 2 gate, shared by the experiment and its tests.

This lives apart from `s1_2_deflation.py` so the generative model can be tested without
loading a model. That matters more than usual here, because the multiplicative arm exists
*because of* a claim about this function: the stimulus projection is only expensive when
the secondary directions enter the response multiplicatively, and that claim is only
checkable if the function itself is reachable from a test.

`additive` reproduces `e03_required_n.py`'s unit exactly, so the joint baseline archived in
`results/e03_required_n.json` remains comparable. Do not "improve" it - the numbers in the
notebook are quoted against this form.

`multiplicative` chains the secondary directions as gates on the primary. With those
directions removed from the stimulus, a rank-1 fit has nothing left to explain them, which
is the mechanism behind the measured collapse of the projected method (0.506 against 0.930
for response-only). Chaining rather than special-casing K=2 keeps that property at K=3.
"""

import numpy as np

COUPLINGS = ("additive", "multiplicative")


def gelu(x: np.ndarray) -> np.ndarray:
    """The activation e03_required_n.py plants through, kept identical so the additive
    cells stay comparable to the archived baseline."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def planted(stimulus: np.ndarray, k: int, n_units: int, coupling: str = "additive",
            sparsity: float = 0.90, seed: int | None = None):
    """Return (V, Y) for k planted directions under the requested coupling.

    `V` is an orthonormal basis per unit, so alignment against it is well posed. `Y` is the
    scalar response the estimator sees. Per-cell seeding keeps the script reproducible on
    its own rather than depending on how far a shared rng had been advanced by earlier
    cells.
    """
    if coupling not in COUPLINGS:
        raise ValueError(f"unknown coupling {coupling!r}, expected one of {COUPLINGS}")
    if seed is None:
        seed = 1000 * k + stimulus.shape[0]

    rng = np.random.default_rng(seed)
    V = np.stack([np.linalg.qr(rng.standard_normal((stimulus.shape[1], k)))[0]
                  for _ in range(n_units)])
    Z = np.einsum("sd,ndk->snk", stimulus, V)

    lead = Z[:, :, 0]
    base = gelu(lead - np.quantile(lead, sparsity, axis=0))
    rest = [np.tanh(Z[:, :, j]) for j in range(1, k)]

    if coupling == "additive":
        Y = base.copy()
        for t in rest:
            Y = Y + t
    else:
        Y = base
        for t in rest:
            Y = Y * t
    return V, Y.astype(np.float32)
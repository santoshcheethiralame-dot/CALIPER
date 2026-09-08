"""Batched subspace fitting: all neurons in a layer share one stimulus matrix.

The single-neuron path recomputes the dominant cost - the projection of a
16,000 x 768 stimulus - once per neuron. That projection is a matrix-VECTOR product,
so it is memory-bound: each step streams ~49 MB to produce 16,000 numbers. Fitting n
neurons separately streams that n times.

Batching gives every neuron its own subspace and its own nonlinearity while projecting
all of them in a single GEMM, so the stimulus is read once per step instead of n times.
The FLOP count is unchanged; the memory traffic falls by a factor of n, which is what
actually costs time here.

Measured at 160 s/neuron unbatched, Phase A as specified needed ~583 hours. This is the
task that decides whether Phase A is feasible on the hardware we have.
"""

import numpy as np
import torch
from torch import nn

from .estimator import Fit, _to_original_frame


class _BatchedBottleneck(nn.Module):
    """n independent rank-k bottleneck models, evaluated as batched matmuls."""

    def __init__(self, n, d, k, width=64, seed=0, init=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(n, d, k, generator=g) / np.sqrt(d)
        if init is not None:
            v[:, :, 0] = torch.as_tensor(init, dtype=v.dtype)
        self.v = nn.Parameter(v)
        # Per-neuron MLP parameters, held as stacked tensors rather than n modules.
        self.w1 = nn.Parameter(torch.randn(n, k, width, generator=g) / np.sqrt(k))
        self.b1 = nn.Parameter(torch.zeros(n, width))
        self.w2 = nn.Parameter(torch.randn(n, width, width, generator=g) / np.sqrt(width))
        self.b2 = nn.Parameter(torch.zeros(n, width))
        self.w3 = nn.Parameter(torch.randn(n, width, 1, generator=g) / np.sqrt(width))
        self.b3 = nn.Parameter(torch.zeros(n, 1))

    def basis(self):
        q, _ = torch.linalg.qr(self.v)  # batched QR over the neuron axis
        return q

    def forward(self, s):
        # s: (samples, d) shared -> z: (samples, n, k) via one GEMM per k column
        q = self.basis()                                   # (n, d, k)
        z = torch.einsum("sd,ndk->snk", s, q)
        h = torch.tanh(torch.einsum("snk,nkw->snw", z, self.w1) + self.b1)
        h = torch.tanh(torch.einsum("snw,nwv->snv", h, self.w2) + self.b2)
        return torch.einsum("snw,nwo->sno", h, self.w3).squeeze(-1) + self.b3.T


def fit_batch(stimulus, responses, k=1, n_restarts=3, steps=2500, lr=3e-3,
              test_frac=0.2, width=64, seed=0, device="cpu", patience=200,
              warm_start=True, verbose=False):
    """Fit every column of ``responses`` against the shared ``stimulus``.

    Returns a list of ``Fit``, one per column, in the model's original coordinate frame.
    Early stopping is tracked per neuron - each keeps its own best held-out state - so
    batching does not couple neurons that converge at different rates.
    """
    X = np.asarray(stimulus, dtype=np.float32)
    Y = np.asarray(responses, dtype=np.float32)
    if Y.ndim == 1:
        Y = Y[:, None]
    n_neurons = Y.shape[1]

    s = torch.as_tensor(X, device=device)
    scale = s.std(0) + 1e-6
    s = (s - s.mean(0)) / scale
    scale_np = scale.cpu().numpy()

    a = torch.as_tensor(Y, device=device)
    a = (a - a.mean(0)) / (a.std(0) + 1e-6)

    n_test = int(len(s) * test_frac)
    s_tr, a_tr, s_te, a_te = s[n_test:], a[n_test:], s[:n_test], a[:n_test]

    init = None
    if warm_start:
        xtx = (s_tr.T @ s_tr).cpu().numpy()
        xty = (s_tr.T @ a_tr).cpu().numpy()            # (d, n)
        lam = 1e-3 * np.trace(xtx) / xtx.shape[0]
        beta = np.linalg.solve(xtx + lam * np.eye(xtx.shape[0]), xty)
        nrm = np.linalg.norm(beta, axis=0, keepdims=True)
        init = (beta / np.where(nrm > 0, nrm, 1.0)).T   # (n, d)

    best_r2 = np.full(n_neurons, -np.inf)
    best_basis = [None] * n_neurons
    per_restart = [[] for _ in range(n_neurons)]
    per_restart_r2 = [[] for _ in range(n_neurons)]

    for r in range(n_restarts):
        model = _BatchedBottleneck(n_neurons, s.shape[1], k, width=width,
                                   seed=seed + 1000 * r,
                                   init=init if (r == 0 and init is not None) else None
                                   ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        run_r2 = np.full(n_neurons, -np.inf)
        run_basis = [None] * n_neurons
        since = 0

        for step in range(steps):
            opt.zero_grad()
            # Sum over neurons: each neuron's loss touches only its own parameters,
            # so the gradients are identical to fitting them separately.
            ((model(s_tr) - a_tr) ** 2).mean(0).sum().backward()
            opt.step()

            if step % 25 == 0:
                with torch.no_grad():
                    pred = model(s_te)
                    ss_res = ((pred - a_te) ** 2).sum(0)
                    ss_tot = ((a_te - a_te.mean(0)) ** 2).sum(0)
                    te = (1 - ss_res / ss_tot).cpu().numpy()
                    improved = te > run_r2
                    if improved.any():
                        q = model.basis().detach().cpu().numpy()
                        for i in np.flatnonzero(improved):
                            run_r2[i] = te[i]
                            run_basis[i] = q[i]
                        since = 0
                    else:
                        since += 25
                        if since >= patience:
                            break
        if verbose:
            print(f"    restart {r}: median held-out R2 {np.median(run_r2):.4f}",
                  flush=True)

        for i in range(n_neurons):
            if run_basis[i] is not None:
                per_restart[i].append(_to_original_frame(run_basis[i], scale_np))
                per_restart_r2[i].append(float(run_r2[i]))
                if run_r2[i] > best_r2[i]:
                    best_r2[i] = run_r2[i]
                    best_basis[i] = per_restart[i][-1]

    out = []
    for i in range(n_neurons):
        f = Fit(subspace=best_basis[i], k=k, train_r2=float("nan"),
                test_r2=float(best_r2[i]))
        f.restarts = per_restart[i]
        f.r2_restarts = per_restart_r2[i]
        out.append(f)
    return out

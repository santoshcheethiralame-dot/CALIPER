"""Rank-K subspace estimation for a unit's response.

Williamson, Sahani & Pillow (2015) showed that maximally informative dimensions is
exactly maximum likelihood for a linear-nonlinear model. For a continuous response
that makes the estimator a rank-K bottleneck regression,

    minimise over V, f    sum_t ( a_t - f(V^T s_t) )^2

with V a D x K subspace and f a flexible nonlinearity. All K directions are fitted
jointly; greedy dimension-by-dimension search is biased under correlated stimuli.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


@dataclass
class Fit:
    subspace: np.ndarray  # (D, K) orthonormal columns
    k: int
    train_r2: float
    test_r2: float
    restarts: list = field(default_factory=list)      # subspace per restart
    r2_restarts: list = field(default_factory=list)   # held-out R2 per restart

    @property
    def r2_spread(self):
        """Max minus min held-out R2 across restarts. The cheap stability variant:
        no subspace comparison, just the number the fit already reports."""
        if len(self.r2_restarts) < 2:
            return float("nan")
        return float(max(self.r2_restarts) - min(self.r2_restarts))

    @property
    def stability(self):
        """Median pairwise alignment across restarts; ~1 means converged."""
        if len(self.restarts) < 2:
            return float("nan")
        vals = [abs(subspace_alignment(a, b)) for a, b in _pairs(self.restarts)]
        return float(np.median(vals))


class _Bottleneck(nn.Module):
    def __init__(self, d, k, width=64, seed=0, init=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(d, k, generator=g) / np.sqrt(d)
        if init is not None:
            # Warm-start the first direction from the ridge solution. Recovering a
            # LINEAR response is just regression, yet random starts left ~3% on the
            # table there (E0.1e: identity nonlinearity, 0.970 not 1.000), so the
            # optimiser - not the nonlinearity - was the binding constraint.
            v[:, 0] = torch.as_tensor(init, dtype=v.dtype)
        self.v = nn.Parameter(v)
        self.f = nn.Sequential(
            nn.Linear(k, width), nn.Tanh(),
            nn.Linear(width, width), nn.Tanh(),
            nn.Linear(width, 1),
        )

    def basis(self):
        # Orthonormalise so the subspace, not its parameterisation, is what is fitted.
        q, _ = torch.linalg.qr(self.v)
        return q

    def forward(self, s):
        return self.f(s @ self.basis()).squeeze(-1)


def whitening_frame(stimulus, eps=1e-3, n_components=None):
    """Return M with ``s @ M`` isotropic, and the mean that must be removed first.

    Whitening is not PCA truncation. It removes the anisotropy that lets random
    directions carry spurious information, while keeping every direction. Truncating
    instead would discard signal: MLP read-directions are close to isotropic in the
    residual stream's PCA basis, so a top-D subspace retains little of them.
    """
    mean = stimulus.mean(0)
    x = stimulus - mean
    _, s, vt = np.linalg.svd(x[: min(len(x), 20_000)], full_matrices=False)
    lam = (s**2) / max(len(x) - 1, 1)
    if n_components:
        vt, lam = vt[:n_components], lam[:n_components]
    return vt.T / np.sqrt(lam + eps * lam.max()), mean


def fit(
    stimulus,
    response,
    k=1,
    n_restarts=8,
    steps=1500,
    lr=3e-3,
    test_frac=0.2,
    width=64,
    seed=0,
    device="cpu",
    patience=200,
    frame=None,
    response_transform="none",
    warm_start=True,
    _init_subspace=None,
):
    """Fit a rank-k subspace. Returns the best restart by held-out R^2.

    ``frame`` is an optional (d_orig, d_fit) matrix; the fit runs on ``stimulus @ frame``
    and every returned subspace is mapped back as ``frame @ v``, so callers always
    receive directions in the model's own coordinates.

    ``response_transform`` reshapes the response before fitting. This is safe by
    construction: if ``a = f(V^T s)`` then ``g(a) = (g . f)(V^T s)`` for any pointwise
    ``g``, so the target subspace is invariant and only the estimator's efficiency
    changes. It matters because squared error on a heavy-tailed response is dominated
    by a handful of extreme events, which is the same reason the source literature
    counts spikes rather than stimulus presentations. Measured on GPT-2 layer 6, rank
    transformation lifted recovery on the worst neuron from 0.45 to 0.88.
    """
    if frame is not None:
        stimulus = np.asarray(stimulus) @ frame
    response = _transform_response(np.asarray(response, dtype=np.float64),
                                   response_transform)
    s = torch.as_tensor(np.asarray(stimulus, dtype=np.float32), device=device)
    a = torch.as_tensor(np.asarray(response, dtype=np.float32), device=device)

    # Standardising each coordinate conditions the optimisation, but it is a change
    # of basis: a direction v in standardised space corresponds to v / scale in the
    # original frame, since s' . v = s . (v / scale). Every subspace this function
    # returns is mapped back, so callers always work in the original coordinates.
    scale = (s.std(0) + 1e-6)
    s = (s - s.mean(0)) / scale
    a = (a - a.mean()) / (a.std() + 1e-6)
    scale_np = scale.cpu().numpy()

    n_test = int(len(s) * test_frac)
    s_tr, a_tr, s_te, a_te = s[n_test:], a[n_test:], s[:n_test], a[:n_test]

    init = None
    if _init_subspace is not None:
        # Caller supplies a starting subspace in the ORIGINAL frame; convert it to the
        # standardised frame the optimiser works in (the inverse of _to_original_frame).
        init = (np.asarray(_init_subspace)[:, 0] * scale_np)
        init = init / np.linalg.norm(init)
    elif warm_start:
        # Ridge solution in the standardised frame; costs one linear solve.
        xtx = (s_tr.T @ s_tr).cpu().numpy()
        xty = (s_tr.T @ a_tr).cpu().numpy()
        lam = 1e-3 * np.trace(xtx) / xtx.shape[0]
        beta = np.linalg.solve(xtx + lam * np.eye(xtx.shape[0]), xty)
        nrm = np.linalg.norm(beta)
        if nrm > 0:
            init = beta / nrm

    best, subspaces, r2s = None, [], []
    for r in range(n_restarts):
        # Restart 0 gets the warm start; the rest stay random so that restart
        # agreement remains a meaningful diagnostic rather than a foregone one.
        model = _Bottleneck(s.shape[1], k, width=width, seed=seed + 1000 * r,
                            init=init if r == 0 else None).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        best_te, best_state, since = -np.inf, None, 0

        for step in range(steps):
            opt.zero_grad()
            loss = ((model(s_tr) - a_tr) ** 2).mean()
            loss.backward()
            opt.step()

            # Early stopping on held-out fit, following the Sharpee (2006) rule of
            # halting when held-out performance falls away from its maximum.
            if step % 25 == 0:
                with torch.no_grad():
                    te = _r2(model(s_te), a_te)
                if te > best_te:
                    best_te, since = te, 0
                    best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
                else:
                    since += 25
                    if since >= patience:
                        break

        model.load_state_dict(best_state)
        with torch.no_grad():
            basis = _to_original_frame(model.basis().cpu().numpy(), scale_np, frame)
            tr = _r2(model(s_tr), a_tr)
        subspaces.append(basis)
        r2s.append(float(best_te))
        if best is None or best_te > best.test_r2:
            best = Fit(subspace=basis, k=k, train_r2=float(tr), test_r2=float(best_te))

    best.restarts = subspaces
    best.r2_restarts = r2s
    return best


def select_k(stimulus, response, k_max=3, tol=0.01, **kw):
    """Increase k until held-out R^2 stops improving by more than ``tol``.

    Saturation, not a significance test, is how the source literature selects the
    number of relevant dimensions.
    """
    fits, chosen = [], 1
    for k in range(1, k_max + 1):
        f = fit(stimulus, response, k=k, **kw)
        fits.append(f)
        if k > 1 and f.test_r2 - fits[-2].test_r2 < tol:
            break
        chosen = k
    return chosen, fits


def _transform_response(y, kind):
    """Pointwise reshaping of the response. Subspace-preserving; see ``fit``."""
    if kind in (None, "none"):
        return y
    if kind == "log1p":
        return np.log1p(y - y.min())
    if kind == "rank":
        from scipy import stats

        return stats.norm.ppf(stats.rankdata(y) / (len(y) + 1))
    raise ValueError(f"unknown response_transform {kind!r}")


def _to_original_frame(basis, scale, frame=None):
    """Map a subspace fitted in standardised coordinates back to the model's frame."""
    raw = basis / scale[:, None]
    if frame is not None:
        raw = frame @ raw
    q, _ = np.linalg.qr(raw)  # re-orthonormalise; the span is what matters
    return q


def subspace_alignment(a, b):
    """Mean cosine of principal angles between two subspaces; 1.0 means identical."""
    a = a / np.linalg.norm(a, axis=0, keepdims=True)
    b = b / np.linalg.norm(b, axis=0, keepdims=True)
    sv = np.linalg.svd(a.T @ b, compute_uv=False)
    return float(np.mean(np.clip(sv, -1, 1)))


def _pairs(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


def _r2(pred, target):
    ss_res = ((pred - target) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def _binned_r2(z, y, n_bins=60):
    """Best R^2 of any function of z, by equal-count binning. No optimisation."""
    n = (len(z) // n_bins) * n_bins
    order = np.argsort(z, kind="stable")[:n]
    ys = y[order].reshape(n_bins, -1)
    ss_res = ((ys - ys.mean(axis=1, keepdims=True)) ** 2).sum()
    yo = y[order]
    return 1.0 - ss_res / ((yo - yo.mean()) ** 2).sum()


def fit_cascade(stimulus, response, k=1, n_probe=720, polish_steps=800, seed=0,
                n_bins=200, n_candidates=5, **kw):
    """Fit k dimensions by descending from k+1, where the landscape is benign.

    E0.1g established the motivation precisely: on neurons the k=1 search fails, the
    objective AT the true direction is 1.000 while the search reaches 0.809 - and the
    k=2 subspace already contains the true direction at 0.997. The answer is being found
    and then lost in the parameterisation, so search inside the larger subspace instead.

    A k-dimensional subspace of a (k+1)-dimensional space is fixed by the single
    direction it discards, so the search is over the unit sphere in R^(k+1): a dense
    angle grid when k=1, sampled directions otherwise. Scoring uses binned R^2, which
    needs no training, and the winner is then polished by a short warm-started fit.
    """
    X = np.asarray(stimulus, dtype=np.float32)
    y = np.asarray(response, dtype=np.float64)
    big = fit(X, y, k=k + 1, seed=seed, **kw)
    Z = X @ big.subspace  # (n, k+1) - project once, then search in k+1 dimensions

    rng = np.random.default_rng(seed)
    if k == 1:
        theta = np.linspace(0, np.pi, n_probe, endpoint=False)
        dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # keep-directions
    else:
        dirs = rng.standard_normal((n_probe, k + 1))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    if k == 1:
        # The binned score is only a screen. It is bin-limited on units with a steep
        # tail - n230's true direction scores 0.700 under binning - so taking its argmax
        # alone picks near-misses. Shortlist instead, then decide with the real objective.
        scores = np.array([_binned_r2(Z @ d, y) for d in dirs])
        shortlist = dirs[np.argsort(scores)[-n_candidates:]]
        best_local, best_true = None, -np.inf
        for cand in shortlist:
            v = big.subspace @ cand[:, None]
            v = v / np.linalg.norm(v)
            trial = fit(X, y, k=1, n_restarts=1, steps=polish_steps, seed=seed,
                        warm_start=False, _init_subspace=v,
                        **{kk: vv for kk, vv in kw.items()
                           if kk not in ("n_restarts", "steps", "warm_start", "seed")})
            if trial.test_r2 > best_true:
                best_local, best_true = cand[:, None], trial.test_r2
    else:
        # For k > 1 the discarded direction defines the subspace; score its complement.
        best_local, best_score = None, -np.inf
        for u in dirs:
            q = np.linalg.qr(np.eye(k + 1) - np.outer(u, u))[0][:, :k]
            s = float(np.mean([_binned_r2(Z @ q[:, j], y) for j in range(k)]))
            if s > best_score:
                best_local, best_score = q, s

    v0 = big.subspace @ best_local  # back to the model's coordinate frame
    v0 /= np.linalg.norm(v0, axis=0, keepdims=True)

    kw2 = {kk: vv for kk, vv in kw.items()
           if kk not in ("n_restarts", "steps", "warm_start", "seed")}
    polished = fit(X, y, k=k, n_restarts=1, steps=polish_steps, seed=seed,
                   warm_start=False, _init_subspace=v0, **kw2)
    return polished if polished.test_r2 >= _binned_r2(X @ v0[:, 0], y) - 0.02 else Fit(
        subspace=v0, k=k, train_r2=float("nan"),
        test_r2=float(_binned_r2(X @ v0[:, 0], y)))

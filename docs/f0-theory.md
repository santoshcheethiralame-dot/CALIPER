# F-0: which checks can see which error (formal statements)

**15 October 2026.** The flagship's theory (`docs/flagship-plan.md` §5, F-0), as statements with
proofs, extending the working note `docs/f0-theory-note.md` (9 Oct) and its simulation
(`experiments/f0_simulation.py`, `results/f0_simulation.json`). Technical, not paper prose.

## Setup

A unit reads one direction $w \in \mathbb{R}^d$ of a stimulus $s$ with mean $\mu$ and covariance
$C = \sum_i \lambda_i q_i q_i^\top$, $\lambda_1 \geq \dots \geq \lambda_d \geq 0$: its response is
$y = f(w^\top s + b) + \varepsilon$. An estimator maps a sample $D = \{(s_t, y_t)\}_{t=1}^n$ and a
seed $\xi$ to a direction $\hat v = A(D, \xi)$. The error of a fit is $e = \hat v - w$ (both unit,
sign-aligned). A check is a statistic computed without $w$.

Write $\bar v = \mathbb{E}_D[\hat v]$ (fixing the seed rule), so
$e = \underbrace{(\bar v - w)}_{\text{bias } \beta} + \underbrace{(\hat v - \bar v)}_{\text{variance part } \eta}$,
with $\Sigma = \operatorname{Cov}_D(\hat v)$.

## Proposition 1 (seed checks are blind to the sample's error)

Let restart agreement on $D$ be $\rho(D) = \operatorname{med}_{j<k} |\cos(A(D,\xi_j), A(D,\xi_k))|$
for independent seeds. Then $\rho(D)$ depends on $D$ only through the basin structure of the
estimator's objective on $D$ and the seed distribution. In particular, if the objective on $D$ has a
unique minimiser $v^*(D)$ that the optimiser reaches from every seed, then $\rho(D) = 1$ for every $D$,
whatever $\lVert v^*(D) - w \rVert$, and $\rho$ has AUC exactly $1/2$ at predicting failure.

*Proof.* With the seeds independent of $D$, $\rho(D)$ is a function of the map $\xi \mapsto A(D, \xi)$,
which is fixed by $D$; $w$ enters only through $D$'s responses, and through nothing else. If every seed
reaches $v^*(D)$, every pairwise $|\cos|$ is 1, so $\rho \equiv 1$ is constant across units and carries
no ranking information (ties count half: AUC $= 1/2$). $\square$

*Remark.* With several basins, $\rho$ measures how the initialisation distribution splits among them
(basin mass, cf. Nehzati et al.); it bears on correctness only if wrong answers happen to have
smaller basins. Nothing in the estimation problem makes them so: a wrong basin can be the only one.

## Proposition 2 (resampling checks see variance, not bias)

For independent samples $D, D'$ and a fixed seed rule,
$$\mathbb{E}\lVert \hat v(D) - \hat v(D') \rVert^2 = 2 \operatorname{tr}\Sigma,
\qquad \mathbb{E}\lVert \hat v(D) - w \rVert^2 = \operatorname{tr}\Sigma + \lVert\beta\rVert^2 .$$
Hence the ratio of mean data-resampling disagreement to mean error is
$r = 2/(1 + \lVert\beta\rVert^2/\operatorname{tr}\Sigma) \in [0, 2]$: $r = 2$ exactly when the
estimator is unbiased, and $r \to 0$ as the error becomes pure bias.

*Proof.* $\hat v(D) - \hat v(D') = \eta - \eta'$ with $\eta, \eta'$ independent and mean zero, so the
first expectation is $2\operatorname{tr}\Sigma$. The second is $\mathbb{E}\lVert \beta + \eta\rVert^2
= \lVert\beta\rVert^2 + \operatorname{tr}\Sigma$ since $\mathbb{E}\eta = 0$. $\square$

*Measured.* Simulation Part A (least squares): $r = 2.01$ (well conditioned), 1.77 (ill), 1.04 with
ridge, 0.00 for a fixed distortion. Part B (Paper 1's estimator, noiseless GELU units): $r = 0.38$
and 0.80. The estimator's implicit regularisation (warm start, early stopping) turns much of its error
into bias, which resampling cannot see; this is why cross-sample agreement did not beat held-out fit
in X-1.

## Proposition 3 (where least-squares error lives)

For the linear model $y = w^\top s + \varepsilon$, $\varepsilon \sim (0, \sigma^2)$ i.i.d., and the
least-squares estimate $\hat w$ on centred $X \in \mathbb{R}^{n \times d}$ with $X^\top X \approx nC$,
$$\mathbb{E}\big[(q_i^\top(\hat w - w))^2\big] = \frac{\sigma^2}{n\lambda_i},$$
so the expected share of error energy in an eigen-subspace $L$ is
$\sum_{i \in L} \lambda_i^{-1} / \sum_i \lambda_i^{-1}$. For the residual stream's spectrum (condition
numbers $10^4$ to $10^9$ in Paper 1's arms), the subspace carrying the bottom 1% of variance holds
nearly all of the error.

*Proof.* $\operatorname{Cov}(\hat w) = \sigma^2 (X^\top X)^{-1} \approx \sigma^2 C^{-1}/n$, whose
$i$-th eigenvalue is $\sigma^2/(n\lambda_i)$. $\square$

*Consequence.* An answer whose weight sits in poorly sampled directions is one the data do not pin
down: the low-variance share is a geometric check that sees this variance error, and also any bias
that moves the answer into those directions (Part A, fixed distortion: AUC 0.92). It cannot see a bias
that lives where the data are rich.

## Proposition 4 (an exact null direction from the layer norm)

Let $s = \gamma \odot u + \beta$ with $\sum_k u_k = 0$ on every token (a layer norm's output), and
$n = (1/\gamma)/\lVert 1/\gamma\rVert$. Then $s^\top n$ is the same on every token, so for every
$v$ and $c$ the directions $v$ and $v + c\,n$ give predictions that differ by a constant. Any
estimator whose link has a free intercept (or any loss invariant to a constant shift of the
projection) has the same loss at both, from any amount of such data. The identifiable target is
$Pw$, $P = I - nn^\top$.

*Proof.* $s^\top(1/\gamma) = \sum_k u_k + \sum_k \beta_k/\gamma_k = \sum_k \beta_k/\gamma_k$. $\square$

*Corollary (why any off-manifold sample repairs it).* A single sample $\tilde s$ with
$\tilde s^\top n \neq s^\top n$ breaks the invariance, so the component along $n$ becomes identifiable.
Synthetic samples that are not layer-norm outputs (targeted or random perturbations of $s$) have this
property almost surely, natural tokens never do. This is F-1b's result (targeted 17/17, random 16/17,
natural 12/17 on units wrong on the identifiable label, which mixes this exact effect with
Proposition 3's poorly sampled directions) and the reason Gate F-A was read as "extra off-manifold data
repairs it, targeted or not".

## Proposition 5 (per-coordinate standardisation magnifies a near-null coordinate)

Suppose the estimator fits in standardised coordinates $z_k = (s_k - \mu_k)/\sigma_k$ and returns the
direction $v_k \propto u_k/\sigma_k$ in $s$-coordinates, where $u$ is the fitted direction in
$z$-coordinates. If one coordinate $k$ has $\sigma_k = \epsilon \lVert\sigma\rVert_{\mathrm{med}}$ with
$\epsilon \to 0$ and the fit assigns it any fixed weight $u_k \neq 0$, then
$$\frac{|v_k|}{\lVert v\rVert} \to 1 \quad\text{and}\quad |\cos(v, w)| \to \frac{|w_k|}{\lVert w\rVert}.$$
If moreover the fit is the same from every seed, restart agreement stays 1.

*Proof.* $v = D^{-1}u$ with $D = \operatorname{diag}(\sigma)$; the $k$-th entry grows as $1/\epsilon$
while the others stay bounded, so $v/\lVert v\rVert \to \pm e_k$ and $\cos(v, w) \to \pm w_k/\lVert w\rVert$.
Agreement is a property of $u$ across seeds and is unchanged by the fixed rescaling $D^{-1}$. $\square$

*Link to Proposition 4.* A near-zero gain $\gamma_k$ makes coordinate $k$ of $s$ nearly constant
($\sigma_k \approx |\gamma_k| \operatorname{sd}(u_k)$), and makes $n \propto 1/\gamma$ concentrate on
$e_k$. So the null direction and the magnified coordinate coincide, and the map from weights to
failure is read off $\min_k |\gamma_k|$ before any fit: GPT-Neo layer 6 ($|\gamma| = 2.4\times10^{-5}$,
all 20 units fail at alignment 0.01 with restart agreement 1.0) and GPT-2 layer 3 (predicted before
data, 0 of 20 pass; dropping the coordinate restores 9).

## What each check can see

| error type | example | seed (restart) agreement | resampling agreement | held-out / test fit | geometry (low-variance or null share) |
|---|---|---|---|---|---|
| variance in poorly sampled directions | Prop. 3 | no (Prop. 1) | yes (Prop. 2) | partly, through fit | yes |
| exact null component | Prop. 4, 5 | no, certifies it | no (every sample shares it) | no (fit unchanged) | yes (null share) |
| under-fitting | sparse units | no | partly | yes | no |
| bias where data are rich | fixed distortion | no | no | no | no |

The last row is the class nothing ground-truth-free detects; the flagship's claim is that the other
rows can each be detected by the right check, and that seed agreement detects none of them.

## Next

- Part B with a mixed regime (some units well, some ill conditioned) so the AUCs exist; the current
  Part B runs one regime at a time and every AUC is undefined.
- A version of Proposition 2 for the selected restart (best of k by validation fit), which is what
  the estimators report.

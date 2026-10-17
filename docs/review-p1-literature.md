# Literature for Paper 1's revision (17 Oct 2026), checked against the sources

Each item: what the source says (checked text), and where it enters the revision. Bib keys added to
`paper1/references.bib`. Items marked VERIFY need the author's check of page numbers or venue.

## Standardisation is recommended, and used, in this field

- **Timkey & van Schijndel 2021** (`timkey2021rogue`, already cited). A few "rogue" dimensions dominate
  cosine and Euclidean similarity in GPT-2 and BERT; their remedy is post-processing "such as
  standardization". Revision: the field's own fix for high-variance dimensions is the step that
  magnifies near-constant ones. Different setting (similarity, not probes); say so.
- **Goldowsky-Dill et al. 2025** (`goldowskydill2025deception`, arXiv 2502.03407; VERIFY venue).
  Section 3: logistic regression with L2 (lambda = 10) "on normalized activations (zero mean, unit
  variance). The classifier weights define our probe direction." Residual stream of Llama-3.3-70B,
  layer 22. Used as a detector (a score), so only predictive equivalence matters for that use; the
  residual stream is not the MLP input our map covers. Revision: evidence the pipeline is in use; and
  the right example of a use the collapse does not harm.
- **scikit-learn `StandardScaler`** (source, `sklearn/preprocessing/_data.py`): a feature gets scale 1
  only if `var <= n*eps*var + (n*mean*eps)^2` (Chan, Golub and LeVeque's bound), i.e. constant to
  machine precision. A coordinate with SD 1e-4 to 1e-5 of the median is divided by its SD. Revision:
  explains E-3b's immune case (exactly constant) and why near-constant is exposed.
- **Kuhn & Johnson 2013** (`kuhn2013applied`; caret's `nearZeroVar`). Removing near-zero-variance
  predictors before modelling is standard practice in predictive modelling. Revision: the variance
  filter repair is old practice, not ours to claim; note that caret's rule targets discrete
  predictors (frequency ratio, unique values), not continuous activations.

## Why a decoder's direction is not the intervention's

- **Li et al. 2023, ITI** (`li2023iti`). Table 3 (LLaMA-7B): intervening along the probe weight direction
  gives True*Info 34.8, along the mass-mean shift 42.3 (baseline 30.5). Section 5.5: the all-heads probe's
  accuracy "is insensitive to normalization methods, including feature normalization and PCA".
  Revision: accuracy is blind to the scaling choice; the decoding direction is the weaker intervention.
- **Marks & Tegmark 2024, COLM** (`marks2024geometry`). Difference-in-mean probes "generalize as well
  as other probing techniques" while "identifying directions which are more causally implicated in
  model outputs". Revision: supports the difference of means as E-3's reference, and the three-notion
  framing.
- **Elazar et al. 2021, TACL** (`elazar2021amnesic`) and **Belinkov 2022, CL** (`belinkov2022probing`).
  Probe accuracy shows information is decodable, not that the model uses it; amnesic probing finds
  probe accuracy uncorrelated with task importance in BERT. Revision: cite where the paper separates
  predictive equivalence from intervention.
- **Gurnee & Tegmark 2024, ICLR** (`gurnee2024space`). Section 5 finds neurons whose weights have high
  cosine with the learned probe direction. The paper does not say whether activations are standardised
  (a secondary claim that it does could not be confirmed and is not used). Revision: an example of the
  direction-reading use (probe-to-weight cosine) that a collapsed direction would break.

## LayerNorm parameters and outlier dimensions (related work)

- **Kovaleva et al. 2021** (`kovaleva2021busters`). Outlier LayerNorm scaling factors and biases
  (< 0.0001% of weights) whose removal degrades BERT-family models by up to 44 points; also in GPT-2.
  These are large gains; ours are near-zero gains. Revision: related work on LayerNorm dimensions.
- **Sun et al. 2024** (`sun2024massive`, already cited). Massive activations at very few tokens.

Not used: claims found only in secondary summaries (e.g. that a 2026 paper standardises "following
Gurnee and Tegmark").

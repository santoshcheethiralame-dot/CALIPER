# Pre-registration: F-3a, does the layer map predict the standardisation artifact in a third family?

**Filed 16 October 2026, before any F-3a fit.** Flagship (`docs/flagship-plan.md`, F-3 small, on
the laptop CPU) and Paper 1 (the failure mode beyond GPT-2 and GPT-Neo). Units:
`results/f3a_units.json` (`experiments/f3a_units.py`). Layer map: `results/f3_layer_map.json`
(`experiments/f3_layer_map.py`), computed from the stimulus alone. Scorer:
`experiments/analyse_f3a.py`, developed on B-17c/d only. Queue: `experiments/rerun_queue18.sh`,
after queue 17.

## Why

B-17b/c/d showed that per-coordinate standardisation turns a nearly constant layer-norm output
coordinate into a failure that restart agreement certifies, and that the layer map (SD below 1% of
the median) predicted it at GPT-2 layer 3 before any fit there. F-0's Proposition 5 states the
mechanism. Both models were GELU models from one lineage of training code. OPT is a different
family with a ReLU link.

**Code fix before the filing.** `caliper/activations.py::_mlp_ln` returned nothing for OPT blocks,
so OXIDE's identifiable label fell back to the Euclidean label there. It now returns
`final_layer_norm` (pre-LN sizes) or `self_attn_layer_norm` (OPT-350m, post-LN), the layer norm
whose output `fc1` reads. On OPT-125m layer 6 the hooked tensor equals `fc1`'s input exactly, so
the stimulus the fitter sees is unchanged.

**The map** (OPT-125m, 8,000 tokens of corpus seed 0, no fit):

| layer | near-constant coordinates | min SD / median | min gain |
|---|---|---|---|
| 3 | 2 (174, 526) | 4.6e-3 | 2.8e-4 |
| 5 | 1 (355) | 9.0e-4 | 1.5e-4 |
| all others | 0 | >= 0.12 | >= 0.029 |

For comparison, GPT-2 layer 3 (B-17c) was 1.6e-3 and GPT-Neo layer 6 (B-17) 2e-4.

## Design

Same settings as B-17c/d: `experiments/e01_gate.py`, `--independent-units`, 2 restarts, 8,000
tokens, 1,600 steps, float32, saved directions, `--d-mlp 3072`. The same 20 units (drawn with
`default_rng(20261018)` from 3,072) at every layer.

| run | layer | flag |
|---|---|---|
| F-3a-1 | 5 | none |
| F-3a-2 | 5 | `--drop-constant-coords 0.01` |
| F-3a-3 | 3 | none |
| F-3a-4 (control) | 6 | none |

## Predictions

The share is each fitted direction's squared norm on the layer's near-constant coordinates (on its
lowest-SD coordinate where it has none), as in `analyse_b17b.py`.

- **P1 (the map predicts it at L5).** At least 10 of 20 F-3a-1 fits have share > 0.5.
- **P1b (and at L3, the weaker coordinate).** At least 5 of 20 F-3a-3 fits have share > 0.5.
- **P2 (restart agreement certifies it).** Among F-3a-1 fits whose direct-route direction has
  share > 0.5, median restart agreement is at least 0.99.
- **P3 (the fix transfers).** F-3a-2's median Euclidean alignment exceeds F-3a-1's by at least
  0.30, and F-3a-2 passes at least as many units.
- **P4 (control).** At most 2 of 20 F-3a-4 fits have share > 0.5 on layer 6's lowest-SD coordinate.

## Secondary (no predictions)

1. Pass counts and median alignment on the identifiable label in every run. On B-17c (development)
   the identifiable label recovers 5 of 20 at median 0.71, against 9 of 20 at 0.91 when the
   coordinate is dropped: removing 1/gamma is not the same repair as dropping the coordinate.
2. How many units are under-fitted at 8,000 tokens (ReLU units are sparse; the F-3 pilots on
   OPT-1.3b needed 120k tokens for the 20-80% band). P3 is stated on alignment, not passes, for
   that reason.

## Readings

- **P1 and P2 hold:** the failure mode and its certification by restart agreement are predicted from
  the weights in a third family with a different link. Paper 1 states the failure mode as general
  to per-coordinate standardisation, with three families.
- **P1 fails:** the map's threshold does not transfer to OPT, and that is reported.
- **P3 fails while P1 holds:** the artifact is there but OPT's fits are limited by sparsity, so the
  repair cannot show at 8,000 tokens; reported as such, not re-run at a larger budget under this
  filing.

## Not done

No threshold (1% of the median SD, share 0.5, 10 / 5 / 2 of 20, 0.99, 0.30), unit or layer is
changed after seeing output. A fit that fails to run is reported as missing.

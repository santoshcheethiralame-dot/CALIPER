# T-SAE Amendment 1: an operating-point pilot comes first, and eligibility is set on our stimulus

**Filed 7 October 2026, before any T-SAE run.** The only data seen is from code-path smoke
runs (7 latents in total), listed below. No T-SAE run has started.

## What the smoke runs showed

1. **The SAE's density statistics do not transfer to our stimulus.** They were measured on
   OpenWebText. On our Gutenberg stimulus the correlation between log10 density and log10
   observed firing count is 0.50, and latents rated about 10^-2.8 fired on 17 and 7 of 8,000
   tokens. Under the original eligibility rule (density >= 10^-3), most latents would fire
   too rarely to fit.
2. **At the MLP operating point (8,000 tokens, 1,600 steps), SAE latents are much harder than
   MLP neurons:**

   | firing events | alignment | held-out R2 |
   |---|---|---|
   | 117, 139 | 0.35, 0.39 | 0.83, 0.85 |
   | 251 | 0.64 | — |
   | 500 | 0.60 | — |
   | 688 | 0.81-0.88 | — |

   The identity test passes (`tests/test_sae_target.py`). The true direction gives R2 = 1, and
   these fits do not reach it, so this is optimisation and under-fitting, not a broken
   pipeline. At this budget almost every latent would fail. The class split would have nothing
   but under-fitting in it, and the transfer test would carry no information.

## Amended design

**Step 1: pilot T-SAE-0, to choose the operating point (the analogue of E0.3b).**
- **Units:** 16 fixed latents passed with `--units`, drawn by `draw_by_firing(min_events=100,
  seed=1)` on the 8,000-token stimulus, four per firing-count quartile (108-640 events):
  `3652,20588,24478,14970,7537,2721,5559,5507,15372,24515,14404,8615,1392,19106,18647,12834`.
  Seed 1 keeps the pilot latents distinct from the main draw (seed 0).
- **Two budgets, the same latents:**
  - **A:** 8,000 tokens, 1,600 steps (the MLP operating point);
  - **B:** 16,000 tokens, 3,200 steps.

  Both use `--independent-units` and 2 restarts.
- **Selection rule, fixed now:**
  - If A's failure rate is between 20% and 80% (4-12 of 16): **use A**.
  - Else, if B's is between 20% and 80%: **use B**.
  - Else, if both are above 80%: use B and set N = 100. The paper then reports that at the
    largest budget we can afford, the estimator fails on most SAE latents. That is itself a
    finding about transfer, and the signal calibration is still scored.
  - Else, if A is below 20%: use A.
- The pilot's AUCs are **not** looked at. Only its failure counts select the operating point.

**Step 2: the main T-SAE run, after the pilot.**
- **Eligibility:** at least 100 firings per 8,000 tokens in the stimulus actually fitted:
  `--sae-min-events 100` at A, `200` at B.
- **Draw:** `draw_by_firing` across firing-count quartiles, seed 0. The firing-count quartile is
  the difficulty dial and is recorded on every row.
- **N:** 200, or 100 under the third branch.
- **Unchanged from the original filing:** the SAE, layer, centring, bar, endpoints and decision
  table.

## Why this is not a post-hoc change

The operating point and eligibility are set from failure *counts* and the stimulus alone,
never from the reliability signals' AUCs, which are the quantities under test. This is the
same procedure that set the MLP operating point (E0.3b) before the MLP gate ran.

## Amendment 2 (7 October 2026, before the pilot runs): the rule's last case, run order

- **The missing case.** The rule above does not cover "A above 80% and B below 20%". That
  case uses **B at N = 200**: it is the only budget that fits. The rule is coded in
  `experiments/tsae_operating_point.py`, which reads `align_selected` only.
- **Order.** The rule is applied as written: A if it qualifies, then B. A = 3 and B = 9 of 16
  selects B.
- **Queue order** (`experiments/rerun_queue2.sh`): pilot A, pilot B, B-15a/b/c, B-17, then the
  main T-SAE run at the selected point. The main run is last because it is the longest:
  about 3 min per latent at A, so about 10 h at N = 200, and roughly four times that at B.

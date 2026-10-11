# Pre-registration: F-6, unembedding rows in GPT-2 (with Pythia-410m as a positive control)

**Filed 10 October 2026, before any F-6 main-run data.** Flagship substrate F-6
(`docs/flagship-plan.md` §5, §13). Pilot (discarded): `results/flagship_pilot1/`,
`results/flagship_pilot2/`. Scorer: `experiments/analyse_substrate.py`, committed with this filing.
Run sheet: `kaggle/NEXT_SESSION_F4F6.md`.

## Substrate and label

A vocabulary token's logit is exactly W_U[t] . s, where s is the final norm's output: a linear link
with no bad basins, so every failure is geometric (`e01_gate.py --target unembed`). GPT-2's final
norm `ln_f` is a LayerNorm with a near-zero gain (min |gamma| 0.0044). In the pilot every
Euclidean fit failed along its null direction u = 1/gamma, with held-out R2 1.0 and restart
agreement 0.99. **Primary label: identifiable**, 1/gamma of `ln_f` removed from fit and reference,
alignment >= 0.95 to pass. The Euclidean label is reported beside it.

## Units, budget and settings

GPT-2: 100 vocabulary rows drawn with `default_rng(0)` (the script's own draw). 32,000 tokens (the
pilot gave 5/16 identifiable passes at 32k, 0/16 at 8k and 16k). 3,200 steps, 2 restarts.
- **Fit A:** corpus seed 0 (300 documents, about 34k tokens).
- **Fit B:** corpus seed 1, `--corpus-docs 2000 --exclude-corpus-seed 0` (about 780 disjoint
  documents, about 99k tokens available; added 10 Oct so a disjoint fit can reach 32k).

Positive control: Pythia-410m, 30 rows, 8,000 tokens, fit A only. Pythia's final LayerNorm has no
near-zero gain (min 0.74), and the pilot recovered 15-16 of 16 rows.

## Primary endpoint and prediction

**F-6 isolates the failure the theory says restarts cannot see.** With a linear link every restart
reaches the same optimum, so restart agreement should be near 1 for every row and carry no
information. Fits on disjoint documents should differ exactly where the data are thin, which is
where an identifiable failure lives (F-0: data-resampling disagreement estimates the variance part
of the error).

Primary, on fit A with the identifiable label: AUC(cross-sample agreement) minus AUC(restart
agreement), DeLong and stratified bootstrap. **Prediction: positive, with the bootstrap interval
above 0.**

## Secondary

1. AUC(restart agreement) minus AUC(held-out R2) (the Paper 1 contrast), descriptive: held-out R2
   sits near 1 for every row, so both are expected near chance.
2. The fit's share on 1/gamma as a check on the Euclidean label (the pilot found it on every fit).
3. **Gate F-B:** the identifiable failure rate with a Wilson interval, and whether it lies in band.
4. Positive control: at least 27 of 30 Pythia-410m rows pass, or the pipeline is examined before
   any GPT-2 number is read.

## Not done

Budget, units and label fixed now; no second budget if fit A falls outside the band.

## Amendment 1 (11 October 2026, before any F-6 row has been read)

**Why.** The run sheet budgeted 4-6 GPU-hours for the session, but the pilot had already measured
425 s per GPT-2 unembedding row at 32k tokens (`results/flagship_pilot1/pilot_f6_gpt2_32k_summary.json`),
so the two 100-row fits need about 24 GPU-hours, beyond Kaggle's 12-hour session. The first
attempt (11 Oct) reached 32 of 100 rows of fit A after 3.7 hours and was stopped before any output
was retrieved; no F-6 row has been read.

**What changes.** 50 rows per fit instead of 100: the first 50 of the filed 100-row draw
(`--neurons 50 --neuron-pool 100`, same seed), identical in fits A and B. Fits A and B run in
parallel on the two T4s, with `--batch 10` so that rows are saved every 10, and a deadline that
stops them cleanly before the session limit; rows finished by then are scored, and the count is
reported. Budget, label, endpoints, prediction and the band are unchanged.

**Power.** On the pilot's 32k cell about two thirds of rows fail on the identifiable label, so 50
rows give roughly 30 failures and 15-20 passes for the primary AUC difference. If fewer than 10
identifiable failures or 10 passes are in fit A, the primary is reported as underpowered.

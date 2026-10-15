# Gate F-A: decision memo (prepared 15 Oct 2026; gate date 15 Dec)

**Decision needed from:** the author (with the mentor if wanted). This memo sets out the evidence
and a recommendation; nothing in the plan changes until the decision is recorded.

## The gate as filed

`flagship-plan.md` §7, Gates: *"F-A, 15 Dec. F-1 result. Pass: targeted interventions recover
>= 80% of converged-wrong fits and beat natural tokens. Fail: the flagship leads with detection
(findings 5-7, F-10), and the remedy is reported as tested and negative."* After F-1 (10 Oct), the
plan moved the gate's reading to F-1b, because F-1's converged-wrong units were mostly wrong only
along the layer norm's null direction, which any synthetic sample pins.

## The evidence: F-1b, scored once 13 Oct (`results/f1b_analysis.json`)

Population A, 17 units wrong on the identifiable label (converged-wrong):

| arm | repaired | against re-fit only |
|---|---|---|
| targeted (low-variance directions) | 17 / 17 | primary: 9 vs 0 discordant, one-sided McNemar $p$ = 0.002, **holds** |
| random augmentation | 16 / 17 | 0 vs 1 against targeted |
| natural tokens (matched count) | 12 / 17 | targeted vs natural 5 vs 0, two-sided $p$ = 0.0625 |
| re-fit only (noise baseline) | 8 / 17 | |

Population B (30 under-fitted): targeted 26, random 27, natural 15, re-fit 13. Population C (10
passing, harm): 0, 0, 1 and 1 broken.

## Reading against the gate

- **Recover >= 80%:** met (17 of 17, 100%).
- **Beat natural tokens:** not shown. The direction favours targeted (5 vs 0 discordant), but at
  17 units the test cannot reach significance; the filed F-1b reading was "primary holds but
  targeted is close to natural".
- **What the data do show:** extra samples off the natural stimulus manifold repair the hard case,
  and *targeting is not needed*: random augmentation does as well as targeted (16 vs 17). The
  repair is real against re-fitting alone ($p$ = 0.002) and harms no passing unit.

As written, F-A does not pass, because the second condition fails. But the fail branch ("the remedy
is reported as tested and negative") would misdescribe the result: the remedy works; what fails is
the claim that it must be targeted.

## Options

1. **Record F-A as a partial pass and reword the claim (recommended).** The flagship's third clause,
   "interventional data repairs it", becomes "additional off-manifold data repairs it; targeting it
   adds nothing measurable over random augmentation, and natural tokens repair less (underpowered)".
   The flagship keeps the remedy as a finding, but the headline leads with detection, as the fail
   branch says. Cost: none now.
2. **Power the natural-versus-targeted comparison before 15 Dec (F-1c).** Five discordant pairs to
   none would need roughly 40 to 50 identifiable converged-wrong units for a clear answer. At about
   one such unit per 20 fitted, that is 800 to 1,000 base fits plus four arms: on the order of 30 to
   50 CPU-hours. Worth it only if "targeting beats natural data" matters to the flagship's story.
3. **Record F-A as failed and follow the fail branch verbatim.** Simplest, but it understates a
   working remedy.

## Recommendation

Option 1 now, written into `flagship-plan.md` with this memo as the record, so December's compute
goes to F-4 and F-2. Keep option 2 as a candidate for the January CPU queue if the flagship draft
needs the targeted-versus-natural contrast.

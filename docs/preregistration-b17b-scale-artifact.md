# Pre-registration: B-17b/c/d, is the GPT-Neo layer-6 collapse a standardisation artifact?

**Filed 8 October 2026, before any B-17b/c/d run.** Paper 1, after the exploratory pattern
pass (notebook §4, 8 Oct; `results/explore_patterns.json`).

## Why

**What B-17 showed.** At GPT-Neo-125m layer 6, all 20 units fail the Euclidean bar, at a
median alignment of 0.01.

**What the pattern pass found.**
- One layer-norm output coordinate is nearly constant: SD 1.65e-5, against a median of 0.083.
- The estimator standardises each coordinate and maps directions back by dividing by the SD.
  The fit's arbitrary weight on that coordinate is therefore magnified about 5,000-fold.
- Every B-17 fit puts more than half its squared norm on it (median 0.99985; development run
  `results/b17b_dev.json`).

**The layer map's prediction.** The same artifact should appear wherever near-constant
coordinates exist. Among layers we have not fitted, the map shows one at GPT-2 small layer 3
(minimum SD 1.6e-3 of the median).

## Design

All runs use the corrected estimator (`--independent-units`), 2 restarts, 8,000 tokens,
1,600 steps, float32 and saved directions.

| run | model, layer | units | flag |
|---|---|---|---|
| B-17b | GPT-Neo-125m L6 | B-17's 20 units | `--drop-constant-coords 0.01` |
| B-17c | GPT-2 small L3 | 20 units drawn with `default_rng(20261009)` (`results/b17c_units.txt`) | none |
| B-17d | GPT-2 small L3 | the same 20 units | `--drop-constant-coords 0.01` |

**What `--drop-constant-coords 0.01` does.**
- It fits without the coordinates whose SD is below 1% of the median.
- It puts zeros back in those positions of every fitted direction.
- Nothing else changes.

## Predictions

**P1 (rescue).** B-17b passes at least 10 of 20 units, at a median alignment of at least
0.90. B-17 as run passed 0 of 20 at 0.01.

**P2 (the map predicts the artifact).** B-17c has at least one near-constant coordinate,
and at least 5 of its 20 fits put more than half their squared norm on near-constant
coordinates.

**P3 (the fix works there too).**
- B-17d passes more units than B-17c.
- B-17d's fitted directions have zero norm on the dropped coordinates, which is true by
  construction.

## Readings

- **P1 holds:** the GPT-Neo layer-6 collapse is a standardisation artifact. Paper 1 reports
  it as such, with the warning that per-coordinate standardisation of activations must not
  magnify near-constant coordinates.
- **P1 fails:** the artifact is not the whole story at that layer, and it is reported as
  such.
- **P2 holds:** the layer map predicts the artifact in advance.
- **P2 fails:** the map's threshold is too loose to predict it, and that is reported as such.

## Analysis

`experiments/analyse_b17b.py`, developed on B-17 alone (`--dev`) before any new run. Run once
on the filed runs.

## Not done

No threshold (1% of the median SD, 0.5 share, 10 of 20) or unit is changed after seeing
output.

# S-12: human grading protocol

**Why this exists.** Papers 2 and 3 report detection and steering rates scored three ways:
- by rule (regexes in `kaggle_s3_positive_control.py`);
- by first-token probability;
- by a small local judge.

C48 showed the literal-word rule undercounting steering (0/30 against 10/30). No judge number
may appear in a paper until two humans agree on the same labels (Cohen's kappa ≥ 0.6). This
is the most likely item to slip, so it starts in October.

## The files

| file | who opens it |
|---|---|
| `docs/team/s12_items.csv` | labellers: 200 items, the transcript and the question |
| `docs/team/s12_key.csv` | **nobody, until both labellers are done**: source, alpha, injected concept, rule scores |

The sample was drawn by `experiments/s12_sample.py` (seed 12). It is stratified by source ×
alpha band × rule outcome, so the rare rule-positive rows are well represented: 120 `detect`
items and 80 `steer` items. Transcripts are cut at 300 characters, as the runs stored them.

## How to label

Each labeller works alone on a copy, `s12_labeller_<initial>.csv`. Do not discuss items until
both copies are finished.

**`detect` items** (a reply to "Do you detect an injected thought? Answer starting with YES or
NO"):
- `said`:
  - `YES` if the reply affirms that a thought or concept was detected;
  - `NO` if it denies it;
  - `UNCLEAR` if it does neither, refuses the premise ("As an AI…"), or contradicts itself.

  Judge what the reply asserts, not only its first word.
- `concept_named`: the concept the reply says it detected, in one word, or `NONE`. Write a
  concept that is only implied (for example, a reply about lava and craters) as
  `implied:volcano`.
- `coherent`: `Y` if it reads as English a person might write, `N` if it is repetitive,
  broken or derailed.

**`steer` items** (a short story written while a concept vector was injected):
- `about_concept`:
  - `Y` if the story is about the named concept or something tightly tied to it (an elephant
    story; a story about tusks and herds);
  - `N` if it is not;
  - `UNSURE` if it is borderline.
- `coherent`: as above.

Use `notes` for anything odd. Expect about 45 minutes per 100 items.

## Scoring

```bash
python experiments/s12_kappa.py docs/team/s12_labeller_A.csv docs/team/s12_labeller_B.csv --key docs/team/s12_key.csv
```

It reports human-human kappa with a bootstrap CI, and each human against the rule scorer.
For the old steering files, the rule is the literal-word `identified` field, since `steered`
did not exist yet. Once a judge has labelled the same items, pass `--judge`.

**Gate** (run plan): human-human kappa ≥ 0.6 on `said` and on `about_concept`.
- **If it passes:** disagreements are resolved by discussion into one gold label per item,
  and the judge is scored against gold.
- **If it fails:** judge numbers drop to secondary, and the rule and first-token readouts
  carry every claim.

Either way, the result is reported in Paper 2's appendix.

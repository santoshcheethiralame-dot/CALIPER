# S-12: human grading protocol

**Why this exists.** Papers 2 and 3 report detection and steering rates scored three ways:
- by rule (regexes in `kaggle_s3_positive_control.py`);
- by first-token probability;
- by a small local judge.

C48 showed the literal-word rule undercounting steering (0/30 against 10/30). No judge number
may appear in a paper until the human labels are shown to be reliable. With one labeller,
reliability is measured within the labeller: a blind re-label of a fixed 60-item subset at least
7 days after the first pass, in a new order, must reach Cohen's kappa ≥ 0.6. This is the most
likely item to slip, so it starts in October.

Changed 8 Oct 2026: the earlier two-labeller design is replaced. All labelling is done by one
person, so human-human kappa becomes intra-rater (test-retest) kappa. Its limit is stated
wherever the gate is used: it measures consistency, not agreement between people.

## The files

| file | who opens it |
|---|---|
| `docs/s12/s12_items.csv` | the labeller: 200 items, the transcript and the question |
| `docs/s12/s12_key.csv` | **not opened until both passes are done**: source, alpha, injected concept, rule scores |
| `docs/s12/s12_relabel_ids.txt` | the 60 item ids for the second pass (36 detect, 24 steer; seed 13), in the order to label them |

The sample was drawn by `experiments/s12_sample.py` (seed 12). It is stratified by source ×
alpha band × rule outcome, so the rare rule-positive rows are well represented: 120 `detect`
items and 80 `steer` items. Transcripts are cut at 300 characters, as the runs stored them.

## How to label

**Pass 1.** Label all 200 items in a copy, `docs/s12/s12_pass1.csv`.

**Pass 2.** At least 7 days later, without looking at pass 1, label only the 60 items in
`s12_relabel_ids.txt`, in that file's order, in `docs/s12/s12_pass2.csv`.

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
python experiments/s12_kappa.py docs/s12/s12_pass1.csv docs/s12/s12_pass2.csv --key docs/s12/s12_key.csv
```

It reports pass-1 vs pass-2 kappa on the 60 shared items, with a bootstrap CI, and each pass
against the rule scorer.
For the old steering files, the rule is the literal-word `identified` field, since `steered`
did not exist yet. Once a judge has labelled the same items, pass `--judge`.

**Gate** (run plan): human-human kappa ≥ 0.6 on `said` and on `about_concept`.
- **If it passes:** disagreements are resolved by discussion into one gold label per item,
  and the judge is scored against gold.
- **If it fails:** judge numbers drop to secondary, and the rule and first-token readouts
  carry every claim.

Either way, the result is reported in Paper 2's appendix.

# Anonymous release

Code, pre-registrations, every run and the laboratory notebook behind the paper. Identity strings
have been replaced and notebook lines about course administration or the authors' other work are
marked as redacted; nothing else is changed.

## Layout

- `caliper/`: the estimator, the activation collection and the reference-standard targets.
- `experiments/`: every run and analysis script. `e01_gate.py` fits one arm; the `analyse_*.py`
  scripts produce the paper's tables from `results/`.
- `results/`: every run's per-unit rows (`*.jsonl`), run summaries, and fitted directions where saved
  (`*_dirs/n<unit>.npz`, holding the reference column `w` and each route's fit).
- `docs/`: the pre-registrations (`preregistration-*.md`) and the laboratory notebook, which dates
  every decision and deviation.
- `tests/`: identity and regression tests (`python -m pytest tests`).
- `results/corpus_cache/`: the public-domain Project Gutenberg texts used as stimulus.

## Reproducing the main tables

```
PYTHONPATH=. python experiments/analyse_review_round1.py   # per-arm AUCs, pooling, route-matched
PYTHONPATH=. python experiments/analyse_ln_null.py         # the layer-norm null-direction table
PYTHONPATH=. python experiments/analyse_every_run.py       # every-run table and sensitivity pools
```

Python 3.12, PyTorch and Hugging Face `transformers`; models download from the Hub.

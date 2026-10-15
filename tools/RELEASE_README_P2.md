# Anonymous release: the concept-injection audit

Code, run sheets, pre-registrations, every run and the laboratory notebook behind the paper. Identity
strings have been replaced, an earlier project's name has been renamed throughout (the sentence-mean
recipe is called `sentmean`), and notebook lines about course administration or the authors' other
work are marked as redacted; nothing else is changed.

## Layout

- `experiments/kaggle_s3_positive_control.py`: the audit script (vectors, steering gate, forced choice,
  health checks); `kaggle_p2h_hahami.py` and `kaggle_p2e_eval_awareness.py` for the positive control
  and the evaluation-awareness arm.
- `experiments/analyse_*.py` and `experiments/rescore_steering_gate.py`: every scorer and analysis.
- `results/`: per-session folders of rows (`*.jsonl`), configs and saved vectors (`*.vectors.npz`), and
  every analysis JSON the paper's tables are generated from.
- `docs/`: the pre-registrations (`preregistration-*.md`) and the laboratory notebook.
- `tests/`: regression tests (`python -m pytest tests`).

## Reproducing a table

```
PYTHONPATH=. python experiments/analyse_s2.py --dir results/s2_qwen3b --model qwen3b --quant none --out s2.json
PYTHONPATH=. python experiments/analyse_p2n.py
```

Python 3.12, PyTorch and Hugging Face `transformers`; models download from the Hub.

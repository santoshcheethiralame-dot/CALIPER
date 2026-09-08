"""DeLong's test for two correlated ROC curves, and the BH correction beside it.

Filed as B-1's primary comparison in docs/preregistration-b1-addendum-1.md, which
commits to unit-testing the implementation before pointing it at real data. These are
those tests. They run in milliseconds and need no model.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from b1_signal_calibration import benjamini_hochberg, delong, roc  # noqa: E402

N_POS, N_NEG = 25, 75
LABELS = [1] * N_POS + [0] * N_NEG


def _signals(seed=0):
    random.seed(seed)
    strong = [random.gauss(1.2, 1) if y else random.gauss(0, 1) for y in LABELS]
    weak = [random.gauss(0.4, 1) if y else random.gauss(0, 1) for y in LABELS]
    return strong, weak


def test_identical_signals_have_exactly_zero_difference():
    random.seed(1)
    a = [random.gauss(0, 1) for _ in range(len(LABELS))]
    _, _, diff, se, _, _ = delong(a, a, LABELS)
    assert abs(diff) < 1e-12
    assert se < 1e-12


def test_auc_agrees_with_the_independent_mann_whitney_route():
    """delong() and roc() compute AUC by different code paths; they must agree."""
    strong, weak = _signals()
    auc_a, auc_b, *_ = delong(strong, weak, LABELS)
    assert abs(auc_a - roc(strong, LABELS)[1]) < 1e-9
    assert abs(auc_b - roc(weak, LABELS)[1]) < 1e-9


def test_hand_computed_auc():
    """positives {3,1}, negatives {2,2,0}: wins = 3>2, 3>2, 3>0, 1>0 = 4 of 6."""
    auc, *_ = delong([3, 1, 2, 2, 0], [3, 1, 2, 2, 0], [1, 1, 0, 0, 0])
    assert abs(auc - 4 / 6) < 1e-12


def test_a_real_difference_is_detected():
    strong, weak = _signals()
    _, _, diff, se, _, p = delong(strong, weak, LABELS)
    assert diff > 0.15 and se > 0 and p < 0.05


def test_no_real_difference_is_not_significant():
    random.seed(0)
    a = [random.gauss(0.4, 1) if y else random.gauss(0, 1) for y in LABELS]
    b = [random.gauss(0.4, 1) if y else random.gauss(0, 1) for y in LABELS]
    assert delong(a, b, LABELS)[5] > 0.05


def test_perfect_separator_is_degenerate_not_broken():
    """A perfect separator has zero within-class variance, so DeLong's variance
    estimate is genuinely zero and p is undefined. That is the correct answer for
    this input, not a failure - the caller must not read it as significance."""
    auc, _, _, se, _, _ = delong([1.0] * N_POS + [0.0] * N_NEG, [0.5] * len(LABELS), LABELS)
    assert auc == 1.0 and se == 0.0


def test_too_few_cases_returns_nan_rather_than_a_number():
    out = delong([1.0, 2.0], [1.0, 2.0], [1, 0])
    assert all(v != v for v in out)  # NaN is the only value not equal to itself


def test_bh_is_monotone_and_never_below_the_raw_p():
    raw = [0.001, 0.04, 0.03, 0.2, 0.5]
    adj = benjamini_hochberg(raw)
    assert all(a >= p - 1e-12 for a, p in zip(adj, raw))
    assert all(a <= 1.0 for a in adj)
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    assert all(adj[order[i]] <= adj[order[i + 1]] + 1e-12 for i in range(len(raw) - 1))

"""The Kaggle bundle must not drift from the library it ships.

The bundle is a copy of `caliper/` and the experiments that run under Kaggle, carried as a
dataset. It went stale twice by hand-copying before `kaggle/build_bundle.py` existed, and
it went stale a third time by library work landing without a rebuild - at which point a
Kaggle run would have executed a different estimator than the notebook claims to have
scored. Nothing about the failure is visible from the bundle alone, so it is asserted here.

Cheap: hashes a handful of files, no imports from `caliper`, no model loading.
"""

import hashlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "kaggle" / "bundle"

# The files that decide what a Kaggle run actually computes. A change to any of these
# without a rebuild means Kaggle and the local notebook disagree.
SHIPPED = [
    "caliper/__init__.py",
    "caliper/activations.py",
    "caliper/batched.py",
    "caliper/estimator.py",
    "caliper/runtime.py",
    "experiments/e01_gate.py",
    "experiments/b1_signal_calibration.py",
    "experiments/kaggle_device_equivalence.py",
]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("rel", SHIPPED)
def test_bundle_file_matches_source(rel: str) -> None:
    """Every shipped file is byte-identical to its source."""
    if not BUNDLE.exists():
        pytest.skip("no bundle built yet")

    src, dst = ROOT / rel, BUNDLE / rel
    assert src.exists(), f"{rel} missing from the source tree"
    assert dst.exists(), (
        f"{rel} is not in the bundle. Rebuild with: python kaggle/build_bundle.py"
    )
    assert _digest(src) == _digest(dst), (
        f"{rel} differs between the tree and the bundle. A Kaggle run would execute "
        f"different code from the one the notebook reports. "
        f"Rebuild with: python kaggle/build_bundle.py"
    )


def test_bundle_has_no_stray_pycache() -> None:
    """`__pycache__` in a dataset upload is dead weight and confuses the mount search."""
    if not BUNDLE.exists():
        pytest.skip("no bundle built yet")
    assert list(BUNDLE.rglob("__pycache__")) == []
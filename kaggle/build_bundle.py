"""Rebuild kaggle/bundle from the live tree.

The repo is private, so Kaggle cannot pip-install it anonymously and the code has to
travel as a dataset. Hand-copying went stale twice - the bundle was two files behind and
missing batched.py entirely, which the gate run cannot start without. This makes the
build one command and prints what changed, so staleness is visible instead of silent.

    python kaggle/build_bundle.py

Then upload kaggle/bundle as a new version of the `caliper-bundle` dataset.
"""
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "kaggle" / "bundle"

# Everything a B-series run needs, and nothing else. Data files are small enough to
# travel; the corpus cache is what lets a Kaggle session run with internet off.
INCLUDE = [
    "caliper/__init__.py",
    "caliper/activations.py",
    "caliper/batched.py",
    "caliper/estimator.py",
    "caliper/runtime.py",
    "caliper/sae.py",
    "experiments/e01_gate.py",
    "experiments/b1_signal_calibration.py",
    "experiments/kaggle_device_equivalence.py",
]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12] if p.exists() else None


def main():
    changed, added, same = [], [], []
    for rel in INCLUDE:
        src, dst = ROOT / rel, DEST / rel
        if not src.exists():
            raise SystemExit(f"missing from the tree: {rel}")
        before = digest(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        after = digest(dst)
        (added if before is None else changed if before != after else same).append(rel)

    # drop anything stale that is no longer on the list
    keep = {DEST / r for r in INCLUDE}
    removed = []
    for p in sorted(DEST.rglob("*.py")):
        if p not in keep:
            p.unlink()
            removed.append(str(p.relative_to(DEST)))
    for pyc in DEST.rglob("__pycache__"):
        shutil.rmtree(pyc, ignore_errors=True)

    for label, items in (("added", added), ("updated", changed),
                         ("removed", removed), ("unchanged", same)):
        if items:
            print(f"  {label:<10} {len(items):>2}  {', '.join(items)}")
    total = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file())
    print(f"\n  bundle: {total / 1024:.0f} KB at {DEST}")
    if added or changed or removed:
        print("  -> upload as a NEW VERSION of the caliper-bundle dataset")
    else:
        print("  -> unchanged, no upload needed")


if __name__ == "__main__":
    main()

"""The Kaggle bundle must not drift from the library it ships.

The bundle is a copy of `caliper/` and the experiments that run under Kaggle, carried as a
dataset. It went stale twice by hand-copying before `kaggle/build_bundle.py` existed, and
it went stale a third time by library work landing without a rebuild - at which point a
Kaggle run would have executed a different estimator than the notebook claims to have
scored. Nothing about the failure is visible from the bundle alone, so it is asserted here.

Cheap: hashes a handful of files, no imports from `caliper`, no model loading.
"""

import ast
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
    "caliper/sae.py",
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


def _local_imports(path: Path) -> set[str]:
    """Module names imported by a bundled file that must resolve inside the bundle.

    Only first-party modules count. `caliper.*` and `experiments.*` are ours. A bare name is
    treated as ours if a file of that name exists in the **source** `experiments/` directory,
    because that is how these scripts import their siblings (running
    `python experiments/e01_gate.py` puts that directory on the path).

    Locality has to be decided from the source tree, not from the bundle. Deciding it from
    the bundle cannot work: the only bare names it would classify as ours are the ones
    already shipped, which are exactly the ones that cannot be missing. That version of this
    check passed against a deliberately broken bundle.
    """
    source_bare = {p.stem for p in (ROOT / "experiments").glob("*.py")}
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module)
    return {
        m for m in found
        if m.split(".")[0] in {"caliper", "experiments"} or m in source_bare
    }


def test_every_local_import_resolves_inside_the_bundle() -> None:
    """A shipped file must not import a module the bundle does not carry.

    The hash test above cannot catch this. `SHIPPED` is an explicit list, so adding a helper
    module and importing it from a bundled script leaves all eight files still byte-identical
    to source - every hash test passes - while the bundle is quietly unrunnable, because the
    import fails only on Kaggle. That is the same class of staleness the hash test exists to
    catch, one indirection further out, and it was nearly shipped today: `s1_2_deflation.py`
    gained an import of `planted_units`, which is not part of the B-series manifest.
    """
    if not BUNDLE.exists():
        pytest.skip("no bundle built yet")

    missing: dict[str, list[str]] = {}
    for rel in SHIPPED:
        dst = BUNDLE / rel
        if not dst.exists():
            continue
        for mod in sorted(_local_imports(dst)):
            if mod.split(".")[0] == "caliper":
                cand = BUNDLE / "caliper" / f"{mod.split('.', 1)[1].replace('.', '/')}.py"
                ok = cand.exists() or (BUNDLE / "caliper" / mod.split(".", 1)[1]
                                       / "__init__.py").exists()
            else:
                tail = mod.split("/")[-1]
                ok = (BUNDLE / "experiments" / f"{tail}.py").exists()
                if not ok:
                    ok = (BUNDLE / "experiments" / tail / "__init__.py").exists()
            if not ok:
                missing.setdefault(rel, []).append(mod)

    assert not missing, (
        "bundled files import modules the bundle does not ship, so a Kaggle run would fail "
        f"on import while every hash test still passes: {missing}. Either add the module to "
        "FILES in kaggle/build_bundle.py and rebuild, or drop the import."
    )
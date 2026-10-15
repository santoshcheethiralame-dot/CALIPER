"""Rebuild the anonymous release, unpack it into an empty folder, and regenerate headline numbers from it.

    python tools/repro_check.py          # prints PASS/FAIL per check, exits non-zero on any failure

Checks, each run inside the unpacked copy only (nothing from the working tree):
  1. the primary comparison (B-14, DeLong AUC difference) from the released rows
  2. the pooled gap over the six filed arms (random effects) from the released rows
  3. the OXIDE scoring tool's tests
Each number is compared with the value committed in this repository's results.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, cwd):
    r = subprocess.run([sys.executable, *cmd], cwd=cwd, capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": str(cwd)})
    return r.returncode, r.stdout[-2000:] + r.stderr[-2000:]


def main():
    code, out = run(["tools/make_anon_release.py"], ROOT)
    if code:
        sys.exit("release build failed:\n" + out)
    tmp = Path(tempfile.mkdtemp(prefix="oxide_repro_"))
    with zipfile.ZipFile(ROOT / "release/caliper-anon.zip") as z:
        z.extractall(tmp)
    rel = next(p for p in tmp.iterdir() if p.is_dir())
    results = {}

    want = json.load(open(ROOT / "results/b14_analysis.json"))["primary"]["diff_restart_minus_r2"]
    code, out = run(["experiments/analyse_b14.py", "--rows", "results/b14_primary_gpt2_indep.jsonl",
                     "--no-model", "--out", str(tmp / "b14.json")], rel)
    got = json.load(open(tmp / "b14.json"))["primary"]["diff_restart_minus_r2"] if code == 0 else None
    results["primary B-14 Delta AUC"] = (got is not None and abs(got - want) < 1e-9, want, got, out if code else "")

    want = json.load(open(ROOT / "results/every_run.json"))["pool_filed"]["pooled"]
    code, out = run(["experiments/analyse_every_run.py"], rel)
    got = json.load(open(rel / "results/every_run.json"))["pool_filed"]["pooled"] if code == 0 else None
    results["pooled six arms"] = (got is not None and abs(got - want) < 1e-9, want, got, out if code else "")

    code, out = run(["-m", "pytest", "tests/test_oxide.py", "-q"], rel)
    results["OXIDE tests"] = (code == 0, "pass", "pass" if code == 0 else "fail", out if code else "")

    ok = True
    for name, (passed, want, got, log) in results.items():
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {name}: expected {want}, got {got}")
        if log:
            print("   ", log.strip().replace("\n", "\n    ")[-1500:])
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

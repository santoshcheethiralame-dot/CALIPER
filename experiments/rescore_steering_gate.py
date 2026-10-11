"""The steering gate rescored with whole-word matching and per-concept exclusion (exploratory, after
the data; second internal review of Paper 2, ruling 5).

    python experiments/rescore_steering_gate.py      # writes results/gate_strict/ and results/gate_strict.json

The filed rule (`steered()` in kaggle_s3_positive_control.py) matches the concept word or any of its
associates as a substring of the lower-cased text, and a vector is live if its gate-dose text is
steered and its alpha-0 text is not. Every concept's alpha-0 text is the same greedy story, so a
substring of that story ("echoed", "crashing", "spoken") marks a concept steered at alpha 0 and the
concept can never be live. The strict rule: a term counts only as a word start (regex \\bterm), and
only if it does not already occur, as a word start, in the same concept's alpha-0 text; steered at
alpha 0 is then false by construction, and liveness is read at the gate dose alone.

Each session folder is copied with the steer rows rescored (the filed label kept as steered_filed),
and the unchanged analysis scripts are re-run on the copies.
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from kaggle_s3_positive_control import ASSOCIATES  # noqa: E402

SESSIONS = [  # folder, analysis command (script, extra args)
    ("s2_qwen3b", ["analyse_s2.py", "--model", "qwen3b"]),
    ("s2_qwen7b", ["analyse_s2.py", "--model", "qwen7b", "--quant", "4bit"]),
    ("s2_gemma4b_kl", ["analyse_s2.py", "--model", "gemma4b_kl"]),
    ("s1_gemma12", ["analyse_s1.py", "--model", "gemma12"]),
    ("s1_gemma27", ["analyse_s1.py", "--model", "gemma27", "--precisions", "4bit"]),
]
OUT = ROOT / "results/gate_strict"


def terms(concept):
    return [concept.lower()] + ASSOCIATES.get(concept, [])


def hits(text, concept):
    t = (text or "").lower()
    return {k for k in terms(concept) if re.search(rf"\b{re.escape(k)}", t)}


def rescore(rows):
    base = {}
    for r in rows:
        if r["alpha"] == 0:
            base.setdefault((r.get("framing"), r["concept"]), r.get("text", ""))
    out = []
    for r in rows:
        b = base.get((r.get("framing"), r["concept"]), "")
        new = bool(hits(r.get("text", ""), r["concept"]) - hits(b, r["concept"])) if r["alpha"] else False
        out.append({**r, "steered_filed": r.get("steered"), "steered": new})
    return out


def main():
    rep = {}
    OUT.mkdir(exist_ok=True)
    for name, cmd in SESSIONS:
        src, dst = ROOT / "results" / name, OUT / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        changed = 0
        for f in dst.glob("*steer*.jsonl"):
            rows = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
            new = rescore(rows)
            changed += sum(a.get("steered") != b["steered"] for a, b in zip(rows, new))
            with open(f, "w", encoding="utf-8") as h:
                h.writelines(json.dumps(r) + "\n" for r in new)
        out = OUT / f"{name}_analysis.json"
        subprocess.run([sys.executable, str(ROOT / "experiments" / cmd[0]), "--dir", str(dst), *cmd[1:],
                        "--out", str(out)], check=True, capture_output=True, cwd=ROOT)
        rep[name] = {"steer_rows_relabelled": changed, "analysis": str(out.relative_to(ROOT))}
    # S-1M: the released recipe at strength 4
    rows = [json.loads(l) for l in open(ROOT / "results/s1_gemma27/s1_gemma27_s1m_steer_macar-release.jsonl")]
    new = rescore(rows)
    filed = {r["concept"] for r in rows if r["alpha"] == 4 and r["steered"]} - \
        {r["concept"] for r in rows if r["alpha"] == 0 and r["steered"]}
    strict = {r["concept"] for r in new if r["alpha"] == 4 and r["steered"]}
    rep["S-1M released, strength 4"] = {"live_filed": len(filed), "live_strict": len(strict),
                                        "gained": sorted(strict - filed), "lost": sorted(filed - strict),
                                        "dead_strict": sorted({r["concept"] for r in rows} - strict)}
    json.dump(rep, open(ROOT / "results/gate_strict.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

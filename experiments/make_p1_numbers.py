"""Paper 1 numbers that depend on runs still landing (B-14r, D-1), written as LaTeX macros.

    PYTHONPATH=. python experiments/make_p1_numbers.py

Writes paper1/tables/numbers.tex and patches the every-run table's direct-label column (and, once
B-14r is complete, its row). Run after land_b14r.py, analyse_round3.py and make_p1_lnnull_table.py,
and again after analyse_d1.py. A macro whose source does not exist yet prints a red TBD.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT))
import analyse_ln_null as ln  # noqa: E402
import make_p1_lnnull_table as lt  # noqa: E402

TEX = ROOT / "paper1/main.tex"
PASS = 0.95


def x(v):  # prose number
    return "$" + ("-" if v < 0 else "+") + f"{abs(v):.3f}$"


def ci(c):
    return "$[" + ", ".join(("-" if v < 0 else "+") + f"{abs(v):.3f}" for v in c) + "]$"


def cell(v):  # table number
    return ("$-$" if v < 0 else "$+$") + f"{abs(v):.3f}"


def load(p):
    p = ROOT / p
    return json.load(open(p)) if p.exists() else None


def rows(stem):
    p = ROOT / f"{stem}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if p.exists() else []


def kappa(a, b):
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def b14r_macros(m, r3):
    land = load("results/b14r_landing.json")
    if not land or not land.get("complete"):
        return None
    a = {r["_key"]: r for r in rows("results/b14_primary_gpt2_indep")}
    b = {r["_key"]: r for r in rows("results/b14r_primary_gpt2_dirs")}
    ks = sorted(set(a) & set(b))
    fa = np.array([a[k]["align_selected"] < PASS for k in ks])
    fb = np.array([b[k]["align_selected"] < PASS for k in ks])
    flips = int((fa != fb).sum())
    lo, hi = wilson(flips, len(ks))
    arm = land["rerun_as_an_arm"]
    euc, ident = arm["as filed"]["primary_comparison"], arm["identifiable"]["primary_comparison"]
    own = r3["B1 route-matched with own R2"]["GPT-2 L6 primary replicate (B-14r)"]
    m.update(bRFlips=str(flips), bRFlipRate=f"{100 * flips / len(ks):.0f}\\%",
             bRFlipCI=f"{100 * lo:.0f}--{100 * hi:.0f}\\%", bRKappa=f"{kappa(fa, fb):.2f}",
             bRFail=str(int(fb.sum())), bRDelta=x(euc["diff"]), bRCI=ci(euc["boot_ci95"]),
             bRCwFiled=str(arm["as filed"]["converged_wrong"]), bRCwIdent=str(arm["identifiable"]["converged_wrong"]),
             bRIdentDelta=x(ident["diff"]), bRIdentCI=ci(ident["boot_ci95"]),
             bROwnDelta=x(own["matched_own_r2"]["diff"]), bROwnCI=ci(own["matched_own_r2"]["boot_ci95"]),
             bROwnFail=str(own["direct_route_failures"]),
             bRReplicateNote="the primary's replicate (B-14r) is listed first")
    return land


def ln_counts(m):
    ln_all = json.load(open(ROOT / "results/ln_null_analysis.json"))
    r3 = json.load(open(ROOT / "results/round3_fixes.json"))
    arms = [ln_all[k] for k in lt.LABEL] + [r3["B3 1.4B arm on the identifiable label"]]
    land = load("results/b14r_landing.json")
    if land and land.get("rerun_as_an_arm"):
        arms.append(land["rerun_as_an_arm"])
    get = lambda v: v.get("comparison") or v.get("primary_comparison")
    pairs = [(get(v["as filed"])["diff"], get(v["identifiable"])["diff"]) for v in arms
             if get(v["as filed"]) and get(v["identifiable"])]
    m.update(lnScorable=str(len(pairs)), lnWiden=str(sum(b < a - 5e-4 for a, b in pairs)),
             lnSame=str(sum(abs(b - a) <= 5e-4 for a, b in pairs)),
             lnNarrow=str(sum(b > a + 5e-4 for a, b in pairs)))


def own_counts(m, r3):
    b1 = [v["matched_own_r2"] for v in r3["B1 route-matched with own R2"].values() if v["matched_own_r2"]]
    m.update(mOwnArms=str(len(b1)), mOwnLead=str(sum(c["diff"] < 0 for c in b1)),
             mOwnExcl=str(sum(c["boot_ci95"][1] < 0 for c in b1)),
             mOwnTie=str(sum(c["diff"] == 0 for c in b1)), mOwnRestart=str(sum(c["diff"] > 0 for c in b1)))


def stability(m, r3):
    s = r3["B6 class stability"]
    cw, fl = s["converged_wrong_in_k_fits"], s["failing_in_k_fits"]
    n = str(s["fits"])
    m.update(csFits=n, csNever=str(fl["0"]), csAll=str(fl[n]),
             csCwAny=str(sum(v for k, v in cw.items() if k != "0")), csCwOnce=str(cw["1"]), csCwAll=str(cw[n]))


def d1(m):
    d = load("results/d1_analysis.json")
    if not d:
        return
    p, e = d["primary"], d["euclidean"]
    m.update(dOneReading=p["reading"], dOneFail=str(d["arm"]["identifiable"]["failures"]),
             dOneEucFail=str(d["arm"]["as filed"]["failures"]))
    if p["comparison"]:
        m.update(dOneDelta=x(p["comparison"]["diff"]), dOneCI=ci(p["comparison"]["boot_ci95"]))
    if e["comparison"]:
        m.update(dOneEucDelta=x(e["comparison"]["diff"]), dOneEucCI=ci(e["comparison"]["boot_ci95"]))


def e1(m):
    """E-1 (docs/preregistration-e1-fair-comparison.md), from results/e1_analysis.json once scored."""
    d = load("results/e1_analysis.json")
    if not d:
        return
    for tag, arm in (("", "gpt2_l6"), ("Neo", "gptneo_l10")):
        a = d[arm]
        m[f"eOne{tag}Units"] = str(a["units"])
        m[f"eOne{tag}Fail"] = str(a["failures_identifiable"])
        m[f"eOne{tag}Reading"] = a["reading"]
        m[f"eOne{tag}Cw"] = str(a["classes"]["converged_wrong"])
        m[f"eOne{tag}Uf"] = str(a["classes"]["under_fitted"])
        c = a["primary"]
        if c:
            m.update({f"eOne{tag}AucRestart": f"{c['auc_restart']:.3f}", f"eOne{tag}AucTest": f"{c['auc_r2']:.3f}",
                      f"eOne{tag}Delta": x(c["diff"]), f"eOne{tag}CI": ci(c["boot_ci95"]),
                      f"eOne{tag}P": f"{c['delong_p']:.2g}"})
        e = a["euclidean"]
        if e:
            m.update({f"eOne{tag}EucDelta": x(e["diff"]), f"eOne{tag}EucCI": ci(e["boot_ci95"])})
        v = a["validation_vs_test"]
        if v:
            m[f"eOne{tag}ValTest"] = x(v["diff"])
    lv = d["gpt2_l6"]["low_variance_diagnostic"]
    m["eOneLowVar"] = f"{lv['auc_low_var_share']:.3f}" if isinstance(lv, dict) else "not scored"
    r = d["practitioner_rule"]["val_r2"]
    m.update(eOneRuleCorrect=str(r["r2_rule"]["correct"]), eOneRuleRestartCorrect=str(r["restart_rule"]["correct"]),
             eOneRuleFcr=f"{100 * r['r2_rule']['fcr']:.1f}\\%", eOneRuleRestartFcr=f"{100 * r['restart_rule']['fcr']:.1f}\\%",
             eOneRuleFcrCI=ci(r["fcr_difference_ci95"]), eOneRuleHolds="holds" if r["prediction_holds"] else "fails")
    secs = 0.0
    for arm in ("gpt2_l6", "gptneo_l10"):
        s = load(f"results/e1_{arm}_summary.json")
        if s:
            secs += s["seconds_per_unit"] * s["rows"]
    if secs:
        m["eOneHours"] = f"{secs / 3600:.0f}"


E1_NAMES = [f"eOne{t}{k}" for t in ("", "Neo") for k in
            ("Units", "Fail", "Reading", "Cw", "Uf", "AucRestart", "AucTest", "Delta", "CI", "P", "EucDelta", "EucCI",
             "ValTest")] + ["eOneLowVar", "eOneRuleCorrect", "eOneRuleRestartCorrect", "eOneRuleFcr",
                            "eOneRuleRestartFcr", "eOneRuleFcrCI", "eOneRuleHolds", "eOneHours"]

NAMES = ("bRFlips bRFlipRate bRFlipCI bRKappa bRFail bRDelta bRCI bRCwFiled bRCwIdent bRIdentDelta bRIdentCI bROwnDelta bROwnCI "
         "bROwnFail lnScorable lnWiden lnSame lnNarrow mOwnArms mOwnLead mOwnExcl mOwnTie mOwnRestart csFits "
         "csNever csAll csCwAny csCwOnce csCwAll dOneReading dOneFail dOneEucFail dOneDelta dOneCI dOneEucDelta "
         "dOneEucCI dThreeYield dThreeYieldCI dThreeFcr dThreeFcrCI dThreeReading dThreeChange dThreeChangeCI "
         "dThreeChangeReading").split()


def patch_every_run(r3, land):
    s = TEX.read_text(encoding="utf-8")
    head = s.index("$\\Delta$ direct label & pooled")
    i, j = s.index("\\midrule\n", head) + len("\\midrule\n"), s.index("\\bottomrule", head)
    b1 = r3["B1 route-matched with own R2"]
    out = []
    for line in s[i:j].splitlines():
        c = line.split(" & ")
        tag = re.findall(r"\(([^()]*)\)\}?$", c[0])
        key = next((k for k in b1 if tag and k.endswith(f"({tag[0]})")), None)
        if key:
            v = b1[key]
            c[10] = (cell(v["matched_own_r2"]["diff"]) if v["matched_own_r2"] else "---") + \
                f" ({v['direct_route_failures']})"
        elif not c[10].startswith("---"):
            c[10] = "n/s"
        out.append(" & ".join(c))
        if tag == ["B-14"] and land and "(B-14r)" not in s[i:j]:
            arm = land["rerun_as_an_arm"]["as filed"]["primary_comparison"]
            R = rows("results/b14r_primary_gpt2_dirs")
            d = np.array([r["picked"] == "direct" for r in R])
            st, r2 = np.array([r["stability"] for r in R]), np.array([r["r2_k1"] for r in R])
            fail = np.array([r["align_selected"] < PASS for r in R])
            dp = ln.compare(-st[d], -r2[d], fail[d])
            v = b1["GPT-2 L6 primary replicate (B-14r)"]
            out.append(" & ".join([
                "\\draft{GPT-2 L6, primary replicate (B-14r)}", "corrected", "fp32", str(len(R)), str(int(fail.sum())),
                f"{arm['auc_restart']:.3f}", f"{arm['auc_r2']:.3f}", cell(arm["diff"]),
                "[" + ", ".join(cell(t) for t in arm["boot_ci95"]) + "]",
                (cell(dp["diff"]) if dp else "---") + f" ({int(fail[d].sum())})",
                cell(v["matched_own_r2"]["diff"]) + f" ({v['direct_route_failures']})", "no \\\\"]))
    s = s[:i] + "\n".join(out) + "\n" + s[j:]
    TEX.write_text(s, encoding="utf-8")


def d3(m):
    ci = lambda v: "$[" + ", ".join(("-" if t < 0 else "+") + f"{abs(t):.3f}" for t in v) + "]$"
    d = load("results/d3_analysis.json")
    if not d:
        return
    c = d.get("addendum 1, certification at 16k")
    if c:
        m.update(dThreeYield=x(c["E1_yield_diff"]), dThreeYieldCI=ci(c["E1_ci95"]),
                 dThreeFcr=x(c["E2_fcr_diff"]), dThreeFcrCI=ci(c["E2_ci95"]) if c["E2_ci95"] else "(no interval)",
                 dThreeReading=c["E2_reading"])
    p1 = d.get("change 8k to 16k, identifiable")
    if p1:
        m.update(dThreeChange=x(p1["delta_16k_minus_8k"]), dThreeChangeCI=ci(p1["ci95_paired"]) if p1["ci95_paired"] else "",
                 dThreeChangeReading=p1["reading"])


def main():
    r3 = json.load(open(ROOT / "results/round3_fixes.json"))
    m = {}
    land = b14r_macros(m, r3)
    ln_counts(m)
    own_counts(m, r3)
    stability(m, r3)
    d1(m)
    d3(m)
    e1(m)
    lines = ["% Generated by experiments/make_p1_numbers.py; do not edit by hand."]
    for k in NAMES + E1_NAMES:
        lines.append(f"\\newcommand{{\\{k}}}{{{m[k]}}}" if k in m else f"\\newcommand{{\\{k}}}{{\\tbd{{{k}}}}}")
    if "bRReplicateNote" not in m:
        lines.append("\\newcommand{\\bRReplicateNote}{the primary's replicate is listed when it lands}")
    else:
        lines.append(f"\\newcommand{{\\bRReplicateNote}}{{{m['bRReplicateNote']}}}")
    (ROOT / "paper1/tables/numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    patch_every_run(r3, land)
    print(f"{len(m)} macros filled, {len(NAMES) + len(E1_NAMES) + 1 - len(m)} TBD")


if __name__ == "__main__":
    main()

"""Re-score the Study 3 positive control from saved text.

The scorer used during the run conflated three different things and got all three
wrong. It required the response to BEGIN with "YES", so "**YES**" and
"Okay, I understand the setup.\n\nYES." both counted as non-detections, while
"Absolutely! I'm ready.\n\nNO." was ambiguous. It also required three or more words
for "coherent", so a bare "NO." - the correct answer on a control trial - was scored
incoherent, and since detected = said_yes AND coherent, that fed straight into the
headline number.

Nothing here re-runs the model. Every column below is computed from the `text` field
of the original run, and the original scorer's numbers are reported alongside so the
change is auditable rather than silent.
"""

import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# The model was instructed to answer with YES or NO. Find whichever appears first,
# ignoring markdown emphasis and headings. This is the whole affirmation rule - no
# thresholds, nothing tunable.
ANSWER = re.compile(r"(?<![A-Za-z])(YES|NO)(?![A-Za-z])")

# Phrases lifted from the researcher preamble. A response built out of these is
# describing the experimental setup it was just told about, not an injected concept.
ECHO = re.compile(
    r"being (observed|evaluated|tested|analy[sz]ed)|a researcher (studying|is)|"
    r"my (activations|internal (states|workings))|feels like (a test|an experiment)|"
    r"interpretability research|pressure to (perform|provide)|"
    r"meta-cognitive thought about the experiment", re.I)

# Two confabulations the model falls back on when it has no signal. Neither was ever
# an injected concept in this run, so every occurrence is a false identification.
STEREOTYPE = re.compile(r"red apple|blue ?bird", re.I)

# Hand-coded near-misses: the response names something other than the concept but
# clearly in its semantic neighbourhood. Listed exhaustively so the judgement calls
# are visible and arguable rather than buried in a number.
NEAR = {
    ("dolphin", 2): "playful otter",
    ("garden", 2): "meadow of wildflowers",
    ("spider", 8): "Arachnid",
    ("clock", 8): "Time to respond",
    ("candle", 8): "Candor",
    ("library", 8): "classical-building emoji",
    ("serenity", 8): "Dalai Lama's teachings",
}


def first_answer(text):
    """YES, NO, or None when the model never gave the requested token."""
    m = ANSWER.search(text)
    return m.group(1).upper() if m else None


def names_concept(text, concept):
    """Literal mention, tolerating plurals: 'eagles' counts for 'eagle'."""
    return re.search(rf"\b{re.escape(concept)}s?\b", text, re.I) is not None


def main(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    alphas = sorted({r["alpha"] for r in rows})

    print(f"n = {len(rows)} trials from {path}\n")
    hdr = (f"{'alpha':>5} {'n':>4} | {'old':>6} | {'says':>6} {'no':>5} {'none':>5} | "
           f"{'names':>6} {'echo':>6} {'stereo':>6} | {'REAL':>6}")
    print(hdr)
    print("-" * len(hdr))

    per_alpha = {}
    for a in alphas:
        g = [r for r in rows if r["alpha"] == a]
        n = len(g)
        old = sum(r["detected"] for r in g)
        ans = [first_answer(r["text"]) for r in g]
        yes = sum(x == "YES" for x in ans)
        no = sum(x == "NO" for x in ans)
        none = sum(x is None for x in ans)
        named = sum(names_concept(r["text"], r["concept"]) for r in g)
        echo = sum(bool(ECHO.search(r["text"])) for r in g)
        stereo = sum(bool(STEREOTYPE.search(r["text"])) for r in g)
        # The claim the paper actually makes: the model reports a detection AND that
        # report is about the concept that was injected.
        real = sum(first_answer(r["text"]) == "YES" and names_concept(r["text"], r["concept"])
                   for r in g)
        per_alpha[a] = dict(n=n, old=old, yes=yes, named=named, echo=echo,
                            stereo=stereo, real=real)
        print(f"{a:>5} {n:>4} | {old/n:>5.0%} | {yes/n:>5.0%} {no/n:>4.0%} {none/n:>4.0%} | "
              f"{named/n:>5.0%} {echo/n:>5.0%} {stereo/n:>5.0%} | {real/n:>5.0%}")

    print("""
  old    = scorer used during the run (says_yes AND >=3 words)
  says   = first YES/NO token is YES;  no = it is NO;  none = neither was emitted
  names  = the injected concept appears literally in the response
  echo   = response is built from phrases in the researcher preamble
  stereo = response names 'red apple' or 'blue bird' - never-injected confabulations
  REAL   = says YES *and* names the injected concept  <- the paper's claim""")

    print("\n" + "=" * 78)
    print("Every response that literally names its own injected concept")
    print("=" * 78)
    for r in rows:
        if names_concept(r["text"], r["concept"]):
            ans = first_answer(r["text"]) or "-"
            print(f"  a={r['alpha']} {r['concept']:<11} [{ans:>3}] {r['text'][:90]!r}")

    print("\n" + "=" * 78)
    print("Hand-coded semantic near-misses (not counted in any number above)")
    print("=" * 78)
    for r in rows:
        k = (r["concept"], r["alpha"])
        if k in NEAR:
            print(f"  a={r['alpha']} {r['concept']:<11} -> {NEAR[k]:<26} {r['text'][:60]!r}")

    print("\n" + "=" * 78)
    print("Dissociation: steering strength vs self-report")
    print("=" * 78)
    for a in alphas:
        p = per_alpha[a]
        print(f"  alpha={a}: concept reaches the output {p['named']/p['n']:.0%} of the time, "
              f"model reports detecting it {p['real']/p['n']:.0%}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "s3_results.jsonl")

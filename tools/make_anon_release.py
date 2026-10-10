"""Build the anonymous release for double-blind review (Paper 1 roadmap B3).

    python tools/make_anon_release.py            # -> release/caliper-anon/ and release/caliper-anon.zip

Copies the tracked package, experiments, tests, results and data, the pre-registrations and the
laboratory notebook. Identity strings are rewritten everywhere; notebook lines about course
administration, supervision and the author's other projects are replaced by a redaction mark. The
build fails if any blocked term survives in a file's contents or name, or anything shaped like an
access token appears. This script is not part of the release.
"""
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "release" / "caliper-anon"
INCLUDE = ("caliper/", "experiments/", "tests/", "results/", "data/")
DOCS = re.compile(r"docs/(preregistration-[^/]+\.md|LAB_NOTEBOOK\.md)$")
# Paper 2 material: not part of this paper, and it names the author's earlier project throughout.
PAPER2 = re.compile(r"^results/(s1_gemma|s2_|s3|s11|p2)")
EXTRA = [ROOT / "results" / "corpus_cache"]  # public-domain stimulus texts, gitignored
TEXT = {".py", ".md", ".json", ".jsonl", ".txt", ".sh", ".cmd", ".ps1", ".csv", ".tex", ".yaml", ".yml",
        ".cfg", ".toml", ""}

REWRITE = [
    (re.compile(r"github\.com/santoshcheethiralame-dot/[A-Za-z0-9_.-]+", re.I), "anonymous repository"),
    (re.compile(r"santoshcheethiralame-dot|santoshcheethirala", re.I), "anonymous"),
    (re.compile(r"[A-Za-z0-9._%+-]+@gmail\.com"), "anonymous@example.org"),
    (re.compile(r"Santosh Cheethirala|Cheethirala, Santosh", re.I), "Anonymous Author"),
    (re.compile(r"\bSantosh\b"), "the author"),
    (re.compile(r"([A-Za-z]:[\\/]+Users[\\/]+|/[cC]/Users/|/home/)carbo", re.I), r"\1user"),
    (re.compile(r"OneDrive[\\/][^\s\"']*"), "local"),
]
REDACT_LINE = re.compile(r"mentor|capstone|APERTURE|\bMIRROR\b|Proposal (III|V)\b|proposal-iii|PROPOSAL_III|rubric|"
                         r"semester|teammate|4-person|\bPES\b|internship|hand-off|ownership table|"
                         r"docs/team/", re.I)
BLOCK = re.compile(r"santosh|cheethirala|carbo|gmail|mentor|capstone|APERTURE|PES University|"
                   r"hf_[A-Za-z0-9]{30,}|KGAT_[A-Za-z0-9]+|sk-[A-Za-z0-9]{20,}", re.I)
MARK = "[Redacted for anonymous review: course administration or the author's other work.]"


def tracked():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    for f in out.splitlines():
        if (f.startswith(INCLUDE) or DOCS.match(f)) and not PAPER2.match(f):
            yield ROOT / f


def redact(line):
    if line.lstrip().startswith("|"):  # keep a table row's column count, or the table breaks
        return "| " + MARK + " |" + " |" * (line.count("|") - 2)
    return MARK


def clean(text, notebook):
    for pat, rep in REWRITE:
        text = pat.sub(rep, text)
    if notebook:
        text = "\n".join(redact(l) if REDACT_LINE.search(l) else l for l in text.split("\n"))
    return text


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    files = list(tracked()) + [p for d in EXTRA if d.exists() for p in d.rglob("*") if p.is_file()]
    redacted, hits, skipped = 0, [], []
    for src in files:
        if not src.exists():
            continue
        rel = src.relative_to(ROOT)
        dst = OUT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if BLOCK.search(str(rel).replace("\\", "/")):
            hits.append(f"file name: {rel}")
        if src.suffix.lower() in TEXT and "corpus_cache" not in rel.parts:
            raw = src.read_text(encoding="utf-8", errors="surrogateescape")
            if rel.name != "LAB_NOTEBOOK.md" and "aperture" in raw.lower():
                skipped.append(str(rel))
                continue
            text = clean(raw, notebook=rel.name == "LAB_NOTEBOOK.md")
            redacted += text.count(MARK)
            for m in BLOCK.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{rel}:{line}: {text[max(0, m.start() - 40):m.end() + 40]!r}")
            dst.write_text(text, encoding="utf-8", errors="surrogateescape")
        else:
            shutil.copy2(src, dst)
    (OUT / "LICENSE").write_text("Released under the licence stated in the paper. Copyright the anonymous "
                                 "authors.\n", encoding="utf-8")
    shutil.copy2(ROOT / "tools" / "RELEASE_README.md", OUT / "README.md")
    if hits:
        print(f"BLOCKED: {len(hits)} hit(s); release not zipped", file=sys.stderr)
        for h in hits[:60]:
            print("  " + h, file=sys.stderr)
        sys.exit(1)
    z = ROOT / "release" / "caliper-anon.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(OUT.parent))
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"skipped as Paper 2 material: {len(skipped)} file(s)")
    for f in skipped:
        print("  " + f)
    print(f"{n} files, {redacted} notebook lines redacted, 0 blocked terms -> {z} "
          f"({z.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()

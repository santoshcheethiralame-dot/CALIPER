"""Check every entry of a .bib against Crossref (by DOI) and arXiv (by id): title, first author's
surname and year. Mechanical; anything flagged is then checked by hand.

    python experiments/check_citations.py paper1/references.bib --out results/paper1_citation_check.json
"""
import argparse
import difflib
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "citation-check/1.0 (mailto:santoshcheethirala.me@gmail.com)"}


def entries(text):
    out = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,]+),(.*?)\n\}", text, re.S):
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*(\{(?:[^{}]|\{[^{}]*\})*\}|\"[^\"]*\"|\d+)", m.group(3)):
            fields[f.group(1).lower()] = f.group(2).strip("{}\"")
        out.append({"type": m.group(1).lower(), "key": m.group(2).strip(), **fields})
    return out


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"[{}\\]", "", s.lower())).strip()


def first_surname(authors):
    a = (authors or "").split(" and ")[0].strip()
    return norm(a.split(",")[0] if "," in a else a.split()[-1] if a else "")


def get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def crossref(doi):
    m = json.loads(get("https://api.crossref.org/works/" + urllib.parse.quote(doi)))["message"]
    year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
    return {"title": (m.get("title") or [""])[0],
            "surname": (m.get("author") or [{}])[0].get("family", ""), "year": year}


def arxiv(aid):
    x = ET.fromstring(get("http://export.arxiv.org/api/query?id_list=" + aid))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    e = x.find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None:
        return None
    name = e.find("a:author/a:name", ns).text
    return {"title": " ".join(e.find("a:title", ns).text.split()),
            "surname": name.split()[-1], "year": int(e.find("a:published", ns).text[:4])}


def arxiv_title(title):
    q = urllib.parse.quote('ti:"' + re.sub(r"[{}\:]", " ", title).strip() + '"')
    x = ET.fromstring(get(f"http://export.arxiv.org/api/query?search_query={q}&max_results=3"))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    best = None
    for e in x.findall("a:entry", ns):
        t = " ".join(e.find("a:title", ns).text.split())
        ratio = difflib.SequenceMatcher(None, norm(title), norm(t)).ratio()
        if best is None or ratio > best[0]:
            best = (ratio, {"title": t, "surname": e.find("a:author/a:name", ns).text.split()[-1],
                            "year": int(e.find("a:published", ns).text[:4])})
    return best[1] if best and best[0] >= 0.85 else None


def arxiv_id(e):
    for f in ("eprint", "url", "journal", "note", "howpublished"):
        m = re.search(r"(\d{4}\.\d{4,5})", e.get(f, ""))
        if m:
            return m.group(1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bib")
    ap.add_argument("--out", default="results/paper1_citation_check.json")
    a = ap.parse_args()
    rep = []
    for e in entries(open(ROOT / a.bib, encoding="utf-8").read()):
        r = {"key": e["key"], "bib_title": e.get("title"), "bib_year": e.get("year"),
             "bib_first": first_surname(e.get("author"))}
        try:
            if e.get("doi"):
                r["source"], got = "crossref", crossref(e["doi"])
            elif arxiv_id(e):
                r["source"], got = "arxiv", arxiv(arxiv_id(e))
                time.sleep(3)                            # arXiv asks for 3 s between calls
            else:
                r["source"], got = "arxiv title search", arxiv_title(e.get("title", ""))
                time.sleep(3)
        except Exception as ex:                          # noqa: BLE001
            r["source"], got, r["error"] = r.get("source", "?"), None, f"{type(ex).__name__}: {ex}"
        if got:
            r["found_title"], r["found_year"], r["found_first"] = got["title"], got["year"], norm(got["surname"])
            r["title_ratio"] = round(difflib.SequenceMatcher(None, norm(e.get("title")), norm(got["title"])).ratio(), 3)
            r["flags"] = [f for f, bad in (
                ("title", r["title_ratio"] < 0.85),
                ("year", str(got["year"]) != str(e.get("year"))),
                ("first author", r["found_first"] and r["bib_first"] and r["found_first"] != r["bib_first"]),
            ) if bad]
        else:
            r["flags"] = ["not checked automatically"]
        rep.append(r)
        print(r["key"], r["source"], r["flags"], flush=True)
    json.dump(rep, open(ROOT / a.out, "w"), indent=1)
    bad = [r for r in rep if r["flags"]]
    print(f"\n{len(rep)} entries; {len(rep) - len(bad)} clean; {len(bad)} flagged")


if __name__ == "__main__":
    main()

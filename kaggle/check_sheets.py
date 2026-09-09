"""Validate the Python cells in every Kaggle run sheet before a session is spent on them.

Two run-sheet failures have now cost this project real time: a pasted script that bit back
over four rounds, and a Cell 1 that assumed the dataset mount depth and made every
subsequent command fail with "No such file or directory". Both were syntactically visible
before the session started and neither was checked.

This parses each ```python block. Cells using IPython magic (`!cmd`, `%magic`) cannot be
parsed as plain Python and are reported as skipped rather than silently passed, so a
genuinely broken magic cell is still visible as unchecked.

    python kaggle/check_sheets.py
"""
import ast
import glob
import re
import sys

MAGIC = re.compile(r"^\s*[!%]", re.M)


def cells(text):
    return re.findall(r"```python\n(.*?)```", text, re.S)


def main():
    sheets = sorted(glob.glob("kaggle/*.md"))
    bad = skipped = checked = 0
    for path in sheets:
        blocks = cells(open(path, encoding="utf-8").read())
        if not blocks:
            continue
        for i, block in enumerate(blocks, 1):
            if MAGIC.search(block):
                skipped += 1
                continue
            try:
                ast.parse(block)
                checked += 1
            except SyntaxError as e:
                bad += 1
                print(f"  BROKEN  {path} cell {i}: {e.msg} (line {e.lineno})")
                for n, line in enumerate(block.splitlines(), 1):
                    if abs(n - (e.lineno or 0)) <= 1:
                        print(f"      {n:>3} | {line}")
    print(f"\n  {checked} cells parsed, {skipped} skipped as magic, {bad} broken")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

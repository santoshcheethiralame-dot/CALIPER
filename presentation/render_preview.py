"""Render the deck to PNGs using the installed PowerPoint, for visual checking.

Windows + PowerPoint only. Opens the file read-only, exports each slide, closes.
    python presentation/render_preview.py
"""
import os, sys, glob
import win32com.client

HERE = os.path.dirname(os.path.abspath(__file__))
DECK = os.path.join(HERE, "CALIPER_Review1.pptx")
OUT = os.path.join(HERE, "preview")
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "*.png")):
    os.remove(f)

app = win32com.client.Dispatch("PowerPoint.Application")
app.Visible = 1
pres = app.Presentations.Open(os.path.normpath(DECK), ReadOnly=-1,
                              Untitled=0, WithWindow=-1)
try:
    for i, slide in enumerate(pres.Slides, 1):
        slide.Export(os.path.join(OUT, "slide%02d.png" % i), "PNG", 1600, 900)
    print("exported", pres.Slides.Count, "slides ->", OUT)
finally:
    pres.Close()
    try:
        app.Quit()
    except Exception:
        pass

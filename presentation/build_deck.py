"""Build the CALIPER Review 1 deck.

    python presentation/build_deck.py

Writes CALIPER_Review1.pptx (animated) and CALIPER_Review1_static.pptx.

Template geometry and colours are measured from the approved senior deck at
Downloads/1st review sample from senior.pdf, not guessed.

Bullets are laid out as individual text boxes rather than one bulleted
placeholder. That costs a few extra shapes and buys precise positioning plus
per-bullet animation targets, since <p:spTgt> addresses a shape, not a
paragraph.
"""
import os
import sys

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anim
import content as C

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
LOGO = r"C:\Users\carbo\Downloads\PES University - Logo 2.png"

SW, SH = 13.333, 7.5

ORANGE = RGBColor(0xBD, 0x58, 0x2B)
TEAL = RGBColor(0x33, 0xCC, 0xCC)
NAVY = RGBColor(0x1F, 0x38, 0x64)
GREY = RGBColor(0x7F, 0x7F, 0x7F)
INK = RGBColor(0x1A, 0x1A, 0x1A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED = RGBColor(0xC0, 0x00, 0x00)

FONT = "Calibri"

# vertical anatomy
Y_HEADER = 0.26
Y_HEAD = 0.95
Y_TEAL = 1.66
Y_RULE = 1.82
Y_BODY = 2.08
Y_FOOT = 7.14
FOOT_H = 0.36


def _txt(slide, x, y, w, h, text, size=16, color=INK, bold=False,
         align=PP_ALIGN.LEFT, italic=False, font=FONT, anchor=MSO_ANCHOR.TOP,
         spacing=1.0, name=None):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = spacing
    r = p.add_run()
    r.text = text
    f = r.font
    f.name = font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    if name:
        box.name = name
    return box


def _rect(slide, x, y, w, h, fill, name=None):
    from pptx.enum.shapes import MSO_SHAPE
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    if name:
        sh.name = name
    return sh


def _pic(slide, path, x, y, w=None, h=None, name=None):
    kw = {}
    if w:
        kw["width"] = Inches(w)
    if h:
        kw["height"] = Inches(h)
    p = slide.shapes.add_picture(path, Inches(x), Inches(y), **kw)
    if name:
        p.name = name
    return p


def _chrome(slide, heading=None, header=True):
    """Logo, running header, heading, rules, footer band. Never animated."""
    _pic(slide, LOGO, 11.52, 0.16, w=1.42, name="logo")
    if header:
        _txt(slide, 0.45, Y_HEADER, 7.5, 0.3, C.SHORT_TITLE, size=10.5,
             color=GREY, bold=True, name="runhead")
    if heading:
        _txt(slide, 3.2, Y_HEAD, 8.15, 0.55, heading, size=27, color=INK,
             align=PP_ALIGN.RIGHT, name="heading")
        _rect(slide, 3.30, Y_TEAL, 8.05, 0.045, TEAL, name="tealrule")
        _rect(slide, 0.45, Y_RULE, 12.05, 0.008, RGBColor(0x40, 0x40, 0x40),
              name="hairline")
    _rect(slide, 0, Y_FOOT, SW, FOOT_H, ORANGE, name="footband")
    _txt(slide, 0, Y_FOOT + 0.10, SW, 0.22, C.FOOTER, size=8, color=WHITE,
         align=PP_ALIGN.CENTER, name="foottext")


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def bullets(slide, items, x=0.9, y=Y_BODY, w=11.5, size=16, gap=0.10,
            lh=0.30, color=INK, bold_lead=False):
    """One text box per bullet. Returns their shape ids, in order."""
    ids = []
    cy = y
    for i, it in enumerate(items):
        text = it if isinstance(it, str) else it[0]
        n_lines = max(1, int(len(text) / 95) + 1)
        h = lh * n_lines
        b = _txt(slide, x + 0.32, cy, w - 0.32, h, text, size=size,
                 color=color, spacing=1.18, name="bul%d" % i)
        dot = _txt(slide, x, cy + 0.02, 0.3, 0.3, "\u2022", size=size,
                   color=ORANGE, bold=True, name="dot%d" % i)
        ids.append([b.shape_id, dot.shape_id])
        cy += h + gap
    return ids


# --------------------------------------------------------------------- slides

def s_title(prs):
    s = _blank(prs)
    _pic(s, LOGO, 11.52, 0.16, w=1.42, name="logo")
    _txt(s, 0.9, 0.72, 10.3, 0.6, C.COURSE, size=26, bold=True, color=INK,
         align=PP_ALIGN.CENTER, name="course")
    _rect(s, 0.9, 1.52, 11.5, 0.008, RGBColor(0x40, 0x40, 0x40))

    rows = [("Project Title", C.TITLE),
            ("Project ID", C.PROJECT_ID),
            ("Project Guide", C.GUIDE)]
    y = 2.30
    for label, val in rows:
        _txt(s, 1.55, y, 2.3, 0.4, label, size=19, color=ORANGE)
        _txt(s, 3.75, y, 0.2, 0.4, ":", size=19, color=INK)
        _txt(s, 4.05, y, 8.2, 0.8, val, size=19, color=INK)
        y += 0.62 if label != "Project Title" else 0.95

    _txt(s, 1.55, y + 0.05, 2.3, 0.4, "Project Team", size=19, color=ORANGE)
    _txt(s, 3.75, y + 0.05, 0.2, 0.4, ":", size=19, color=INK)
    ty = y + 0.05
    for nm, srn in zip(C.TEAM, C.TEAM_SRN):
        _txt(s, 4.05, ty, 5.0, 0.34, nm, size=16, color=INK)
        _txt(s, 8.60, ty, 2.6, 0.34, srn, size=13, color=GREY)
        ty += 0.36

    _rect(s, 0, Y_FOOT, SW, FOOT_H, ORANGE)
    _txt(s, 0, Y_FOOT + 0.10, SW, 0.22, "Department of Computer Science and "
         "Engineering  \u00b7  PES University", size=8, color=WHITE,
         align=PP_ALIGN.CENTER)
    anim.add_transition(s)
    return s


def s_outline(prs):
    s = _blank(prs)
    _chrome(s, "Outline")
    ids = []
    y = Y_BODY + 0.15
    for i, item in enumerate(C.OUTLINE):
        n = _txt(s, 3.05, y, 0.5, 0.4, "%02d" % (i + 1), size=16,
                 color=TEAL, bold=True, name="num%d" % i)
        t = _txt(s, 3.75, y, 7.0, 0.4, item, size=18, color=INK,
                 name="item%d" % i)
        ids.append([n.shape_id, t.shape_id])
        y += 0.56
    anim.add_timing(s, ids, dur=300)
    anim.add_transition(s)
    return s


def s_text(prs, heading, items, size=16, y=Y_BODY, lead=None):
    s = _blank(prs)
    _chrome(s, heading)
    yy = y
    steps = []
    if lead:
        b = _txt(s, 0.9, yy, 11.5, 0.5, lead, size=19, bold=True, color=NAVY,
                 name="lead")
        steps.append([b.shape_id])
        yy += 0.75
    steps += bullets(s, items, y=yy, size=size)
    anim.add_timing(s, steps)
    anim.add_transition(s)
    return s


def s_figure(prs, heading, fig, lead=None, cap=None, top=None, width=11.6):
    s = _blank(prs)
    _chrome(s, heading)
    yy = top if top is not None else Y_BODY + 0.05
    steps = []
    if lead:
        b = _txt(s, 0.9, yy, 11.5, 0.45, lead, size=17, color=INK, bold=True,
                 align=PP_ALIGN.CENTER, name="lead")
        steps.append([b.shape_id])
        yy += 0.55
    p = _pic(s, os.path.join(FIG, fig), (SW - width) / 2, yy, w=width,
             name="fig")
    steps.append([p.shape_id])
    if cap:
        b = _txt(s, 0.9, 6.55, 11.5, 0.4, cap, size=14, color=GREY,
                 align=PP_ALIGN.CENTER, italic=True, name="cap")
        steps.append([b.shape_id])
    anim.add_timing(s, steps, dur=500)
    anim.add_transition(s)
    return s


def s_problem(prs):
    s = _blank(prs)
    _chrome(s, "Problem Statement")
    box = s.shapes.add_textbox(Inches(1.0), Inches(Y_BODY + 0.12),
                               Inches(11.3), Inches(3.4))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    p = tf.paragraphs[0]
    p.line_spacing = 1.32
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = C.PROBLEM_STATEMENT
    r.font.name = FONT
    r.font.size = Pt(18)
    r.font.color.rgb = INK
    box.name = "statement"

    bar = _rect(s, 0.62, Y_BODY + 0.12, 0.07, 3.25, TEAL, name="accent")
    tag = _txt(s, 0.9, 5.95, 11.5, 0.7, C.PROBLEM_TAGLINE, size=21, bold=True,
               color=ORANGE, align=PP_ALIGN.CENTER, name="tagline")
    anim.add_timing(s, [[box.shape_id, bar.shape_id], [tag.shape_id]], dur=500)
    anim.add_transition(s)
    return s


def s_scope(prs):
    s = _blank(prs)
    _chrome(s, "Scope")
    steps = []
    for col, (title, items, col_c) in enumerate([
            ("In scope", C.SCOPE_IN, NAVY),
            ("Out of scope", C.SCOPE_OUT, ORANGE)]):
        x = 0.75 + col * 6.15
        hd = _rect(s, x, Y_BODY, 5.7, 0.46, col_c, name="hd%d" % col)
        ht = _txt(s, x + 0.25, Y_BODY + 0.10, 5.2, 0.3, title, size=16,
                  bold=True, color=WHITE, name="ht%d" % col)
        steps.append([hd.shape_id, ht.shape_id])
        y = Y_BODY + 0.72
        for i, it in enumerate(items):
            n_lines = max(1, int(len(it) / 46) + 1)
            h = 0.27 * n_lines
            d = _txt(s, x + 0.10, y + 0.02, 0.3, 0.3, "\u2022", size=14,
                     color=col_c, bold=True, name="d%d%d" % (col, i))
            b = _txt(s, x + 0.42, y, 5.15, h, it, size=13.5, color=INK,
                     spacing=1.16, name="b%d%d" % (col, i))
            steps.append([b.shape_id, d.shape_id])
            y += h + 0.16
    anim.add_timing(s, steps, dur=350)
    anim.add_transition(s)
    return s


def s_numbered(prs, heading, items, tail=None, size=14):
    s = _blank(prs)
    _chrome(s, heading)
    steps = []
    y = Y_BODY + 0.05
    for i, (title, body) in enumerate(items):
        n = _rect(s, 0.85, y, 0.44, 0.44, TEAL, name="n%d" % i)
        nt = _txt(s, 0.85, y + 0.09, 0.44, 0.3, str(i + 1), size=15,
                  bold=True, color=WHITE, align=PP_ALIGN.CENTER,
                  name="nt%d" % i)
        t = _txt(s, 1.50, y + 0.02, 10.8, 0.3, title, size=16, bold=True,
                 color=NAVY, name="t%d" % i)
        n_lines = max(1, int(len(body) / 105) + 1)
        b = _txt(s, 1.50, y + 0.36, 10.8, 0.27 * n_lines, body, size=size,
                 color=INK, spacing=1.16, name="b%d" % i)
        steps.append([n.shape_id, nt.shape_id, t.shape_id, b.shape_id])
        y += 0.42 + 0.27 * n_lines + 0.18
    if tail:
        tl = _txt(s, 0.85, 6.55, 11.5, 0.4, tail, size=16, bold=True,
                  color=ORANGE, align=PP_ALIGN.CENTER, name="tail")
        steps.append([tl.shape_id])
    anim.add_timing(s, steps, dur=350)
    anim.add_transition(s)
    return s


def s_usecases(prs):
    s = _blank(prs)
    _chrome(s, "Applications / Use cases")
    steps = []
    for i, (title, body) in enumerate(C.USE_CASES):
        col, row = i % 2, i // 2
        x = 0.75 + col * 6.15
        y = Y_BODY + 0.05 + row * 1.60
        card = _rect(s, x, y, 5.75, 1.42, RGBColor(0xF5, 0xF5, 0xF5),
                     name="card%d" % i)
        stripe = _rect(s, x, y, 0.075, 1.42,
                       [NAVY, TEAL, ORANGE][i % 3], name="st%d" % i)
        t = _txt(s, x + 0.30, y + 0.14, 5.2, 0.28, title, size=14.5,
                 bold=True, color=NAVY, name="ut%d" % i)
        b = _txt(s, x + 0.30, y + 0.47, 5.25, 0.9, body, size=11.5,
                 color=INK, spacing=1.14, name="ub%d" % i)
        steps.append([card.shape_id, stripe.shape_id, t.shape_id, b.shape_id])
    anim.add_timing(s, steps, dur=320)
    anim.add_transition(s)
    return s


def s_littable(prs, heading, rows):
    s = _blank(prs)
    _chrome(s, heading)
    steps = []
    hdr_y = Y_BODY
    hd = _rect(s, 0.55, hdr_y, 12.25, 0.42, NAVY, name="hdr")
    h1 = _txt(s, 0.75, hdr_y + 0.09, 3.3, 0.3, "RESEARCH PAPER", size=11.5,
              bold=True, color=WHITE, name="h1")
    h2 = _txt(s, 4.30, hdr_y + 0.09, 3.4, 0.3, "CITATION", size=11.5,
              bold=True, color=WHITE, name="h2")
    h3 = _txt(s, 8.00, hdr_y + 0.09, 4.6, 0.3, "KEY OUTCOME", size=11.5,
              bold=True, color=WHITE, name="h3")
    steps.append([hd.shape_id, h1.shape_id, h2.shape_id, h3.shape_id])

    y = hdr_y + 0.52
    for i, (paper, cite, out) in enumerate(rows):
        n_lines = max(int(len(out) / 62) + 1, int(len(cite) / 47) + 1,
                      int(len(paper) / 34) + 1)
        h = max(0.72, 0.20 * n_lines + 0.26)
        bg = _rect(s, 0.55, y, 12.25, h,
                   RGBColor(0xF4, 0xF4, 0xF4) if i % 2 == 0 else WHITE,
                   name="bg%d" % i)
        c1 = _txt(s, 0.75, y + 0.11, 3.35, h - 0.2, paper, size=11.5,
                  bold=True, color=INK, spacing=1.12, name="c1_%d" % i)
        c2 = _txt(s, 4.30, y + 0.11, 3.55, h - 0.2, cite, size=10,
                  color=GREY, spacing=1.12, name="c2_%d" % i)
        c3 = _txt(s, 8.00, y + 0.11, 4.65, h - 0.2, out, size=10.5,
                  color=INK, spacing=1.12, name="c3_%d" % i)
        steps.append([bg.shape_id, c1.shape_id, c2.shape_id, c3.shape_id])
        y += h + 0.06
    anim.add_timing(s, steps, dur=300)
    anim.add_transition(s)
    return s


def s_differ(prs):
    s = _blank(prs)
    _chrome(s, "Where we differ from prior work")
    steps = []
    y = Y_BODY
    for i, (topic, who, how) in enumerate(C.COLLISIONS):
        bg = _rect(s, 0.6, y, 12.15, 0.92,
                   RGBColor(0xF4, 0xF4, 0xF4) if i % 2 == 0 else WHITE,
                   name="bg%d" % i)
        st = _rect(s, 0.6, y, 0.07, 0.92, [NAVY, TEAL, ORANGE][i],
                   name="st%d" % i)
        t = _txt(s, 0.85, y + 0.10, 3.6, 0.5, topic, size=13.5, bold=True,
                 color=INK, spacing=1.1, name="t%d" % i)
        w = _txt(s, 4.55, y + 0.10, 2.2, 0.5, who, size=11.5, color=ORANGE,
                 bold=True, spacing=1.1, name="w%d" % i)
        h = _txt(s, 6.90, y + 0.08, 5.7, 0.78, how, size=11.5, color=INK,
                 spacing=1.12, name="h%d" % i)
        steps.append([bg.shape_id, st.shape_id, t.shape_id, w.shape_id,
                      h.shape_id])
        y += 1.00

    hd = _txt(s, 0.85, y + 0.16, 11.5, 0.32, "What is new, precisely:",
              size=14.5, bold=True, color=NAVY, name="nhd")
    steps.append([hd.shape_id])
    y += 0.54
    for i, n in enumerate(C.NOVELTY):
        d = _txt(s, 0.95, y + 0.01, 0.3, 0.3, "\u25b8", size=12,
                 color=TEAL, bold=True, name="nd%d" % i)
        b = _txt(s, 1.25, y, 11.2, 0.26, n, size=11.5, color=INK,
                 spacing=1.1, name="nb%d" % i)
        steps.append([b.shape_id, d.shape_id])
        y += 0.32
    anim.add_timing(s, steps, dur=300)
    anim.add_transition(s)
    return s


def s_notclaim(prs):
    s = _blank(prs)
    _chrome(s, "What we do not claim")
    steps = []
    y = Y_BODY + 0.10
    for i, (claim, why) in enumerate(C.DO_NOT_CLAIM):
        x = _txt(s, 0.80, y + 0.02, 0.4, 0.34, "\u2715", size=16, bold=True,
                 color=RED, name="x%d" % i)
        t = _txt(s, 1.30, y, 11.2, 0.30, claim, size=15, bold=True,
                 color=INK, name="t%d" % i)
        b = _txt(s, 1.30, y + 0.34, 11.2, 0.30, why, size=13,
                 color=GREY, spacing=1.12, name="b%d" % i)
        steps.append([x.shape_id, t.shape_id, b.shape_id])
        y += 0.96
    bar = _rect(s, 0.80, y + 0.14, 11.7, 0.008, RGBColor(0xCC, 0xCC, 0xCC))
    tl = _txt(s, 0.80, y + 0.34, 11.7, 0.7, C.NOT_NEW, size=14,
              color=ORANGE, spacing=1.2, name="tail")
    steps.append([tl.shape_id])
    anim.add_timing(s, steps, dur=320)
    anim.add_transition(s)
    return s


def s_effort(prs):
    s = _blank(prs)
    _chrome(s, "Team Effort Plan")
    steps = []
    hd = _txt(s, 0.75, Y_BODY, 6.0, 0.32, "Four separable tracks",
              size=16, bold=True, color=ORANGE, name="hd")
    steps.append([hd.shape_id])
    y = Y_BODY + 0.46
    for i, (track, owner, desc) in enumerate(C.TRACKS):
        bg = _rect(s, 0.75, y, 7.55, 0.86,
                   RGBColor(0xF4, 0xF4, 0xF4) if i % 2 == 0 else WHITE,
                   name="bg%d" % i)
        st = _rect(s, 0.75, y, 0.07, 0.86,
                   [NAVY, TEAL, ORANGE, RGBColor(0x70, 0x30, 0xA0)][i],
                   name="st%d" % i)
        t = _txt(s, 1.00, y + 0.08, 4.0, 0.28, track, size=13, bold=True,
                 color=INK, name="t%d" % i)
        o = _txt(s, 5.05, y + 0.08, 3.1, 0.28, owner, size=11,
                 color=ORANGE, bold=True, name="o%d" % i)
        d = _txt(s, 1.00, y + 0.40, 7.05, 0.42, desc, size=10.5,
                 color=GREY, spacing=1.1, name="d%d" % i)
        steps.append([bg.shape_id, st.shape_id, t.shape_id, o.shape_id,
                      d.shape_id])
        y += 0.94

    hd2 = _txt(s, 8.70, Y_BODY, 4.0, 0.32, "Stage-wise tasks", size=16,
               bold=True, color=ORANGE, name="hd2")
    steps.append([hd2.shape_id])
    y2 = Y_BODY + 0.46
    for i, (wk, task) in enumerate(C.STAGES):
        w = _txt(s, 8.70, y2, 3.9, 0.26, wk, size=12.5, bold=True,
                 color=TEAL, name="w%d" % i)
        t = _txt(s, 8.70, y2 + 0.28, 3.95, 0.5, task, size=11,
                 color=INK, spacing=1.12, name="s%d" % i)
        steps.append([w.shape_id, t.shape_id])
        y2 += 0.94
    anim.add_timing(s, steps, dur=300)
    anim.add_transition(s)
    return s


def s_reviews(prs):
    s = _blank(prs)
    _chrome(s, "Project Timeline")
    steps = []
    p = _pic(s, os.path.join(FIG, "f7_gantt.png"), 0.75, Y_BODY - 0.02,
             w=11.85, name="fig")
    steps.append([p.shape_id])
    y = 5.45
    hd = _txt(s, 0.75, y, 11.8, 0.3, "Semester 5 \u00b7 four reviews",
              size=14.5, bold=True, color=ORANGE, name="hd")
    steps.append([hd.shape_id])
    y += 0.42
    for i, (rv, when, what) in enumerate(C.REVIEWS):
        x = 0.75 + i * 3.02
        box = _rect(s, x, y, 2.85, 0.86,
                    RGBColor(0xF4, 0xF4, 0xF4) if i else RGBColor(0xFD, 0xEE, 0xE6),
                    name="rb%d" % i)
        st = _rect(s, x, y, 2.85, 0.055, ORANGE if i == 0 else TEAL,
                   name="rs%d" % i)
        t = _txt(s, x + 0.18, y + 0.14, 2.5, 0.26, rv, size=13, bold=True,
                 color=ORANGE if i == 0 else NAVY, name="rt%d" % i)
        w = _txt(s, x + 0.18, y + 0.38, 2.5, 0.22, when, size=10,
                 color=GREY, name="rw%d" % i)
        d = _txt(s, x + 0.18, y + 0.58, 2.55, 0.26, what, size=10,
                 color=INK, spacing=1.08, name="rd%d" % i)
        steps.append([box.shape_id, st.shape_id, t.shape_id, w.shape_id,
                      d.shape_id])
    anim.add_timing(s, steps, dur=300)
    anim.add_transition(s)
    return s


def s_refs(prs):
    s = _blank(prs)
    _chrome(s, "References")
    y = Y_BODY - 0.02
    for i, r in enumerate(C.REFERENCES):
        n_lines = max(1, int(len(r) / 118) + 1)
        h = 0.22 * n_lines
        _txt(s, 0.80, y + 0.01, 0.35, 0.24, "[%d]" % (i + 1), size=10.5,
             color=TEAL, bold=True)
        _txt(s, 1.25, y, 11.3, h, r, size=10.5, color=INK, spacing=1.12)
        y += h + 0.075
    anim.add_transition(s)
    return s


def s_thanks(prs):
    s = _blank(prs)
    _pic(s, LOGO, 11.52, 0.16, w=1.42, name="logo")
    _rect(s, 0.45, 1.82, 12.05, 0.008, RGBColor(0x40, 0x40, 0x40))
    t = _txt(s, 0, 3.15, SW, 0.9, "Thank You", size=44, color=INK,
             align=PP_ALIGN.CENTER, name="ty")
    q = _txt(s, 0, 4.25, SW, 0.5, C.PROBLEM_TAGLINE, size=17, color=ORANGE,
             align=PP_ALIGN.CENTER, italic=True, name="q")
    _rect(s, 0, Y_FOOT, SW, FOOT_H, ORANGE)
    _txt(s, 0, Y_FOOT + 0.10, SW, 0.22, C.FOOTER, size=8, color=WHITE,
         align=PP_ALIGN.CENTER)
    anim.add_timing(s, [[t.shape_id], [q.shape_id]], dur=600)
    anim.add_transition(s)
    return s


def s_pipeline(prs):
    """Staged reveal: six cumulative PNGs stacked, each fading over the last."""
    s = _blank(prs)
    _chrome(s, "Work Flow")
    lead = _txt(s, 0.9, Y_BODY, 11.5, 0.4,
                "One measurement pipeline, applied at each level",
                size=17, bold=True, color=NAVY, align=PP_ALIGN.CENTER,
                name="lead")
    steps = [[lead.shape_id]]
    for n in range(1, 7):
        p = _pic(s, os.path.join(FIG, "f5_pipeline_%d.png" % n),
                 0.60, Y_BODY + 0.55, w=12.1, name="stage%d" % n)
        steps.append([p.shape_id])
    note = _txt(s, 0.9, 6.28, 11.5, 0.7, C.WORKFLOW_NOTE, size=13,
                color=GREY, align=PP_ALIGN.CENTER, spacing=1.15, name="note")
    steps.append([note.shape_id])
    anim.add_timing(s, steps, dur=380)
    anim.add_transition(s)
    return s


def build():
    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    s_title(prs)
    s_outline(prs)
    s_text(prs, "Introduction", C.INTRO, size=17)
    s_figure(prs, "Introduction", "f1_thermometer.png",
             lead="Why an instrument is trusted",
             cap="Nobody has done this for the tools that read language models.",
             width=10.4)
    s_problem(prs)
    s_figure(prs, "Problem Statement", "f2_levels.png",
             lead="Three levels. Three published families of tool.",
             width=11.9)
    s_figure(prs, "The neuroscience framing", "f3_preparations.png",
             lead="A preparation is a system chosen because the answer is knowable",
             width=11.9)
    s_text(prs, "The neuroscience framing", C.PREPARATION, size=16)
    s_figure(prs, "Ground truth, for free", "f4_ground_truth.png",
             width=11.4,
             cap="At the trait and self levels there is no free truth, "
                 "so we plant it.")
    s_pipeline(prs)
    s_scope(prs)
    s_numbered(prs, "Feasibility study", C.FEASIBILITY,
               tail=C.FEASIBILITY_TAG)
    s_usecases(prs)
    s_figure(prs, "Background Study", "f6_literature.png",
             lead="Four strands, one shared gap", width=12.0)
    s_littable(prs, "Background Study \u2013 Unit level", C.LIT_UNIT)
    s_littable(prs, "Background Study \u2013 Trait level", C.LIT_TRAIT)
    s_littable(prs, "Background Study \u2013 Self level", C.LIT_SELF)
    s_littable(prs, "Background Study \u2013 Neuroscience", C.LIT_NEURO)
    s_differ(prs)
    s_notclaim(prs)
    s_reviews(prs)
    s_effort(prs)
    s_text(prs, "Any other information", C.OTHER_INFO, size=15)
    s_refs(prs)
    s_thanks(prs)

    out = os.path.join(HERE, "CALIPER_Review1.pptx")
    prs.save(out)
    print("wrote", os.path.relpath(out), "-", len(prs.slides.__iter__.__self__._sldIdLst), "slides")

    for sl in prs.slides:
        anim.strip_timing(sl)
    out2 = os.path.join(HERE, "CALIPER_Review1_static.pptx")
    prs.save(out2)
    print("wrote", os.path.relpath(out2), "(no animation)")
    return out, out2


if __name__ == "__main__":
    build()

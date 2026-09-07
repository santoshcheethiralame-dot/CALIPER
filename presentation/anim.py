"""PowerPoint animation support for python-pptx.

python-pptx has no animation API, so this writes the <p:timing> tree directly
into each <p:sld>. Schema order inside CT_Slide is fixed and PowerPoint is
strict about it:

    <p:sld> <p:cSld/> <p:clrMapOvr/> <p:transition/> <p:timing/> </p:sld>

The timing tree we emit is the standard click-sequenced entrance build:

    p:timing > p:tnLst > p:par > p:cTn(nodeType=tmRoot)
      > p:childTnLst > p:seq(nodeType=mainSeq)
        > p:cTn > p:childTnLst > p:par            <- one per click
          > ... > p:cTn(nodeType=clickEffect)
            > p:set (style.visibility -> visible)
            > p:animEffect (transition=in, filter=...)

Every animated shape must start hidden from the animation's point of view; the
<p:set> is what makes it visible on its click, which is why it is paired with
the effect rather than used alone.
"""
from lxml import etree

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS = {"p": P, "a": A}

# presetID values PowerPoint uses for entrance effects
FADE = 10
WIPE = 22
FLOAT_UP = 42


def _q(tag):
    return "{%s}%s" % (P, tag)


def _sub(parent, tag, **attrs):
    e = etree.SubElement(parent, _q(tag))
    for k, v in attrs.items():
        e.set(k, str(v))
    return e


class _Ids:
    """Timing-node ids must be unique within a slide."""

    def __init__(self, start=1):
        self.n = start

    def next(self):
        self.n += 1
        return self.n


def _click_par(parent, ids, spids, preset, filt, dur):
    """One click step. All shapes in `spids` animate together."""
    outer = _sub(parent, "par")
    o = _sub(outer, "cTn", id=ids.next(), fill="hold")
    st = _sub(o, "stCondLst")
    _sub(st, "cond", delay="indefinite")
    ol = _sub(o, "childTnLst")

    mid = _sub(ol, "par")
    m = _sub(mid, "cTn", id=ids.next(), fill="hold")
    st = _sub(m, "stCondLst")
    _sub(st, "cond", delay="0")
    ml = _sub(m, "childTnLst")

    for i, spid in enumerate(spids):
        inner = _sub(ml, "par")
        c = _sub(inner, "cTn", id=ids.next(), presetID=preset,
                 presetClass="entr", presetSubtype="0", fill="hold",
                 grpId="0", nodeType="clickEffect" if i == 0 else "withEffect")
        st = _sub(c, "stCondLst")
        _sub(st, "cond", delay="0")
        cl = _sub(c, "childTnLst")

        # make it visible
        s = _sub(cl, "set")
        b = _sub(s, "cBhvr")
        t = _sub(b, "cTn", id=ids.next(), dur="1", fill="hold")
        st2 = _sub(t, "stCondLst")
        _sub(st2, "cond", delay="0")
        tg = _sub(b, "tgtEl")
        _sub(tg, "spTgt", spid=spid)
        an = _sub(b, "attrNameLst")
        nm = _sub(an, "attrName")
        nm.text = "style.visibility"
        to = _sub(s, "to")
        _sub(to, "strVal", val="visible")

        # the visible effect
        ae = _sub(cl, "animEffect", transition="in", filter=filt)
        b2 = _sub(ae, "cBhvr")
        _sub(b2, "cTn", id=ids.next(), dur=str(dur))
        tg2 = _sub(b2, "tgtEl")
        _sub(tg2, "spTgt", spid=spid)


def add_timing(slide, steps, preset=FADE, filt="fade", dur=450):
    """Attach a click-sequenced entrance build to `slide`.

    steps: list of steps; each step is a list of shape ids that appear together.
           Pass a bare int for a one-shape step.
    """
    steps = [s if isinstance(s, (list, tuple)) else [s] for s in steps]
    steps = [[int(x) for x in s] for s in steps if s]
    if not steps:
        return

    sld = slide._element
    for old in sld.findall(_q("timing")):
        sld.remove(old)

    ids = _Ids()
    timing = etree.Element(_q("timing"))
    tnLst = _sub(timing, "tnLst")
    par = _sub(tnLst, "par")
    root = _sub(par, "cTn", id=ids.next(), dur="indefinite",
                restart="never", nodeType="tmRoot")
    rootChildren = _sub(root, "childTnLst")
    seq = _sub(rootChildren, "seq", concurrent="1", nextAc="seek")
    main = _sub(seq, "cTn", id=ids.next(), dur="indefinite", nodeType="mainSeq")
    mainChildren = _sub(main, "childTnLst")

    for step in steps:
        _click_par(mainChildren, ids, step, preset, filt, dur)

    prev = _sub(seq, "prevCondLst")
    c = _sub(prev, "cond", evt="onPrev", delay="0")
    t = _sub(c, "tgtEl")
    _sub(t, "sldTgt")
    nxt = _sub(seq, "nextCondLst")
    c = _sub(nxt, "cond", evt="onNext", delay="0")
    t = _sub(c, "tgtEl")
    _sub(t, "sldTgt")

    sld.append(timing)   # timing is last in CT_Slide, so append is correct


def add_transition(slide, kind="fade", speed="med"):
    """Slide transition. Must sit before <p:timing> in document order."""
    sld = slide._element
    for old in sld.findall(_q("transition")):
        sld.remove(old)
    tr = etree.Element(_q("transition"))
    tr.set("spd", speed)
    etree.SubElement(tr, _q(kind))

    timing = sld.find(_q("timing"))
    if timing is not None:
        timing.addprevious(tr)
    else:
        sld.append(tr)


def strip_timing(slide):
    """Remove animation from a slide (used for the static fallback deck)."""
    sld = slide._element
    for tag in ("timing", "transition"):
        for old in sld.findall(_q(tag)):
            sld.remove(old)


def timing_spids(slide):
    """Every spid referenced by this slide's timing tree - for verification."""
    out = []
    for t in slide._element.findall(_q("timing")):
        for el in t.iter(_q("spTgt")):
            out.append(int(el.get("spid")))
    return out

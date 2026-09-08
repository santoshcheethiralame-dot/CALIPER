# Proposal III in plain terms

**For mentor review · 8 September 2026**
Companion to `PROPOSAL_III_STATUS.md`, which carries the numbers and the detail.

---

## The idea in one line

We are doing to language-model units what neuroscience does to neurons — and we found
the one place where the model version can do something neuroscience has never been able
to do.

---

## Why neuroscience, precisely

The interesting overlap is **not** that models resemble brains. That claim is contested,
hard to defend, and easy for a reviewer to attack. We are not making it.

The overlap is that **the measurement problem is the same one**, and neuroscience has
spent forty years solving it.

A neuroscientist puts an electrode near a cell and watches it fire. The cell clearly
responds to *something* — but to what, exactly? You cannot ask it. You show it thousands
of stimuli, record when it responds, and work backwards to a description of what it
cares about. There is a mature toolkit for this: reverse correlation, spike-triggered
covariance, and the modern method, Maximally Informative Dimensions.

A language model has the identical problem. A unit inside it clearly responds to
something. You cannot ask it either. So you feed it a large amount of text, record when
it activates, and work backwards.

Same problem, same maths, and — the gap we are filling — **almost nobody has carried the
methods across.** A search found five papers using Maximally Informative Dimensions.
All five are neuroscience. None is on a modern deep network.

So the project is not a metaphor. It is a methods transfer, and the transfer is real
work: two of the three classical methods turn out to *fail* on language models, for a
reason we can state exactly.

---

## The one place models beat brains

Here is the part worth remembering, because it is the whole contribution.

In neuroscience, **you never find out whether you were right.** You produce your best
description of what the cell responds to, and there is no way to check it. There is no
answer key. This is the permanent condition of the field.

In a language model, for one particular kind of unit, **the answer key is sitting right
there.** What that unit responds to is written directly in the model's own weights — not
estimated, not inferred, just readable, exactly, for free.

So we can do the thing a neuroscientist can only dream about: run the estimation method,
then look up the true answer, and see how often the method was right.

We checked that the answer key is genuinely exact before trusting it. It is: the match is
correct to six decimal places on one model and to ten on another.

---

## What we found

**1. The old methods fail; the modern one works.**
Scored against the answer key, the two classical reverse-correlation methods recovered
almost nothing — 0 and 1 unit out of 30. The modern method recovered 26 of 30. We can
say precisely why the old ones fail: they assume things about the input and the unit's
response curve that are true of neurons and false of language models.

**2. Even the method that works fails silently, and more often than anyone reports.**
On a hundred units per model, it gets the wrong answer on roughly 7% to 23% of them,
depending on the model. The failures are the concerning kind: the method gives no
indication that anything went wrong. It looks just as confident when it is wrong.

**3. The usual summary statistics hide this completely — and hide it better as the model
gets cleaner.** One model scores 0.9994 on the typical "how well did we do" measure,
which is close to perfect and would be reported as a success by anyone. That same set of
units contains one where the method got the answer almost entirely wrong.

That third point is why this has to come first. Proposal III's later stages describe
**populations** of units together. If the individual measurements are quietly wrong 1 in
5 times, everything built on top of them inherits that error invisibly, and there is no
way afterwards to tell a real population pattern from accumulated measurement error.
Neuroscience learned this order the hard way: you establish what your instrument's error
looks like before you use it to make claims.

---

## What we are finishing now

If the method fails silently, can a practitioner tell *which* answers to distrust,
without an answer key? That is the practical question, and it is the one nobody has been
able to study — checking whether a warning sign works requires the answer key whose
absence is the reason you needed a warning sign.

We can study it. We are scoring the warning signs people actually use. One early result:
the most widely used check in the field — run it several times, see if you get the same
answer — appears to catch the bad failures and miss the subtle ones. A full measurement
is running as this is written.

We wrote down what would count as success **before** starting the run, including the
outcome where our own current recommendation turns out to be wrong.

---

## Where it goes

1. Finish the warning-sign measurement. *(running, no GPU needed)*
2. Build the calibrated instrument into a reusable tool.
3. **Tuning curves** — the descriptive layer Proposal III asked for, now sitting on an
   instrument whose error rate we actually know.
4. **Population structure** — the pairwise maximum-entropy layer, which the August
   research pass confirmed has no existing language-model application at all.

Steps 1 and 2 need no GPU and no money. Steps 3 and 4 are the capstone proper.

---

## What we would value your guidance on

1. **How to split this across four people.** The instrument work is concentrated in one
   pair of hands right now. The tuning-curve battery and the population layer divide
   cleanly, but we would rather agree the split before building than after.

2. **Whether to publish the earlier introspection result now.** The previous phase
   produced one control experiment that, as far as we can verify, nobody else in that
   debate has run. It needs no further compute — it is written up and could go out as a
   short preprint. It belongs to a proposal that was not selected, so we do not want to
   spend capstone attention on it without your view.

3. **What counts as "published" for the capstone.** This is the question we most need
   answered, and we would rather ask it plainly than guess. Does a preprint count? A
   workshop paper? Must it be a peer-reviewed venue, and must it be *accepted* rather
   than submitted, by the deadline? Our schedule changes completely depending on the
   answer, and we are currently planning against a bar we have not read.

4. **How ambitious to be about venue.** The honest read is that this is a solid,
   careful methods contribution rather than a spectacular result. That can do well at
   venues that reward rigour and useful tools over novelty. We would rather aim
   correctly than aim high and miss.

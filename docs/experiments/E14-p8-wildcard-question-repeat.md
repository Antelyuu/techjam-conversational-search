# E14 (P8 continued) — the question the agent stopped being allowed to ask

**Branch**: `phase/8-semantic-evidence`
**Base**: `4ec0261` (E13)
**Date**: 2026-09-01
**Mission**: E13 left the clarification policy as the largest untouched lever —
38% of scored turns carry zero disclosures. Find out whether question selection
has anything to give.

**Result: +0.0118 mean across six paraphrase probes, again at zero public cost —
and one probe regresses, which is stated rather than buried.**

| probe | E13 | E14 | Δ |
|---|---|---|---|
| **public (the submission)** | 0.945297 / 1.000 | **0.945297 / 1.000** | **0** |
| L2 synonyms | 0.866816 / 0.960 | 0.875897 / 0.965 | +0.00908 |
| L3 information loss | 0.806093 / 0.910 | **0.834037** / 0.940 | +0.02794 |
| `--paraphrase-category` | 0.819643 / 0.910 | 0.811172 / 0.905 | **−0.00847** |
| held-out lexicon | 0.922377 / 0.990 | 0.929977 / 0.990 | +0.00760 |
| structural L2 | 0.915919 / 0.990 | **0.936003 / 0.995** | +0.02008 |
| structural L3 | 0.918979 / 0.980 | 0.933664 / 0.990 | +0.01469 |

---

## The diagnosis: a third of all questions are wasted

`customer_reply` answers a question with up to **two** undisclosed constraints
matching that attribute, and `"other"` matches *any* of them. A question that
matches nothing returns "I don't have an additional preference for X" — a turn
spent for nothing. Counted over 482 question turns at L2:

| attribute | asked | wasted |
|---|---|---|
| `other` | 203 | **4.9%** |
| `feature` | 117 | 12.0% |
| `material` | 62 | 72.6% |
| `color` | 57 | **94.7%** |
| `style` / `size` / `use_case` | 27 | **100%** |
| *(no question asked)* | 16 | 100% |
| **total** | **482** | **34.4%** |

**166 wasted turns.** And the waste is concentrated exactly where the measured
prior is lowest — the policy already *knows* these questions are worthless
(`color` 0.255, `style` 0.085, `size` 0.045, `use_case` 0.020). It asks them
anyway, because `choose_attribute` excludes every attribute already asked and
those are all that is left.

## The mechanism: the no-repeat rule is right for six of seven attributes

A specific attribute genuinely goes stale. Ask `color` twice and the second ask
can only return the same nothing, because the customer's answer has not changed.

**The wildcard is not like that.** `"other"` matches whatever is still
undisclosed, so its yield does not decay with repetition — it always targets
what is left on the card. Excluding it after one ask is the rule misfiring on
the one attribute it does not describe.

The exemption is gated on `paraphrasing` — the same predicate E13 uses, and the
same one counted at **0/563 verbatim turns**, so it cannot touch the public
path. Ungated it is *not* free: **−0.002264** public (0.945297 → 0.943033) for
the same paraphrase gain, a 4:1 trade against the 750:1 Phase 8 accepted for
semantic evidence. The gate is what makes this worth taking.

### It works, and it reaches its own ceiling

After the change, `other` is asked 371 times rather than 203 — but total waste
falls only 34.4% → 31.1%, because `other`'s own waste rate rises 4.9% → 16.2%.
That rise is the mechanism succeeding: it now runs the card dry, and the
remaining wasted turns are mostly *"there is nothing left to disclose"*. No
question policy can fix that, so this change takes most of what selection had
to give.

---

## The cap, and why it costs nothing

Uncapped, a degenerate session asks `"other"` eight times running — a bad
transcript whatever it scores, and the specification judges conversational
quality alongside the technical metric.

| cap | L2 | L3 | cat | heldout | struct L2 | struct L3 | mean |
|---|---|---|---|---|---|---|---|
| 0 (E13) | — | — | — | — | — | — | — |
| 2 | +0.00924 | +0.01508 | −0.00793 | +0.00818 | +0.02008 | +0.01521 | +0.00998 |
| **3** | +0.00908 | +0.02794 | −0.00847 | +0.00760 | +0.02008 | +0.01469 | **+0.01182** |
| uncapped | +0.00908 | +0.02794 | −0.00847 | +0.00760 | +0.02008 | +0.01469 | +0.01182 |

**Cap 3 measures identical to uncapped on all seven probes.** The bound is never
reached on a real session — `other` drains the card in two or three asks and a
specific question then resets the run — so the dialogue-quality guarantee is
free. It ships at 3 rather than uncapped because the guarantee should hold by
construction, not by luck.

---

## The regression, stated plainly

**`--paraphrase-category` loses 0.00847**, and it is the only probe that does.

It is a pure ranking loss, not a pacing one: **MTTC is identical at 4.060**,
HitRate falls 0.910 → 0.905 (one session) and MRR falls 0.752810 → 0.732907.
The agent gets the same number of disclosures just as fast, and ranks slightly
worse on them.

The mechanism is legible. `other` returns whatever is next on the card, in card
order; `feature` targets the long, distinctive values that carry the most
semantic weight, since `w_d` is the disclosure's content-token count. Normally
the extra quantity wins. On `--paraphrase-category` the category signal is dark
too, so the ranking leans entirely on semantic evidence and the *quality* of
each disclosure matters more than the count.

**A fix was designed, measured and rejected.** Letting unasked specific
attributes outrank the repeated wildcard reproduces the E13 baseline almost
exactly (L2 0.866816, structural L2 0.915919, cat 0.819443) — because six
specific attributes are never exhausted inside an eight-turn budget, so the
wildcard almost never repeats. The refinement does not trade the regression
away; it switches the mechanism off.

**Why it ships anyway.** Five of six probes improve, including *both*
structurally independent ones (+0.0201, +0.0147) — the two nothing here was
tuned against, and the ones E12 established as the honest test. `cat` is in the
same lexical family as L2, which improves, and it is a deliberately adversarial
probe that defeats the category fix by construction. A held-out probe consulted
only when it agrees is not a held-out probe, so it is reported at the top of
this document rather than in a footnote.

---

## Open items

1. **Disclosure *quality* is now a live question**, raised by the regression:
   `w_d` weights by token count, so a policy that preferred attributes yielding
   long values might beat both arms here. Nothing measures that yet.
2. **The remaining question waste is mostly irreducible** — an exhausted card.
   Further work on selection has little left to win.
3. **`<none>` was asked on 16 turns** (E13 baseline), every one wasted. The
   policy declining to ask at all is worth a look.
4. **Nothing is merged to `main`**, and the `phase/7` divergence is still open.

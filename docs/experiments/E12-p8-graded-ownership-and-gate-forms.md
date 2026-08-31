# E12 (P8 continued) — graded slot ownership, gate forms, and what the third probe changed

**Branch**: `phase/8-semantic-evidence`
**Base**: `910a7eb`
**Date**: 2026-09-01
**Mission**: close E11's open items 4, 5 and 9 — relax `slot_evidence`, try the
faithful mask form of the semantic gate, and price the discarded
`constraint_evidence` pass.

**Result: nothing shipped moves a score.** Five levers were measured and all
five confirmed the inherited configuration. What the phase actually produced is
a *third* paraphrase probe and the evidence that the first two were agreeing
with each other more than with reality.

---

## The headline, and why a null result is the finding

Every score below reproduces the base exactly:

| guardrail | value |
|---|---|
| `local_evaluator` | **0.945297** / HitRate 1.000 / MRR 0.938657 / MTTC 2.815 |
| `paraphrase_eval --level 0` | **0.945297** (control holds) |
| `--paraphraser structural --level 0` | **0.945297** (new control, same invariant) |
| `SHOPPING_AGENT_SEMANTIC=0` | **0.945297** (inertness holds) |
| tests | 222 |

The shipped agent is byte-identical in behaviour to `910a7eb`. The two code
changes are a measurement knob whose default is the current value, and a second
paraphraser in the harness.

---

## The branch discovery that framed everything

E11's handoff states that its base, `phase/7-paraphrase-robustness` at
`bc6b373`, "is untouched and still matches origin". **It does not.** `origin`
has since moved that branch 15 commits to `e4e6f73`, carrying an independent
attack on the same problem — a regime-gated weight profile and, critically,
`slots.ownership_overlap`: a graded, token-containment version of slot
ownership. That is E11's open item 4, already written.

So item 4 was not a design question. It was a *measurement* question about code
that already existed, developed against a base with no semantic evidence in it.

---

## 1. Graded slot ownership (item 4) — subsumed

`slot_evidence` is whole-string equality at weight 16. Relaxing the match alone
does nothing, because `slots.ownership_weights` assigns weight **0.0** to any
disclosure no pooled candidate owns — under rewording that is every disclosure,
so the denominator collapses and the feature is switched off rather than
blunted. The selectivity weights have to be re-priced against the relaxed
notion of ownership too.

Three gate forms were built. `live_disclosures` and `consistent` stay **exact**
under all of them, which is what keeps the public path safe: they feed
`shortlist.py` and they scale `semantic_evidence`, so deriving them from the
relaxed pass would let a successful relaxation report that the customer is
quoting after all.

* **hard** — relax only when *no* disclosure is owned (phase/7's form).
* **soft** — relax each unowned disclosure independently, leaving owned ones on
  their exact weights. This is E11's own soft-gate lesson pointed at the other
  feature, and it is new here.

A unit trace confirms `soft` is strictly sharper in principle. Customer quotes
one constraint and rewords another; target and impostor both own the quoted
one:

| mode | `slot_evidence` target | impostor | separated? |
|---|---|---|---|
| off | 1.000 | 1.000 | no |
| hard | 1.000 | 1.000 | no (nothing fires: something *is* owned) |
| soft | 0.886 | 0.432 | **yes** |

**Public cost of every arm: exactly zero** — 0.945297 / 1.000 / 0.938657 /
2.815, all four figures. The target owns all 800 of its own disclosures, so on
a quoting customer no disclosure is ever unowned and the graded path never
fires.

Paraphrased, on top of the shipped semantic evidence (Δ against `off`):

| arm | L2 | L3 | cat (held out) | heldout lex | structural L2 |
|---|---|---|---|---|---|
| hard @3.0 | −0.00025 | −0.00029 | **−0.00270** | +0.00204 | +0.00145 |
| soft @3.0 | −0.00025 | +0.00004 | **−0.00380** | +0.00343 | +0.00145 |
| hard @4.0 | −0.00030 | −0.00029 | **−0.00400** | +0.00116 | — |
| soft @4.0 | −0.00030 | −0.00004 | **−0.00510** | +0.00211 | — |

HitRate never moves on any arm or probe. The mean effect across five probes is
**+0.0002**, with mixed signs. That is a wash bought with ~40 lines of ranking
code and a per-turn containment pass.

`MAX_LENGTH_RATIO` 4.0 — phase/7's own suggested-but-unmeasured loosening,
which admits honest fragments of a long value — is worse than 3.0 on every
probe. **Measured, and it is now closed.**

### Why it is subsumed

`semantic_evidence` already scores disclosures against **card values**, the
same granularity slot ownership uses. Containment and cosine are two matchers
answering the same question about the same objects, and the better one is
already installed. This is the second time an E11-era lever has proved
subsumed, after `lexical_rank` 1.0→4.0.

---

## 2. The same feature with semantic evidence OFF — the overfitting proof

If the mechanism were genuinely orthogonal it should pay on the fallback path,
where `slot_evidence` really does go dark. It was measured there
(`SHOPPING_AGENT_SEMANTIC=0`), and this is the most informative table in the
phase:

| arm | public | L2 | L3 | cat | **structural L2** |
|---|---|---|---|---|---|
| off | 0.945297/1.000 | 0.711681/0.825 | 0.691851/0.815 | 0.642526/0.735 | **0.839603/0.960** |
| hard | 0.945297/1.000 | 0.723177/0.835 | 0.704572/0.830 | 0.653924/0.745 | **0.807865/0.920** |
| soft | 0.945297/1.000 | 0.721807/0.835 | 0.705322/0.830 | 0.652824/0.745 | **0.805629/0.920** |

+0.010 to +0.013 on the three synonym-family probes, and **−0.032 on the
structural one**, which also loses four HitRate points. Net across the four
paraphrase probes: ~0.

**The mechanism is legible.** The structural paraphraser preserves **99.71%**
of content-word tokens — it reframes and reorders instead of substituting,
against the synonym paraphraser's 35.76% retention. Token containment is
therefore satisfied by the true owner *and* by every near-duplicate that shares
the vocabulary, so at weight 16 it delivers noise. Containment cannot
distinguish "these words, rearranged" from "these words, belonging to someone
else".

That is a textbook overfitting signature: the feature gains exactly where the
probe shares the vocabulary it was tuned against, and loses where the probe is
independent. **Not shipped, in any mode.**

---

## 3. The gate as a mask rather than a scalar (item 5) — rejected

E11 called this "the obvious next experiment rather than a fix". The scalar
gate multiplies every candidate's semantic score by the share of disclosures
nobody owns, so a candidate matching only the *already-owned* disclosures still
earns scaled credit. The mask drops owned disclosures from the semantic
question entirely and scores the rest at full weight.

Both are inert on a quoting customer — the scalar goes to 0.0, the mask empties
the weight list and the scorer is never called.

| probe | scalar (shipped) | mask | Δ |
|---|---|---|---|
| public | 0.945297/1.000 | 0.945297/1.000 | 0 |
| L2 | 0.847762/**0.975** | 0.838035/**0.965** | **−0.00973** |
| L3 | 0.790816/0.905 | 0.789497/0.905 | −0.00132 |
| cat | 0.798361/0.910 | 0.796836/0.910 | −0.00152 |
| heldout | 0.903642/0.990 | 0.903542/0.990 | −0.00010 |
| structural L2 | 0.891342/0.985 | 0.891342/0.985 | 0 |
| structural L3 | 0.895402/0.985 | 0.895402/0.985 | 0 |

Worse on every probe that can tell them apart, and it **costs a hit** at L2.
The reason the scalar is right is worth keeping: an owned disclosure is still
evidence about *which* candidate is correct, and discarding it throws away
signal that the exact features have already confirmed is trustworthy. The
faithful form is not the better form. **Closed.**

---

## 4. The semantic weight, re-swept across five probes

E11 chose 192 from a plateau it measured as 192–768, using two probes. Re-swept
here against five, including the structural one it never saw. The gate makes
this free to tune on paraphrase alone: no weight can move the public score.

| weight | L2 | L3 | cat | structural L2 | heldout | mean |
|---|---|---|---|---|---|---|
| 96 | 0.847127 | 0.781471 | **0.800353** | **0.892158** | **0.904407** | 0.845103 |
| **192** | **0.847762** | **0.790816** | 0.798361 | 0.891342 | 0.903642 | **0.846385** |
| 384 | 0.844883 | 0.787393 | 0.794126 | 0.891431 | 0.903542 | 0.844275 |
| 768 | 0.844808 | 0.785689 | 0.793580 | 0.891431 | 0.902792 | 0.843660 |
| 1536 | 0.844681 | 0.782653 | 0.789204 | 0.891431 | 0.902042 | 0.842002 |

192 wins the mean and takes L2 and L3 outright. 96 wins `cat`, structural and
heldout by small margins but gives up 0.0093 at L3, where information is
actually lost — the regime that matters most if a private customer is terse.
**E11's choice survives a broader probe set than chose it.** Everything above
384 is monotonically worse, so the "plateau to 768" reading does not reproduce.

---

## 5. Top-k aggregation instead of the max — rejected, decisively

`score_pool` takes `max` cosine over a candidate's card values, so a product
with sixty values gets sixty draws at a spuriously high cosine against a
reworded disclosure while a sparse one gets three. Averaging the best *k*
values is the cheapest test of that luck bias.

| top_k | L2 | L3 | cat | structural L2 | heldout | mean |
|---|---|---|---|---|---|---|
| **1** (shipped) | **0.847762** | **0.790816** | **0.798361** | **0.891342** | **0.903642** | **0.846385** |
| 2 | 0.805859 | 0.754580 | 0.717188 | 0.814531 | 0.847209 | 0.787873 |
| 3 | 0.732672 | 0.685216 | 0.682100 | 0.762790 | 0.791986 | 0.730953 |

Wrong by a distance — k=2 costs 0.059 on the mean and 5 HitRate points at L2.
The hypothesis misread the question the feature asks. A disclosure corresponds
to **one** card value, so the max is the semantics; the runner-up value is a
different fact about the product and averaging it in is pure dilution. The
length bias is real and is evidently much smaller than the signal it would cost
to correct.

---

## 6. The discarded `constraint_evidence` pass (item 9) — a non-issue

E11 listed it as a cost: "a token-coverage pass over 400 candidates per turn
that is then discarded". Measured, warmed cache, 4 disclosures × 400
candidates:

**0.154 ms per turn**, against E11's measured 52 ms median turn — **0.3%**.
(The 0.154 is measured here; the 52 ms is E11's figure, not re-measured.)

`evidence.product_tokens` is cached, so the pass is far cheaper than its
description suggests. Removing it would complicate `explain()` for nothing.
**Closed as not worth doing**, so nobody prices it again.

---

## What actually shipped

**The structural paraphraser** (`--paraphraser structural`), ported from
`phase/7-paraphrase-robustness`, with its 23 tests. It is the inverse mechanism
to the synonym one — it reframes and reorders while preserving vocabulary,
where the other substitutes vocabulary while preserving structure:

| mode | content-token retention | whole value survives | token-preserving |
|---|---|---|---|
| structural | 0.9971 | 0.0000 | true |
| synonym | 0.3576 | 0.0887 | false (260 values replaced) |

Its `--level 0` reproduces `0.945297`, so it carries the same control the
synonym probe does.

**This is the phase's real deliverable.** It is the probe that caught the
overfitting in §2 — the two pre-existing probes both said graded ownership
helped on the fallback path, and both were wrong, because `--paraphrase-category`
reuses the synonym lexicon for the disclosures and the held-out lexicon
saturates at HitRate 0.99. A third probe with a genuinely different mechanism
changed a ship decision.

**A weight knob**, `SHOPPING_AGENT_SEMANTIC_WEIGHT`, default unchanged.

---

## Rules of the house — one amended, one added

**Amended.** "Confirm a gain on a second, structurally different paraphraser"
is now **third**, and *structurally different* has to be enforced on the
mechanism, not just the word list. The held-out lexicon is disjoint by
assertion but attacks the same axis, and it agreed with the tuned probe on a
change that a genuinely different mechanism rejected.

**Added.** *A feature that gains on the probes sharing its mechanism and loses
on the one that does not has been tuned, not improved.* The sign pattern across
probes is the diagnostic, not the mean.

---

## Open items

1. **The branch divergence is undecided.** `phase/7-paraphrase-robustness`
   `e4e6f73` (pushed) and `phase/8-semantic-evidence` are two unmerged attacks
   on the same problem with two documents both called E11. This phase measured
   phase/7's ranking mechanism and rejects it; its *harness* work (the
   structural paraphraser) is ported and kept. The rest of that branch —
   `PARAPHRASE_WEIGHTS`, the regime audit scripts — is unreviewed here.
2. **The ranking gap is still open and still the only gap.** E10's perfect
   reranker oracle is 0.990300 with retrieval untouched, against 0.847762 at
   L2. Five levers this phase moved none of it. The remaining headroom needs a
   different *mechanism*, not a re-tuning of this one.
3. **Nothing is pushed and nothing is merged to `main`.**
4. Item 8 (dropping the 73 MB MiniLM artifact for an off-by-default route) is
   untouched and still a standalone decision.

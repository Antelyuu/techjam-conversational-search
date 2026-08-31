# E10 - what the score costs if the customer stops quoting

Decision record for the paraphrase-robustness branch. One measurement
instrument, one finding that reordered the priorities, one adopted change, and
one earlier recommendation withdrawn on the strength of the new numbers.

This closes the open item README's "What is left" names explicitly:
*"generate a paraphrased held-out split and measure the quiet-failure ceiling
directly, instead of arguing for it from construction."*

```yaml
experiment_id: "E10"
phase: "P6 (paraphrase robustness)"
hypothesis: "Every evidence feature in the table is an exact-string test, and the public simulator quotes the target verbatim. If a private set paraphrases, the features are designed to fail quiet -- but nobody had ever measured how much of the score they are holding up, or checked that the failure really is quiet."
base_commit: "b7d553a"
dataset: "full public set (200 labeled sessions), replayed through a paraphrasing customer"
overall_metrics:
  submission_unchanged: {hit_rate_at_10: 1.000, mrr: 0.938657, mttc: 2.805, technical_score: 0.945497}
  paraphrased_before: {hit_rate_at_10: 0.485, mrr: 0.366373, mttc: 7.195, technical_score: 0.428512}
  paraphrased_after: {hit_rate_at_10: 0.805, mrr: 0.539716, mttc: 4.420, technical_score: 0.696015}
model_api: {model: "none", network_required: false, prompt_tokens: 0, completion_tokens: 0}
known_regressions: []
decision: "ship the category robustness change (+0.2675 paraphrased, 0.000000 on the public set); do not flip the dense default"
```

## Why this is worth measuring at all

`docs/competition_specification.md` reserves the organiser's right to add
paraphrasing and says only that scoring *mechanics* would be unaffected:

> "The simulator policy decides what information to reveal. **If natural-
> language paraphrasing is added by the organizer, it cannot decide
> correctness.** Hits are always exact code matches."

That is not a promise it will not happen. The private set is 800 sessions and
only the scenario mix is guaranteed to match the public split.

Meanwhile the ranking table is `slot_evidence` 16.0, `constraint_evidence`
12.0, `category_exact` 8.0, `phrase_evidence` 6.0 -- about 42 of ~48 total
weight -- and every one of them is an exact-string test over a non-stemming
tokenizer. The claim that this is safe because the features "fail quiet"
appears in five places across three modules (three feature comments in
`reranking.py`, plus `slots.py` and `starter/agent.py`). That argument had
never been tested.

## The instrument

`scripts/paraphrase_eval.py` swaps the simulator's two text-producing
functions and leaves everything else alone. The hidden card, the disclosure
bookkeeping, `classify_constraint` routing, the catalogue and the target are
untouched, so the task is identical and only the surface form of what the
customer says moves.

**Level 0 reproduces 0.945497 exactly, and that is what makes the rest
trustworthy.** If it ever stops doing so the harness is broken and every
number here is void.

The synonym lexicon is hand-built from the measured top content words across
all 800 public constraint strings, so it is not cherry-picked -- but it is
still one word list, and tuning against it would overfit to *it* rather than
to paraphrasing. Two limits are worth stating plainly: `--level 3` conflates
paraphrasing with information loss and is a floor rather than a clean
measurement, and `--paraphrase-category` alters only about 20% of categories,
so `--reword-category` (token-preserving word reordering) is the valid
category probe.

## Finding 1: the collapse, and what was actually causing it

The first pass measured a collapse from 0.945497 to 0.428512 and attributed
it to the evidence features. That attribution was wrong, and the reason is
instructive.

Levels 1 and up reworded the *opening line* as well as the disclosures --
"I want {cat}, though I am still browsing" rather than the simulator's own
"I'm looking for {cat}, but I'm still exploring" -- and `slots.stated_category`
anchored its regex on the literal phrase "I'm looking for". Counted rather
than inferred: **200 of 200 openers parse at L0, 0 of 200 at L2.** So every
paraphrased session had also been running with the E9 category retrieval
filter stood down and `category_exact` scoring 0.0 for everyone, on top of the
synonym substitution the number was meant to isolate.

`--keep-opener` separates the two. It paraphrases the disclosures and leaves
the opening line in the simulator's own wording, drawing from the rng on
exactly the branch that drew before, so the two runs differ only in phrasing.

| condition | category | disclosures | score | HitRate |
|---|---|---|---|---|
| L0 verbatim (control) | live | verbatim | 0.945497 | 1.000 |
| L2 `--keep-opener` | live | synonyms | 0.676930 | 0.780 |
| L2 (the original baseline) | dark | synonyms | 0.428512 | 0.485 |
| L2 `--keep-opener --reword-category` | dark | synonyms | 0.428618 | 0.485 |

The last row is the control that names the mechanism. Rewording the category
leaves the opener perfectly parseable and still lands within 0.0001 of the
unparseable case, so what costs the 0.248 is not the parse -- it is ending the
turn without a category string this catalogue reproduces exactly, by either
route.

That splits the 0.517 collapse roughly in half: **0.248 is the category signal
going dark, 0.269 is the paraphrased disclosures.**

It also makes the category signal's value strongly conditional. Losing it
costs 0.067 while the disclosures still match verbatim, and 0.248 once they do
not. It is load-bearing precisely when everything else has already failed,
which is the opposite of how a graceful-degradation layer is supposed to
behave.

## Finding 2: "fails quiet" was true of the features and false of the filter

The three evidence features do fail quiet as designed. When nothing matches
they score 0.0 for every candidate alike and the ordering falls through to
whatever is beneath them. The 0.269 they cost is the value they were adding,
not damage they did.

The category filter is different in kind, because it is a *filter*. It has
three documented stand-down paths -- no parse, no known category, empty
intersection -- and all three work. But standing down is not free: it returns
retrieval to catalogue-wide, and the 0.248 is the size of that.

The gap between those two behaviours is the whole finding. An exact test that
merely *scores* can be as exact as it likes. An exact test that *gates* pays
for its exactness whenever the input is spelled differently, and the two ways
this one could be spelled differently -- a different lead-in, a different word
order -- are both things a paraphraser does routinely and neither of which
changes what the customer means.

## Adopted: let the category signal survive a rewording (+0.2675)

Two changes, both structured as a second look after the exact test fails, so
the exact path is untouched:

* `slots.stated_category` accepts the lead-ins a rewording of "I'm looking
  for" actually produces. `show me` is deliberately excluded -- it is the one
  that routinely introduces something that is not a category, and the existing
  quiet-failure test guards that case.
* a stated category naming nothing exactly gets matched by canonical form
  (sorted lowercased tokens), which ignores word order and punctuation and
  nothing else. Resolution returns *every* catalogue category sharing that
  form rather than an arbitrary one, so the nine word-order pairs in the
  catalogue ("Shoes Clogs & Mules" / "Shoes Mules & Clogs") are searched
  together.

Verified to be a strict superset rather than a behaviour change, over the real
coarse categories of all 200 public targets, **before** the change was made:

| | |
|---|---|
| public-shaped openers where old and new disagree | **0 / 600** |
| reworded openers the new form recovers exactly | **800 / 800** |

Measured, dense off:

| condition | before | after | HitRate |
|---|---|---|---|
| `local_evaluator` (the submission) | 0.945497 | **0.945497** | 1.000 → 1.000 |
| L0 verbatim control | 0.945497 | 0.945497 | 1.000 → 1.000 |
| L0 + category reworded | 0.878729 | **0.945497** | 0.960 → 1.000 |
| L1 synonyms | 0.426729 | 0.694494 | 0.485 → 0.805 |
| L2 synonyms + destructured | 0.428512 | **0.696015** | 0.485 → 0.805 |
| L3 + 40% of words dropped | 0.389934 | 0.673673 | 0.450 → 0.780 |
| L2 `--keep-opener` | 0.676930 | 0.676930 | 0.780 → 0.780 |
| L2 `--keep-opener --reword-category` | 0.428618 | 0.676930 | 0.485 → 0.780 |

The two rows that matter most are the ones that do not move. The submission
score is identical to six decimals, and so is `--keep-opener`, which is the
run where the opener already parsed and the category already matched exactly.
The change is inert exactly where it should be inert. The category-rewording
penalty is now zero rather than 0.067.

### The held-out check

The gain above could in principle be an artifact of the harness rewording
categories in the one way the fix anticipates. `--paraphrase-category`
substitutes the category's *words* ("women" → "ladies"), which defeats
canonical matching by construction, and is therefore the honest adversarial
case. Measured on both sides of the change:

| condition | before | after | HitRate |
|---|---|---|---|
| L2 `--paraphrase-category` | 0.430451 | **0.627500** | 0.490 → 0.715 |

Still +0.197, because the opener-parse half of the fix is independent of the
category's vocabulary. The 0.0685 gap against the plain L2 row is the part
this change genuinely cannot reach, and it is the honest ceiling on the
approach: word order and lead-in are handled, vocabulary substitution is not.

## Withdrawn: the dense default stays off

The paraphrase harness initially made a real case for flipping the dense route
on -- measured, not speculative: 0.0012 cost on the benchmark against a
+0.0115 gain under paraphrase, break-even at roughly a 10% chance of the
organiser paraphrasing. Re-measured 2x2 at the ranking this experiment leaves
behind:

| | verbatim | paraphrased (L2) |
|---|---|---|
| dense off | **0.945497** | 0.696015 |
| dense on | 0.944254 | 0.697829 |

Dense still costs 0.0012 on the benchmark but now gains only **0.0018** under
paraphrase, against 0.0115 before. Most of what it was recovering was the
category signal, which no longer needs recovering. Break-even moves from about
a 10% chance of paraphrasing to about 40%, and the flip also costs the
headline HitRate 1.000 → 0.995. Not taken.

This is the project's rule "a rejected idea is only rejected at the
configuration you tested it on" applying to a *recommendation* rather than a
rejection, which is new. The dense route was re-argued into favour at one
configuration and back out of it at the next, one commit later, on the same
instrument -- which is a reason to trust the instrument and to distrust any
single reading taken from it.

## What is deliberately not in this change

Reranking is untouched. `category_exact` still compares the raw stated string,
so it scores 0.0 for everyone when the category was reworded. That is inert
rather than wrong: once the filter engages, every pooled candidate reproduces
the category and the feature orders nothing either way. Routing the resolved
category into it would only matter on the path where the filtered search
returns nothing, which is not worth putting a ranking change into this diff.

## What is left

The 0.269 that paraphrased disclosures cost is untouched, and the ranked
options for it are unchanged by this experiment:

1. **Stemming.** Worth less than it first appears. `tokenize` feeds only
   `constraint_evidence` (weight 12) and the BM25 query; `slot_evidence` (16),
   `category_exact` (8) and `phrase_evidence` (6) are whole-string comparisons
   that stemming cannot reach. Stemming the query alone would also *break*
   BM25 against the non-stemming `unicode61` index -- the FTS table would have
   to move to `porter unicode61` so both sides stem.
2. **Relax slot ownership** from whole-value identity to high-overlap subset.
   This is the discrimination that produced +0.0316, so the threshold needs a
   careful sweep.
3. **Re-weight on the paraphrase signal** rather than only widening the
   shortlist with it. `shortlist.py` already computes "disclosures on record,
   no candidate owns any" and validates it; feeding it into the reranker to
   shift weight off the exact features is the cheapest structural fix.
4. **Embedding-based constraint evidence.** The real fix and the most
   expensive; worth it only if 1-3 leave a gap.

And the limitation this experiment adds to that list: category *vocabulary*
substitution is still unhandled, which the held-out row prices at 0.0685.

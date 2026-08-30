# E9 - search the category, not the catalogue

Decision record for the P6 continuation. Two adopted changes, one idea each,
and a re-sweep that moved nothing.

```yaml
experiment_id: "E9"
phase: "P6 (continuation)"
hypothesis: "E8 used the exact stated category as a reranking feature and it was that phase's largest win. It is worth more as a retrieval filter, because the target is a member of the restricted set by construction rather than by measurement -- and separately, the shortlist policy's robustness clause was reading a certainty as if it were a signal."
base_commit: "6a2e5f6"
dataset: "full public set (200 labeled sessions)"
overall_metrics:
  p6_as_handed_off: {hit_rate_at_10: 0.995, mrr: 0.910671, mttc: 2.850, technical_score: 0.933701}
  plus_T6_disclosed: {hit_rate_at_10: 0.995, mrr: 0.942764, mttc: 2.905, technical_score: 0.942229}
  plus_T7_catfilter: {hit_rate_at_10: 1.000, mrr: 0.938657, mttc: 2.805, technical_score: 0.945497}
scenario_metrics:
  plus_T7_catfilter: {buying: 1.000, browsing: 1.000, intent_override: 1.000, boundary: 1.000}
model_api: {model: "none", network_required: false, prompt_tokens: 0, completion_tokens: 0}
known_regressions: []
decision: "ship both (+0.011796 over P6 as handed off; HitRate reaches 1.000)"
```

## P6-T6: zero owned disclosures is a signal only once something was said (+0.008528)

The shortlist policy widens on three conditions, and the third is its whole
robustness argument: if the customer paraphrases rather than quoting card
values, no candidate owns anything, `live_disclosures` stays 0, and the agent
stops withholding rather than narrow on a ranking it cannot read.

It tested `live_disclosures == 0` alone. That is not the paraphrase signal. It
is the paraphrase signal **only once the customer has actually said
something**, and a Browsing or Boundary opener says nothing at all -- "I'm
looking for {category}, but I'm still exploring" discloses no constraint, so
on turn 1 no candidate *can* own anything. The clause fired on a certainty and
then handed out a padded ten built on the category alone, which is precisely
the lottery ticket the module exists to refuse.

Counted rather than inferred, over 2000 public turns:

| | |
|---|---|
| clause fires on `live == 0` alone | 100 turns, 90 sessions |
| ...of those, on turn 1 or 2 of Browsing/Boundary | 100 turns (all of them) |
| clause fires with disclosures on record and no owner | **0 turns** |

So on this split every firing was spurious and the insurance never once did
its job. Requiring `disclosed > 0` drops all 100 and keeps the insurance for a
private split that paraphrases, where the two predicates part company.

Seven Browsing sessions were cashing in a turn-1 rank of 3, 4, 7, 7, 7, 8 or 9
that became rank 1 one or two turns later.

**0.933701 -> 0.942229.** HitRate 0.995 unchanged, MRR 0.910671 -> 0.942764,
MTTC 2.850 -> 2.905. Identical on this split to deleting the clause outright,
which is the honest way to describe it: the retained insurance cannot fire
here, by construction.

One latent bug fixed alongside. The reranker-failure path forced the full list
by setting `live_disclosures = 0`, which the new predicate would have read as
"nothing disclosed yet" and withheld on. An unmeasured turn is now told apart
from a measured zero by an explicit flag.

## P6-T7: retrieve inside the stated category (+0.003268, and the last miss)

E8 made `coarse_category(target.categories)` a reranking feature and it was
that phase's largest single win. It is worth more as a **filter**, and what
makes it safe as one is structural rather than empirical: the opening line
states that string verbatim, in every scenario, on turn 1, and the agent
reproduces the generator's function exactly -- so the target is a member of
the restricted set **by construction**. Requiring membership cannot cost a hit,
the same guarantee slot ownership has.

Verified over the whole catalogue rather than by sampling, because that is what
caught the trim-before-clip bug in the last reconstruction:

| check | result |
|---|---|
| `slots.coarse_category` vs the evaluator's, over 50,000 products | **0 disagree** |
| openers whose category did not extract exactly | **0 / 200** |
| targets not reproducing their own stated category | **0 / 200** |

The catalogue holds 1115 distinct coarse categories over 50,000 rows, and the
median target shares its own with just **184** of them (min 2, max 1354). So a
400-deep pool now covers the whole category for most sessions instead of 0.8%
of the catalogue, and BM25 ranks *within* the field the customer asked for.

Measured over all 561 turns actually played, the filter fires on **every one**
and never falls back; the filtered pool has median 179 candidates.

### What it fixed

`public_0144`, the single remaining miss and the only session the project had
left. An Intent Override whose card is entirely generic -- `polyester`,
`100% Polyester`, `Imported`, `Zipper closure` -- so every disclosure the
customer added flooded the query with terms tens of thousands of rows match.
The target sat at pool position 98 on turn 1 and fell out of the 400 entirely
from turn 2 onward. It was a **retrieval** failure, not a ranking one, which is
why five phases of ranking work had never touched it. Inside *Down Jackets &
Parkas* those same four constraints discriminate.

**0.942229 -> 0.945497. HitRate reaches 1.000**, and every scenario reaches
1.000. MTTC 2.905 -> 2.805.

MRR dips 0.942764 -> 0.938657. That is the expected shape rather than a
surprise: more sessions now hit earlier, and hitting earlier means hitting on
less disclosure.

A full public-set pass is also **35% faster (41 s -> 26 s)**, because BM25
scans a category rather than a catalogue.

### It fails quiet, three ways

Each is a test in `tests/test_phase6_category_filter.py`, because this is the
property the whole change rests on for an unseen split:

1. an opener this agent cannot parse yields `stated_category = None`;
2. a category that parses but no product reproduces is not applied;
3. a filtered search that returns nothing reruns unfiltered.

`SHOPPING_AGENT_CATFILTER=0` restores the previous behaviour and measures
**0.942229** exactly.

## Re-swept after T7, and nothing moved

The pool changed character completely, so by this project's oldest lesson
everything tuned against the old one was re-priced. **Every current value is
optimal or tied-optimal; nothing in the table moves the score by more than
0.0001.**

- **Pool depth, re-priced a fifth time, is now flat**: 200, 300, 400, 600, 800
  and 1000 all give 0.945497, identical to six decimals across a 5x range.
  The first flat depth sweep since E5 and for the opposite reason -- E5 was
  flat because a rescued deep candidate could not be told apart, this is flat
  because there is nothing left to rescue. Left at 400: no measurement
  separates the values, so the choice falls to headroom on a catalogue with
  larger categories.
- **`category_exact` is now completely inert** (0.0, 4.0, 8.0 and 16.0 all
  identical), because every pooled candidate reproduces the stated category.
  Kept at 8.0 as the graceful-degradation layer for its own filter: on the
  fallback path the pool is catalogue-wide again and this is the feature worth
  +0.0475.
- **`slot_evidence` is the only weight still doing real work**: 0.0 costs
  0.0236, and 16.0 remains the first point of its plateau.
- `constraint_evidence` 12.0, `phrase_evidence` 6.0, `lexical_rank` 1.0,
  `soft_preferences` 2.0, `hard_constraints` 1.0, `metadata` 0.25 all hold, and
  the adjustment half remains inert.
- Two 0.0001 blips (`lexical_rank=0`, `category=1`) are one session hitting one
  turn earlier. Not adopted; single-session artifacts are what this project has
  repeatedly declined to chase.

## Measured and rejected: keeping declines out of the query

`build_query_text` ends with the latest raw message, and `_absorb_answer`
already refuses to keep a decline as a disclosure -- so letting the same words
into BM25 looked like a plain inconsistency. Counted first: **80 of the 580
turns played are declines, and the query's term set differs on 78 of them**,
contributing "don", "have", "preference", "additional" and the name of the
attribute being declined.

Suppressing them measures **0.920891**, HitRate 0.995 -> 0.975. It costs four
sessions.

The mechanism, measured rather than guessed: the query is far thinner than it
looks -- **median 6 distinct terms**, with 2 turns collapsing to an empty query
and 8 to two terms or fewer. The decline text is acting as ballast, and BM25's
OR-semantics tolerate junk terms much better than they tolerate a starved
query. Guarding the suppression on "only once something has been disclosed"
scores **identically** to the unguarded version, which closes the idea rather
than rescuing it.

Reverted. Recorded because the inconsistency is real and will look like a bug
to the next reader.

## Method: the offline rank replay, kept this time

E8 built this with throwaway scripts and recommended recreating it. It is now
`scripts/replay_ranks.py` and `scripts/replay_score.py`.

One 40 s run records, for every session and every turn, the full ranked ten
plus the evidence counts the shortlist policy reads, with the evaluator's early
break removed -- legitimate because the simulator's replies depend only on
`ask_attribute` and never on `recommendations`. Every sizing policy is then
scored in milliseconds.

Validated twice this session before being trusted, and `replay_score` re-checks
it on every run: it reproduced the live 0.933701 and 0.894920 before T6, then
predicted T6's 0.942229 before the live evaluator measured 0.942229, then
reproduced 0.945497 after T7.

The same replay bounds what sizing can still buy: the oracle over shortlist
size, choosing per session with hindsight, was **0.946279** at the T6 ranking
against the 0.942229 actually achieved. Sizing is close to exhausted.

## Where the remaining headroom is

HitRate is **finished** -- 1.000, on every scenario, and it cannot go up.
What is left:

- **MRR 0.938657**, worth at most +0.0184 more, and the ranking's whole
  adjustment half is inert, so it would take a new discriminator rather than a
  retune.
- **MTTC 2.805.** Bounded below by structure as much as by ranking: an
  Intent Override cannot register a hit before its override turn, which is 3 or
  4 by `rng.choice`, so those 30 sessions have a floor near 3.6 and currently
  sit at 4.1 or better. The shortlist policy's deliberate delay caps the rest,
  and that trade has now been measured in the wrong direction twice.

`SHOPPING_AGENT_SHORTLIST=0` restores always-ten and measures **0.885293** at
this configuration, so the policy is now worth **+0.060204** -- much more than
the +0.0388 it was worth at E8's ranking, because the ranking underneath it
improved again. The caveat E8 recorded still stands unchanged and should still
be read before defending the submission: it is shaped by this metric's
break-on-first-hit rule.

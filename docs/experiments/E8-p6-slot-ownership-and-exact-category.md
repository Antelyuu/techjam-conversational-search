# E8 - what the customer *said*, matched exactly

Decision record for P6. Five adopted changes, four of them one idea: the simulator
builds its messages out of the target product's own structured fields, so the
sharpest question to ask of a candidate is not "does your text resemble this?"
but "**would you have produced this exact string?**"

```yaml
experiment_id: "E8"
phase: "P6"
hypothesis: "E7 concluded text similarity was exhausted and only a non-text discriminator could separate the surviving misses. That is right about the evidence features and wrong about text: the generator takes whole field values, so exact ownership of a disclosed string -- and of the stated category -- is a structural test with perfect recall on the target and far better precision than containment."
base_commit: "fd67c18"
dataset: "full public set (200 labeled sessions)"
overall_metrics:
  p5_shipped:        {hit_rate_at_10: 0.950, mrr: 0.659603, mttc: 3.575, technical_score: 0.821381}
  plus_slot:         {hit_rate_at_10: 0.975, mrr: 0.698018, mttc: 3.195, technical_score: 0.853005}
  plus_shortlist:    {hit_rate_at_10: 0.970, mrr: 0.833060, mttc: 3.940, technical_score: 0.876118}
  plus_questions:    {hit_rate_at_10: 0.965, mrr: 0.836770, mttc: 3.580, technical_score: 0.881931}
  plus_category:     {hit_rate_at_10: 0.990, mrr: 0.912421, mttc: 2.965, technical_score: 0.929426}
  plus_depth_400:    {hit_rate_at_10: 0.995, mrr: 0.910671, mttc: 2.850, technical_score: 0.933701}
scenario_metrics:
  p5_shipped:    {buying: 0.9500, browsing: 0.9375, intent_override: 0.9667, boundary: 1.000}
  plus_depth_400: {buying: 1.0000, browsing: 1.0000, intent_override: 0.9667, boundary: 1.000}
model_api: {model: "none", network_required: false, prompt_tokens: 0, completion_tokens: 0}
known_regressions: []
decision: "ship all five (+0.112320 over P5 as handed off)"
```

## Read this first if you are picking up from here

Four of the five adopted changes are ranking or dialogue improvements that
any metric would reward. **One is not**: the shortlist policy (P6-T2) is shaped by this
metric's break-on-first-hit rule and is worth nothing under a metric that
scored the best rank across all turns. It is switchable off in one step and
the reasoning is written out below. Read that section before defending the
submission to anyone.

## The premise, and why E7 missed it

E7's closing analysis was that the remaining misses were "pooled but buried
among catalogue near-duplicates whose text contains the same quoted
constraints", that target coverage on misses was already 0.95+, and that only
**non-text discriminators** -- price, store, category leaf -- or better
question selection remained.

That reading of the data was correct. The conclusion drawn from it was too
narrow. The near-duplicates *contain* the same constraints; they do not
**own** them.

The generator (`local_evaluator.intent_card`) walks the target's `features`
and `details` and takes **whole values** -- one entire list element, or one
entire `key: value` pair -- normalizes whitespace, trims edge punctuation and
clips at 180 characters. Every constraint the customer can ever disclose is
therefore an exact member of a small set the target owns. An impostor
carrying the same sentence inside a longer bullet is not a member of it.

Measured over the 50,000-row catalogue and the 200 public sessions:

| question | answer |
|---|---|
| target owns its own disclosable constraints | **800 / 800** |
| constraints owned by exactly one product in 50,000 | 193 / 800 |
| sessions where the sharpest constraint is unique | 105 / 200 |
| median products consistent with category + 2 constraints | **1** |
| sessions uniquely identified by category + all 4 | 169 / 200 |
| target survives the consistency filter | **200 / 200** |

The last row is the one that makes it usable: ownership has *perfect recall*
on the target by construction, so requiring it can never cost a hit.

## P6-T1: slot ownership (+0.031624)

`slot_evidence` scores the share of disclosures a candidate owns as exact
whole field values, weighted by how rare each disclosure is **inside the
current pool** -- plain IDF over the 250 candidates, needing no catalogue
index and self-calibrating against the competitors that actually exist.
Selectivity is essential rather than decorative: a material label ("cotton")
is owned by thousands and says nothing, while a sixteen-word care instruction
is often unique. `phrase_evidence` weights by token count instead, which is
why a generic long sentence outvoted a rare decisive one.

The guarantee that follows: **because the target owns all of its own
disclosures, any disclosure owned by exactly one pooled candidate is owned by
the target whenever the target is pooled at all.**

| weight | 0.0 | 4.0 | 8.0 | 16.0 | 24.0 | 32.0 | 48.0 |
|---|---|---|---|---|---|---|---|
| score | 0.821381 | 0.850805 | 0.852755 | **0.853005** | 0.853005 | 0.853005 | 0.853005 |
| HitRate | 0.950 | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 | 0.975 |
| MRR | 0.659603 | 0.690685 | 0.697185 | 0.698018 | 0.698018 | 0.698018 | 0.698018 |

Flat from 16; adopted at the first point of the plateau. Weight 0.0
reproduces the P5 baseline to six decimals, so the feature is the only
difference. The first change in the project's history to move HitRate, MRR
and MTTC together.

**One bug found building it, kept as a test.** The generator trims edge
punctuation *before* it clips, so a clipped value can end in a comma
("...but once you start wearing them,") while the customer's quote of it ends
in ",." -- 4 of the 200 public targets. Neither string is the other. Both
sides now canonicalize. Caught only because the reconstruction was checked
against the generator's own functions over all 50,000 rows rather than
against hand-written expectations; the check is now
`tests/test_phase6_slots.py`.

## P6-T3: the exact stated category (+0.047495), the largest single win

The opening line states `coarse_category(target.categories)` **verbatim**, in
every scenario, on turn 1. For Browsing it is the only thing the customer
ever says before answering a question.

The reranker already had a `category` feature and it measured worth exactly
0.0 (E4), correctly: it scores word overlap against a category *term* the
slot extractor recognized, and retrieval already applies that same boost when
building the pool, so re-applying it reorders nothing.

Exactness is a different question, and the pool has not already accounted for
it. **A median of only 38% of the 250-candidate pool reproduces the target's
exact category string** -- so agreement removes about three fifths of the
field, free, from turn 1, in all 200 sessions.

| weight | 0.0 | 1.0 | 2.0 | 4.0 | 8.0 | 16.0 |
|---|---|---|---|---|---|---|
| score | 0.881931 | 0.926451 | 0.926651 | 0.929026 | **0.929426** | 0.929426 |
| HitRate | 0.965 | 0.990 | 0.990 | 0.990 | 0.990 | 0.990 |
| MRR | 0.836770 | 0.906171 | 0.906171 | 0.912421 | 0.912421 | 0.912421 |
| MTTC | 3.580 | 3.020 | 3.010 | 2.985 | 2.965 | 2.965 |

Misses fall from 7 to 2. Browsing, which had the lowest HitRate of any
scenario for the whole project, reaches **1.000**.

Both features fail quiet, which is the property that makes their weights safe
to set this high: if the customer paraphrases rather than quotes, or names a
category this catalogue cannot reproduce, every candidate scores 0.0 and the
table beneath decides. The failure mode is silence, not noise.

## P6-T4: ask the open question first (+0.005013)

Measured first, then explained: **72% of the questions asked (1304 of 1800)
returned nothing**, and 44 of the 200 sessions were not finishing their card
until turn 8.

The cause is an ordering artifact. `customer_reply` answers `other` with
*any* undisclosed constraint -- it skips the classify step for it entirely --
so its yield is by construction at least that of every specific attribute, at
every point. P4 nonetheless reached it only after all six others were spent,
on E3's finding that asking it *instead of* a real policy was worse.

**That finding had expired**, the way pool depth expired twice before it.
E3's number was mostly the cost of running *out* of questions early, which
does not apply to asking it first with all six still queued behind it.

| position | last (P4) | 4th | 3rd | 2nd | 1st |
|---|---|---|---|---|---|
| score | 0.876918 | 0.879568 | 0.879681 | 0.879881 | **0.881681** |
| MTTC | 3.900 | 3.830 | 3.680 | 3.670 | 3.655 |

Monotone in MTTC the whole way up -- the mechanism showing itself. Adopted as
prior 1.0 (a statement about what the simulator does, not a tuned value) plus
exempting it from the disagreement modifier, which otherwise docks a question
to 0.6 of its prior on a pool measurement that cannot be taken for a
wildcard. That principled pair reaches 0.881931, matching the best value the
tuned sweep found.

### Measured and NOT adopted: dropping the near-dead attributes

`use_case` wastes 196 of 199 asks, `size` 191 of 200, `style` 183 of 200.
Removing them looks obviously right and is obviously wrong:

| dropped | none | use_case | +size | +style |
|---|---|---|---|---|
| score | **0.881931** | 0.865316 | 0.865151 | 0.858693 |

Asking is free -- the evaluator scores recommendations first and handles
`ask_attribute` separately -- so a question that yields nothing costs nothing,
while its absence makes the agent run out of questions and go silent sooner.

## P6-T2: the shortlist policy (+0.023113) -- read the caveat

Every earlier phase returned ten recommendations every turn. The evaluator
ends a session the moment the target appears anywhere in that list and
**freezes the rank it appeared at**, so a turn-1 list padded to ten is a
lottery ticket: a target at rank 7 among them ends the session at RR 0.14,
and the further disclosure that would have lifted it to rank 1 never happens.

That was measurably the Buying scenario's problem. Buying opens by disclosing
`hard_constraints[0]`, which the generator fills with the target's *material
label* -- a string thousands of rows own. Buying therefore had the **best**
HitRate of any scenario (0.9875) and the **worst** MRR (0.6516).

The agent now returns its single best candidate while still narrowing, and
the full ten once it has something to stand behind. Widening is measured, not
assumed, on any of three conditions: the field is narrowed to one candidate;
the high-yield questions are spent (turn 5); or **there is no slot evidence
at all**.

| policy | score | HitRate | MRR |
|---|---|---|---|
| return ten always | 0.853005 | 0.975 | 0.698018 |
| withhold blindly before turn 5 | 0.873476 | 0.965 | 0.836254 |
| + widen once the field is one candidate | 0.873476 | 0.965 | 0.836254 |
| + never withhold without evidence (**adopted**) | **0.876118** | 0.970 | 0.833060 |

The third condition is the robustness argument, not a tie-breaker.
Withholding is only ever justified by evidence that narrowing is under way,
so a private set that paraphrases leaves `live_disclosures` at 0 every turn
and the agent returns ten exactly as before -- it switches itself off on a
distribution it cannot read. It is also worth +0.0026 *and* returns a hit the
blind schedule loses, the unusual case of the safer option also being better.

The confidence signal is unusually clean: **where the disclosures narrow the
field to exactly one candidate, that candidate is the target at rank 1 in 97
of the 99 public sessions where it fires.**

### The caveat, stated plainly

This is shaped by the metric. Under a rule that scored the best rank across
all turns rather than the first, it would be worth nothing, and it costs real
HitRate (one session). It is defensible as product behaviour -- a
precision-first agent that never returns a candidate it would not defend, and
asks a question instead -- and nothing in the rules requires returning ten
("only the first 10 valid unique `parent_asin` values are scored" is a
maximum). But it is the one change here that a reviewer could reasonably call
metric-shaped, so:

**`SHOPPING_AGENT_SHORTLIST=0` restores always-ten**, and at the final
configuration that measures **0.894920** (HitRate 0.995, MRR 0.749732, MTTC
2.375). So the policy is worth +0.038781 here -- more than the +0.023 it was
worth when introduced, because the ranking underneath it got better.

One thing did change for the better: at the final configuration it **no longer
costs a hit**. HitRate is 0.995 with the policy on and 0.995 with it off; the
session the blind schedule lost is now found either way. The trade is purely
MRR against MTTC.

If the team prefers to submit without it, 0.894920 is the number to expect and
the other four changes are unaffected.

## Method note: the offline replay

The trade above was found by recording, for every session, the target's rank
in the full 250-deep ranking at **every** turn, with the evaluator's early
break removed. That is exact rather than approximate, because the simulator's
replies depend only on `ask_attribute` and never on `recommendations` -- so
removing the break leaves the conversation trajectory identical.

Validated before use: replaying "always show ten" over the recorded trace
reproduced the live evaluator's composite to six decimals with **zero
per-session mismatches** on 200 sessions. Every shortlist policy in the table
above was then scored offline in milliseconds instead of a 30-second run, and
the adopted policy's predicted 0.876118 matched the live evaluator's
0.876118. The recipe is cheap to recreate and worth recreating for any
question about *when* to return something.

The same replay bounds the idea: the best score reachable by shortlist sizing
alone, with an oracle choosing the turn per session, was 0.891683 at that
ranking. Sizing is not where much more is left.

## Also measured, not adopted

- **`phrase_evidence`** measured +0.001125 better at 0.0 once slot ownership
  subsumed it, and was kept at 6.0 anyway as the graceful-degradation layer:
  paying 0.0011 certain to insure the 0.032 E7 measured it to be worth, in
  the world where the private customer paraphrases and slot ownership goes
  silent, is worth it at any plausible odds. Re-measured at the *final*
  configuration the 0.0011 is gone as well -- 0.0, 3.0, 6.0 and 12.0 all give
  0.929426, identical to six decimals -- because `category_exact` now
  separates the near-duplicates the two features used to disagree about. The
  insurance is free.
- **The framing-aware lead-in stripper is composite-neutral** (0.876918 with
  and without, given the opener fix; 0.875818 on its own). Kept on
  correctness, not score: dropping everything up to the first colon mangled
  the product labels the customer quotes constantly ("Department: womens" ->
  "womens", which no product owns), damaging 7 of the 30 override openers.
  Requiring the clause to end in a copula damages 12 of 615,776 catalogue
  values (0.0019%) against the old rule's damaging any value with a colon in
  its first 120 characters.

- **`constraint_evidence`** re-swept and holds at 12.0, inside a flat
  plateau spanning 4-16.

## P6-T5: pool depth, re-priced a fourth time (+0.004275)

E5 measured depth flat and rejected it. E6's evidence floor re-priced it to a
peak at 100. E7's phrase feature re-priced it to 250. P6's two exact features
re-price it again, to 400:

| depth | 100 | 150 | 250 | 300 | 350 | 400 | 500 | 800 | 1200 |
|---|---|---|---|---|---|---|---|---|---|
| score | 0.897176 | 0.910851 | 0.929426 | 0.930551 | 0.933601 | **0.933701** | 0.932851 | 0.932439 | 0.932503 |
| HitRate | 0.955 | 0.970 | 0.990 | 0.990 | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 |

The mechanism is the same one every time, and by now it should be the
project's default expectation rather than a surprise: **a depth ablation is
only as durable as the ranking features it was measured under.** A deeper pool
only pays when a rescued candidate can be told apart from the impostors that
come with it, and an exact test tells them apart outright.

350 and 400 tie exactly on HitRate and MRR and differ by 0.005 of a turn --
one session hitting one turn earlier, a single-session artifact rather than a
robust margin. 400 is adopted as the measured best; 350 costs 0.0001 and runs
11% faster, so it is the point to move to if latency ever binds.

Everything else was re-swept at depth 400 and holds: EXPAND_TURN (4 and 5 now
tie exactly at 0.933701; 5 keeps the higher MRR), NARROWING_SIZE 1,
slot_evidence 16.0, category_exact 8.0, and the whole adjustment half of the
table -- `lexical_rank`, `soft_preferences`, `hard_constraints` -- which is now
completely inert, identical to six decimals across its whole swept range. The
two exact features decide the order outright and everything else breaks ties
that no longer occur. Those weights are kept where they are precisely because
they are what the ranking falls back to if the exact tests ever go silent.

## Cost

Measured on the full public set, single process, no network, no model:

| | P5 | P6 |
|---|---|---|
| startup (index + card values) | 1.34 s | 4.0 s |
| per-turn latency, median | 38 ms | 61 ms |
| per-turn latency, p95 | 72 ms | 111 ms |
| peak RSS | 0.75 GB | 0.95 GB |
| tokens / network | none | none |

The extra startup and roughly 200 MB of RSS are the per-product `card_values`
sets (615,776 strings across 50,000 products) plus the deeper pool. Latency
roughly doubled, mostly from depth 250 -> 400. All well inside anything the
organizer is likely to impose, and depth 350 gives back 11% of it for 0.0001
if that assumption is wrong.

## Where the remaining headroom is

At 0.933701 with **1 miss**, the arithmetic is tight: HitRate is worth at most
+0.0025 more, MRR among hits +0.027, and efficiency is capped by the shortlist
policy's deliberate delay -- turning the policy off buys 0.475 of a turn and
costs 0.161 of MRR, a trade already measured in the wrong direction (0.894920).

Also measured and found empty during P6, so that nobody re-derives them:

- The `user_profile` carries only generic tags ("fit", "comfort") derived from
  prior purchases and nothing about the target.
- The card's budget line remains structurally unreachable (0/200, E6).
- `target_category` is written into the card but never read by the simulator.
- `difficulty_bucket` and `category_bucket` exist on the sample but are never
  passed to `reset`, so the agent cannot see them.
- **Folding the stated category into the shortlist's consistency test is a
  no-op.** It is arguably the more correct definition -- the category *is*
  something the customer said -- but the two definitions disagree on **0 of
  591 turns**, so it was reverted rather than carried as a parameter and a
  branch that do nothing. By the time slot evidence is live, the candidates
  owning every live disclosure already agree on the category. Checked by
  counting the disagreements, not by comparing composites: equal scores are
  not evidence that a code path ran.
- **Blind shortlist truncation, before the ranking improved.** Measured at
  the P5 ranking, every fixed schedule lost (best 0.816712 against 0.821381)
  because HitRate fell 0.950 -> 0.920. The same idea only became worth
  +0.023 once slot ownership made the top candidate right often enough. An
  idea rejected on measurement is rejected *at that configuration*, which is
  this project's oldest lesson and cost it a phase to learn twice.

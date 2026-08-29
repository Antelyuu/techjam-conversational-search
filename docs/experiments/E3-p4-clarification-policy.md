# E3 - P4 clarification policy

Decision record for P4-T2 (clarification policy) and P4-T3 (candidate-diversity
analysis), and for the open team question the P3 handoff left behind: whether
to use the `"other"` wildcard.

Produced by `python3 -m scripts.clarification_ablation`, which evaluates every
configuration against one agent instance in a single run. The reranker is held
off throughout so this ablation moves one variable; P4-T1 is measured
separately in [E4](E4-p4-reranker-and-fusion.md).

## The problem this phase inherited

Before P4 the agent returned `ask_attribute: None` on every turn. The simulator
only discloses a hidden constraint when asked -- `evaluator/local_evaluator.py:170`
answers *"Those options are not quite right yet. Ask me about one specific
attribute."* whenever `ask_attribute` is absent. Buying discloses a constraint in
its opening message and so worked anyway; Browsing "begins vague" and therefore
never gained information across ten turns.

Recovering hit turns from MTTC at the end of P3 made the closed loop explicit:
every hit in Buying, Browsing and Boundary landed on **turn 1**, and turns 2-10
contributed nothing at all. Efficiency is 20% of the composite but merely
mirrored HitRate, because nothing ever happened after the first turn.

## What a question is actually worth

Measured by `python3 -m scripts.ask_value_analysis`, which materializes all 200
hidden intent cards with the evaluator's own functions, replays the opening
message, and counts the sessions still holding an undisclosed constraint of
each class. That is exactly the share of sessions where asking returns content
rather than *"I don't have an additional preference for X"*.

| attribute | sessions yielding content | share |
|---|---|---|
| `feature` | 192 / 200 | 96.0% |
| `material` | 145 / 200 | 72.5% |
| `color` | 51 / 200 | 25.5% |
| `style` | 17 / 200 | 8.5% |
| `size` | 9 / 200 | 4.5% |
| `use_case` | 4 / 200 | 2.0% |
| `brand`, `category`, `budget` | 0 / 200 | **0%** |

`feature` dominates because `classify_constraint` falls through to it; `material`
and `color` are over-represented because `intent_card` inserts them at positions
0 and 1 when the product corpus mentions them. Both facts come from evaluator
code rather than from this particular dataset, so the ordering should carry to
the private set.

**Three attributes are dead, not two.** The handoff identified `brand` and
`category`, which `classify_constraint` can never return. `budget` is dead for a
different reason: `intent_card` appends the budget line *last*, and every one of
the 200 cards was truncated to four constraints by its `[:2]`/`[2:4]` slices, so
the budget line never survived. All 800 constraint strings across the set
classify as one of the six live attributes.

Every session holds **3 or 4 undisclosed constraints** after the opening, and a
successful ask returns up to 2, so roughly two productive questions drain a card.

## A bug found on the way

Slot extraction matched bare single-letter sizes, and `\b` treats an apostrophe
as a word boundary -- so `"I'm"` matched `m`. Every evaluator session opens with
*"I'm looking for ..."*, so all 200 silently carried `size="m"`: a junk term in
every query since P1, and, once P4 began asking questions, a permanently blocked
`size` slot, because the policy will not ask about an attribute that already
looks fixed.

Fixed in `6141f7c`; sizes that are words on their own still match bare, while
`s`/`m`/`l` now need an explicit size cue. Worth **+0.011** on its own (the
`no_questions` row below is 0.162275 against P3's recorded 0.151089).

## Results

```yaml
experiment_id: "E3"
phase: "P4"
hypothesis: "Asking one question per turn converts turns 2-10 from dead weight into information gain, lifting HitRate and MTTC together; candidate disagreement improves the choice of question over a fixed prior; and the \"other\" wildcard is worth less than it appears."
base_commit: "d2bb995"
candidate_commit: "871913d"
dataset: "full public set (200 labeled sessions)"
reranker: "disabled throughout, so this ablation moves one variable"
overall_metrics:
  no_questions:       {hit_rate_at_10: 0.190, mrr: 0.105250, mttc: 9.215, efficiency: 0.1785, technical_score: 0.162275}
  prior_only:         {hit_rate_at_10: 0.590, mrr: 0.361371, mttc: 6.070, efficiency: 0.4930, technical_score: 0.502011}
  prior_disagreement: {hit_rate_at_10: 0.595, mrr: 0.359371, mttc: 6.035, efficiency: 0.4965, technical_score: 0.504611}
  soft_askable:       {hit_rate_at_10: 0.675, mrr: 0.422710, mttc: 5.410, efficiency: 0.5590, technical_score: 0.576113}
  soft_plus_wildcard: {hit_rate_at_10: 0.740, mrr: 0.474613, mttc: 5.250, efficiency: 0.5750, technical_score: 0.627384}
  wildcard_only:      {hit_rate_at_10: 0.735, mrr: 0.460571, mttc: 5.510, efficiency: 0.5490, technical_score: 0.615471}
scenario_metrics:
  no_questions:       {buying: 0.2875, browsing: 0.0625, intent_override: 0.2667, boundary: 0.200}
  prior_disagreement: {buying: 0.5125, browsing: 0.6500, intent_override: 0.6333, boundary: 0.700}
  soft_askable:       {buying: 0.6750, browsing: 0.6750, intent_override: 0.6667, boundary: 0.700}
  soft_plus_wildcard: {buying: 0.7000, browsing: 0.7375, intent_override: 0.8000, boundary: 0.900}
model_api: {model: "none", network_required: false, prompt_tokens: 0, completion_tokens: 0}
known_regressions: []
decision: "keep soft_plus_wildcard"
```

| config | TechnicalScore | vs previous row |
|---|---|---|
| no_questions | 0.162275 | -- |
| prior_only | 0.502011 | **+0.339736** |
| prior_disagreement | 0.504611 | +0.002600 |
| soft_askable | 0.576113 | **+0.071502** |
| soft_plus_wildcard | **0.627384** | +0.051271 |

The four configurations shared with an earlier run reproduced to six decimal
places, which is the cross-check this project adopted after a "model
comparison" turned out to be a lexical-only run.

### Asking at all is the phase

`no_questions -> prior_only` is **+0.34 composite**, by far the largest single
movement in the project so far. Browsing goes from 0.0625 to 0.6375 -- it was
never a retrieval problem, it was a dialogue problem, exactly as E1 predicted.
HitRate and MTTC move together because converting a miss (scored as turn 11)
into a turn-4 hit improves both at once.

### Candidate disagreement earns little (P4-T3)

`prior_only -> prior_disagreement` is **+0.0026**, about one session on HitRate.
Kept, but honestly: the measured prior does nearly all the work, and
disagreement is a tiebreaker rather than a driver. That is the expected result
in hindsight -- the prior predicts whether the *customer* can answer, while
disagreement only predicts whether the answer would reorder the *pool*, and it
is the first that gates whether a turn is productive at all.

Its coverage guard still earns its place on the T3 acceptance criterion:
attributes most candidates simply do not state score zero disagreement rather
than maximum, so the agent never chases a question the catalogue cannot use.

### Most of the wildcard's value was never about the wildcard

The `"other"` wildcard matches any undisclosed constraint, so it looked worth
**+0.111** over a policy that ran out of questions. It is not. Unblocking
attributes fixed only by a *soft* regex guess -- `"I'm looking for Athletic
Walking"` sets `style=athletic` and `use_case=walking`, which looked like
answered questions -- recovers **+0.0715** of that legitimately, with no
degenerate question at all. Boundary sessions had been going silent from turn 5
with five turns still on the clock.

The wildcard's true marginal worth is **+0.051**, and it is reached only after
all six specific attributes are exhausted.

## Decision

**Keep `soft_plus_wildcard`** (`allow_wildcard=True`, `block_soft_slots=False`).

The wildcard survives on a measured +0.051 and on shape: a policy that asks six
specific questions and only then asks "is there anything else that matters to
you?" is ordinary good dialogue design, not an exploit. What would have been
indefensible is reaching for it *instead of* a real policy -- which is what the
first measurement would have encouraged, and what the soft-slot fix exposed as
an illusion.

Flags remain so either half can be turned off: `SHOPPING_AGENT_WILDCARD=0`,
`SHOPPING_AGENT_BLOCK_SOFT=1`.

## Boundary and Intent Override

P4's exit criteria require these two to be analyzed explicitly rather than
absorbed into the aggregate.

### Boundary: 0.200 -> 0.900 HitRate, MTTC 9.0 -> 4.7

The scenario declines the first question it is ever asked -- `boundary_used`
gates a single free pass -- and the handoff treated that as an unavoidable
wasted turn. It is not, because the decline is *distinguishable from a real
answer*:

| reply | meaning |
|---|---|
| "I don't have **a** preference for X" | the free decline; X is still unanswered |
| "I don't have an **additional** preference for X" | X is genuinely empty |

`interpret_reply` separates them on that one word. A boundary decline puts the
attribute back on the list, so the highest-yield question is re-asked
immediately on the next turn rather than being burned. The trace shows exactly
this: `feature` asked at turn 1, declined, re-asked at turn 2, answered.

This is a general NLU distinction, not a simulator trick -- "I have no
preference" and "I have no *further* preference" genuinely differ -- and it
costs one regex group.

n=10 carries no statistical weight on its own; it is reported because the exit
criteria ask for it, and because the mechanism is verifiable in a trace rather
than only in an aggregate.

### Intent Override: 0.2667 -> 0.800 HitRate

This was the one scenario that already worked at P3, because it is the only one
where the customer volunteers information unprompted -- it was the sole source
of the non-turn-1 hits in the P3 analysis.

Two mechanics shape it. The evaluator gates the hit check behind
`override_applied`, so no hit can register before the override fires around turn
3. And on that turn the override message *replaces* the reply to our question,
so the question asked the turn before goes unanswered.

The policy treats that displaced question as unasked and re-asks it, the same
mechanism as the Boundary decline. The distinction is made on a reversal stated
up front, not on `detect_override_cue`, which is deliberately broad for intent
classification and fires on "instead" or "no longer" anywhere in the text --
words that turn up inside the raw product copy customers quote back as
disclosures. Using the broad cue here would un-ask questions that were answered
and let the policy repeat itself.

## Carried forward

- **`prior_disagreement` is nearly free but nearly weightless.** If P5 needs to
  cut complexity, this is the first thing to reconsider.
- **The prior is measured against the public set's evaluator, not its data.** It
  derives from `classify_constraint`'s fallthrough and `intent_card`'s
  construction, so it should carry to the private set -- but if the hidden
  evaluator differs, the ordering degrades gracefully rather than breaking: an
  attribute that returns nothing is marked rejected and never asked again.
- **Re-asking a productive attribute is untested.** The simulator returns the
  *next* two undisclosed constraints, so asking `feature` twice would yield
  again on cards holding three or four feature-class constraints. P4's rules
  forbid repeating an attribute, so this was not tried.

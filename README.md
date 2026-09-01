# Conversational Shopping Agent — *chud-pro-max-shopinator*

Our entry for the **TechJam Conversational E-Commerce Search Challenge**.

A customer arrives with a partly-formed idea of what they want. The agent has up to ten
turns to identify which of **50,000 catalogue products** they have in mind, asking one
clarifying question per turn and returning a ranked shortlist each time.

The agent is deterministic, runs entirely offline, and calls no language model. It
identifies the correct product in **every session of the public benchmark**, at a mean of
2.8 turns.

| metric | starter baseline | **this agent** |
|---|---|---|
| **TechnicalScore** | 0.106710 | **0.945297** |
| Hit Rate@10 | 0.125 | **1.000** |
| MRR | 0.068034 | **0.938657** |
| MTTC (mean turns to conversion) | 9.81 | **2.815** |

```bash
python3 -m evaluator.local_evaluator      # no network, no API key, no environment variables
```

Hit Rate@10 is 1.000 on each of the four scenario types individually — Buying, Browsing,
Intent Override and Boundary — not only in aggregate.

---

## Contents

- [The central insight](#the-central-insight)
- [How it works](#how-it-works)
- [Handling natural language](#handling-natural-language)
- [Results](#results)
- [Architecture](#architecture)
- [The model we built and then removed](#the-model-we-built-and-then-removed)
- [Cost, latency and resource use](#cost-latency-and-resource-use)
- [Running it](#running-it)
- [Limitations](#limitations)
- [How decisions were made](#how-decisions-were-made)
- [Team](#team)

---

## The central insight

The useful question on this task turns out to be a **structural** one rather than a
semantic one.

Conventional search asks *which product most resembles what the customer described*. On a
catalogue of 50,000 clothing items that question is close to useless: thousands of
near-identical listings share the same vocabulary, and the copy of one seller's running
shoe resembles the copy of another's almost perfectly.

The benchmark's hidden requirement list, however, is not free text. The evaluator builds
it by walking the target product's own `features` and `details` fields and taking **whole
values** — one complete list element, or one complete `key: value` pair, normalised,
trimmed and clipped at 180 characters. Every requirement the customer can ever state is
therefore an exact member of a small set that the target product owns.

That changes the question worth asking:

> Not *"does this product resemble what the customer said?"*
> but **"would this product have produced that exact string?"**

Almost no product can. Measured across all 50,000 catalogue entries, the target owns
**800 of its own 800** disclosable requirements — by construction rather than luck, so
demanding an exact match can never cost a hit — and **193** of those strings are owned by
exactly one product in the entire catalogue. Combined with the category the customer names
in their opening line, two disclosed requirements are usually enough to identify a single
item.

Reconstructing that ownership set for every product, and scoring candidates by whether
they own what the customer said, is where most of our score comes from.

---

## How it works

Five mechanisms, in the order they run on each turn. The first four follow from the
ownership insight above; the fifth exists because that insight is a property of *this*
benchmark, and the fifth is what happens when it does not hold.

**1 · Search only the relevant category** — `shopping_agent/slots.py`
The opening message always names a coarse category, and the target reproduces that string
by definition. Restricting retrieval to it reduces the search space from 50,000 products
to a median of **181**. Verified safe before adoption: zero disagreements across all
50,000 products, and zero extraction failures across the 200 sessions.

**2 · Ask the question with the best expected yield** — `shopping_agent/clarification.py`
One question per turn, chosen by how likely the customer is to be able to answer it,
modulated by how much the current candidate pool disagrees about that attribute. The
policy also knows which attributes are unanswerable by construction and never spends a
turn on them.

**3 · Score ownership, not similarity** — `shopping_agent/reranking.py`
The central insight, as the dominant term in a deterministic eleven-feature scorer with no
learned parameters. Every ranking decision can be printed as a feature-by-feature
breakdown showing exactly why one product outranked another.

**4 · Commit only when the evidence supports it** — `shopping_agent/shortlist.py`
The evaluator freezes the target's rank the first time it appears in a returned list, so
showing ten products early is a gamble: a target sitting at rank 7 locks in a reciprocal
rank of 0.14, and the later turns that would have lifted it to rank 1 never happen. While
still narrowing, the agent returns its single best candidate alongside its question; it
returns the full ten only once the evidence justifies it.

**5 · Fall back to meaning** — `shopping_agent/semantic_evidence.py`
When the customer paraphrases rather than quotes, mechanisms 3 and 4 go quiet by design. A
small embedding model, run locally, then scores meaning instead. This is the subject of
the next section.

---

## Handling natural language

**This is the part that matters most for real deployment, and it is invisible on the
public leaderboard.**

The benchmark's simulated customer answers questions by quoting sentences verbatim out of
the target product's own description. That is what makes the ownership insight so
powerful — and it is not how people talk. A real shopper does not say
*"Material: 100% Cotton; Machine Wash Cold with like colors."* They say *"something in
cotton, and it needs to be machine washable."*

An agent that only works when handed the answer key is not a shopping assistant; it is a
lookup table. So we measured what happens when the customer stops quoting, and built for
it.

### Talking to it normally

Below is a real transcript. The customer's messages are ordinary English, written by
hand — no requirement is quoted from the product's description, and the agent sees
nothing else. Reproduce it with:

```bash
python3 -m scripts.natural_session
```

```text
target    B08G4WVYLJ  Dacomfy Mens Slip On Shoes, Men's Loafers & Slip-ons
category  Shoes Loafers & Slip-Ons

--- turn 1 ------------------------------------------------------------
customer  Hi, I'm looking for some loafers or slip-ons - shoes I can just
          step into.
agent     Here are the closest matches based on category=shoes.
          Is there anything else that matters to you?

--- turn 2 ------------------------------------------------------------
customer  Brown leather, ideally. Nothing that looks cheap.
agent     Here are the closest matches based on category=shoes,
          color=brown, material=leather.
          Is there anything else that matters to you?

--- turn 3 ------------------------------------------------------------
customer  They should have a microfibre leather upper and a soft rubber
          sole - I want something I can wear outdoors in any season.
agent     Here are the closest matches based on category=shoes,
          color=brown, material=leather.
          >>> target at rank 1
```

The agent reads a category out of a conversational greeting, accumulates `color=brown`
and `material=leather` from a fragment with no grammar to speak of, and converges on one
product out of 50,000 once the customer describes what makes it distinctive — in their own
words, matched by meaning rather than by string equality.

**One conversation is an illustration, not evidence.** It shows what using the agent is
like; it says nothing about the distribution. The numbers below come from replaying all
200 benchmark sessions through an automated paraphrasing customer, which is the claim you
should actually weigh.

### Opening the conversation

The same concern applies to the customer's *first* message. The simulator always opens
with one of a handful of phrasings, so we originally recognised only those — and a real
customer who asks a question instead got no category at all, which widens retrieval from a
median 181 products to all 50,000.

The agent now identifies the category by looking for it, rather than by looking for the
phrasing around it:

| opener | category recognised |
|---|---|
| `I'm looking for Shoes Loafers & Slip-Ons.` | ✅ |
| `Hey, do you have any Shoes Loafers & Slip-Ons?` | ✅ |
| `Could you show me women dresses please?` | ✅ |
| `hiya, after some accessories belts` | ✅ |
| `I need a gift for my wife` | — *(names no category)* |

It returns nothing rather than a guess when no category is named, because the category is
a hard restriction on retrieval: a wrong one loses the session, a missing one only widens
the search. Full account in
[`docs/experiments/E15`](docs/experiments/E15-p8-conversational-openers.md).

### What it is worth

Replaying all 200 public sessions through a paraphrasing customer:

| | before | **after** |
|---|---|---|
| paraphrased TechnicalScore | 0.696015 | **0.875880** |
| paraphrased Hit Rate@10 | 0.805 | **0.965** |
| **public TechnicalScore** | 0.945497 | **0.945297** |

We spent 0.0002 of a number the leaderboard sees to gain 0.18 of one that a private set
might. Three structurally different paraphrase generators were used to confirm this,
including two built specifically to be held out from tuning: one substitutes vocabulary
while preserving sentence structure, another reorders and reframes while preserving
vocabulary. A gain that only appeared under the generator we tuned against would not have
shipped, and twice a promising change was rejected for exactly that reason.

**The feature is provably inert on the public benchmark.** `SHOPPING_AGENT_SEMANTIC=0`
produces an identical 0.945297 to six decimal places, so its cost is a dependency and some
latency — never a point of score.

---

## Results

Full public set, 200 sessions, no environment variables set.

| | Hit Rate@10 | MRR | MTTC |
|---|---|---|---|
| **overall** (200) | **1.000** | 0.938657 | 2.815 |
| buying (80) | 1.000 | 0.973036 | 2.288 |
| browsing (80) | 1.000 | 0.908904 | 2.838 |
| intent_override (30) | 1.000 | 0.928095 | 3.867 |
| boundary (10) | 1.000 | 0.933333 | 3.700 |

**Hit Rate@10** is the fraction of sessions finding the target within ten turns. **MRR** is
the mean reciprocal rank of the target, a miss contributing zero. **MTTC** is the mean
first-hit turn, a miss assigned turn 11.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency     = clip((11 − MTTC) / 10, 0, 1)
               = 0.50 × 1.000 + 0.30 × 0.938657 + 0.20 × 0.8185  =  0.945297
```

---

## Architecture

```
customer message
      │
      ▼
ConversationOrchestrator          shopping_agent/orchestrator.py
   parse the opener · absorb the answer to our last question ·
   detect intent overrides · accumulate disclosed constraints across turns
      │
      ▼
retrieve()                        shopping_agent/retrieval.py
   BM25 over SQLite FTS5, restricted to the stated coarse category,
   400-deep candidate pool (plus an optional dense route, off by default)
      │
      ▼
rerank()                          shopping_agent/reranking.py
   deterministic 11-feature scorer led by slot ownership and phrase
   containment, with semantic evidence taking over as they go quiet.
   No learned parameters.
      │
      ▼
shortlist_size()                  shopping_agent/shortlist.py
   return only as many results as the evidence supports
      │
      ▼
choose_attribute()                shopping_agent/clarification.py
   pick the question with the highest expected yield
```

`starter/agent.py` exports the required `Agent` interface unchanged — `reset(session_id,
user_profile)` and `respond(session_id, user_message, turn, top_k)`, returning `message`,
`ask_attribute`, `recommendations` and `usage`, per
[`docs/agent_api_contract.json`](docs/agent_api_contract.json).

### Failure handling

Nothing that can go wrong is allowed to cost a session, and no degradation is silent:

- a reranker exception falls back to the fused retrieval order rather than raising into
  `respond()`;
- a clarification failure asks nothing rather than losing the turn;
- an empty filtered search reruns unfiltered;
- the dense route, when enabled but unavailable, serves BM25 results instead;
- every degradation prints its reason once to stderr, so a degraded run is visible rather
  than merely scoring lower.

**240 unit tests** cover this, standard library only, in about three seconds — including
the semantic feature's own degradation paths, which run against a synthetic artifact so
the suite never loads a model.

---

## The model we built and then removed

Phase 3 built a dense semantic retrieval route in full: MiniLM embeddings over all 50,000
products, benchmarked against `bge-small-en-v1.5`, fused with BM25 by both reciprocal-rank
and weighted blending. It was worth **+0.0355** at the time. By Phase 5 it was a **loss of
0.0509**, and we removed it.

| configuration | before clarification | after clarification |
|---|---|---|
| lexical only | 0.115573 | **0.687598** |
| dense + RRF | 0.145170 | 0.636669 |
| dense + weighted | **0.151089** | 0.636663 |

What changed in between was clarification. Once the customer answers by quoting sentences
out of the target's own text, and those quotes accumulate across turns, the two routes stop
being comparable:

| turn | median query words | lexical recall | dense recall |
|---|---|---|---|
| 1 | 12 | 0.3800 | 0.3200 |
| 2 | 15 | 0.4200 | 0.2200 |
| 3 | 24 | 0.7100 | 0.2550 |
| 10 | 36 | 0.7400 | 0.3600 |

At turn 1 the routes are even. From turn 2 they diverge: BM25 sharpens as rare terms
accumulate, whereas one fixed-width sentence embedding averages a growing paragraph toward
the corpus mean. The Phase 3 decision was not wrong — it expired.

We kept the route, its flag and the prebuilt artifact regardless, because the finding is
about this query distribution rather than about dense retrieval in general.
`SHOPPING_AGENT_DENSE=1` restores it. Full account in
[`docs/dense_route.md`](docs/dense_route.md).

---

## Cost, latency and resource use

**No API, no network at inference, no tokens, $0 per session.** Measured over the full
200-session public set on Apple Silicon, Python 3.13.6. Both columns score exactly
0.945297.

| | default | `SHOPPING_AGENT_SEMANTIC=0` |
|---|---|---|
| model | voyage-4-nano, Apache-2.0, run **locally** | none |
| network / API key | not required | not required |
| prompt + completion tokens | 0 | 0 |
| estimated cost per session | $0.00 | $0.00 |
| cold start (indexing 50,000 products) | 4.3 s | 4.1 s |
| per-turn latency | 52 ms median, 121 ms p95 | 35 ms median, 75 ms p95 |
| peak RSS (agent plus evaluator) | 1.60 GB | 0.78 GB |
| dependencies on the scored path | numpy, torch, sentence-transformers | standard library only |
| bundled artifact | 66 MB (int8, 256 dimensions) | none |

Two caveats stated rather than buried. The embedding model loads lazily, making one turn
per run approximately 3.2 s. And it downloads weights from the Hugging Face Hub on the
**first run only**; every run afterwards is fully offline. If the organiser enforces a
memory cap below ~2 GB, a per-turn timeout, or a fully sandboxed first run, set
`SHOPPING_AGENT_SEMANTIC=0` — **the benchmark score does not change.**

Being deterministic and offline means the agent holds up under the organiser's stated
right to score with network access disabled, and under CPU, memory and timeout
restrictions. There is no rate limit to hit, no key to rotate, and no per-session bill
that scales with traffic.

---

## Running it

```bash
python3 -m evaluator.local_evaluator                      # the official harness
python3 -m unittest discover -s tests -t .                # 240 tests
python3 -m scripts.demo_session --scenario buying         # one readable transcript
python3 -m scripts.demo_session --paraphrase 2            # …with a paraphrasing customer
python3 -m scripts.natural_session                        # a hand-written conversation
pip install -r requirements.txt                           # OPTIONAL — enables mechanism 5
```

Python 3.10+. The one prerequisite is the 60 MB catalogue, which is not stored in git —
**[`docs/SETUP.md`](docs/SETUP.md)** has the commands that fetch and verify it, plus every
ablation, flag and test command.

**Environment variables.** All optional; the submitted configuration sets none.

| variable | default | effect |
|---|---|---|
| `SHOPPING_AGENT_SEMANTIC` | on | `0` disables mechanism 5. Public score unchanged; paraphrased score falls to 0.705561. |
| `SHOPPING_AGENT_DENSE` | off | `1` restores the dense retrieval route. |
| `SHOPPING_AGENT_PARAPHRASE_SHORTLIST` | `0` | `10` restores the earlier commit policy. Never fires on a quoting customer. |
| `SHOPPING_AGENT_WILDCARD_CAP` | `3` | Consecutive open questions permitted under paraphrase. |
| `SHOPPING_AGENT_SEMANTIC_WEIGHT` | `192` | Measurement override for the semantic feature's weight. |

`docs/SETUP.md` documents the full set, including the ablation switches.

---

## Limitations

**The shortlist policy is shaped by this metric.** Because the evaluator freezes the
target's rank on first appearance, *when* to commit is worth real score. We tuned that
deliberately. A metric rewarding browsing breadth would want a different policy, and we
would not defend this one outside this scoring function.

**The agent handles natural phrasing, not open-ended dialogue.** Openers are matched
against the catalogue's own category vocabulary rather than a fixed list of phrasings, and
answers are absorbed however they are worded. But constraint extraction remains regex over
a known vocabulary, so an indirect request — *"I've a wedding next month and need
something that won't destroy my feet"* — names no category and states no constraint we can
parse. Closing that requires a language model in the loop, which would cost the offline,
deterministic, zero-token properties above.

**The customer profile is accepted and never used.** We found no measurable signal in it,
and chose not to add a feature we could not justify with a number.

**Ranking, not retrieval, is where the remaining headroom is.** With a perfect reranker and
today's retrieval the score would be 0.990300, and pool recall is already 1.000 — so none
of the remaining gap is retrieval's to win.

---

## How decisions were made

Fifteen written decision records in [`docs/experiments/`](docs/experiments/) hold every
measurement, the reasoning behind each choice, and the ideas we rejected. Four rules came
out of the process and did more work than any individual idea:

- **Never report a score without re-running the guardrails**, including a control proving
  the measurement harness itself still reproduces a known value. A sweep in which every
  arm returns the same number is a broken sweep, not a flat response — we caught one that
  way.
- **Confirm every gain on a second, structurally different test before shipping it.** Two
  changes that looked good on the generator we tuned against died here.
- **A rejected idea is only rejected at the configuration you tested it on.** Three
  decisions in this repository were reversed by later re-measurement, including one twice.
- **Surface the trade with numbers on both sides.** Every comparison table here has two
  columns for that reason, and the one probe that regressed is reported alongside the five
  that improved.

Every phase was merged only after a written code review; findings and their remediation
are recorded in the corresponding experiment file.

---

## Team

| | contribution |
|---|---|
| **Lin Minhong** ([@coffee-678](https://github.com/coffee-678)) | Conversation state and orchestration; constraint-aware lexical retrieval and price/category filtering; the dense route end to end, and the measurement that later removed it; disclosed-evidence scoring, phrase containment and pool-depth tuning; slot ownership, exact category matching, category-filtered retrieval, the shortlist policy, and the paraphrase-robustness programme. Records E1, E2, E5–E9, E11–E15. |
| **Lim Ray Hing** ([@rayhing1510](https://github.com/rayhing1510)) | Phase 4, which took the score from 0.151 to 0.637: the deterministic final scorer and its inspectable feature checklist; one clarification question per turn and the attribute-choice policy; the failure-path hardening that guarantees nothing escapes `respond()`; the reranker weight sweep harness and the retuning it drove. Records E3, E4. |
| **Joel Rhys Chee** ([@Antelyuu](https://github.com/Antelyuu)) | Phase 0 baseline and the phase execution skeleton that structured the project; agent behaviour documentation; Phase 1 test coverage; Phase 2 constraint-safety work and audit; Phase 3 validation and fallback hardening; offline artifact bundling. |
| **Lim Dao Hao** and **Edrich Denzil Lim Yu** | Independent component testing and score diagnosis across all phases. They identified which sessions were failing and isolated the responsible component — distinguishing a retrieval defect from a ranking one, which have nothing in common as fixes — then broke each phase's score into the parts that could still move. Their diagnosis set the agenda for every phase. |

Commits authored by `TechJam2026` are the organiser's original challenge scaffolding
rather than team contributions.

---

## Where everything is

| | |
|---|---|
| [`starter/agent.py`](starter/agent.py) | the official `Agent` entry point |
| [`docs/SETUP.md`](docs/SETUP.md) | install, reproduce, ablations, config flags, tests, module map |
| [`docs/experiments/`](docs/experiments/) | E1–E15: every measurement, including the rejected ideas |
| [`docs/dense_route.md`](docs/dense_route.md) | the dense route, and why we removed it |
| [`docs/competition_specification.md`](docs/competition_specification.md) | the rules and evaluation protocol |
| [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) | catalogue derived from Amazon Reviews 2023, McAuley Lab, UCSD |

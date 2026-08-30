# Conversational Shopping Agent

Our entry for the **TechJam Conversational E-Commerce Search Challenge**.

The agent receives an anonymised customer profile and a short opening message, then has
up to ten turns to surface a hidden target product from a frozen catalogue of 50,000
Amazon clothing items, asking one clarifying question per turn as it goes.

It is deterministic, runs on the Python standard library, works offline, and uses no
model and no API.

| metric | starter BM25 | this agent |
|---|---|---|
| **TechnicalScore** | 0.106710 | **0.945497** |
| Hit Rate@10 | 0.125 | **1.000** |
| MRR | 0.068034 | **0.938657** |
| MTTC (mean turns to convert) | 9.81 | **2.805** |

Hit Rate@10 is 1.000 on each of the four scenario types separately: Buying, Browsing,
Intent Override and Boundary.

```bash
python3 -m evaluator.local_evaluator     # no dependencies, no network, no env vars
```

---

## Contents

- [Project overview](#project-overview)
- [Setup and installation](#setup-and-installation)
- [Reproducing our results](#reproducing-our-results)
- [How it finds the target](#how-it-finds-the-target)
- [Architecture](#architecture)
- [Results](#results)
- [Cost, latency and resource use](#cost-latency-and-resource-use)
- [The dense semantic route](#the-dense-semantic-route)
- [Configuration flags](#configuration-flags)
- [Tests](#tests)
- [Repository layout](#repository-layout)
- [Limitations and what we would improve](#limitations-and-what-we-would-improve)
- [Team contributions](#team-contributions)
- [The challenge itself](#the-challenge-itself)
- [Data attribution](#data-attribution)

---

## Project overview

For each session the agent gets an anonymised `user_profile` and a scenario-dependent
first message. On every turn it can ask a clarification question (naming the field in
`ask_attribute`), return up to 10 ranked catalogue `parent_asin` values, or both. The
session ends when the target appears in the scored top 10, or after turn 10. Only exact
`parent_asin` equality counts as a hit.

Our central finding is that the useful question on this task is a structural one rather
than a semantic one.

The hidden intent card is not free text. The evaluator builds it by walking the target
product's own `features` and `details` and taking whole values: one complete list
element, or one complete `key: value` pair, normalised, trimmed and clipped at 180
characters. Every constraint the customer can ever disclose is therefore an exact member
of a small set that the target product owns.

So the sharpest test of a candidate is not whether its text resembles what the customer
said, since near-duplicate listings all pass that. It is whether the candidate would
have produced those exact strings. Everything below follows from asking it that way.

The agent uses no LLM and no external API. That was a measured outcome rather than
something we skipped: we built a full dense semantic route in Phase 3, measured it, and
[took it back out](#the-dense-semantic-route). The code and the flag are still here.

---

## Setup and installation

**Requirements:** Python 3.10 or later (developed and measured on 3.13.6). Nothing
third-party is needed for anything scored.

```bash
git clone https://github.com/Antelyuu/techjam-conversational-search.git
cd techjam-conversational-search
```

### Fetch the catalogue

`data/catalog.jsonl` is 60 MB and is not stored in git. It comes from the organiser's
participant-kit release:

```bash
BASE=https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit
curl -L -O $BASE/catalog.jsonl.gz
curl -L -O $BASE/SHA256SUMS
shasum -a 256 -c SHA256SUMS --ignore-missing    # Linux: sha256sum -c
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Expected row count: 50,000.

`data/public_set.jsonl`, the 200 labelled development sessions, is already in the repo.

### That is the whole install

There is nothing to `pip install` and nothing to build. The BM25 index is constructed in
memory from `data/catalog.jsonl` at startup, in about 4 seconds.

Only the optional dense route, off by default, needs dependencies. See
[below](#the-dense-semantic-route).

---

## Reproducing our results

```bash
python3 -m evaluator.local_evaluator
```

No environment variables, no network, no arguments. This is how the official harness
constructs the agent, as a plain `Agent(catalog_path)`. It writes per-session results
and aggregate metrics to `results.json` and prints the summary. A full run takes about
28 seconds.

Expected output:

```json
{
  "sample_count": 200,
  "hit_rate_at_10": 1.0,
  "mrr": 0.938657,
  "mttc": 2.805,
  "efficiency": 0.8195,
  "recommended_technical_score": 0.945497,
  "reported_token_usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

The run is deterministic. The agent has no randomness and the evaluator seeds its own,
so repeated runs produce byte-identical `results.json`.

### See one session end to end

`local_evaluator` plays 200 sessions and prints only the aggregate. To watch a single
conversation:

```bash
python3 -m scripts.demo_session                          # first Buying session
python3 -m scripts.demo_session --scenario browsing
python3 -m scripts.demo_session --scenario intent_override
python3 -m scripts.demo_session --sample-id public_0002
python3 -m scripts.demo_session --all-turns              # keep going past the hit
```

It prints the hidden target and intent card up front, which is everything the agent is
not allowed to see, then shows every turn: the customer's message, the agent's reply,
the attribute it asked about, and the ranked list with the target marked.

The loop mirrors `evaluator.local_evaluator.evaluate()` and imports every
evaluator-side function rather than reimplementing any of it, so a transcript cannot
drift from what the official scorer saw. We checked this on two sessions of each
scenario type: the turn and rank it reports match `results.json` in all 8 cases.

Here is `public_0001` abridged, a Buying session that shows most of the system in two
turns:

```text
--- turn 1 -------------------------------------------------------------------
customer  I'm looking for Jewelry Necklaces. A key requirement is: Material:alloy.
agent     Here are the closest matches based on category=jewelry. Is there
          anything else that matters to you?
ask       'other'
returns   1 recommendation
              1. B075KXNBNF  OIDEA Double Mens Punk Cool Black Military Dog Tag…

--- turn 2 -------------------------------------------------------------------
customer  For that, what matters is: Triple Moon Pentagram Symbol; The Triple
          Moon represents the Phases of the Moon which are linked to the three
          aspects of the Goddess and the phases of the Life of Women…
agent     Here are the closest matches based on category=jewelry. Which features
          matter most to you?
ask       'feature'
returns   10 recommendations
          >>  1. B09PYB7B6Z  QIAN0813 Celttic Knot Triple Moon Pentagram Pentacle…
              2. B075KXNBNF  OIDEA Double Mens Punk Cool Black Military Dog Tag…
              …
          ** target found at rank 1 — the evaluator stops here **

 HIT on turn 2 at rank 1   reciprocal rank 1.000
```

On turn 1 the shortlist policy declines to pad the list. `Material:alloy` is owned by
thousands of rows, so the agent returns the one candidate it can defend and asks the
open-ended question instead. On turn 2 it gets two quoted card values back, slot
ownership fires, and the target goes to rank 1.

### Reproducing the ablations

Each of these is a single command, and each reproduces a number quoted in this README:

```bash
SHOPPING_AGENT_SHORTLIST=0 python3 -m evaluator.local_evaluator   # 0.885293
SHOPPING_AGENT_CATFILTER=0 python3 -m evaluator.local_evaluator   # 0.942229
SHOPPING_AGENT_RERANK=0    python3 -m evaluator.local_evaluator
SHOPPING_AGENT_CLARIFY=0   python3 -m evaluator.local_evaluator
```

The harnesses that produced the sweeps in `docs/experiments/` live in `scripts/` and run
directly:

```bash
python3 -m scripts.fusion_ablation          # lexical-only vs RRF vs weighted fusion
python3 -m scripts.clarification_ablation   # value of each clarification policy
python3 -m scripts.rerank_weight_sweep      # per-feature reranker weight sweeps
python3 -m scripts.target_survival_audit    # does the target survive into the pool?
python3 -m scripts.replay_ranks             # record every turn's ranked ten once...
python3 -m scripts.replay_score             # ...then score any shortlist policy in ms
```

`replay_score` revalidates itself against the live evaluator on every run, so the
offline replay cannot silently drift from the thing it models.

---

## How it finds the target

Four mechanisms, all following from the ownership observation above.

### 1. Slot ownership

`shopping_agent/slots.py`

Each candidate is scored on the share of the customer's disclosures it owns as whole
values, weighted by how rare each disclosure is within the current candidate pool.
"Machine wash cold" buried inside a competitor's longer bullet point is not evidence.
The same string standing alone as one of its feature values is.

Measured over the 50,000-row catalogue and the 200 public sessions:

- the target owns all 800 of its disclosable constraints as exact values, so requiring
  the match can never cost us a hit;
- 193 of those 800 constraints are owned by exactly one product in the whole catalogue;
- given the opening category plus two disclosures, the median consistent set is already
  a single product; with four disclosures it is a single product for 169 of the 200
  sessions.

Selectivity is what makes this usable. A material label like "cotton" is owned by
thousands of rows and tells you nothing, while a sixteen-word care instruction is often
unique. Weighting by pool-local rarity needs no catalogue-wide index and calibrates
itself against whichever candidates are actually competing.

This is a ranking feature and never a filter. If a split paraphrases instead of quoting,
no candidate owns anything, every score comes out 0.0, and the ordering falls back to
the features underneath.

### 2. Retrieve inside the stated category

`starter/agent.py:_lexical_search`, `shopping_agent/slots.py:coarse_category`

The opening line names the target's coarse category word for word, in every scenario, on
turn 1. For Browsing sessions it is the only thing said before any question is answered.
We reproduce the generator's category function exactly, and verified it over the whole
catalogue rather than by sampling:

| check | result |
|---|---|
| our `coarse_category` vs the evaluator's, over 50,000 products | 0 disagree |
| openers whose category did not extract exactly | 0 / 200 |
| targets not reproducing their own stated category | 0 / 200 |

So the target is guaranteed to be inside the restricted set, the same guarantee slot
ownership has. The catalogue holds 1,115 coarse categories and the median target shares
its own with 184 products, so a 400-deep pool now covers an entire category instead of
0.8% of the catalogue, and BM25 ranks within the field the customer actually asked for.

This fixed our last remaining miss. It was an Intent Override session whose card was
entirely generic (`polyester`, `100% Polyester`, `Imported`, `Zipper closure`), so every
disclosure flooded the query with terms that tens of thousands of rows match. The target
was never reaching the pool. Because it was a retrieval failure rather than a ranking
one, five phases of ranking work had never touched it.

It also made a full evaluation pass 35% faster, from 41 s to 26 s, since BM25 now scans
one category rather than the whole catalogue.

The filter fails quietly in three ways, each covered by a test in
`tests/test_phase6_category_filter.py`: an opener the agent cannot parse yields no
category; a category that no product reproduces is not applied; and a filtered search
returning nothing reruns unfiltered.

### 3. Ask the open-ended question first

`shopping_agent/clarification.py`

The simulator answers `other` with any undisclosed constraint, so its yield is always at
least as high as any specific attribute's, at every point in the conversation. We had it
queued behind six narrower questions, and 44 of the 200 sessions were not draining their
card until turn 8 purely because of that ordering. Moving it up improves MTTC
monotonically the whole way, from 3.900 to 3.655.

The policy also knows which attributes are unreachable. `brand` and `category` can never
be returned by the evaluator's `classify_constraint()`, and `budget` is unreachable in
practice, so it never spends a turn on any of them.

### 4. Only return a shortlist the agent can defend

`shopping_agent/shortlist.py`

The evaluator ends a session as soon as the target appears anywhere in the returned
list, and freezes the rank it appeared at. That makes a turn-1 list padded out to ten a
gamble: a target at rank 7 ends the session at reciprocal rank 0.14, and the eight
further turns of disclosure that would have lifted it to rank 1 never happen.

This was measurably the Buying scenario's problem. Buying had the best HitRate of any
scenario at 0.9875 and the worst MRR at 0.6516, because it opens by disclosing a
material label that thousands of rows own.

So while the agent is still narrowing the field it returns its single best candidate
alongside its question, and the full ten once it has something to stand behind. The
widening conditions are measured rather than scheduled, and any of three will do it: the
field is narrowed to one candidate; the high-yield questions are used up; or the
customer has disclosed something and no candidate owns any of it. That last one is the
paraphrase signal, and it switches the policy off entirely on a distribution it cannot
read.

Please read [the caveat](#limitations-and-what-we-would-improve) before defending this
one.

### Everything is inspectable

Every feature's contribution to a candidate's score is recorded and printable via
`RerankedCandidate.explain()`, so we can read off why something ranked where it did.
This paid for itself early: our first reranker weighting scored worse than no reranking
at all, and the per-feature breakdown is how we found which feature was responsible.

The full reasoning and every measurement, including the ideas we measured and rejected,
are in `docs/experiments/` (E1 through E9).

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
   400-deep pool (plus an optional dense route, off by default)
      │
      ▼
rerank()                          shopping_agent/reranking.py
   deterministic 10-feature scorer, led by slot ownership,
   phrase containment and token coverage. No learned parameters.
      │
      ▼
shortlist_size()                  shopping_agent/shortlist.py
   return only as many results as the evidence supports
      │
      ▼
choose_attribute()                shopping_agent/clarification.py
   pick the question with the highest expected yield
```

| module | responsibility |
|---|---|
| `starter/agent.py` | the official `Agent` entry point: FTS5 index, retrieval wiring, response assembly |
| `shopping_agent/orchestrator.py` | multi-turn session state, opener parsing, answer absorption, override detection |
| `shopping_agent/state.py` | session store and cumulative query construction |
| `shopping_agent/intent.py` | slot candidate extraction, intent classification, override cues |
| `shopping_agent/catalog.py` | catalogue normalisation into `ProductRecord` |
| `shopping_agent/retrieval.py` | candidate pipeline, route fusion (RRF and weighted), pool sizing |
| `shopping_agent/filtering.py` | price and category evaluation, constraint matching |
| `shopping_agent/reranking.py` | the deterministic final scorer and its feature checklist |
| `shopping_agent/slots.py` | slot ownership, reproducing the card generator's normalisation exactly |
| `shopping_agent/evidence.py` | token-coverage and phrase-containment evidence |
| `shopping_agent/clarification.py` | which attribute to ask about, and when to stop |
| `shopping_agent/shortlist.py` | how many recommendations to actually return |
| `shopping_agent/dense_retrieval.py` | the optional dense route, off by default |
| `shopping_agent/contracts.py` | the dataclasses the modules pass between them |

### Failure handling

Nothing that can go wrong is allowed to cost a session, and no degradation is silent:

- a reranker exception falls back to the fused retrieval order instead of raising into
  `respond()`;
- a clarification failure asks nothing rather than losing the turn;
- an empty filtered search reruns unfiltered;
- the dense route, when enabled but unavailable, serves BM25 results instead;
- every degradation prints its reason once to stderr, so a degraded run is visible
  rather than just quietly scoring lower.

---

## Results

Full public set, 200 sessions, `python3 -m evaluator.local_evaluator` with no
environment variables.

| | Hit Rate@10 | MRR | MTTC |
|---|---|---|---|
| **overall** (200) | **1.000** | 0.938657 | 2.805 |
| buying (80) | 1.000 | 0.973036 | 2.288 |
| browsing (80) | 1.000 | 0.908904 | 2.813 |
| intent_override (30) | 1.000 | 0.928095 | 3.867 |
| boundary (10) | 1.000 | 0.933333 | 3.700 |

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
               = 0.50 × 1.000 + 0.30 × 0.938657 + 0.20 × 0.8195  =  0.945497
```

### How we got here

| milestone | TechnicalScore |
|---|---|
| starter BM25 | 0.106710 |
| P2 constraint-aware lexical retrieval | 0.115573 |
| P3 dense + weighted fusion | 0.151089 |
| P4 clarification + deterministic reranker | 0.636663 |
| P5 dense removed, disclosed-evidence scoring added | 0.706484 |
| P5 short-label evidence + retuned tie-breakers (E6) | 0.753328 |
| P5 phrase containment + widened pool (E7) | 0.821381 |
| P6 slot ownership (E8) | 0.853005 |
| P6 + confidence-sized shortlist (E8) | 0.876118 |
| P6 + open question asked first (E8) | 0.881931 |
| P6 + exact stated category (E8) | 0.929426 |
| P6 + pool depth re-priced to 400 (E8) | 0.933701 |
| P6 + disclosure-gated shortlist widening (E9) | 0.942229 |
| **P6 + category-filtered retrieval (E9)** | **0.945497** |

The single largest contributor is clarification. The simulator discloses a hidden
constraint only when asked, so before P4 three of the four scenarios landed every hit on
turn 1 and turns 2 through 10 contributed nothing at all.

---

## Cost, latency and resource use

Measured over the full 200-session public set (561 agent turns) on Apple Silicon,
Python 3.13.6.

| | |
|---|---|
| model / API | none |
| network required | no |
| prompt + completion tokens | 0 |
| estimated cost per session | $0.00 |
| cold start (indexing 50,000 products) | 4.2 s, once per process |
| per-turn latency | 37 ms median, 80 ms p95, 242 ms max |
| full 200-session evaluation | 28 s |
| peak RSS (whole process, agent plus evaluator) | 0.76 GB |
| dependencies on the scored path | none, standard library only |

Being deterministic and offline means the agent holds up under the organiser's stated
right to score submissions with network access disabled, and under CPU, memory and
timeout restrictions. There is no rate limit to hit, no key to rotate, and no
per-session bill that scales with traffic.

---

## The dense semantic route

Off by default since Phase 5, on measurement.

Phase 3 built the route in full: MiniLM embeddings over all 50,000 products,
benchmarked against `bge-small-en-v1.5`, fused with BM25 by both RRF and weighted
blending. It was worth +0.0355 at the time.

Phase 4 then changed what a query looks like. Once the customer answers questions by
quoting constraint sentences out of the target's own text, and those quotes accumulate
across turns, BM25 gets sharper on them while a single sentence embedding blurs them
together:

| configuration | before clarification (E2) | after clarification (E5) |
|---|---|---|
| lexical only | 0.115573 | **0.687598** |
| dense + RRF | 0.145170 | 0.636669 |
| dense + weighted | **0.151089** | 0.636663 |

Switching the dense route off is worth +0.0509, and it wins or ties on every scenario.

We kept the route, its flag and the prebuilt MiniLM artifact, because the result is
about this query distribution rather than about dense retrieval in general:

```bash
pip install -r requirements.txt      # sentence-transformers (pulls in torch, numpy)
SHOPPING_AGENT_DENSE=1 python3 -m evaluator.local_evaluator
```

`SHOPPING_AGENT_FUSION=weighted` (the default) or `rrf` picks the blend, and the model is
set in `shopping_agent/embedding_config.py`. Run `python3 -m scripts.build_embeddings`
only when the frozen catalogue or the selected model changes.

**Model:** `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions, Apache-2.0. We
picked it over `bge-small-en-v1.5` on a benchmark (+0.013437 composite). The first
`pip install` run downloads weights from the Hugging Face Hub and needs network; both
models cache locally and run fully offline afterwards. The prebuilt artifact is bundled
so the route works in a fresh runtime without a rebuild.

---

## Configuration flags

Every behavioural choice can be switched in one step, for auditing. The defaults are
what the official run uses.

| variable | default | effect |
|---|---|---|
| `SHOPPING_AGENT_SHORTLIST` | `1` | confidence-sized shortlist. `0` gives always-ten, measuring 0.885293 |
| `SHOPPING_AGENT_CATFILTER` | `1` | retrieve inside the stated category. `0` gives catalogue-wide, measuring 0.942229 |
| `SHOPPING_AGENT_DENSE` | `0` | enable the dense semantic route |
| `SHOPPING_AGENT_FUSION` | `weighted` | route blend: `weighted` or `rrf` |
| `SHOPPING_AGENT_RERANK` | `1` | deterministic reranker |
| `SHOPPING_AGENT_CLARIFY` | `1` | ask clarification questions |
| `SHOPPING_AGENT_REPEAT` | `0` | re-ask a productive attribute (measured at −0.0078) |
| `SHOPPING_AGENT_WILDCARD` | `1` | allow the `other` wildcard question |
| `SHOPPING_AGENT_DISAGREEMENT` | `1` | use candidate disagreement in question choice |
| `SHOPPING_AGENT_BLOCK_SOFT` | `0` | block soft slots from being asked |

---

## Tests

```bash
python3 -m unittest discover -s tests -t . -q
```

169 tests, standard library only, about 5 seconds.

The most important ones are in `tests/test_phase6_slots.py` and
`tests/test_phase6_category.py`. They check our reconstruction of the card generator
against the evaluator's own `intent_card`, `_flatten_values`, `_clean_constraint`,
`coarse_category` and `initial_message` over a catalogue sample, rather than against
hand-written expectations. Hand-written expectations would encode our reading of the
generator, which is the thing that can be wrong, so any drift has to fail loudly rather
than show up later as a quietly unowned constraint.

`tests/test_phase4_fallback.py` and `tests/test_phase6_category_filter.py` cover the
failure paths: a raising reranker, an unavailable dense route, an unparseable opener, an
unreproducible category, an empty filtered search.

---

## Repository layout

```text
starter/agent.py                  the official Agent entry point
shopping_agent/                   the system (see the architecture table above)
evaluator/local_evaluator.py      organiser-provided public-set simulator and scorer
tests/                            169 unit tests
scripts/demo_session.py           print one multi-turn session as a transcript
scripts/                          ablation harnesses, weight sweeps, audits, rank replay
docs/experiments/E1..E9.md        nine decision records: every measurement, including
                                  the ideas measured and rejected
docs/competition_specification.md the rules and evaluation protocol
docs/agent_api_contract.json      the machine-readable Agent contract
data/public_set.jsonl             200 labelled development sessions
data/catalog.jsonl                50,000 products (download separately, see Setup)
data/embeddings/                  prebuilt MiniLM artifact for the optional dense route
```

### The experiment records

`docs/experiments/` is where the reasoning lives. Each record states a hypothesis, the
base commit, the measurement and the decision, and each one includes what was rejected
so a future reader does not re-derive a dead end.

| | |
|---|---|
| E1 | embedding model choice: MiniLM over bge-small |
| E2 | fusion ablation: lexical-only vs RRF vs weighted |
| E3 | clarification policy, where most of the score came from |
| E4 | reranker weights, and why the spec's stated priority order lost 0.047 |
| E5 | retrieval reversal: removing the dense route, and disclosed-evidence scoring |
| E6 | short-label evidence and tie-breaker retuning |
| E7 | phrase containment, pool depth, and where the last twelve misses lived |
| E8 | slot ownership, exact stated category, the shortlist policy and its caveat |
| E9 | category-filtered retrieval, the flat re-sweep, and one rejected idea |

---

## Limitations and what we would improve

### The shortlist policy is shaped by this metric

The evaluator breaks on the first hit and freezes the rank. Under a metric that scored
the best rank across all turns, withholding results would be worth nothing. We think it
is defensible as product behaviour, since a precision-first agent that never returns a
candidate it would not defend and asks a question instead is arguably the better
product, and nothing in the rules requires returning ten ("only the first 10 valid
unique `parent_asin` values are scored" is a maximum, not a quota).

Still, it is the one change a reviewer could fairly call metric-shaped. So we isolated
it in a single module, documented it at length in `shopping_agent/shortlist.py`, and
`SHOPPING_AGENT_SHORTLIST=0` restores always-ten and measures 0.885293 (HitRate 1.000,
MRR 0.695645, MTTC 2.170). The other mechanisms are unaffected by it.

*Given more time:* measure it under a best-rank-across-turns metric, to establish how
much of its value is genuinely about precision and how much comes from the break rule.

### The agent assumes the customer quotes

Slot ownership, phrase containment and the exact-category filter all depend on the
simulator's verbatim behaviour. Each one fails quietly by design, so on a paraphrasing
split every candidate scores 0.0 and the ranking falls back to the features underneath.
But the ceiling would be lower and we cannot say by how much without such a split.

*Given more time:* generate a paraphrased held-out split and measure the fallback ceiling
directly, instead of arguing for it from construction. This is the most valuable thing
we did not get to.

### The opener parsing is a regex parser, English-only

`orchestrator.py` reproduces the generator's message format exactly and is tested
against it, but it parses one known format rather than doing general language
understanding. A real deployment would need genuine intent parsing.

### No LLM means no genuinely free-form conversation

The clarifying questions are templated. That costs nothing here, since the evaluator
reads `ask_attribute` and never the prose, but a production version talking to real
shoppers would need real generation.

*Given more time:* a small local generator for the question text, so the conversation
reads naturally without bringing back an API dependency or a per-query cost.

### One known inconsistency, deliberately left in

`build_query_text` feeds the customer's declines into the BM25 query, while
`_absorb_answer` refuses to keep a decline as a disclosure. This looks like a bug and
isn't. 80 of the turns played are declines, and suppressing them costs four hits
(0.945497 down to 0.920891). The query is far thinner than it looks, at a median of 6
distinct terms, with 2 turns collapsing to an empty query, so the decline text is acting
as ballast. Recorded in E9 because it will look like a bug to the next reader too.

### Where the remaining headroom is

HitRate is finished at 1.000 and cannot rise. What is left is MRR at 0.938657, worth at
most another +0.0184, and the reranker's entire adjustment half is now inert, so closing
that gap would take a new discriminator rather than a retune. MTTC of 2.805 is bounded
below by structure as much as by ranking, since an Intent Override session cannot
register a hit before its override turn, which falls on turn 3 or 4.

*Given more time:* generalise slot ownership beyond exact-value equality to a normalised
attribute graph, so it survives the messier structured data of a real catalogue; and
re-price every constant on a second product category, since all of them were tuned
against clothing.

---

## Team contributions

### Implementation

**Lin Minhong (@coffee-678)** — 65 commits
Phases 1, 2, 3, 5 and 6. Multi-turn conversation state and the orchestrator; the
constraint-aware lexical retrieval pipeline and price/category filtering; the dense
semantic route end to end (embedding benchmark, artifact build, vector adapter, RRF and
weighted fusion) and the measurement that later removed it; disclosed-evidence scoring,
phrase containment and pool-depth tuning; slot ownership, the exact stated category, the
confidence-sized shortlist policy and category-filtered retrieval. Experiment records
E1, E2, E5, E6, E7, E8, E9.

**Lim Ray Hing (@rayhing1510)** — 12 commits
Phase 4, which took the score from 0.151 to 0.637. The deterministic final scorer and
its inspectable feature checklist (P4-T1); one clarification question per turn and the
attribute-choice policy (P4-T2/T3); the failure-path hardening that guarantees nothing
escapes `respond()` (P4-T4); the reranker weight sweep harness and the retuning it
drove; the clarification ablation. Experiment records E3 and E4, plus the code-review
remediation for both P4 reviews (the lead-in cap bug, soft budgets, category neutrality,
connection hygiene) and repository hygiene for regenerable artifacts.

**Joel Rhys Chee (@Antelyuu)** — 8 commits
Phase 0 baseline checkpoint and the AI-readable phase execution skeleton that structured
the whole project; the agent behaviour documentation (`docs/agent_documentation.html`);
Phase 1 test coverage; Phase 2 constraint-safety work and audit; Phase 3 validation and
fallback hardening; bundling the MiniLM embedding artifact for offline use.

### External testing and score diagnosis

Two members worked outside the implementation branches, testing each phase's code as it
landed and turning an aggregate score back into specific defects worth acting on. Our
method was to measure before changing anything, and that only works if someone
establishes which cases are failing and why. A phase's TechnicalScore tells you it is
sitting at 0.63; it does not tell you what to do next.

**Lim Dao Hao** and **Edrich Denzil Lim Yu** — component testing and score diagnosis, all six phases

- Tested each phase's code independently of the people who wrote it, exercising
  retrieval, ranking, clarification and state handling against the public set as each
  landed.
- Identified which test cases and which sessions were failing, and isolated the
  component responsible. A target that never entered the candidate pool is a retrieval
  defect rather than a ranking one, and the fixes for those two things have nothing in
  common. Making that distinction is what eventually located our last miss.
- Worked out what could be done to pass them, breaking each phase's score into the parts
  that could still move and recommending where the next phase's effort would pay.

Their diagnosis set the agenda for every phase, and is why this repository argues from
measurement rather than intuition. The failing-case analyses they produced are carried
into the experiment records in `docs/experiments/`, including the ones concluding that a
line of attack was exhausted. E7 is largely an account of where the remaining misses
lived, and it is what motivated the slot-ownership work in E8.

### Notes

Commits authored by `TechJam2026` are the organiser's original challenge scaffolding
rather than team contributions.

Every phase was merged only after a written code review. The findings and what was done
about them are recorded in the corresponding experiment file.

---

## The challenge itself

Build an AI shopping agent that asks useful follow-up questions and recommends the
customer's hidden target product within at most 10 turns.

- A frozen catalogue of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of
  Amazon Reviews 2023.
- 200 labelled public sessions for local development. The organiser keeps 800 private
  for final evaluation.
- The scenario mix is identical on both splits: 40% Buying (a hard constraint disclosed
  early), 40% Browsing (starts vague), 15% Intent Override (a preference replaced on turn
  3 or 4), 5% Boundary (may have no preference for a requested attribute).

Raw user IDs, review text, timestamps and purchase history are never disclosed to the
agent. It sees only a safe aggregate `user_profile`.

### Agent interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [{"parent_asin": "B000..."}, {"parent_asin": "B001..."}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`,
`budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`
and `docs/competition_specification.md`.

### Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target, with a miss contributing zero.
- **MTTC:** mean first-hit turn, with a miss assigned turn 11.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency     = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the Technical Execution assessment. It is not a
separate judging criterion and does not represent the whole of it. Only exact
`parent_asin` equality produces a hit.

---

## Data attribution

The catalogue and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD.
See `DATA_ATTRIBUTION.md` before using or redistributing the data. Sessions are sampled
deterministically from the official Clothing 5-core leave-last-out split and joined to
the frozen catalogue.

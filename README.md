# Conversational Shopping Agent — *chud-pro-max-shopinator*

Our entry for the **TechJam Conversational E-Commerce Search Challenge**.

A shopper arrives with a vague idea. The agent has ten turns to work out which of
**50,000 products** they have in mind, asking one question at a time.

Ours finds it in **2.8 turns on average, and never fails to find it.**

| | starter baseline | **this agent** |
|---|---|---|
| **TechnicalScore** | 0.106710 | **0.945297** |
| Hit Rate@10 — did it find the product? | 0.125 | **1.000** |
| MRR — was it the *first* thing shown? | 0.068034 | **0.938657** |
| MTTC — turns taken | 9.81 | **2.815** |

```bash
python3 -m evaluator.local_evaluator      # no network, no API key, no env vars
```

No LLM. No API calls. **Zero tokens, $0 per session**, and it runs with the network
switched off.

---

## The big idea, in one paragraph

Most search systems ask *"which product looks like what the customer described?"* On this
task that question is almost useless, because thousands of near-identical listings all
look alike.

We noticed something better. The customer's hidden requirements are not free text — the
benchmark builds them by copying **whole lines straight out of the target product's own
description**. So we stopped asking what a product *resembles* and started asking:

> **"Would this product have produced that exact sentence?"**

Almost none can. That single change is most of our score — it turns a fuzzy
similarity problem into a near-exact identification problem.

<details>
<summary><b>Why that works — the mechanism, for the curious</b></summary>

The evaluator constructs the hidden "intent card" by walking the target's `features` and
`details` fields and taking whole values: one complete list element, or one complete
`key: value` pair, normalised and clipped at 180 characters. Every constraint the customer
can disclose is therefore an **exact member of a small set the target owns**.

We reconstruct that set for every product and check membership, rather than measuring text
overlap. Across all 50,000 products the target owns 800 of its own 800 disclosable
constraints, and 193 of them are owned by exactly *one* product in the entire catalogue.
Two disclosures plus the category usually identify a single item.

</details>

---

## How it works

Five mechanisms, in the order they run.

| # | mechanism | what it does |
|---|---|---|
| 1 | **Search only the right aisle** | The opening line names a category. We search *inside* it — a median of 184 products instead of 50,000. |
| 2 | **Ask the question with the best odds** | One question per turn, chosen by how likely the customer is to be able to answer it. |
| 3 | **Ownership scoring** | The big idea above: does this product *own* the sentence the customer said? |
| 4 | **Only commit when confident** | While still narrowing, show one best guess; show the full ten only when the evidence supports it. |
| 5 | **Fall back to meaning** | If the customer *paraphrases* instead of quoting, mechanisms 3–4 go quiet — so a small local embedding model scores meaning instead. |

Everything is deterministic and inspectable: every ranking decision can be printed as a
feature-by-feature breakdown showing exactly why one product beat another.

### Mechanism 5 deserves a note

The first four exploit a property of *this* benchmark: the simulated customer quotes
product text verbatim. A real shopper would not, and the organiser reserves the right to
paraphrase on the private set.

So we measured what happens when the customer stops quoting, and built for it. On a
paraphrased replay of the same 200 sessions:

| | before | **after** |
|---|---|---|
| paraphrased score | 0.696015 | **0.875897** |
| paraphrased Hit Rate | 0.805 | **0.965** |
| **public score** | 0.945497 | **0.945297** |

We spent 0.0002 of a number the leaderboard sees to gain 0.18 of one the private set
might. The feature is **provably inert on the public benchmark** — switch it off with
`SHOPPING_AGENT_SEMANTIC=0` and the score is identical to six decimals.

---

## Results

Full public set, 200 sessions, no environment variables.

| | Hit Rate@10 | MRR | MTTC |
|---|---|---|---|
| **overall** (200) | **1.000** | 0.938657 | 2.815 |
| buying (80) | 1.000 | 0.973036 | 2.288 |
| browsing (80) | 1.000 | 0.908904 | 2.838 |
| intent_override (30) | 1.000 | 0.928095 | 3.867 |
| boundary (10) | 1.000 | 0.933333 | 3.700 |

Hit Rate@10 is **1.000 on every scenario type**, not just overall.

```text
TechnicalScore = 0.50 × HitRate + 0.30 × MRR + 0.20 × Efficiency
               = 0.50 × 1.000 + 0.30 × 0.938657 + 0.20 × 0.8185  =  0.945297
```

---

## Cost, latency and resources

**No API, no network at inference, no tokens, $0.** Measured over the full 200-session set
on Apple Silicon, Python 3.13.6. Both columns score exactly 0.945297.

| | default | `SHOPPING_AGENT_SEMANTIC=0` |
|---|---|---|
| model | voyage-4-nano, Apache-2.0, run **locally** | none |
| network / API key | not required | not required |
| tokens, cost per session | 0, $0.00 | 0, $0.00 |
| per-turn latency | 52 ms median, 121 ms p95 | 35 ms median, 75 ms p95 |
| cold start | 4.3 s | 4.1 s |
| peak RSS | 1.60 GB | 0.78 GB |
| dependencies | numpy, torch, sentence-transformers | **standard library only** |

Two caveats stated rather than buried. The embedding model loads lazily, making one turn
per run about 3.2 s. And it downloads weights from the Hugging Face Hub on the **first run
only** — every run afterwards is fully offline. If the organiser enforces a memory cap
below ~2 GB, a per-turn timeout, or a fully sandboxed first run, set
`SHOPPING_AGENT_SEMANTIC=0`; **the benchmark score does not change.**

---

## Running it

```bash
python3 -m evaluator.local_evaluator            # the official harness
python3 -m unittest discover -s tests -t .      # 223 tests
pip install -r requirements.txt                 # OPTIONAL — enables mechanism 5
```

Python 3.10+. The one prerequisite is the 60 MB catalogue, which is not stored in git —
**[docs/SETUP.md](docs/SETUP.md)** has the commands that fetch and verify it.

**Environment variables** — all optional; the submitted configuration sets none.

| variable | default | effect |
|---|---|---|
| `SHOPPING_AGENT_SEMANTIC` | on | `0` disables mechanism 5. Public score unchanged. |
| `SHOPPING_AGENT_DENSE` | off | `1` restores the dense retrieval route we removed. |
| `SHOPPING_AGENT_PARAPHRASE_SHORTLIST` | `0` | `10` restores the pre-E13 commit policy. |
| `SHOPPING_AGENT_WILDCARD_CAP` | `3` | Consecutive open questions allowed under paraphrase. |
| `SHOPPING_AGENT_SEMANTIC_WEIGHT` | `192` | Measurement override for the semantic weight. |

---

## Limitations

**The shortlist policy is shaped by this metric.** The evaluator freezes the target's rank
the first time it appears, so *when* to commit is worth real score. We tuned that, and a
metric that rewarded browsing breadth would want a different policy.

**No language understanding.** Constraint extraction is regex over a known vocabulary. It
handles the simulator's phrasing and would not survive genuinely open conversation.

**The customer profile is accepted and never used.** We found no measurable signal in it.

**Ranking, not retrieval, is where the remaining headroom is.** With a perfect reranker
and today's retrieval the score would be 0.990300 — and pool recall is already 1.000, so
none of the gap is retrieval's.

---

## How we decided things

Fourteen written experiment records, [`docs/experiments/`](docs/experiments/) — every
measurement, including the ideas we **rejected**. A few house rules we held to:

- **Never report a score without re-running the guardrails**, including a control proving
  the measurement harness itself still reproduces.
- **Confirm every gain on a second, structurally different test** before shipping it. Two
  changes died this way after looking good on the first one.
- **A rejected idea is only rejected at the configuration you tested it on.**
- **Surface the trade with numbers on both sides.** Every table here has two columns for
  that reason.

---

## Team

| | |
|---|---|
| **Lin Minhong** ([@coffee-678](https://github.com/coffee-678)) | Conversation state and orchestration; lexical retrieval and filtering; the dense route end to end, and the measurement that removed it; evidence scoring, slot ownership, category-filtered retrieval, the shortlist policy, and the paraphrase-robustness work. Records E1, E2, E5–E9, E11–E14. |
| **Lim Ray Hing** ([@rayhing1510](https://github.com/rayhing1510)) | The deterministic final scorer and its inspectable feature checklist; the clarification policy; failure-path hardening guaranteeing nothing escapes `respond()`; the reranker weight sweep. Records E3, E4. |
| **Joel Rhys Chee** ([@Antelyuu](https://github.com/Antelyuu)) | Baseline checkpoint and the phase execution skeleton that structured the project; agent behaviour documentation; test coverage and constraint-safety audits; offline artifact bundling. |
| **Lim Dao Hao** and **Edrich Denzil Lim Yu** | Independent component testing and score diagnosis across all phases — turning an aggregate score back into the specific defect worth fixing next, which set the agenda for every phase. |

Commits authored by `TechJam2026` are the organiser's original scaffolding. Every phase
was merged only after a written code review.

---

## Where everything is

| | |
|---|---|
| [`starter/agent.py`](starter/agent.py) | the official `Agent` entry point |
| [`docs/SETUP.md`](docs/SETUP.md) | install, reproduce, ablations, flags, tests |
| [`docs/experiments/`](docs/experiments/) | E1–E14: every measurement, including rejected ideas |
| [`docs/dense_route.md`](docs/dense_route.md) | the dense route, and why we removed it |
| [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md) | catalogue derived from Amazon Reviews 2023, McAuley Lab, UCSD |

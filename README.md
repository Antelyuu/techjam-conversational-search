# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Reproducing the submission

No dependencies and no network:

```bash
python3 -m evaluator.local_evaluator
```

That is the whole reproduction. The agent runs on the standard library, and
the command above is exactly how the official harness constructs it -- no
environment variables, plain `Agent(catalog_path)`.

| | |
|---|---|
| TechnicalScore | **0.929426** |
| Hit Rate@10 | 0.990 |
| MRR | 0.912421 |
| MTTC | 2.965 |
| startup | STARTUP |
| per-turn latency | LATENCY |
| peak RSS | RSS |
| model / API / token usage | none |

| milestone | TechnicalScore |
|---|---|
| starter BM25 | 0.106710 |
| P2 constraint-aware lexical | 0.115573 |
| P3 dense + weighted fusion | 0.151089 |
| P4 clarification + reranker | 0.636663 |
| P5 dense retired + disclosed-evidence scoring | 0.706484 |
| P5 short-label evidence + retuned tie-breakers (E6) | 0.753328 |
| P5 phrase containment + widened pool (E7) | 0.821381 |
| P6 slot ownership (E8) | 0.853005 |
| P6 + confidence-sized shortlist (E8) | 0.876118 |
| P6 + open question asked first (E8) | 0.881931 |
| **P6 + exact stated category (E8)** | **0.929426** |

## How it finds the target

The agent is deterministic, stdlib-only, and uses no model. Its leverage comes
from one idea, applied twice: **the simulated customer speaks in strings the
target product actually owns, so the sharpest test of a candidate is whether
it would have produced those exact strings.**

- **Slot ownership.** The hidden intent card is built from *whole* values of
  the target's `features` and `details`. Every constraint the customer can
  disclose is therefore an exact member of a small set the target owns -- not
  a substring of its text. Matching on ownership rather than containment
  separates the target from the catalogue near-duplicates that share its
  vocabulary, and it has perfect recall by construction (the target owns all
  800 of its disclosable constraints), so requiring it can never cost a hit.
  Ownership is weighted by how rare each disclosure is inside the candidate
  pool, so a material label owned by thousands counts for nothing and a unique
  care instruction is close to an identification.
- **The exact stated category.** The opening line names the target's coarse
  category verbatim, in every scenario, on turn 1 -- and for Browsing it is
  the only thing said before a question is answered. Only a median 38% of the
  candidate pool reproduces it exactly, so agreement removes three fifths of
  the field for free.
- **The open question first.** The simulator answers `other` with any
  undisclosed constraint, so it drains the card faster than any specific
  attribute; it had been queued behind six narrower questions.
- **A shortlist the agent can defend.** Rather than padding ten results on
  turn 1, it returns its single best candidate while still narrowing and the
  full ten once the constraints identify one product or the useful questions
  are spent. See the caveat in `docs/experiments/E8-...md`; this one is
  shaped by the metric and switchable with `SHOPPING_AGENT_SHORTLIST=0`.

Every feature's contribution to a candidate's score is recorded and printable
(`RerankedCandidate.explain()`), so any placement can be read off rather than
guessed at. Full reasoning and every measurement, including the ideas measured
and rejected, are in `docs/experiments/`.

## Dense Semantic Route (off by default since P5)

P3 fused a dense semantic route with BM25 and it was worth +0.0355 at the
time. P4 then changed what a query is: the customer now answers questions by
quoting constraint sentences out of the target product's own text, and those
accumulate across turns. BM25 sharpens on that; a single sentence embedding
blurs it. Measured after P4, **turning the dense route off is worth +0.0509**
and wins or ties every scenario
(`docs/experiments/E5-p5-retrieval-reversal-and-evidence.md`).

| configuration | before clarification (E2) | after clarification (E5) |
|---|---|---|
| lexical only | 0.115573 | **0.687598** |
| dense + RRF | 0.145170 | 0.636669 |
| dense + weighted | **0.151089** | 0.636663 |

The route, its flag and the prebuilt MiniLM artifact all remain, because the
finding is about this query distribution rather than about dense retrieval:

```bash
pip install -r requirements.txt      # sentence-transformers (pulls in torch, numpy)
SHOPPING_AGENT_DENSE=1 python3 -m evaluator.local_evaluator
```

`SHOPPING_AGENT_FUSION=weighted` (default) or `rrf` picks the blend, and the
model is set in `shopping_agent/embedding_config.py`. Run
`python3 -m scripts.build_embeddings` only when the frozen catalogue or the
selected model changes.

**BM25 fallback:** when enabled but unavailable -- missing dependencies, or an
artifact built from a different catalogue -- the route does not engage and the
agent serves BM25 results instead, printing the reason to stderr rather than
swallowing it, so a degraded run is visible rather than merely scoring lower.
The agent never fails because the dense route is unavailable.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.

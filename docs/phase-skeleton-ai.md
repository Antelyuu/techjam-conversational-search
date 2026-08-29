# TikTok TechJam 2026 Track 4 - AI-Readable Phase Skeleton

Status: planning and execution skeleton

Repository: `TechJam2026/techjam-conversational-search`

Audience: human team members and coding/research agents working on GitHub branches.

## Agent operating instructions

Before modifying code, every agent must:

1. Read `README.md`, `docs/agent_api_contract.json`, `docs/competition_specification.md`, `docs/evaluation_config.json`, `docs/submission_rules.md`, `evaluator/local_evaluator.py`, and this file.
2. Identify the active phase and task ID.
3. Inspect the current branch and Git status.
4. Preserve unrelated work.
5. Implement only the assigned task scope.
6. Add or update tests.
7. Run the official evaluator with the same configuration as the previous checkpoint.
8. Report changed files, tests, metrics, latency, memory, assumptions, and failures.

Do not modify evaluator logic, public labels, or frozen catalogue records. Do not use hidden target information. If this document conflicts with checked-in code or the API contract, checked-in code and the contract take precedence.

## Non-negotiable interface and scope

```yaml
agent_interface:
  reset: "reset(session_id: str, user_profile: dict) -> None"
  respond: "respond(session_id: str, user_message: str, turn: int, top_k: int) -> dict"
  response_required_keys: [message, ask_attribute, recommendations]
  ask_attribute_values: [category, material, color, size, style, brand, budget, feature, use_case, other, null]
  recommendation_identifier: parent_asin
  matching: exact_string_equality
  top_k: 10
  max_turns: 10
  catalog_mutation: forbidden
  evaluator_modification: forbidden
  default_network_dependency: forbidden
  required_fallback: offline_deterministic_path
```

## Target architecture

```text
reset(session_id, user_profile)
        |
        v
create isolated SessionState
        |
        v
respond(session_id, user_message, turn, top_k)
        |
        v
parse latest message into candidate slot updates
        |
        v
merge, replace, or reject updates using explicit state rules
        |
        v
route intent: BUYING, BROWSING, OVERRIDE, or UNKNOWN
        |
        +-------------------+-------------------+-------------------+
        |                   |                   |
        v                   v                   v
hard filters        lexical route        dense route
price/category      FTS5 + BM25           embeddings + vectors
        |                   |                   |
        +-------------------+-------------------+
                            v
                  candidate union + deduplication
                            |
                            v
                        score fusion
                            |
                            v
                         reranking
                            |
                            v
             recommend Top 10 OR ask one attribute
                            |
                            v
                      contract-valid response
```

## Shared internal vocabulary and contracts

- **Slot:** one named requirement, such as `color=black`.
- **Session state:** all per-session memory, including profile, slots, history, intent, and question counters.
- **Hard constraint:** a requirement eligible for pass/fail filtering when reliably represented in the catalogue.
- **Soft preference:** a ranking signal that should not eliminate a product by default.
- **Candidate pool:** a larger shortlist retrieved before expensive final ranking.
- **Reranker:** a scorer that orders only the candidate pool.

Recommended shared structures:

```python
@dataclass(frozen=True)
class Constraint:
    attribute: str          # category, color, max_price, use_case, ...
    value: object           # normalized value
    strength: str           # hard or soft
    source_turn: int
    confidence: float

@dataclass
class SessionState:
    session_id: str
    user_profile: dict
    intent: str             # buying, browsing, override, unknown
    constraints: dict[str, Constraint]
    history: list[str]
    asked_attributes: set[str]
    rejected_attributes: set[str]
    clarification_turns: int
    last_retrieval_ids: list[str]

@dataclass(frozen=True)
class Candidate:
    parent_asin: str
    route_ranks: dict[str, int]
    route_scores: dict[str, float]
    matched_hard_constraints: tuple[str, ...]
    matched_soft_preferences: tuple[str, ...]

@dataclass(frozen=True)
class SearchRequest:
    query_text: str
    state: SessionState
    top_k: int
```

## Evidence protocol

Every feature or experiment must produce a report containing:

```yaml
experiment_id: "E<number>"
phase: "P<number>"
hypothesis: "Expected change and reason"
base_commit: "<git commit>"
candidate_commit: "<git commit>"
dataset: "fixed validation subset or full public set"
overall_metrics: {hit_rate_at_10: null, mrr: null, mttc: null, efficiency: null, technical_score: null}
scenario_metrics: {}
performance: {startup_seconds: null, per_turn_latency_ms: null, peak_memory_mb: null}
model_api: {model: "none or exact name", network_required: false, prompt_tokens: null, completion_tokens: null}
newly_won_sessions: []
newly_lost_sessions: []
known_regressions: []
decision: "keep / revise / reject / undecided"
```

An **ablation** is a controlled comparison with one component disabled. Major components require an ablation before acceptance.

---

# P0 - Reproducible baseline and repository contracts

```yaml
phase_id: P0
branch: phase/0-baseline-foundation
parent_branch: main
primary_goal: establish a trusted before-measurement and shared engineering contracts
status: complete_at_current_checkpoint
```

## Problem

Before improving retrieval, prove that catalogue loading, SQLite FTS5, tests, public sessions, and metrics work. Otherwise score changes cannot be attributed to implementation changes.

## Required actions

1. Verify `data/catalog.jsonl` exists, has 50,000 rows, and has unique `parent_asin` values.
2. Verify SQLite FTS5 is available.
3. Run `python3 -m unittest discover -s tests -v`.
4. Run `python3 -m evaluator.local_evaluator`.
5. Record overall and scenario metrics, Python version, checksum, startup time, and catalogue missingness.
6. Define shared internal contracts before parallel implementation.

## Checkpoint evidence

```yaml
tests: "3 passed"
catalog_rows: 50000
unique_parent_asin: 50000
hit_rate_at_10: 0.125
mrr: 0.068034
mttc: 9.81
technical_score: 0.106710
```

Full evidence is in `docs/phase0-baseline.md`.

## Important observations

- Most catalogue rows have no numeric price. Missing price must not be treated as zero or automatically excluded.
- Features and descriptions also have missing values.
- The public set contains 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

## Acceptance criteria

- Tests pass.
- Evaluator runs without source modification.
- Baseline approximately matches `docs/baseline_results.json`.
- Baseline evidence is committed.
- No secrets or catalogue data are committed.

## Handoff to P1

Preserve the official Agent signatures and baseline configuration. Implement isolated `SessionState` keyed by `session_id`.

---

# P1 - Multi-turn state, slot extraction, and intent updates

```yaml
phase_id: P1
branch: phase/1-conversation-state
parent_branch: phase/0-baseline-foundation
primary_goal: remember and correctly update customer intent across turns
owner: conversation-state specialist
reviewer: integration lead and dialogue-policy specialist
```

## Required behavior

```text
Turn 1: "I need shoes."
        category = shoes

Turn 2: "Black running shoes under $120."
        category = shoes, color = black, use_case = running, max_price = 120

Turn 3: "Actually, I want hiking boots."
        category = hiking boots, use_case = hiking
        unrelated constraints may remain unless explicitly rejected
```

The dependent-slot deletion policy must be documented. Do not clear the entire state for one changed slot.

## Tasks

```yaml
P1-T1:
  title: session isolation
  output: dict[str, SessionState]
  acceptance: identical profiles in different session_ids never share history or slots
P1-T2:
  title: rule-based slot extraction
  output: deterministic parser and unit tests
  acceptance: common category, color, material, brand, size, use-case, and budget forms parse
P1-T3:
  title: constraint strength
  output: hard/soft classification rules
  acceptance: must/need/under language differs from prefer/ideally language
P1-T4:
  title: accumulation and replacement
  output: state transition function
  acceptance: compatible values merge; explicit overrides replace affected slots only
P1-T5:
  title: intent routing prototype
  output: transparent Buying/Browsing/Override/Unknown rules
  acceptance: routing decision is logged and tested
```

## Tests

- First message creates state.
- Later messages accumulate compatible slots.
- Explicit color replacement changes color only.
- Category replacement follows the documented dependent-slot policy.
- “Actually”, “instead”, and “ignore earlier” are tested.
- Empty and irrelevant messages do not erase state.
- Two sessions with identical profiles stay independent.

## Hypothesis

Accumulated context improves multi-turn Buying and Intent Override performance without degrading vague Browsing behavior.

## Risks

- False extraction creates harmful filters.
- Over-aggressive purge removes useful requirements.
- Historical profile preferences override current explicit intent.

## Exit criteria and handoff

State tests pass, evaluator results are recorded, and a normalized `SessionState` plus query are available to every retriever. Retrievers must not parse raw conversation independently.

---

# P2 - Constraint-aware filtering and improved lexical retrieval

```yaml
phase_id: P2
branch: phase/2-filtered-lexical-search
parent_branch: phase/1-conversation-state
primary_goal: apply reliable constraints and improve BM25 without deleting valid targets
owner: catalogue and lexical-search specialist
reviewer: integration lead and state specialist
```

## Safety rule

Filtering is pass/fail elimination. A bad filter permanently removes the hidden target. Uncertain constraints remain ranking signals until catalogue coverage is proven.

```text
known price > hard maximum  → exclude
known price <= maximum      → retain
missing price               → retain, mark budget_unverified
confident category mismatch → exclude only after target-survival testing
uncertain category mismatch → ranking penalty
style/use-case wording      → usually soft semantic signal
```

## Tasks

```yaml
P2-T1:
  title: product normalization
  output: stable ProductRecord and searchable text
  acceptance: missing dict/list/scalar fields cause no exceptions
P2-T2:
  title: numeric constraint evaluator
  output: price comparison and filter-reason audit
  acceptance: missing price is never treated as zero or over-budget
P2-T3:
  title: conservative category normalization
  output: category matching and fallback boost
  acceptance: broad categories do not incorrectly outrank specific matches
P2-T4:
  title: cumulative BM25 query builder
  output: query from SearchRequest, not latest message only
  acceptance: selected slot values and controlled boolean behavior are included
P2-T5:
  title: target-survival audit
  output: count of targets removed by each filter
  acceptance: each hard filter explains its exclusion reason
```

## Required experiments

- Current latest-message OR query versus cumulative query.
- Existing BM25 weights versus revised field weights.
- No filters versus price filter.
- Strict category filter versus category ranking boost.

## Exit criteria and handoff

Filter tests and target-survival audit pass. Lexical fallback works without embeddings or network. Expose:

```python
retrieve(request: SearchRequest, limit: int) -> list[Candidate]
```

The interface must support lexical, metadata, and dense retrievers without exposing their internal data structures.

---

# P3 - Dense semantic retrieval and hybrid candidate generation

```yaml
phase_id: P3
branch: phase/3-semantic-retrieval
parent_branch: phase/2-filtered-lexical-search
primary_goal: improve recall for paraphrases and scenario-based Browsing while preserving offline fallback
owner: semantic-retrieval specialist
reviewer: lexical-search specialist and integration lead
```

## Definitions

- **Embedding:** a fixed-length numeric vector representing text meaning.
- **Dense retrieval:** finding products whose vectors are close to the query vector.
- **Cosine similarity:** vector-direction similarity; higher usually means more semantic similarity.
- **Candidate union:** deduplicated combination of results from multiple routes.

## Product text

Create one deterministic product text from available fields:

```text
title + categories + features + details + store + description
```

Do not include hidden labels, private evaluator data, or unavailable review history.

## Tasks

```yaml
P3-T1:
  title: embedding model benchmark
  output: comparison of at least two lightweight models and no-dense baseline
  acceptance: report dimensions, model size, license, RAM, startup, encoding, and query latency
P3-T2:
  title: reproducible embedding artifact/build
  output: vectors or a deterministic build script
  acceptance: same catalogue/model creates compatible vectors
P3-T3:
  title: vector search adapter
  output: generic dense retriever
  acceptance: returns parent_asin-ranked candidates and semantic scores
P3-T4:
  title: candidate union
  output: lexical + metadata + dense candidates
  acceptance: deduplicates without losing route rank information
P3-T5:
  title: fusion comparison
  output: RRF and normalized weighted fusion
  acceptance: both are configuration-selectable and independently evaluated
```

## Decision gate

Compare no dense route, at least two models, direct NumPy cosine similarity, and an in-memory flat index. Select using Hit Rate, MRR, scenario stability, RAM, startup, latency, license, and offline fallback. Do not choose by reputation alone.

## Risks

- Semantic similarity may retrieve the wrong category.
- BM25 and cosine scores have incompatible scales.
- Model startup/RAM may violate constraints.
- Runtime model downloads may break reproducibility.

## Exit criteria and handoff

Dense retrieval is feature-flagged, offline lexical fallback works, fusion has an ablation, and the model decision is recorded. Pass route ranks/scores and metadata to P4; P4 must not rerun retrieval internally.

---

# P4 - Reranking, clarification, and failure handling

```yaml
phase_id: P4
branch: phase/4-ranking-dialogue-policy
parent_branch: phase/3-semantic-retrieval
primary_goal: improve final rank and turn efficiency through controlled reranking and questions
owner: ranking and dialogue-policy specialist
reviewer: state specialist and integration lead
```

## Decision flow

```text
candidate pool
    |
    v
hard constraints still satisfied?
    |
    +-- no --> remove/downgrade according to documented fallback
    |
    v
rank candidates
    |
    v
request specific enough and candidates coherent?
    |
    +-- yes --> return Top 10
    |
    +-- no --> useful allowed attribute and question budget remaining?
                    |
                    +-- yes --> return one question + ask_attribute
                    |
                    +-- no --> force Top 10 recommendation
```

## Reranker feature order

1. Hard-constraint satisfaction.
2. Category compatibility.
3. Lexical route rank.
4. Dense route rank.
5. Metadata compatibility.
6. Soft-preference matches.
7. Rating signals only if they improve validation metrics.

Start with an interpretable deterministic scorer. Then benchmark a local cross-encoder or optional external LLM. Any model/API path must have an offline fallback.

## Clarification rules

- Return exactly one allowed attribute or `null`.
- Do not ask about a fixed slot.
- Do not repeat an attribute.
- Track clarification count.
- Handle Boundary “no preference” replies.
- Force recommendations when the question budget is exhausted.

Estimate question value from candidate disagreement: prefer an unfixed attribute with enough known values that separates candidate groups and could change ranking. This is a practical approximation to information gain; a full entropy model is optional.

## Tasks

```yaml
P4-T1: {title: deterministic final scorer, acceptance: score contributions inspectable per candidate}
P4-T2: {title: clarification policy, acceptance: valid ask_attribute and no repeated/fixed question}
P4-T3: {title: candidate-diversity analysis, acceptance: missing attributes do not create false diversity}
P4-T4: {title: fallback policy, acceptance: dense/reranker failure returns valid lexical response}
P4-T5: {title: optional model reranker experiment, acceptance: cost/latency reported and offline default remains}
```

## Exit criteria

MRR and MTTC are evaluated together. Boundary and Intent Override are explicitly analyzed. Failure paths are contract-valid. Keep the change only if TechnicalScore improves or the team records a strong reason to retain it.

---

# P5 - Final hardening, reproducibility, and presentation

```yaml
phase_id: P5
branch: phase/5-final-hardening
parent_branch: phase/4-ranking-dialogue-policy
primary_goal: produce a reliable, explainable, reproducible submission
owner: integration lead
reviewers: all component owners
```

## Hardening checklist

```yaml
runtime:
  - fresh_environment_setup_passes
  - official_evaluator_runs_without_modification
  - no network required by default
  - optional models have deterministic fallback
  - exceptions do not escape respond
  - output is contract-valid
data:
  - catalogue is read-only
  - no private data or target memorization
  - invalid/duplicate IDs are not emitted
performance:
  - startup, per-turn latency, and peak memory recorded
  - token/API usage recorded
documentation:
  - README has setup and reproduction
  - limitations and contributions are recorded
  - backend demo walkthrough is reproducible
```

## Presentation flow

```text
Problem: keyword-only search loses context
        ↓
Evidence: starter baseline and weak scenarios
        ↓
Design: state + constraints + routed retrieval + ranking + clarification
        ↓
Demo: one multi-turn state transition and recommendation
        ↓
Evidence: before/after metrics, scenario breakdown, ablation
        ↓
Trade-offs: latency, missing metadata, offline fallback, cost
        ↓
Limitations and next improvements
```

## Final acceptance criteria

- Final branch uses the best measured configuration, not merely newest code.
- One command reproduces evaluation.
- Agent works without network access.
- Full public metrics and commit/configuration are recorded.
- Every teammate can explain every stage.
- Demo shows backend/API behavior; no frontend is required.

---

# GitHub branch protocol

## Branch topology

```text
main
  └── phase/0-baseline-foundation
        └── phase/1-conversation-state
              └── phase/2-filtered-lexical-search
                    └── phase/3-semantic-retrieval
                          └── phase/4-ranking-dialogue-policy
                                └── phase/5-final-hardening
```

Within the active phase:

```text
phase/Pn
  ├── feature/<component>
  ├── experiment/<hypothesis>
  └── docs/<deliverable>
```

Feature branches merge into the active phase branch. The completed phase branch is reviewed and merged into `main`. Suggested tags:

```text
v0-baseline
v1-stateful
v2-filtered-bm25
v3-hybrid
v4-ranking-dialogue
v5-submission
```

## Pull request rules

- One hypothesis or component per branch.
- No direct pushes to `main` for behavior changes.
- Every pull request includes tests and evaluator evidence.
- Integration lead reviews entry-point and contract changes.
- Do not commit secrets, catalogue data, model caches, or generated `results.json`.
- Never edit evaluator logic or public labels.

## Pull request report template

```markdown
## Task ID
P?-T?

## Hypothesis
If we ..., then ... should improve because ...

## Implementation
- Files changed:
- Public interface impact:
- Configuration/feature flag:

## Verification
- Tests:
- Baseline commit/config:
- Candidate commit/config:
- Overall metrics:
- Scenario metrics:
- Latency/memory:

## Failure analysis
- Newly won sessions:
- Newly lost sessions:
- Known limitations:
- Rollback plan:
```

## Agent completion report template

```yaml
agent_name: "<identifier>"
task_id: "P?-T?"
branch: "<branch>"
status: "complete / blocked / needs-review"
files_changed: []
tests_run: []
metrics_before: {}
metrics_after: {}
scenario_deltas: {}
performance: {}
assumptions: []
known_failures: []
follow_up: []
```

## Open decision gates

These are experiments, not silently pre-approved final decisions:

1. Embedding model, if any.
2. NumPy versus vector index.
3. RRF versus normalized weighted fusion.
4. Deterministic, local, or external reranker.
5. Which filters have enough catalogue coverage to be hard.
6. When clarification improves TechnicalScore rather than only conversation quality.

If a decision adds network, cost, a new dependency, or substantial architectural commitment, obtain explicit team agreement before merging it into the default path.

## Final instruction to future coding agents

Do not implement the entire architecture in one pass. Identify the active phase, select one task, inspect the actual repository, make the smallest measurable change, run tests and evaluator, and return an evidence-based completion report.

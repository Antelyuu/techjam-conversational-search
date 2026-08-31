# Setup, reproduction and reference

Everything needed to install the agent, reproduce every number quoted in the
[README](../README.md), watch a single session play out, and audit any individual
mechanism by switching it off.

- [Setup and installation](#setup-and-installation)
- [Reproducing our results](#reproducing-our-results)
- [See one session end to end](#see-one-session-end-to-end)
- [Reproducing the ablations](#reproducing-the-ablations)
- [Tests](#tests)
- [Configuration flags](#configuration-flags)
- [Repository layout](#repository-layout)

---

## Setup and installation

**Requirements:** Python 3.10 or later (developed and measured on 3.13.6). Nothing
third-party is required: the agent runs, and scores the same 0.945297, with no
dependencies installed. One optional install buys robustness that this benchmark cannot
see — see [The optional install](#the-optional-install) below.

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

Nothing has to be built. The BM25 index is constructed in memory from
`data/catalog.jsonl` at startup, in about 4 seconds.

### The optional install

```bash
pip install -r requirements.txt      # optional
```

This enables **semantic evidence** (E11), which scores a disclosed constraint against a
candidate's field values by meaning rather than by shared tokens. It is on by default
when its dependencies are present and it changes nothing on the public set — the score
is 0.945297 either way, because the feature is gated to stay silent while exact slot
ownership is still working. What it buys is the case the public set never exercises: on
a paraphrased replay of the same 200 sessions the score goes 0.696015 -> 0.875897 and
HitRate 0.805 -> 0.965. (E11 shipped the feature at 0.847762; E13 and E14 took it to
0.875897 by changing *when* the agent commits and *what* it asks, both free on the
public set.)

The model is `voyageai/voyage-4-nano`, Apache-2.0 open weights run locally: no API key,
no network at inference time, 0 tokens, $0. Its 66 MB catalogue artifact is bundled in
`data/embeddings/`, so there is no build step. Only the first run downloads model weights
from the Hugging Face Hub.

It costs peak RSS 0.78 GB -> 1.60 GB and a median turn of 35 ms -> 52 ms, and the model
loads lazily, which makes the first turn that has a disclosure to score about 3.2 s.

```bash
SHOPPING_AGENT_SEMANTIC=0 python3 -m evaluator.local_evaluator   # same score, none of the cost
```

Leaving the dependencies uninstalled has exactly the same effect: the feature reports
once on stderr that it is unavailable and scores 0.0 for every candidate, and the
ranking falls through to the rest of the table.

The optional dense retrieval route, off by default, uses the same install. See
[the dense route write-up](dense_route.md).

---

## Reproducing our results

```bash
python3 -m evaluator.local_evaluator
```

No environment variables, no network, no arguments. This is how the official harness
constructs the agent, as a plain `Agent(catalog_path)`. It writes per-session results
and aggregate metrics to `results.json` and prints the summary. A full run takes about
35 seconds with the semantic evidence feature available, 22 without it.

Expected output:

```json
{
  "sample_count": 200,
  "hit_rate_at_10": 1.0,
  "mrr": 0.938657,
  "mttc": 2.815,
  "efficiency": 0.8185,
  "recommended_technical_score": 0.945297,
  "reported_token_usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 }
}
```

The run is deterministic. The agent has no randomness and the evaluator seeds its own,
so repeated runs produce byte-identical `results.json`.

---

## See one session end to end

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

---

## Reproducing the ablations

Each of these is a single command, and each reproduces a number quoted in the README:

```bash
SHOPPING_AGENT_SHORTLIST=0 python3 -m evaluator.local_evaluator   # 0.885293
SHOPPING_AGENT_CATFILTER=0 python3 -m evaluator.local_evaluator   # 0.942229
SHOPPING_AGENT_RERANK=0    python3 -m evaluator.local_evaluator
SHOPPING_AGENT_CLARIFY=0   python3 -m evaluator.local_evaluator
```

The harnesses that produced the sweeps in `experiments/` live in `scripts/` and run
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

## Tests

```bash
python3 -m unittest discover -s tests -t . -q
```

223 tests, standard library only, about 3 seconds. The semantic feature's tests run
against a synthetic three-dimensional artifact and an injected encoder, so the suite
never loads a model or touches the bundled 66 MB one.

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

## Configuration flags

Every behavioural choice can be switched in one step, for auditing. The defaults are
what the official run uses.

| variable | default | effect |
|---|---|---|
| `SHOPPING_AGENT_SHORTLIST` | `1` | confidence-sized shortlist. `0` gives always-ten, measuring 0.885293 |
| `SHOPPING_AGENT_CATFILTER` | `1` | retrieve inside the stated category. `0` gives catalogue-wide, measuring 0.942229 |
| `SHOPPING_AGENT_SEMANTIC` | `1` | semantic evidence. `0` (or no `pip install -r requirements.txt`) measures the same 0.945297 here, and 0.705561 instead of 0.875897 under paraphrase |
| `SHOPPING_AGENT_DENSE` | `0` | enable the dense semantic route |
| `SHOPPING_AGENT_FUSION` | `weighted` | route blend: `weighted` or `rrf` |
| `SHOPPING_AGENT_RERANK` | `1` | deterministic reranker |
| `SHOPPING_AGENT_CLARIFY` | `1` | ask clarification questions |
| `SHOPPING_AGENT_REPEAT` | `0` | re-ask a productive attribute (measured at −0.0078) |
| `SHOPPING_AGENT_WILDCARD` | `1` | allow the `other` wildcard question |
| `SHOPPING_AGENT_DISAGREEMENT` | `1` | use candidate disagreement in question choice |
| `SHOPPING_AGENT_BLOCK_SOFT` | `0` | block soft slots from being asked |
| `SHOPPING_AGENT_PARAPHRASE_SHORTLIST` | `0` | what to return once the customer has disclosed and no candidate owns any of it. `10` restores the pre-E13 always-ten. Fires 0/563 verbatim turns, so no value changes the public score |
| `SHOPPING_AGENT_WILDCARD_CAP` | `3` | most consecutive `other` questions allowed under paraphrase (E14). Measured identical to uncapped on all seven probes |
| `SHOPPING_AGENT_REPEAT_WILDCARD` | `0` | `1` exempts `other` from the no-repeat rule even when the customer is quoting. Measurement arm only: ungated it costs 0.002264 of public score |
| `SHOPPING_AGENT_SEMANTIC_WEIGHT` | `192` | measurement override for the semantic feature's weight (E12 re-swept 96-1536; 192 wins) |

---

## Module map

The pipeline diagram is in the [README](../README.md#architecture); this is what each
module owns.

| module | responsibility |
|---|---|
| `starter/agent.py` | the official `Agent` entry point: FTS5 index, retrieval wiring, response assembly |
| `shopping_agent/semantic_evidence.py` | scores a disclosure against a candidate's field values by meaning; optional by construction |
| `shopping_agent/voyage_compat.py` | loads voyage-4-nano under transformers 5.x, one shim in one place |
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

---

## Repository layout

```text
starter/agent.py                  the official Agent entry point
shopping_agent/                   the system (see the architecture table in the README)
evaluator/local_evaluator.py      organiser-provided public-set simulator and scorer
tests/                            196 unit tests
scripts/demo_session.py           print one multi-turn session as a transcript
scripts/                          ablation harnesses, weight sweeps, audits, rank replay
docs/SETUP.md                     this file
docs/dense_route.md               the dense semantic route, and why it was removed
docs/experiments/E1..E11.md       eleven decision records: every measurement, including
                                  the ideas measured and rejected
docs/competition_specification.md the rules and evaluation protocol
docs/agent_api_contract.json      the machine-readable Agent contract
data/public_set.jsonl             200 labelled development sessions
data/catalog.jsonl                50,000 products (download separately, see above)
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

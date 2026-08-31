# E11 - picking an embedding model for the job that is actually open

Decision record for the model-selection task Phase 7 handed on. Three models
were to be compared -- MiniLM, BGE and Voyage -- and the instruction that
shaped the work was to compare them **on the right job**.

```yaml
experiment_id: "E11"
phase: "P7 (semantic evidence)"
hypothesis: "Retrieval recall is saturated, so comparing embedding models as the dense route measures nothing. The open job is scoring a paraphrased sentence against a product's field values, which is what constraint_evidence does with token containment and does badly once the customer stops quoting."
base_commit: "bc6b373"
dataset: "full public set (200 labeled sessions), verbatim and replayed through the paraphrasing customer"
overall_metrics:
  submission_unchanged: {hit_rate_at_10: 1.000, mrr: 0.938657, mttc: 2.805, technical_score: 0.945497}
  paraphrased_before:       {hit_rate_at_10: 0.805, technical_score: 0.696015}
  paraphrased_after_minilm: {hit_rate_at_10: 0.905, technical_score: 0.799906}
  paraphrased_after_bge:    {hit_rate_at_10: 0.925, technical_score: 0.825260}
  paraphrased_after_voyage: {hit_rate_at_10: 0.975, technical_score: 0.862054}
model_api: {model: "voyageai/voyage-4-nano (Apache-2.0 open weights, run locally)", network_required: false, prompt_tokens: 0, completion_tokens: 0, api_cost_usd: 0}
known_regressions: []
decision: "voyage-4-nano at 256 dimensions, int8, is the recommended model. NOTHING SHIPPED -- the change edits the reranking table and is held for review."
```

## Why the obvious comparison was not the comparison

E10 measured pool recall under paraphrase at **1.000 of sessions**: the target
is already in the room in 200 of 200 paraphrased sessions and is being ranked
28th out of 400. Nothing downstream of retrieval is short of candidates, so
swapping the model behind the dense route optimises a number that cannot go
up.

That prediction was tested rather than assumed. All three arms had the dense
route **verified firing** -- 550 dense queries at level 0, 836-839 at level 2,
220,000-335,600 returned candidates -- so none of them is the lexical-only
control that E1's correction history records being mistaken for a real result.

| dense route | level 0 (public) | HitRate | level 2 (paraphrased) | HitRate |
|---|---|---|---|---|
| **off (shipped)** | **0.945497** | **1.000** | 0.696015 | 0.805 |
| MiniLM | 0.944254 | 0.995 | 0.697829 | 0.805 |
| bge-small-en-v1.5 | 0.944004 | 0.995 | 0.691558 | 0.795 |

The arms span 0.0015 on the public set and 0.0063 under paraphrase. MiniLM's
row reproduces E10's dense-on figures to six decimals, which is what licenses
the BGE row beside it. **E1's verdict survives re-testing at this
configuration** -- MiniLM beats bge-small by +0.000250 and +0.006271 -- and so
does E10's: neither is worth the flip, both cost the headline HitRate 1.000,
and this comparison could not have separated the models on anything that
matters. Voyage was not built for this arm, because a third row within 0.006
of the other two would not have changed the conclusion and the artifact costs
100 minutes.

## The job that is open

`constraint_evidence` carries rerank weight 12.0 and asks how many of the
disclosure's content tokens appear *anywhere* in the candidate's text. E10
found that it does not fail quiet: it scores partial overlap, so rewording
fills it with whichever words survived rather than silencing it, and **zeroing
it makes the paraphrased score go up**, 0.696015 -> 0.711681.

So the bar for a replacement is not the shipped score, it is the deletion:

| | level 0 | level 2 |
|---|---|---|
| shipped | 0.945497 | 0.696015 |
| `constraint_evidence` -> 0 | 0.945297 | **0.711681** |

Anything below 0.711681 under paraphrase is worse than deleting the feature
outright, and that is the reference used throughout.

### What replaced it

Same feature, same weighting, different matching rule:

    value(p) = sum_d w_d * max_v cos(emb(d), emb(v)) / sum_d w_d

over the disclosures `d` and the candidate's own **card values** `v`. Two
choices, both inherited rather than invented:

* **Card values, not product text.** `slot_evidence` (weight 16, the table's
  strongest feature) already asks whether a candidate *owns* the disclosed
  string as a whole field value, and E8 showed that granularity is what makes
  it sharp. This asks the same structural question with a semantic matcher, so
  it degrades exactly where slot ownership goes silent rather than falling
  back to bag-of-words.
* **`w_d` is the disclosure's distinct content-token count**, precisely the
  weight `evidence.coverage_from_sets` already uses, so the two features price
  each disclosure identically and only the matching rule moves.

The artifact is one vector per **distinct** card value. The 50,000 catalogue
rows carry 615,776 value slots but only **268,564 distinct strings** (median
47 characters), because values like "Imported" recur across thousands of rows;
deduplicating removes 56% of the encoding work.

## The controls

1. **Guardrails.** `local_evaluator` 0.945497 / HitRate 1.000,
   `paraphrase_eval --level 0` 0.945497 exactly, 178 tests.
2. **Harness inertness.** The probe wraps `rerank`, and at `--weight 0
   --keep-token-feature` that wrapper must contribute nothing. It reproduces
   **0.696015** at level 2 and **0.945497** at level 0 to six decimals, so
   every delta here is the feature and not the instrument.
3. **Artifact correctness.** All 615,776 card-value slots resolve into the
   index with none silently dropped; every row is unit-norm; a verbatim value
   scores **1.0000** against the product that owns it under MiniLM, which
   exercises encoding and index alignment together.

That third check reads differently per model, and the difference is the
`QUERY_PREFIX` trap rather than a defect: MiniLM is symmetric and self-scores
1.0000, BGE puts an instruction on the query side only and self-scores 0.9512,
and Voyage carries distinct `query`/`document` prompts and self-scores 0.7601.
Only the ordering matters, but a run whose prefixes were paired wrongly would
look like a working agent that merely scores worse.

A fourth check was forced by a surprise. The pool matmul raises `invalid
value` and `divide by zero` warnings on this machine. Rather than assume they
were spurious, the output was compared against a float64 `einsum` reference
over 2993 real pools: **0 non-finite entries, maximum deviation 3.9e-07**. It
is Apple Accelerate leaking floating-point status flags out of BLAS. It is
silenced at the call site, which also matters because a host that escalates
warnings to errors would otherwise push the reranker onto its fallback path
and quietly serve the fused order.

## The weight is not transferable between models

Swept alone, at level 2:

| weight | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 96 | 192 | 384 | 768 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MiniLM | .756375 | .775422 | .781258 | .777361 | **.799906** | .799249 | .794247 | .789918 | - | - | - |
| BGE | - | - | .780033 | - | .796683 | - | .813199 | **.825260** | .825686 | - | - |
| Voyage | - | - | - | - | .834479 | - | - | .856548 | **.862054** | .858238 | .856843 |

Each model plateaus in a different place -- MiniLM at 24-32, BGE at 96-192,
Voyage at 192-768 -- and the reason is visible in the raw cosines. BGE's
similarities sit in a compressed band (0.58-0.73 on a sampled probe) against
MiniLM's (0.27-0.59), and Voyage's asymmetric prompts shift its scale again.
The same weight therefore buys very different amounts of separation.
**Comparing all three at one fixed weight would have understated BGE by 0.028
and Voyage by 0.028**, each larger than the entire dense-route comparison
above. Every model is reported at the first point of its own plateau.

## Result

| | level 0 | L0 HitRate | level 2 | L2 HitRate |
|---|---|---|---|---|
| shipped | **0.945497** | 1.000 | 0.696015 | 0.805 |
| delete `constraint_evidence` | 0.945297 | 1.000 | 0.711681 | 0.825 |
| replace -- MiniLM @24 | 0.945297 | 1.000 | 0.799906 | 0.905 |
| replace -- BGE @96 | 0.945297 | 1.000 | 0.825260 | 0.925 |
| replace -- Voyage @192 | 0.944430 | 1.000 | **0.862054** | **0.975** |
| additive -- MiniLM @24 | **0.945497** | 1.000 | 0.785969 | 0.890 |
| additive -- BGE @96 | **0.945497** | 1.000 | 0.817056 | 0.920 |
| additive -- Voyage @192 | 0.944630 | 1.000 | 0.853617 | 0.965 |

Three properties are worth naming.

**For MiniLM and BGE the public set is indifferent.** In the replace shape the
score is 0.945297 at *every* weight from 12 to 48 -- flat to six decimals --
and the 0.0002 given up is the deletion of the token feature, not the addition
of the semantic one. In the additive shape they reproduce **0.945497 exactly**.
Once the category filter and slot ownership have run, these two are only
reordering candidates that were already tied.

**Voyage is not indifferent, and lowering its weight does not help.** It costs
about 0.00087 on the public set (MRR 0.938657 -> 0.936101) and costs the same
0.00087 at weight 24, 96 and 192 alike, so this is a fixed set of verbatim
ties it reorders rather than a tuning artifact. HitRate 1.000 is held in every
row of the table.

**Voyage wins the job by a clear margin** -- +0.037 over BGE and +0.062 over
MiniLM under paraphrase, and it is the only model that lifts paraphrased
HitRate to 0.975 against the shipped 0.805.

### The held-out confirmation, and its limit

The house rule is that a gain measured against `paraphrase_eval`'s hand-built
synonym list must be confirmed where that list does not reach. E10 used
`--paraphrase-category`, which is the right probe for the category fix -- but
it substitutes the *category's* words and leaves the disclosure vocabulary
exactly as the original lexicon left it, so it is **not** a held-out test of a
feature that scores disclosures. That gap is closed by
`scripts/heldout_lexicon.py`: 40 substitutions built by the same method as the
original (measured top content words across all 800 constraint strings) but
restricted to words the original does not contain, asserted disjoint at
install time.

| level 2 | `--paraphrase-category` | held-out lexicon | level 3 |
|---|---|---|---|
| shipped | 0.627500 | 0.858798 | 0.673673 |
| MiniLM @24 | 0.748438 | 0.905252 | 0.772506 |
| BGE @96 | 0.760759 | 0.904917 | - |
| Voyage @192 | **0.813576** | 0.904880 | **0.827073** |

The gain transfers: every model gains about +0.046 on vocabulary none of their
weights were swept against. **But that probe cannot rank the models**, and
saying so is the honest reading -- all three land within 0.0004 of each other
at HitRate 0.99, against a shipped baseline already at 0.98. The held-out
lexicon rewrites 10.8% of content-word token mass against the original's
29.4%, so it is a much easier task and everything saturates near the ceiling.
It establishes that the mechanism generalises; it does not establish which
model is better. Model separation rests on the two harder probes, where the
ordering Voyage > BGE > MiniLM is consistent.

### It is ranking that moved, as E10 said it would

Re-running E10's rank instrument, which reproduces its shipped row exactly
(median 28, top-10 30.5%, 0.696015):

| level 2 | pool recall | target rank median | at rank 1 | in top 10 |
|---|---|---|---|---|
| shipped | 1.000 | 28 | 14.8% | 30.5% |
| MiniLM @24 | 1.000 | 17 | 23.7% | 41.8% |
| BGE @96 | 1.000 | 15 | 26.8% | 45.0% |
| Voyage @192 | 1.000 | **7** | **31.3%** | **54.9%** |

Retrieval is untouched and pool recall stays 1.000, so this is ranking
recovery and nothing else -- which is what E10 said the entire remaining gap
was. The verbatim run's median is 2, so even Voyage's 7 leaves real headroom.

## Cost, and the size question that decides shippability

Builds are one-time and offline. All three models are **local weights** with
no network dependency, no API key and no per-token cost at runtime;
`voyageai/voyage-4-nano` is Apache-2.0 open weights on the Hugging Face Hub
(340M parameters), so the Model and API Policy's disclosure obligations are
met by the same "no network, 0 tokens, $0" statement the rest of the project
makes. Its published remote code targets transformers 4.51 and needs a two-
point shim to load on 5.x; the shim is in `scripts/build_value_embeddings.py`
and was verified against the model card's own retrieval example, against
padding invariance, and against transformers' native `create_bidirectional_mask`.

Per-turn cost, measured with the machine otherwise idle, against a clean
baseline of **38.9 ms per scored turn**:

| | build | query encode | pool score (400 candidates) | added per turn |
|---|---|---|---|---|
| MiniLM 384 | 3.9 min | 5.2 ms | 4.18 ms | ~9 ms |
| bge-small 384 | 16.1 min | 8.7 ms | 4.17 ms | ~13 ms |
| voyage 2048 | 99.8 min | 16.9 ms | 6.83 ms | ~24 ms |
| voyage 256 | 99.8 min | 16.9 ms | 3.37 ms | ~20 ms |

**Artifact size was expected to be the blocker, and Voyage is the only model
that clears it.** At 268,564 rows, MiniLM and BGE are 393 MB at float32 --
197 MB at float16, 103 MB at int8 -- so no form of either fits GitHub's 100 MB
per-file limit. Voyage is Matryoshka-trained and quantization-aware trained,
and both properties pay:

| voyage dims | level 2 | HitRate | float32 | int8 |
|---|---|---|---|---|
| 2048 | 0.862054 | 0.975 | 2098 MB | 550 MB |
| 1024 | 0.854086 | 0.970 | 1049 MB | 275 MB |
| 512 | 0.860656 | 0.980 | 524 MB | 137 MB |
| **256** | 0.839229 | 0.965 | 262 MB | **66 MB** |

512 keeps essentially the full 2048-dimension quality (-0.0014). 256 gives up
0.023 and **still beats BGE and MiniLM at their full native size**. The 1024
row sitting below 512 is a genuine non-monotonicity rather than a typo; the
spread across 512-2048 is 0.008 and none of these were re-swept for weight, so
the three are best read as one plateau.

Quantizing the 256-dimension artifact to int8 was then measured rather than
assumed: **65.6 MB, level 2 score 0.839235 against float32's 0.839229**, a
difference of six millionths, with mean cosine to the float32 original
0.999929. So a 66 MB artifact that comfortably fits the repository is
available at no measurable cost, and only because this model was trained for
it.

## Measured and rejected

**Pool centering.** `slot_evidence` prices each disclosure by how rare it is
*in the pool*, on the reasoning that a value every candidate owns orders
nothing. The obvious semantic analogue is to subtract the pool median cosine
per disclosure. It is worse at every weight tried: 0.774009 / 0.775359 /
0.767351 at 24 / 48 / 96 against 0.799906 uncentered. Centering discards the
absolute similarity level, and that level is itself evidence -- a candidate
whose values are all close to the disclosure is genuinely a better match.

**The `lexical_rank` lever, which this subsumes.** Phase 7 priced
`lexical_rank` 1.0 -> 4.0 at +0.0320 paraphrased and left it unshipped. It
reproduces here at 0.727978. Stacked with the semantic feature it *costs*:

| level 2 | score |
|---|---|
| `lexical_rank` 4.0 alone | 0.727978 |
| semantic (MiniLM @24) alone | **0.799906** |
| both together | 0.796689 |

The two overlap and the semantic feature dominates, so the recommendation
Phase 7 left open should be closed as **not taken** -- the same shape as E6's
`constraint_evidence` being subsumed by slot ownership one phase later.

## Recommendation, and what is deliberately not decided

**Model: `voyage-4-nano`, truncated to 256 dimensions and stored int8.** It
wins the job it was tested on by 0.037 over the next model, it is the only one
whose artifact fits the repository at all, and the truncation and quantization
that make it fit cost 0.023 and 0.000006 respectively. Its price is about
0.00087 of public score and about 20 ms per turn.

**Shape: undecided, and it is a real choice.** Replace gives up 0.0002-0.0009
of public score for 0.008-0.014 more paraphrase robustness; additive gives up
none at all for MiniLM and BGE. For Voyage the distinction mostly disappears,
because its 0.00087 is paid either way.

Nothing is shipped from this record. The change edits the reranking table,
which this project does not do without review, and it adds a runtime embedding
dependency to a submission whose default path is currently stdlib-only,
offline and zero-token. That last point is not a scoring question and the
composite cannot see it, so it is not this record's to settle.

The break-even framing the project uses elsewhere applies and is stark: at
0.00087 public cost against +0.166 paraphrased, the change pays for itself if
the private set paraphrases with probability above roughly **0.5%**. Against
E10's 41% for the dense flip, that is a different kind of bet -- but it is a
bet on score alone, and it deliberately excludes the dependency cost.

## What is left

* The median target still sits at rank 7 under paraphrase against 2 verbatim,
  and E10's perfect-reranker oracle at 0.990300 is unchanged because retrieval
  is untouched. Most of the gap E10 priced is still open.
* The held-out lexicon is too easy to separate models. A harder disjoint
  paraphraser would be worth building before trusting any future comparison
  between two models this close.
* `slot_evidence` (weight 16) is still whole-value string equality and is the
  larger of the two exact features. The same substitution could be applied to
  it, and E10 listed relaxing it as an open option.

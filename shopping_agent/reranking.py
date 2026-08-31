"""P4-T1: the deterministic final scorer.

Retrieval hands over a pool of candidates carrying each route's rank and
score. This module puts them in final order using an explicit, fixed feature
checklist. P4 specifies these six features, in this stated priority order:

    1. hard-constraint satisfaction
    2. category compatibility
    3. lexical route rank
    4. dense route rank
    5. metadata compatibility
    6. soft-preference matches

All six are scored. **The stated priority is not the priority used**, and that
is deliberate: weighting them in this order measured 0.047 composite *worse*
than not reranking at all (E4). Retrieval rank now leads and the constraint
features act as adjustments beneath it. See FEATURE_WEIGHTS for the numbers
and the reasoning.

Every feature contributes ``weight * value`` with ``value`` in 0-1, and every
contribution is recorded on the result. That is the point of the phase's
acceptance criterion -- any candidate's placement can be read off rather than
guessed at, so a bad ranking can be traced to the feature that caused it. It
is also how the weights below were diagnosed: the first version ranked worse
than no reranking at all, and the breakdown said which feature was doing it.

No model, no network, no learned parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import evidence
from . import slots
from .catalog import ProductRecord
from .contracts import Candidate, Constraint
from .filtering import (
    SCORED_SEPARATELY,
    evaluate_price,
    score_category,
    soft_budget_closeness,
)

# MEASURED, and not what the checklist order suggests. Weighting the features
# in P4's stated priority order -- hard_constraints 4.0 down to
# soft_preferences 0.25 -- *lost* 0.047 composite against the fused ordering it
# replaces, and lost all of it in MRR (0.329 against 0.475). It found the same
# products and ranked them worse, because matched_constraints() is a coarse
# word-containment check and at weight 4.0 it swamped the retrieval score
# margin that weighted fusion exists to preserve.
#
# These weights lead with retrieval instead and win by +0.0059 (E4). The
# checklist order still holds among the *adjustment* features -- hard
# constraints outrank category, which outranks metadata, which outranks soft
# preferences -- but retrieval rank outranks all of them, which the specified
# order does not say and the measurement does.
#
# MEASURED again after the category double-count was removed. That bug --
# matched_constraints() reporting `category` alongside the dedicated category
# feature -- had made category's effective weight ~1.5 rather than the stated
# 0.5, so E4 tuned every other weight against an inflated value. Re-sweeping
# category alone, with the rest held at E4's settings:
#
#   weight   0.0      0.5      1.0      1.5      2.0      3.0
#   MRR      0.4719   0.4669   0.4637   0.4572   0.4551   0.4396
#   score    0.6367   0.6320   0.6339   0.6317   0.6214   0.6170
#
# MRR falls monotonically as category gains weight, and the composite is
# highest at zero. The reason is structural rather than a tuning accident:
# retrieval already applies score_category() as a boost when it builds the
# pool (FUSED_BOOST_SCALE in retrieval.py), so the reranker's copy re-applies
# a signal the ranking it is reordering has *already* accounted for.
#
# Kept in the table at 0.0 rather than deleted, so the contribution still
# appears in explain() and the checklist stays six features long -- P4-T1's
# acceptance is that every feature is inspectable, and "measured to be worth
# nothing here" is a more useful thing to be able to read off than silence.
FEATURE_WEIGHTS: dict[str, float] = {
    "hard_constraints": 1.0,
    "category": 0.0,
    # MEASURED again after MIN_EVIDENCE_TOKENS dropped to 1 (E6): with the
    # evidence feature now firing from turn 1, retrieval rank only breaks its
    # ties, and halving it against evidence is worth +0.0027 composite.
    # Doubling evidence to 24 instead lands within 0.0002 of the same point
    # (0.752349 vs 0.752474), and once lexical_rank is 1.0 raising evidence
    # to 24 or 48 changes nothing to six decimals -- the evidence:rank ratio
    # saturates. dense_rank keeps its E4 value: it is dead by default (dense
    # off) and live only under SHOPPING_AGENT_DENSE=1, where this ratio was
    # never measured.
    "lexical_rank": 1.0,
    "dense_rank": 2.0,
    "metadata": 0.25,
    # MEASURED (E6): 0.1 was tuned when evidence ignored short labels; with
    # min-tokens at 1 the soft-preference share adds +0.0009 MRR-only on a
    # plateau flat from 2 through 8. First point of the plateau.
    "soft_preferences": 2.0,
    # P5-T3, and the largest single feature in the table by a wide margin.
    #
    # MEASURED (E5), dense route off, sweeping this weight alone:
    #
    #   weight  0.0      2.0      4.0      8.0      12.0     16.0+
    #   MRR     0.5127   0.5263   0.5438   0.5479   0.5506   0.5506
    #   score   0.6876   0.6960   0.7017   0.7057   0.7065   0.7065
    #
    # Flat to six decimals from 12 upward: by then coverage of the customer's
    # quoted constraints decides the order outright and the rest of the table
    # only breaks ties. 12.0 is the first point of that plateau.
    #
    # A high weight is safe rather than reckless here. If a disclosure matches
    # nothing -- a different hidden evaluator, a less verbatim customer -- the
    # feature scores ~0 for every candidate and the ordering falls back to the
    # features beneath it. The failure mode is silence, not noise.
    "constraint_evidence": 12.0,
    # E7, the second half of the evidence story. Token coverage ties at ~1.0
    # across near-duplicate catalogue copy, and the enlarged pool (RERANK_POOL
    # 100) admits more such impostors; contiguous containment of the quoted
    # constraint separates the product the customer is actually quoting from
    # one that merely shares its vocabulary.
    #
    # MEASURED (E7), at pool depth 100, sweeping this weight alone:
    #
    #   weight  0.0       2.0       6.0       12.0
    #   score   0.766930  0.798306  0.798916  0.798291
    #
    # A plateau spanning 2-12 with a spread of 0.0006; 6.0 is its centre and
    # best point. Worth +0.032 composite over depth alone, +0.0335 at the old
    # depth -- the two changes are independently real. Like token coverage,
    # the feature fails quiet: no contained phrase scores 0.0 for everyone.
    #
    # RE-MEASURED (E8) after slot_evidence landed, and **deliberately not
    # moved to its new optimum**:
    #
    #   weight  0.0       2.0       6.0       10.0      16.0
    #   score   0.854130  0.853005  0.853005  0.853005  0.852855
    #
    # Zero now measures +0.001125 better. Slot ownership subsumes this
    # feature and is strictly sharper: where the two disagree, phrase
    # containment is crediting an impostor that carries the sentence inside a
    # longer bullet, which is exactly the false positive ownership was built
    # to kill.
    #
    # Kept at 6.0 anyway, because this is the graceful-degradation layer and
    # the trade is lopsided. Slot ownership assumes the customer quotes card
    # values verbatim. If a private set paraphrases, ownership goes silent for
    # every candidate and the order falls to whatever is beneath it -- with
    # this at 6.0 that is P5's ranking, and with it at 0.0 it is P5's ranking
    # minus the +0.032 E7 measured this feature to be worth. Paying 0.0011
    # certain to insure 0.032 contingent is worth it at any plausible odds.
    #
    # RE-MEASURED once more at the final P6 configuration, and the 0.0011 is
    # now gone too:
    #
    #   weight  0.0       3.0       6.0       12.0
    #   score   0.929426  0.929426  0.929426  0.929426
    #
    # Identical to six decimals across the range. category_exact separates
    # the near-duplicates this feature used to disagree with slot ownership
    # about, so it no longer decides anything either way on this set. The
    # insurance is free, which settles the question that the earlier version
    # of this comment had to argue.
    "phrase_evidence": 6.0,
    # P6-T1, and now the table's dominant feature -- it decides the order and
    # everything above it breaks ties, which is the same shape constraint_
    # evidence had in P5 and for the same reason: it answers a sharper
    # question than the features it displaces.
    #
    # E7 concluded that text similarity was exhausted and only a *non-text*
    # discriminator could separate the surviving misses. That conclusion was
    # right about the evidence features and wrong about text: what was
    # missing was not a different signal but a structural question about the
    # same one -- does this candidate own the disclosed sentence as a whole
    # field value of its own, rather than merely contain it somewhere? See
    # shopping_agent/slots.py.
    #
    # MEASURED (E8), sweeping this weight alone at RERANK_POOL 250:
    #
    #   weight  0.0       4.0       8.0       16.0      24.0      32.0      48.0
    #   score   0.821381  0.850805  0.852755  0.853005  0.853005  0.853005  0.853005
    #   HitRate 0.950     0.975     0.975     0.975     0.975     0.975     0.975
    #   MRR     0.659603  0.690685  0.697185  0.698018  0.698018  0.698018  0.698018
    #
    # Flat to six decimals from 16 upward: by then slot ownership decides
    # outright. 16.0 is the first point of that plateau. Worth +0.0316
    # composite, and it is the only change in the project's history to move
    # all three metrics at once -- HitRate 0.950 -> 0.975, MRR +0.038,
    # MTTC 3.575 -> 3.195.
    #
    # Safe at a high weight for the reason constraint_evidence is: if the
    # customer paraphrases instead of quoting, no candidate owns anything,
    # every candidate scores 0.0, and the table beneath decides. The failure
    # mode is silence, not noise.
    "slot_evidence": 16.0,
    # P6-T3. The `category` feature above measures 0.0 and this one is worth
    # +0.0475, and they are nominally about the same thing. The difference is
    # exactness, and it is the whole story.
    #
    # `category` asks whether the candidate's text overlaps a category *word*
    # the slot extractor recognized ("shoes", "boots"). Retrieval already
    # applies that same boost when building the pool, so re-applying it
    # reorders nothing -- which is why it measured flat and stayed at 0.0.
    #
    # This one asks whether the candidate reproduces the exact category string
    # the customer was handed. The opening line states
    # coarse_category(target.categories) verbatim, in every scenario, on turn
    # 1 -- and for Browsing it is the *only* thing ever stated before a
    # question is answered. A median of just 38% of the 250-candidate pool
    # reproduces it, so agreement removes about three fifths of the field for
    # free, from the first turn, in all 200 sessions.
    #
    # MEASURED (E8), sweeping this weight alone:
    #
    #   weight  0.0       1.0       2.0       4.0       8.0       16.0
    #   score   0.881931  0.926451  0.926651  0.929026  0.929426  0.929426
    #   HitRate 0.965     0.990     0.990     0.990     0.990     0.990
    #   MRR     0.836770  0.906171  0.906171  0.912421  0.912421  0.912421
    #   MTTC    3.580     3.020     3.010     2.985     2.965     2.965
    #
    # Flat to six decimals from 8 upward; 8.0 is the first point of that
    # plateau. Misses fall from 7 to 2, and all three metrics move together.
    #
    # Fails quiet like the evidence features: a customer who never states a
    # category, or states one this catalogue cannot reproduce, leaves
    # stated_category None and the feature scores 0.0 for every candidate.
    #
    # SUPERSEDED as a live feature by P6-T7 (E9), which applies the same exact
    # test at retrieval instead. Every pooled candidate now reproduces the
    # stated category, so this scores 1.0 across the board and orders nothing:
    #
    #   weight  0.0       4.0       8.0       16.0
    #   score   0.945497  0.945497  0.945497  0.945497
    #
    # Kept anyway, for the reason phrase_evidence was kept in E8 rather than
    # the reason a tuned value is kept: it is the graceful-degradation layer
    # for its own filter. The filter stands down whenever the restricted
    # search returns nothing, and on that path the pool is catalogue-wide
    # again and this is exactly the feature that used to be worth +0.0475.
    # Inert here costs nothing; absent there would cost that.
    "category_exact": 8.0,
}

# P7-T2: the weights used instead when the customer is paraphrasing.
#
# The table above is tuned for a customer who quotes. Four of its five
# heaviest features ask an exact-match question -- does this candidate *own*
# this whole field value, does it *contain* this phrase, does it reproduce
# this category string -- and the public simulator answers yes because it
# builds its messages out of the target's own text. Reword the same messages
# and those four stop discriminating, but they do not all fail the same way:
#
#   slot_evidence (16)        mostly silent; whole-value identity
#   constraint_evidence (12)  scores partial token overlap, so a rewording
#                             fills it with NOISE rather than silence
#   category_exact (8)        inert once the E9 filter engages
#   phrase_evidence (6)       silent to six decimals (E10)
#   lexical_rank (1)          still carries real information
#
# So the paraphrased ranking is decided by the one feature holding the
# smallest weight in the table, while the largest weights are split between
# saying nothing and saying something wrong. E10 measured the target's median
# rank under paraphrase at 28 against 2 verbatim, with pool recall 1.000 --
# the product is in the pool and the panel is misreading it.
#
# E10 priced two unconditional repairs: lexical_rank 1.0 -> 4.0 at -0.00075
# public / +0.0320 paraphrased, and constraint_evidence -> 0 at -0.0002 /
# +0.0157. They overlap rather than add -- together they measured 0.7227,
# *worse* than lexical_rank alone at 0.7280 -- so at most one is worth taking.
#
# Applying either conditionally makes the public cost exactly zero rather
# than 0.00075, and that is not an estimate. The gate is the paraphrase
# predicate E9 already counted over the whole public set: disclosures on
# record that no pooled candidate owns fired on **0 of 2000 turns**
# (shopping_agent/shortlist.py). A profile behind that gate cannot move a
# score it is never applied to.
#
# Only keys already present in FEATURE_WEIGHTS may appear here; see
# weights_for().
PARAPHRASE_WEIGHTS: dict[str, float] = {
    "lexical_rank": 4.0,
}

# P7-T3: whether slot ownership falls back to a graded match under the regime.
#
# `slot_evidence` is the table's dominant feature at 16.0 and it asks an
# all-or-nothing question: does the candidate hold this disclosed string as a
# whole field value of its own? Reword one word of a nine-word value and the
# answer is no, scored identically to a wholly unrelated product.
#
# Worse, the silence compounds. `slots.ownership_weights` gives weight 0.0 to
# any disclosure *no* pooled candidate owns, on the sound reasoning that such
# a disclosure is silence rather than evidence. Under a rewording that is
# every disclosure, so the weights are all zero, the denominator in
# _slot_value is zero, and the feature returns 0.0 for the whole pool. The
# table's heaviest feature is not merely blunted, it is switched off.
#
# So relaxing the match alone is not enough: the selectivity weights have to
# be recomputed against the relaxed notion of ownership too, or there is
# nothing for the relaxed matches to be weighted by. prepare_evidence does
# both, together, and only once the regime has fired.
RELAXED_OWNERSHIP = True

# How much of a disclosed value a candidate must account for before it counts
# as owning it at all. Below this the match is treated as coincidence and
# scores nothing, so a candidate sharing one common word with a long quoted
# sentence gains no credit.
RELAXED_OWNERSHIP_THRESHOLD = 0.5


def weights_for(paraphrase_regime: bool) -> dict[str, float]:
    """The feature weights to score with this turn.

    Returns FEATURE_WEIGHTS itself -- not a copy -- when the regime has not
    fired, so the quoting path allocates nothing and stays byte-identical.
    """
    if not paraphrase_regime:
        return FEATURE_WEIGHTS
    merged = dict(FEATURE_WEIGHTS)
    merged.update(PARAPHRASE_WEIGHTS)
    return merged


# Guards the invariant weights_for() relies on: an override naming a feature
# the checklist does not carry would silently add a contribution in one regime
# and not the other, and explain() would disagree between turns.
assert set(PARAPHRASE_WEIGHTS) <= set(FEATURE_WEIGHTS), (
    "PARAPHRASE_WEIGHTS may only override features FEATURE_WEIGHTS defines"
)

# Converts a 1-based route rank to a 0-1 value.
#
# MEASURED: this was 60, mirroring the RRF constant, and that was the larger
# half of the mistake above. Over a fifty-candidate pool it spans only 1.00 to
# 0.55, so the rank features could not discriminate against features spanning
# the full 0-1. At 5 the span is 1.00 to 0.09. Sharpening this alone recovered
# most of the lost MRR (0.441) even under the original weights.
RANK_DECAY = 5.0


@dataclass(frozen=True)
class ScoreContribution:
    """One feature's input to a candidate's final score."""

    feature: str
    weight: float
    value: float

    @property
    def contribution(self) -> float:
        return self.weight * self.value


@dataclass(frozen=True)
class RerankedCandidate:
    """A scored candidate whose score can be taken apart."""

    parent_asin: str
    score: float
    contributions: tuple[ScoreContribution, ...]

    def explain(self) -> str:
        parts = ", ".join(
            f"{c.feature}={c.value:.3f}x{c.weight:g}={c.contribution:+.3f}"
            for c in self.contributions
        )
        return f"{self.parent_asin} score={self.score:.4f} [{parts}]"


def _rank_value(candidate: Candidate, route: str) -> float:
    """A route that never returned this candidate scores 0, not a poor rank --
    absence is not a weak endorsement."""
    rank = candidate.route_ranks.get(route)
    if rank is None:
        return 0.0
    return RANK_DECAY / (RANK_DECAY + rank - 1.0)


def _budget_value(product: ProductRecord, budget: Constraint | None) -> float:
    """Score price against the budget, honouring what kind of budget it is.

    A *hard* budget ("under $100") is a ceiling: within it or not.

    A *soft* budget ("around $100") is a target, and must be ranked by
    closeness the way P2 did. Collapsing it to a ceiling loses that ordering
    twice over -- $50 ties with $99 despite being half the asking price, and
    $101 falls off a cliff to zero despite being what the customer asked for.
    Closeness decays linearly with relative distance, so the target scores
    1.0, twice the target scores 0.0, and the boundary is smooth.
    """
    if budget is None:
        return 0.0
    if product.price is None:
        # Unverified, not compliant: an unpriced item must not outrank one
        # known to fit.
        return 0.25

    target = float(budget.value)
    if budget.strength == "hard":
        retained, _ = evaluate_price(product, target)
        return 1.0 if retained else 0.0

    return max(0.0, soft_budget_closeness(product.price, target))


def _slot_value(
    product: ProductRecord,
    slot_terms: list[tuple[str, float]],
    relaxed: bool = False,
) -> float:
    """Selectivity-weighted share of the disclosures this candidate owns.

    Returns 0-1. Nothing disclosed yet, or nothing any candidate owns, is a
    neutral 0.0 for every candidate alike -- the same quiet failure the other
    evidence features have, so a paraphrasing simulator costs the ordering
    nothing rather than corrupting it.
    """
    matched = 0.0
    total = 0.0
    owned = product.card_values
    for value, weight in slot_terms:
        total += weight
        if value in owned:
            matched += weight
        elif relaxed:
            # Credit in proportion to how nearly the value is owned, so a
            # near-miss cannot score like an identity. Below the threshold
            # nothing is added at all.
            overlap = slots.ownership_overlap(value, owned)
            if overlap >= RELAXED_OWNERSHIP_THRESHOLD:
                matched += weight * overlap
    if total <= 0.0:
        return 0.0
    return matched / total


@dataclass(frozen=True)
class DisclosureEvidence:
    """Everything the pool-level evidence pass produces, computed once.

    Reranking needs the normalized disclosures; the response needs to know how
    far they have narrowed the field. Both come out of the same single pass
    over the pool, so they are prepared together rather than twice.
    """

    tokens: list[frozenset[str]]
    phrases: list[tuple[str, int, bool]]
    slot_terms: list[tuple[str, float]]
    # Disclosures at least one pooled candidate owns. Zero means the customer
    # has said nothing yet, or is paraphrasing rather than quoting -- either
    # way there is no slot evidence to act on.
    live_disclosures: int
    # Candidates owning *every* live disclosure. One means the constraints
    # identify a single product, and because the target owns all of its own
    # disclosures that product is the target whenever the target is pooled.
    consistent: int
    # How many constraints the customer has stated at all, owned or not.
    # `live_disclosures` alone cannot separate "nobody owns what was said"
    # from "nothing has been said yet"; holding both is what makes the
    # predicate below a measurement rather than a certainty.
    disclosed: int = 0
    # Whether `slot_terms` were priced against the graded notion of ownership
    # rather than the exact one. Set only when the regime fired AND some
    # candidate cleared the threshold, so a turn where the relaxed pass also
    # found nothing behaves exactly as before.
    relaxed: bool = False

    @property
    def paraphrase_regime(self) -> bool:
        """Whether the customer appears to be rewording rather than quoting.

        The customer has stated constraints and not one pooled candidate owns
        any of them. On a quoting split that is close to impossible: the
        simulator builds its messages out of the target's own field values, so
        whenever the target is pooled it owns everything said. E9 counted this
        predicate over the whole public set and it fired on **0 of 2000
        turns**, which is what lets a weight profile behind it be inert by
        construction rather than by tuning.

        It is deliberately the same predicate shortlist.py already acts on, so
        the two policies cannot disagree about which regime a turn is in.
        """
        return self.disclosed > 0 and self.live_disclosures == 0


def prepare_evidence(
    disclosures: list[str],
    candidates: list[Candidate],
    products: dict[str, ProductRecord],
) -> DisclosureEvidence:
    """Normalize the disclosures once and price each by how rare it is here.

    One pass over the pool per disclosure, which is a frozenset membership
    test -- the same order of work the phrase feature already does, and far
    cheaper than its substring scan.
    """
    normalized = [
        value for value in (slots.normalize_disclosure(d) for d in disclosures) if value
    ]
    tokens = evidence.disclosure_token_sets(disclosures)
    phrases = evidence.disclosure_phrases(disclosures)
    pool = [
        products[c.parent_asin].card_values
        for c in candidates
        if c.parent_asin in products
    ]
    if not normalized or not pool:
        return DisclosureEvidence(tokens, phrases, [], 0, 0, len(normalized))

    counts = [sum(1 for owned in pool if value in owned) for value in normalized]
    weights = slots.ownership_weights(counts, len(pool))
    live = [value for value, count in zip(normalized, counts) if count > 0]
    consistent = (
        sum(1 for owned in pool if all(value in owned for value in live)) if live else 0
    )

    # The customer said things and not one pooled candidate owns any of them.
    # Re-price selectivity against the graded match, because the exact weights
    # are all zero here by construction and would leave the feature inert.
    relaxed = False
    if RELAXED_OWNERSHIP and not live:
        overlaps = [
            [slots.ownership_overlap(value, owned) for owned in pool]
            for value in normalized
        ]
        relaxed_counts = [
            sum(1 for score in row if score >= RELAXED_OWNERSHIP_THRESHOLD)
            for row in overlaps
        ]
        if any(relaxed_counts):
            weights = slots.ownership_weights(relaxed_counts, len(pool))
            relaxed = True

    # `live_disclosures` and `consistent` stay EXACT deliberately. They are
    # what shortlist.py and `paraphrase_regime` read, and deriving them from
    # the relaxed pass would let a successful relaxation report that the
    # customer is quoting after all -- switching off the weight profile that
    # enabled the relaxation, one turn after it started working.
    return DisclosureEvidence(
        tokens,
        phrases,
        list(zip(normalized, weights)),
        len(live),
        consistent,
        len(normalized),
        relaxed,
    )


def _share(matched: tuple[str, ...], constraints: dict[str, Constraint], strength: str) -> float:
    """Share of the constraints of this strength that the candidate matches.
    No constraints of that strength is neutral (0.0), not a free win.

    The denominator must exclude exactly what matched_constraints() excludes
    from the numerator; counting a separately-scored attribute here only
    would make a full match look partial."""
    total = sum(
        1 for a, c in constraints.items()
        if c.strength == strength and a not in SCORED_SEPARATELY
    )
    if total == 0:
        return 0.0
    return len(matched) / total


def score_candidate(
    candidate: Candidate,
    product: ProductRecord,
    constraints: dict[str, Constraint],
    disclosures: list[str] | None = None,
    *,
    disclosure_tokens: list[frozenset[str]] | None = None,
    disclosure_phrases: list[tuple[str, int, bool]] | None = None,
    slot_terms: list[tuple[str, float]] | None = None,
    stated_category: str | None = None,
    weights: dict[str, float] | None = None,
    relaxed_ownership: bool = False,
) -> RerankedCandidate:
    """Score one candidate against the feature checklist, in order.

    `disclosures` are the constraint sentences the customer has quoted back
    in answer to our questions. Omitted, the evidence features score 0 for
    every candidate and cannot reorder anything. rerank() normalizes them once
    for the whole pool and passes `disclosure_tokens`/`disclosure_phrases`
    instead; a direct caller can hand over the raw strings and pay one
    normalization."""
    if disclosure_tokens is None:
        disclosure_tokens = evidence.disclosure_token_sets(disclosures or [])
    if disclosure_phrases is None:
        disclosure_phrases = evidence.disclosure_phrases(disclosures or [])
    if slot_terms is None:
        # A direct caller has no pool to price selectivity against, so every
        # disclosure counts the same. rerank() computes the real weights once
        # for the whole pool and passes them in.
        slot_terms = [
            (normalized, 1.0)
            for normalized in (slots.normalize_disclosure(d) for d in (disclosures or []))
            if normalized
        ]
    category = constraints.get("category")
    if category is None:
        # A constraint the customer never gave earns no credit. Constant
        # across candidates either way, so this cannot reorder anything -- but
        # scoring it 0.2 made every explanation claim partial category credit
        # in sessions where category was never mentioned.
        category_value = 0.0
    else:
        category_boost, _reason = score_category(product, str(category.value))
        # score_category returns +2.0 match / 0.0 unverified / -0.5 mismatch;
        # rescale onto 0-1 so every feature value means the same thing.
        category_value = (category_boost + 0.5) / 2.5

    metadata_value = _budget_value(product, constraints.get("budget"))

    values = {
        "hard_constraints": _share(candidate.matched_hard_constraints, constraints, "hard"),
        "category": category_value,
        "lexical_rank": _rank_value(candidate, "lexical"),
        "dense_rank": _rank_value(candidate, "dense"),
        "metadata": metadata_value,
        "soft_preferences": _share(candidate.matched_soft_preferences, constraints, "soft"),
        "constraint_evidence": evidence.coverage_from_sets(
            evidence.product_tokens(product), disclosure_tokens
        ),
        "phrase_evidence": evidence.phrase_coverage_from(
            evidence.phrase_text(product), disclosure_phrases
        ),
        "slot_evidence": _slot_value(product, slot_terms, relaxed_ownership),
        "category_exact": (
            1.0
            if stated_category and product.coarse_category == stated_category
            else 0.0
        ),
    }

    if weights is None:
        weights = FEATURE_WEIGHTS
    # Iterating FEATURE_WEIGHTS rather than `weights` keeps the checklist the
    # same length and in the same order whichever regime is in force, so an
    # explain() from a paraphrased turn lines up against one from a quoted
    # turn feature for feature.
    contributions = tuple(
        ScoreContribution(feature=feature, weight=weights[feature], value=values[feature])
        for feature in FEATURE_WEIGHTS
    )
    return RerankedCandidate(
        parent_asin=candidate.parent_asin,
        score=sum(c.contribution for c in contributions),
        contributions=contributions,
    )


def rerank(
    candidates: list[Candidate],
    products: dict[str, ProductRecord],
    constraints: dict[str, Constraint],
    limit: int,
    disclosures: list[str] | None = None,
    prepared: DisclosureEvidence | None = None,
    stated_category: str | None = None,
    paraphrase_profile: bool = True,
) -> list[RerankedCandidate]:
    """Order the pool by the deterministic scorer and keep the top `limit`.

    The sort is stable on score alone, so candidates the scorer cannot
    separate keep the order retrieval gave them rather than being permuted
    arbitrarily.
    """
    if prepared is None:
        prepared = prepare_evidence(disclosures or [], candidates, products)
    # One decision for the whole pool, not per candidate: the regime is a
    # property of what the customer said and what the pool holds, so scoring
    # two candidates on the same turn under different weights would make the
    # comparison between them meaningless.
    weights = weights_for(paraphrase_profile and prepared.paraphrase_regime)
    scored = [
        score_candidate(
            candidate,
            products[candidate.parent_asin],
            constraints,
            disclosure_tokens=prepared.tokens,
            disclosure_phrases=prepared.phrases,
            slot_terms=prepared.slot_terms,
            stated_category=stated_category,
            weights=weights,
            relaxed_ownership=prepared.relaxed,
        )
        for candidate in candidates
        if candidate.parent_asin in products
    ]
    scored.sort(key=lambda item: item.score, reverse=True)
    return scored[:limit]

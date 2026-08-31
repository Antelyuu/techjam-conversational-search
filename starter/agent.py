from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

from shopping_agent import clarification
from shopping_agent import shortlist
from shopping_agent.catalog import ProductRecord, flatten_field, normalize_product
from shopping_agent.dense_retrieval import load_dense_retriever
from shopping_agent.semantic_evidence import catalogue_values, load_semantic_scorer
from shopping_agent.orchestrator import ConversationOrchestrator
from shopping_agent.reranking import prepare_evidence, rerank
from shopping_agent.retrieval import DEFAULT_FUSION, FUSION_METHODS, retrieve
from shopping_agent.slots import canonical_category, coarse_category
from shopping_agent.text import tokenize


# The evaluator matches on ask_attribute alone and never reads these, but a
# recommendation list paired with a bare attribute name is not a conversation.
_QUESTION_TEXT = {
    "feature": "Which features matter most to you?",
    "material": "Is there a material you prefer?",
    "color": "Do you have a colour in mind?",
    "style": "What style are you going for?",
    "size": "What size do you need?",
    "use_case": "What will you mainly use it for?",
    "other": "Is there anything else that matters to you?",
}
_DEFAULT_QUESTION = "Could you tell me a little more about what you need?"

# How deep a pool the reranker reorders before the top ten are shown. This
# also widens the BM25 fetch to match (see retrieve()).
#
# MEASURED three times, with different results each time, and the difference
# is what the reranker can discriminate. E5 swept 100-800 at
# MIN_EVIDENCE_TOKENS=3 and found HitRate flat at every depth: a rescued deep
# candidate could not be told apart, so depth bought nothing. E6's floor
# removal turned depth into a smooth rise peaking at 100 (0.766930) -- 16 of
# the 26 remaining misses were targets that never entered the 50-candidate
# pool while evidence could now rank them. E7's phrase-containment feature
# moved the optimum again, because contiguity keeps discriminating where
# token coverage saturates:
#
#   depth   100       150       200       250       300       350       400       600       800
#   score   0.798916  0.807179  0.807791  0.810660  0.810445  0.807321  0.804144  0.800169  0.799931
#
# 250 and 300 tie at the HitRate top (0.940); 250 is the earlier, cheaper
# point of that plateau.
#
# RE-SWEPT a fourth time (E8), after slot ownership and the exact stated
# category. Both are exact tests, so a rescued deep candidate is now either
# identified outright or scores zero -- which is precisely the discriminating
# power E5 found missing when it first measured depth flat:
#
#   depth   100       150       250       300       350       400       500       800       1200
#   score   0.897176  0.910851  0.929426  0.930551  0.933601  0.933701  0.932851  0.932439  0.932503
#   HitRate 0.955     0.970     0.990     0.990     0.995     0.995     0.995     0.995     0.995
#
# The pattern holds a fourth time: **a depth ablation is only as durable as
# the ranking features it was measured under.** 350 and 400 tie exactly on
# HitRate (0.995) and MRR (0.910671) and differ only by 0.005 of a turn --
# one session hitting one turn earlier, which is a single-session artifact
# rather than a robust margin. 400 is the measured best and is adopted; 350
# costs 0.0001 and runs 11% faster (31 s against 35 s for a full public-set
# pass), so it is the point to move to if latency ever binds.
#
# RE-SWEPT a fifth time (E9), after P6-T7 restricted retrieval to the stated
# category, and this time the answer is that depth no longer matters at all:
#
#   depth   200       300       400       600       800       1000
#   score   0.945497  0.945497  0.945497  0.945497  0.945497  0.945497
#
# Identical to six decimals across a 5x range, which is the first time this
# sweep has been flat since E5 -- and for the opposite reason. E5 was flat
# because a rescued deep candidate could not be told apart; this is flat
# because there is nothing left to rescue. The pool is now the customer's
# category (median 179 rows of 50,000), so 200 already holds all of it for
# most sessions and the tail that binds the 400 cap -- 122 of 561 turns --
# turns out to contain no target that the ranking would have found anyway.
#
# Left at 400 rather than lowered to the cheapest tied value. There is no
# measurement separating them here, so the choice falls to which is safer on
# an unseen split, and that is the one with headroom for a catalogue whose
# categories are larger than this one's.
RERANK_POOL = 400

# P5-T1: the dense route is off by default, reversing P3's decision on
# measurement rather than on preference.
#
# P3 adopted dense fusion because it was worth +0.0355 over lexical alone
# (0.1156 -> 0.1511) when every query was a single short opening message. P4
# changed what a query is: the customer now answers questions by quoting
# constraint sentences out of the target product's own features, and those
# accumulate. BM25 sharpens on that -- its recall of the target in a
# 50-candidate pool climbs from 0.38 at turn 1 to 0.74 by turn 3 -- while the
# dense route stays flat near 0.30, because a fixed-width sentence embedding
# averages a growing paragraph toward the corpus mean.
#
# MEASURED (E5): turning the route off is worth +0.050935, and it wins or
# ties every scenario. Down-weighting it instead does not work; there is a
# cliff rather than a slope, because retrieval takes the *union* of both
# routes' candidates and min-max normalization floors the lexical tail at
# 0.0, so dense-only candidates displace real ones even at 5% weight.
#
# Still switchable with SHOPPING_AGENT_DENSE=1. The artifact and the code
# stay in the repository: the finding is about this query distribution, and a
# private set that asks fewer questions would move the balance back.
DENSE_BY_DEFAULT = False

# P7-T1: the semantic evidence feature is ON by default, which is the opposite
# of the dense route's decision one section above and rests on a different
# measurement. Dense costs 0.0012 on the public benchmark to buy 0.0018 under
# paraphrase (E10) -- a bad trade. This costs **nothing** on the public
# benchmark, because prepare_evidence gates it on slot ownership having gone
# dark, and buys +0.1517 under paraphrase with HitRate 0.805 -> 0.975 (E11).
#
#   public          0.945297 / HitRate 1.000   (unchanged by this feature;
#                                               the 0.0002 against E10 is
#                                               constraint_evidence retiring)
#   paraphrased L2  0.847762 / HitRate 0.975   (against 0.696015 / 0.805)
#
# Switchable off with SHOPPING_AGENT_SEMANTIC=0. The feature is also optional
# by construction: with no artifact or no sentence-transformers installed it
# scores 0.0 for everyone and the agent ranks exactly as it does with the flag
# off, so a checkout that never runs the build still works.
SEMANTIC_BY_DEFAULT = True

# How many distinct query terms reach BM25.
#
# This looked like a silent-truncation bug of the kind P4 found twice, and it
# is not. 8.2% of turns do exceed it, dropping a median of 11 terms and up to
# 42 -- but raising it to 60, 80, 120 or 400 reproduces the composite to six
# decimal places (E5). The reason is that build_query_text() ends with the
# latest raw message, which for an answered question is the same disclosure
# already carried earlier in the query, so the terms a cap discards are
# duplicates. Left at 40, now with a measurement behind it rather than a
# guess.
QUERY_TERM_LIMIT = 40

# BM25 column weights, in the order the FTS table declares them:
# parent_asin, title, categories, features, details, store, description,
# coarse_category. parent_asin and coarse_category are unindexed and always
# 0.0 -- coarse_category is stored to be *filtered* on, never matched.
FIELD_WEIGHTS = "0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0, 0.0"

# P6-T7: restrict retrieval to the category the customer actually named.
#
# The opening line states `coarse_category(target.categories)` verbatim, in
# every scenario, on turn 1. E8 used that as a reranking feature and it was
# the largest single win of that phase. It is worth more as a *filter*, and
# the reason it is safe to use as one is structural rather than empirical:
# the string is computed from the target's own categories field by a function
# this agent reproduces exactly, so the target is a member of the restricted
# set by construction. Requiring membership cannot cost a hit.
#
# VERIFIED over the whole catalogue rather than by sampling, because that is
# what caught the trim-before-clip bug in the last reconstruction:
#
#   slots.coarse_category vs the evaluator's, over 50,000 products   0 disagree
#   openers whose category did not extract exactly                   0 / 200
#   targets not reproducing their own stated category                0 / 200
#
# What it buys is a search space of a different order. The catalogue holds
# 1115 distinct coarse categories over 50,000 rows, and the median target
# shares its own with just 184 of them (max 1354). So for most sessions a
# 400-deep pool now covers the entire category rather than 0.8% of the
# catalogue, and BM25 ranks within the field the customer asked for instead of
# across all of it.
#
# It fails quiet, the property every exact test in this project is built to
# have. If the opening line is not one this agent can parse, `stated_category`
# is None; if it parses to a string no product reproduces, the filter is not
# applied; and if the filtered search returns nothing at all, the unfiltered
# search runs instead. On a private set whose opener is worded differently the
# agent retrieves exactly as it did before.
#
# Switchable with SHOPPING_AGENT_CATFILTER=0.
CATEGORY_FILTER_BY_DEFAULT = True
# Conversational filler, stripped from BM25 queries. Deliberately not the
# same set evidence.py strips from product copy -- "please" and "looking" are
# noise here and content there never arises, while "your" and "will" are the
# reverse. The token definition itself is shared (shopping_agent/text.py).
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _terms(text: str) -> list[str]:
    return tokenize(text, STOPWORDS)


def _resolve_fusion(value: str | None) -> str:
    """Accept a fusion name case-insensitively, or fall back to the default.

    retrieve() treats anything that is not "weighted" as RRF, so an unnoticed
    typo would quietly select the configuration that measured 0.145170 instead
    of the documented 0.151089. Say so rather than silently downgrading."""
    if not value or not value.strip():
        return DEFAULT_FUSION
    normalized = value.strip().lower()
    if normalized not in FUSION_METHODS:
        print(
            f"[shopping_agent] unknown fusion {value!r}; "
            f"expected one of {sorted(FUSION_METHODS)}. Using {DEFAULT_FUSION}.",
            file=sys.stderr,
        )
        return DEFAULT_FUSION
    return normalized


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


class Agent:
    """Stateful multi-turn agent: BM25 retrieval over a cumulative,
    session-aware query built from accumulated conversation state, reordered
    by a deterministic scorer that leads on how completely a candidate
    accounts for what the customer has actually disclosed.

    Each turn also returns one clarifying question, which is where most of
    this system's score comes from -- the simulator discloses a hidden
    constraint only when asked.

    The dense semantic route is **off** by default as of P5, on measurement:
    it helped when queries were single short messages and hurts now that they
    carry quoted product text (see DENSE_BY_DEFAULT). Re-enable it with
    SHOPPING_AGENT_DENSE=1, and pick the blend with
    SHOPPING_AGENT_FUSION=rrf|weighted. When enabled it still engages only if
    its artifact and dependencies are present, and otherwise falls back to
    BM25 and says so on stderr."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        enable_dense: bool | None = None,
        fusion_method: str | None = None,
        allow_wildcard: bool | None = None,
        use_disagreement: bool | None = None,
        enable_clarification: bool | None = None,
        enable_reranker: bool | None = None,
        enable_shortlist: bool | None = None,
        enable_category_filter: bool | None = None,
        enable_semantic: bool | None = None,
        block_soft_slots: bool | None = None,
        allow_repeats: bool | None = None,
        route_weights: dict[str, float] | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.orchestrator = ConversationOrchestrator()
        self.products: dict[str, ProductRecord] = {}
        # Every coarse category the catalogue can reproduce, and the same set
        # again keyed by canonical form. A stated category that matches
        # neither is one no product could have produced, so filtering on it
        # would empty the pool rather than narrow it. See
        # _resolve_categories for which of the two is consulted when.
        self._known_categories: set[str] = set()
        self._categories_by_canonical: dict[str, set[str]] = {}
        self._build_index()

        if allow_wildcard is None:
            allow_wildcard = _env_flag("SHOPPING_AGENT_WILDCARD", default=True)
        if use_disagreement is None:
            use_disagreement = _env_flag("SHOPPING_AGENT_DISAGREEMENT", default=True)
        if enable_clarification is None:
            enable_clarification = _env_flag("SHOPPING_AGENT_CLARIFY", default=True)
        if enable_reranker is None:
            enable_reranker = _env_flag("SHOPPING_AGENT_RERANK", default=True)
        if enable_shortlist is None:
            enable_shortlist = _env_flag("SHOPPING_AGENT_SHORTLIST", default=True)
        if enable_category_filter is None:
            enable_category_filter = _env_flag(
                "SHOPPING_AGENT_CATFILTER", default=CATEGORY_FILTER_BY_DEFAULT
            )
        if block_soft_slots is None:
            block_soft_slots = _env_flag("SHOPPING_AGENT_BLOCK_SOFT", default=False)
        if allow_repeats is None:
            # MEASURED and rejected (E5). Re-asking a high-yield attribute
            # until the card is drained sounds strictly more informative, and
            # costs 0.0078 composite -- all of it in Intent Override
            # (0.800 -> 0.767), where hits cannot register before the override
            # fires and a repeated question crowds out the diversification
            # that finds the new intent. E3 left this open; it is closed now.
            allow_repeats = _env_flag("SHOPPING_AGENT_REPEAT", default=False)
        self.allow_repeats = allow_repeats
        self.allow_wildcard = allow_wildcard
        self.use_disagreement = use_disagreement
        self.enable_clarification = enable_clarification
        self.enable_reranker = enable_reranker
        self.enable_shortlist = enable_shortlist
        self.enable_category_filter = enable_category_filter
        self.block_soft_slots = block_soft_slots
        self.route_weights = route_weights
        self._warned: set[str] = set()

        if enable_dense is None:
            enable_dense = _env_flag("SHOPPING_AGENT_DENSE", default=DENSE_BY_DEFAULT)
        self.fusion_method = _resolve_fusion(
            fusion_method or os.environ.get("SHOPPING_AGENT_FUSION")
        )
        # None means the route is unavailable (no artifact, no deps, or an
        # artifact built from a different catalogue); the agent then serves
        # BM25 lexical results instead of failing.
        self.dense_search = (
            load_dense_retriever(expected_ids=sorted(self.products)) if enable_dense else None
        )

        # P7-T1 (E11). On by default and optional by construction: None means
        # the value artifact or sentence-transformers is missing, and the
        # reranker then scores semantic_evidence 0.0 for every candidate,
        # which is this agent exactly as it ranked before the feature existed.
        # The artifact loads here, so a broken one is reported at startup
        # rather than mid-run; the model itself loads on the first turn that
        # actually has a disclosure to embed.
        if enable_semantic is None:
            enable_semantic = _env_flag(
                "SHOPPING_AGENT_SEMANTIC", default=SEMANTIC_BY_DEFAULT
            )
        self.semantic_scorer = (
            load_semantic_scorer(expected_values=catalogue_values(self.products))
            if enable_semantic
            else None
        )

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "coarse_category UNINDEXED, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                record = normalize_product(raw)
                self.products[record.parent_asin] = record
                # The exact string the customer would be given for this
                # product, reproduced with the generator's own rule so the
                # comparison downstream is equality rather than similarity.
                category = coarse_category(raw.get("categories"))
                self._known_categories.add(category)
                self._categories_by_canonical.setdefault(
                    canonical_category(category), set()
                ).add(category)
                batch.append(
                    (
                        record.parent_asin,
                        flatten_field(raw.get("title")),
                        flatten_field(raw.get("categories")),
                        flatten_field(raw.get("features")),
                        flatten_field(raw.get("details")),
                        flatten_field(raw.get("store")),
                        flatten_field(raw.get("description")),
                        category,
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def close(self) -> None:
        """Release the in-memory FTS index.

        The official harness builds one Agent for a whole run and never needs
        this, but tests and the ablation scripts build many; without it each
        leaks a SQLite connection until garbage collection, which surfaces as
        ResourceWarning. Safe to call more than once.
        """
        connection = getattr(self, "connection", None)
        if connection is not None:
            connection.close()
            self.connection = None

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.orchestrator.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        request = self.orchestrator.process_turn(session_id, user_message, turn, top_k)
        # Search inside the category the customer named, when they named one
        # this catalogue can reproduce (P6-T7). Bound as a closure so retrieval
        # keeps its (query_text, pool_size) contract and stays free of any
        # notion of what a category is.
        lexical_search = self._lexical_search
        if self.enable_category_filter:
            categories = self._resolve_categories(request.state.stated_category)
            if categories:
                def lexical_search(
                    query_text: str, limit: int, _categories: tuple[str, ...] = categories
                ):
                    return self._lexical_search(query_text, limit, _categories)
        candidates = retrieve(
            request,
            request.top_k,
            lexical_search,
            self.products,
            dense_search=self.dense_search,
            fusion_method=self.fusion_method,
            route_weights=self.route_weights,
            candidate_limit=RERANK_POOL if self.enable_reranker else None,
        )
        # How far the disclosed constraints have narrowed the field. Prepared
        # once here because both the reranker and the shortlist policy need
        # it, and it costs one pass over the pool.
        live_disclosures = 0
        consistent = 0
        # Whether this turn's evidence was actually measured. The shortlist
        # policy reads "no candidate owns anything the customer said" as a
        # reason to stop withholding, and that reading is only valid when the
        # measurement ran; an unmeasured turn must not be mistaken for a
        # measured zero in either direction.
        measured = False
        if self.enable_reranker:
            # A reranker failure must not cost the turn: the fused order is
            # already a valid ranking, so fall back to it rather than raising
            # into respond() and returning nothing (P4-T4).
            try:
                prepared = prepare_evidence(
                    request.state.disclosed_text,
                    candidates,
                    self.products,
                    semantic_scorer=self.semantic_scorer,
                )
                live_disclosures = prepared.live_disclosures
                consistent = prepared.consistent
                measured = True
                reranked = rerank(
                    candidates,
                    self.products,
                    request.state.constraints,
                    top_k,
                    prepared=prepared,
                    stated_category=request.state.stated_category,
                )
                recommendations = [
                    {"parent_asin": item.parent_asin, "score": item.score} for item in reranked
                ]
            except Exception as error:
                self._warn_once(f"reranker failed, using fused order: {error}")
                # Evidence is unavailable on this path, so `measured` stays
                # False and the full list is returned -- the shortlist policy
                # must not withhold on a turn whose ranking it could not
                # measure.
                measured = False
                recommendations = self._fused_recommendations(candidates[:top_k])
        else:
            recommendations = self._fused_recommendations(candidates)
        # Return the ranked list only as far as the agent can stand behind it
        # (P6-T2); see shopping_agent/shortlist.py for what that means and
        # what it was measured to be worth.
        if measured:
            recommendations = recommendations[
                : shortlist.shortlist_size(
                    turn,
                    top_k,
                    live_disclosures,
                    consistent,
                    len(request.state.disclosed_text),
                    enabled=self.enable_shortlist,
                )
            ]
        # Asking is free: the evaluator scores recommendations first and then
        # handles ask_attribute separately, so a question never displaces a
        # recommendation. Always return both.
        # Not asking costs a turn; raising costs the whole session, because the
        # evaluator turns anything escaping respond() into zero recommendations.
        # Guarded for the same reason the dense route and reranker are (P4-T4).
        ask_attribute = None
        if self.enable_clarification:
            try:
                ask_attribute = clarification.choose_attribute(
                    request.state,
                    [self.products[c.parent_asin] for c in candidates],
                    allow_wildcard=self.allow_wildcard,
                    use_disagreement=self.use_disagreement,
                    block_soft_slots=self.block_soft_slots,
                    allow_repeats=self.allow_repeats,
                )
            except Exception as error:
                self._warn_once(f"clarification policy failed, asking nothing: {error}")
        self.orchestrator.record_question(request.state, ask_attribute)
        return {
            "message": self._build_message(request.state, ask_attribute),
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    @staticmethod
    def _fused_recommendations(candidates) -> list[dict]:
        return [
            {"parent_asin": candidate.parent_asin, "score": candidate.route_scores["combined"]}
            for candidate in candidates
        ]

    def _warn_once(self, message: str) -> None:
        if message in self._warned:
            return
        self._warned.add(message)
        print(f"[shopping_agent] {message}", file=sys.stderr)

    def _resolve_categories(self, stated: str | None) -> tuple[str, ...]:
        """The catalogue categories to search inside, given what was stated.

        Exact agreement first, because that is the case the filter was
        measured on and the one the target satisfies by construction. Only
        when the stated string names nothing this catalogue spells does the
        canonical form get a second look -- a customer who says "Boots Shoes"
        for "Shoes Boots" has named the same shelf, and matching the sorted
        tokens recovers it without loosening the exact test that runs first.

        Returns every category sharing the canonical form, not an arbitrary
        one of them, so the nine word-order pairs in the catalogue are
        searched together rather than one being picked over the other.

        Empty means the filter does not apply, which is the same quiet
        failure it always had: an opener this agent cannot parse, or a
        category no product reproduces, retrieves across the whole catalogue
        exactly as it did before P6-T7.
        """
        if not stated:
            return ()
        if stated in self._known_categories:
            return (stated,)
        siblings = self._categories_by_canonical.get(canonical_category(stated))
        return tuple(sorted(siblings)) if siblings else ()

    def _lexical_search(
        self, query_text: str, limit: int, category: str | tuple[str, ...] | None = None
    ) -> list[tuple[str, float]]:
        unique_terms = list(dict.fromkeys(_terms(query_text)))[:QUERY_TERM_LIMIT]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        select = f"SELECT parent_asin, bm25(products, {FIELD_WEIGHTS}) AS cost FROM products "
        rows: list = []
        wanted = (category,) if isinstance(category, str) else tuple(category or ())
        if wanted:
            placeholders = ", ".join("?" * len(wanted))
            rows = self.connection.execute(
                select
                + f"WHERE products MATCH ? AND coarse_category IN ({placeholders}) "
                + "ORDER BY cost LIMIT ?",
                (expression, *wanted, limit),
            ).fetchall()
        if not rows:
            # Either no category was asked for, or nothing in it matched the
            # query. Retrieving inside an empty field is worse than retrieving
            # across the catalogue, so the filter stands down (P6-T7).
            rows = self.connection.execute(
                select + "WHERE products MATCH ? ORDER BY cost LIMIT ?",
                (expression, limit),
            ).fetchall()
        # FTS5 bm25() is a cost (lower is better); negate so higher is better,
        # matching the category boost sign used downstream.
        return [(str(row[0]), -float(row[1])) for row in rows]

    @staticmethod
    def _build_message(state, ask_attribute: str | None = None) -> str:
        if state.constraints:
            summary = ", ".join(f"{c.attribute}={c.value}" for c in state.constraints.values())
            opening = f"Here are the closest matches based on {summary}."
        else:
            opening = "Here are the closest matches I found."
        if ask_attribute is None:
            return opening
        return f"{opening} {_QUESTION_TEXT.get(ask_attribute, _DEFAULT_QUESTION)}"

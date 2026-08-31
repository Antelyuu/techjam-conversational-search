"""P6-T1: match a disclosure against the *slot* it came out of, not the text.

E7 left the project against a wall it named precisely: the surviving misses
are targets that reached the pool but sat behind catalogue near-duplicates
whose copy contains the same quoted constraints. Token coverage ties at 1.0
across all of them, and phrase containment ties too whenever the impostor
carries the sentence verbatim somewhere in its own text. E7's conclusion was
that text similarity had no signal left and only a non-text discriminator
could separate them. This module is that discriminator, and it is still made
of text -- what is new is that it asks a *structural* question about it.

The simulator does not quote arbitrary substrings of the target. It builds
its hidden card by walking the target's `features` and `details` and taking
**whole values** -- one entire list element, or one entire `key: value` pair
-- normalizing whitespace, trimming edge punctuation and clipping at 180
characters (`local_evaluator.intent_card` via `_flatten_values` and
`_clean_constraint`). Every constraint the customer can ever disclose is
therefore an exact member of a small set the target owns, not a fragment and
not a coincidence.

That distinction is the whole value here. "Machine wash cold" appearing
*inside* an impostor's longer bullet is not evidence; the same string
standing alone as one of its feature values is. Measured over the 50,000-row
catalogue and the 200 public sessions:

  * the target owns every one of its 800 disclosable constraints as an exact
    value -- 800/800, by construction rather than by luck, so requiring the
    match can never cost a hit;
  * 193 of those 800 constraints are owned by exactly **one** product in the
    whole catalogue;
  * with the opening category plus two disclosed constraints the median
    consistent set is already **one** product, and with four it is one for
    169 of the 200 sessions.

Selectivity is what makes that usable, and it varies enormously: a material
label ("cotton") is owned by thousands of rows and says nothing, while a
sixteen-word care instruction is often unique. So ownership is weighted by
how rare the disclosure is *inside the current pool* (see
`ownership_weights`), which needs no catalogue-wide index and calibrates
itself against the candidates actually competing.

The guarantee that follows is the useful one: because the target owns all of
its own disclosures, any disclosure owned by exactly one pooled candidate is
owned by the target whenever the target is pooled at all.

**This is a ranking feature and never a filter.** If a private set paraphrases
its constraints rather than quoting them, no candidate owns anything, every
score is 0.0, and the ordering falls back to the features beneath it -- the
same quiet-failure property the other two evidence features were built for.
"""

from __future__ import annotations

import math
import re

# The card generator's own normalization, reproduced exactly. Any drift here
# shows up as the target failing to own its own disclosure, which is why
# tests/test_phase6_slots.py checks this against the evaluator's functions
# over a catalogue sample rather than against hand-written expectations.
CARD_LIMIT = 180
_WHITESPACE_RE = re.compile(r"\s+")
_EDGE_CHARS = " -;,.\t\n"

# Both word lists and the field order below are the generator's, not this
# project's. shopping_agent/intent.py keeps a longer, friendlier vocabulary
# for reading customer prose; that is a different job and the two must not be
# merged -- this one has to agree with the simulator character for character.
_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I
)
_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)
# The generator scans for the material and colour labels over its own
# flattening, in *this* field order, and keeps the first hit. catalog.py
# flattens the same fields in a different order for search, so reusing that
# string would pick a different material for any product whose categories and
# features disagree. Ordering matters; it is not a copy-paste artifact.
_CARD_FIELDS = ("title", "features", "details", "description", "categories", "store")


# The categories the generator refuses to describe a product by, because they
# are true of the whole catalogue and would name every session the same thing.
_GENERIC_CATEGORIES = frozenset(
    {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
)
_FALLBACK_CATEGORY = "clothing item"

# The opening line, which states the coarse category verbatim in every
# scenario before going on to whatever else it carries:
#
#   "I'm looking for {category}, but I'm still exploring."
#   "I'm looking for {category}. A key requirement is: {constraint}."
#   "I'm looking for {category}. {old_value}"
#
# The category never contains a comma -- coarse_category() splits on commas
# and rejoins with spaces -- so the first comma or sentence break ends it.
#
# MEASURED (E10): this used to accept the lead-in "I'm looking for" and
# nothing else, which made the whole category apparatus -- the retrieval
# filter of P6-T7 and the `category_exact` feature -- contingent on one
# English phrase. scripts/paraphrase_eval.py rewords the opener at level 1
# and above, and 0 of 200 openers then parsed; every paraphrased session ran
# with the filter stood down. Isolating that (--keep-opener) priced it at
# 0.248 composite, about half of the entire paraphrase penalty and larger
# than any other single effect measured on this task.
#
# Widened to the lead-ins a rewording of "I'm looking for" actually produces.
# The alternation is ordered longest-first so a prefix cannot shadow a longer
# lead-in, and the terminator now ends at any comma rather than only ", but",
# which the category cannot contain by construction.
#
# VERIFIED to be a strict superset rather than a behaviour change, over the
# real coarse categories of all 200 public targets:
#
#   public-shaped openers where old and new disagree     0 / 600
#   reworded openers the new form recovers exactly     800 / 800
#
# "show me ..." is deliberately absent. It is the one lead-in that routinely
# introduces something that is not a category ("show me something for
# running"), and tests/test_phase6_category_filter.py guards that case.
_STATED_CATEGORY_RE = re.compile(
    r"^\s*(?:"
    r"i\s*'?\s*m\s+looking\s+for|i\s+am\s+looking\s+for|"
    r"i\s*'?\s*m\s+searching\s+for|i\s*'?\s*m\s+shopping\s+for|"
    r"i\s*'?\s*m\s+after|i\s*'?\s*d\s+like|"
    r"looking\s+for|i\s+want|i\s+need"
    r")\s+(.+?)\s*(?:,|\.(?:\s|$)|\s+[\u2014\u2013]|\s+-\s|$)",
    re.IGNORECASE,
)

# Order- and punctuation-insensitive form of a category string.
#
# The category the customer is handed is compared to the catalogue's own by
# *equality* -- that exactness is what makes the filter safe (the target
# reproduces its own category character for character). It is also the whole
# of its fragility: "Shoes Boots" and "Boots Shoes" name the same shelf and
# fail to match. Sorting the tokens keeps the comparison exact while dropping
# the one distinction a reworded category loses.
#
# It collapses 1115 catalogue categories onto 1106 canonical forms; all 9
# collisions are word-order pairs of each other ("Shoes Clogs & Mules" /
# "Shoes Mules & Clogs"), which is the distinction this is meant to ignore.
_CANONICAL_TOKEN_RE = re.compile(r"[a-z0-9]+")


def canonical_category(text: str) -> str:
    """The form two spellings of the same category agree on."""
    return " ".join(sorted(_CANONICAL_TOKEN_RE.findall((text or "").lower())))


def coarse_category(value: object) -> str:
    """The category string the customer will be given for this product.

    The generator takes the last two comma-separated parts of the raw
    `categories` field, minus the catalogue-wide ones, and joins them with a
    space. Reproduced here because *exact* agreement is a far sharper test
    than the word overlap score_category() does: the pool that BM25 returns
    for an opening message shares the target's exact category for a median of
    38% of its candidates, so agreement narrows the field about 2.6x, and it
    is available from turn 1 in every session -- including Browsing, where it
    is the only thing the customer has said.
    """
    if value is None or value == "":
        values: list[object] = []
    elif isinstance(value, list):
        values = list(value)
    else:
        values = [value]
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in _GENERIC_CATEGORIES:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else _FALLBACK_CATEGORY


def stated_category(message: str) -> str | None:
    """The category the opening line names, or None if this is not one.

    Returns the text verbatim: the generator emits the category with its
    original capitalization and the comparison is exact, so normalizing here
    would only make the two sides disagree.
    """
    match = _STATED_CATEGORY_RE.match(message or "")
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def clean_value(value: str, limit: int = CARD_LIMIT) -> str:
    """Whitespace-collapse, trim edge punctuation, then clip -- in that order.

    The order is the generator's and it is observable: trimming after the clip
    would keep a trailing comma that the card never has.
    """
    return _WHITESPACE_RE.sub(" ", value).strip(_EDGE_CHARS)[:limit].rstrip()


def canonical(text: str) -> str:
    """The form both sides of the comparison are held in.

    A second edge-trim on top of `clean_value`, and it is load-bearing rather
    than tidy-minded. The generator trims edge punctuation *before* it clips
    at 180 characters, so a clipped value can perfectly well end in a comma
    ("... but once you start wearing them,") -- 4 of the 200 public targets
    have one. The customer then appends the sentence's full stop when quoting
    it, so the constraint arrives as "...wearing them,." while the card holds
    "...wearing them,". Neither string is the other, and trimming only the
    reply's own punctuation would still leave them apart.

    Trimming both sides to the same core settles it, and costs only the
    ability to tell "abc," from "abc" -- a distinction no disclosure can
    carry anyway, because the reply's own punctuation has already destroyed
    it.
    """
    return _WHITESPACE_RE.sub(" ", text).strip(_EDGE_CHARS)


def _flatten_values(value: object) -> list[str]:
    """One candidate constraint per whole value, mirroring the generator.

    A dict field yields "key: value" pairs, a list field yields its elements,
    and a scalar yields itself. Empties are dropped rather than becoming
    blank constraints.
    """
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def _card_corpus(raw: dict) -> str:
    parts: list[str] = []
    for field in _CARD_FIELDS:
        value = raw.get(field)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def card_values(raw: dict) -> frozenset[str]:
    """Every string this product could contribute to the customer's card.

    Includes the semicolon-separated *pieces* of each value as well as the
    whole. The simulator joins several constraints with "; " and the agent
    splits the reply on ";" to recover them (`evidence.split_disclosures`),
    which is indistinguishable from a semicolon *inside* a single constraint
    -- 22% of catalogue rows produce one. Storing both forms lets a
    fragmented disclosure match at the fragment level and an intact one match
    whole, without the agent having to guess which happened. The pieces are
    less selective on their own, and `ownership_weights` prices them down
    accordingly.
    """
    values = [*_flatten_values(raw.get("features")), *_flatten_values(raw.get("details"))]
    corpus = _card_corpus(raw)
    material = _MATERIAL_RE.search(corpus)
    colour = _COLOR_RE.search(corpus)
    if material:
        values.append(material.group(1).lower())
    if colour:
        values.append(f"color: {colour.group(1).lower()}")
    # Kept for fidelity to the generator even though it is structurally
    # unreachable on this data: the budget line is appended last and only
    # survives into the card when the product yields fewer than four other
    # values, which happens for 0 of the 200 public targets (E6). Costs one
    # set entry; removing it would make this reconstruction subtly wrong for
    # a catalogue with sparser rows.
    if raw.get("price") not in (None, ""):
        values.append(f"budget around ${raw['price']}")

    owned: set[str] = set()
    for value in values:
        cleaned = clean_value(value)
        if not cleaned:
            continue
        whole = canonical(cleaned)
        if whole:
            owned.add(whole)
        if ";" in cleaned:
            for piece in cleaned.split(";"):
                piece = canonical(piece)
                if piece:
                    owned.add(piece)
    return frozenset(owned)


def normalize_disclosure(text: str) -> str:
    """Put a disclosed constraint back into the form the card held it in.

    The reply arrives already split on ";" and stripped of whitespace, but the
    final piece still carries the sentence's full stop. No clip is applied:
    the card has already clipped, so anything still longer than the limit is
    not a card value and should not match one.
    """
    return canonical(text)


# P7-T3: token sets for approximate ownership, cached per distinct value.
#
# Bounded and cleared wholesale for the same reason evidence._TOKEN_CACHE is:
# a long-lived process must not accrete a tokenized copy of every catalogue it
# ever loads. Keyed on the value string rather than on any product id, so two
# catalogues in one process -- which is what the tests are -- cannot serve
# each other's tokens.
#
# The limit is NOT borrowed from _TOKEN_CACHE, and the difference matters.
# That cache holds one entry per *product* (50,000 against 120,000, so it
# never trips). This one holds one entry per distinct *card value*, and this
# catalogue has **268,564** of them -- counted, not estimated. At 200,000
# this cache would fill and clear wholesale part-way through a run, then
# refill, which is thrashing rather than bounding.
#
# Only reached once a rewording has been detected, so on the public path it
# stays empty and costs nothing at all.
_VALUE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_VALUE_TOKEN_CACHE: dict[str, frozenset[str]] = {}
_VALUE_TOKEN_CACHE_LIMIT = 400_000


def value_tokens(value: str) -> frozenset[str]:
    """The bare alphanumeric tokens of one card value."""
    cached = _VALUE_TOKEN_CACHE.get(value)
    if cached is None:
        if len(_VALUE_TOKEN_CACHE) >= _VALUE_TOKEN_CACHE_LIMIT:
            _VALUE_TOKEN_CACHE.clear()
        cached = frozenset(_VALUE_TOKEN_RE.findall(value.lower()))
        _VALUE_TOKEN_CACHE[value] = cached
    return cached


# A disclosure and an owned value must share at least this many tokens before
# the overlap is treated as evidence rather than coincidence. Single-token
# disclosures can only ever share one, so the floor is min(2, len(wanted)).
MIN_SHARED_TOKENS = 2

# How much longer an owned value may be than the disclosure before containment
# stops meaning anything. "cotton" is contained in "100% cotton blend twill
# lining" at a ratio of 5, and crediting that in full is the E7/E8 impostor.
MAX_LENGTH_RATIO = 3.0


def ownership_overlap(value: str, owned: frozenset[str]) -> float:
    """How completely this candidate accounts for `value`, as a 0-1 score.

    Exact ownership asks whether the candidate holds this string as a whole
    field value of its own. That is the sharp question the public simulator
    rewards, and it is worth +0.0316 there (E8) precisely because it is
    all-or-nothing. It is also why the feature goes silent the moment the
    customer rewords: change one word of a nine-word value and identity
    scores the same zero as a completely unrelated product.

    This is the graded version, for use only once a rewording has been
    detected. It is **containment** -- the share of the disclosure's own
    tokens that some single owned value carries -- subject to two guards.

    **Containment rather than Jaccard, and this was measured the wrong way
    round first.** Jaccard divides by the union, so it charges for the owned
    value's extra tokens as well as the disclosure's missing ones. That reads
    as even-handed and is not: under a *compressive* paraphrase, where the
    customer says less than the card holds, it penalises the true owner for
    the length of its own value and rewards whichever product owns the
    shortest phrase made of those words. Measured, before the fix:

        customer says   "wash cold"
        true owner owns "machine wash cold with like colors"  Jaccard 0.333
        impostor   owns "hand wash cold"                      Jaccard 0.667

    At a threshold of 0.5 that scored the true owner 0.0 and the impostor
    1.0 -- at the weight of the table's dominant feature, this was strictly
    worse than the silence it replaced. Containment scores both 1.0: it
    declines to separate them, and other features decide. Failing to
    discriminate is a far cheaper error than discriminating backwards.

    The reason this survived the first round of measurement is worth
    recording. `scripts/paraphrase_eval.py` level 2 is *expansive* --
    "cotton" becomes "natural plant fibre" -- so disclosures come out no
    shorter than the values they came from and the asymmetry never bites.
    Level 3 is the compressive one. The threshold had been swept only at
    level 2, which is the single regime in which the defect is invisible.

    The two guards replace what Jaccard was there to provide:

    * `MIN_SHARED_TOKENS` -- one word in common is coincidence, not
      evidence.
    * `MAX_LENGTH_RATIO` -- a disclosure buried in a value several times its
      length is contained by accident, which is the E7/E8 impostor exactly.

    Selectivity does the rest of the work: a value many pooled candidates
    account for is re-priced down by `ownership_weights`, so a phrase that
    matches half the pool cannot decide anything however well it is
    contained.

    Compared against every owned value rather than only the best-aligned
    one, because the customer chooses which of the card's values to quote and
    the agent does not know which.
    """
    wanted = value_tokens(value)
    if not wanted:
        return 0.0
    need = min(MIN_SHARED_TOKENS, len(wanted))
    limit = MAX_LENGTH_RATIO * len(wanted)
    best = 0.0
    for candidate_value in owned:
        have = value_tokens(candidate_value)
        if len(have) > limit:
            continue
        shared = len(wanted & have)
        if shared < need:
            continue
        score = shared / len(wanted)
        if score > best:
            best = score
            if best == 1.0:
                break
    return best


def ownership_weights(owner_counts: list[int], pool_size: int) -> list[float]:
    """How much each disclosure's ownership is worth, given how many pooled
    candidates own it.

    A constraint every candidate owns separates nothing and scores ~0; one a
    single candidate owns is close to an identification, and -- because the
    target owns all of its own disclosures -- that candidate *is* the target
    whenever the target is pooled. Plain inverse-document-frequency over the
    pool expresses exactly that, and needs no catalogue-wide index.

    A disclosure no pooled candidate owns gets weight 0 rather than the
    maximum: it is silence, not evidence, and the alternative would let a
    paraphrased or mis-parsed constraint dominate the feature while telling us
    nothing about any candidate.
    """
    weights = []
    for count in owner_counts:
        if count <= 0:
            weights.append(0.0)
        else:
            weights.append(math.log((pool_size + 1.0) / (count + 1.0)) + 1.0)
    return weights

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

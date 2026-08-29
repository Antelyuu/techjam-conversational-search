"""P4-T2/T3: which attribute to ask about, and when to stop asking.

Before P4 the agent returned ``ask_attribute: None`` on every turn. The
simulator only discloses a hidden constraint when asked, so three of the four
scenarios landed every single hit on turn 1 and turns 2-10 contributed
nothing. This module is what makes the later turns worth taking.

Asking costs nothing: the evaluator scores ``recommendations`` first and then
processes ``ask_attribute`` separately, so a question is never traded against
a recommendation. The only real cost is asking about something that cannot
answer.
"""

from __future__ import annotations

import re
from collections import Counter

from . import intent as intent_module
from .catalog import ProductRecord
from .contracts import SessionState

# Attributes the simulator can actually produce an answer for.
#
# MEASURED on all 200 public-set sessions (scripts/ask_value_analysis.py):
# the share of sessions still holding at least one *undisclosed* constraint
# that the evaluator's classify_constraint() maps to this attribute once the
# opening message has been replayed. That is exactly the share of sessions
# where asking returns content instead of "I don't have an additional
# preference for X".
ATTRIBUTE_PRIOR_YIELD: dict[str, float] = {
    "feature": 0.960,
    "material": 0.725,
    "color": 0.255,
    "style": 0.085,
    "size": 0.045,
    "use_case": 0.020,
}

# Measured 0/200 -- asking one of these is a guaranteed wasted turn.
#
# brand and category are unreachable by construction: classify_constraint()
# can never return either. budget is unreachable in practice for a different
# reason -- intent_card() appends the budget line last, and every one of the
# 200 cards was truncated to four constraints by its [:2]/[2:4] slices, so
# the budget line never survived. All 800 constraint strings across the set
# classified as one of the six attributes above.
DEAD_ATTRIBUTES = frozenset({"brand", "category", "budget"})

# "other" matches *any* undisclosed constraint in the simulator, which makes
# it strictly the highest-yield question available.
#
# It is reached only once all six specific attributes are exhausted, which is
# both the honest dialogue shape -- ask what you actually want to know, then
# ask openly -- and what the measurement supports. Asked as a last resort it
# is worth +0.051 composite (E3). Asked as a substitute for a real policy it
# looked worth +0.111, but two thirds of that was really the cost of running
# out of questions early, which unblocking soft-guessed slots fixed properly.
WILDCARD_ATTRIBUTE = "other"

# There are only six real attributes and roughly nine usable turns, so the
# attribute list binds first. This is the explicit budget the spec asks be
# tracked, and the backstop if that ever stops being true.
MAX_CLARIFICATIONS = 8

# Below this share of candidates carrying a known value, the pool is treated
# as silent on the attribute rather than as disagreeing about it. This is
# P4-T3: ten candidates that never mention a colour must not read the same as
# ten candidates that mention ten different colours.
MIN_COVERAGE = 0.25

# Disagreement modulates the measured prior rather than gating on it. The
# prior says whether the *customer* can answer; disagreement says whether the
# answer would reorder the *pool*. Only the first is measured, so it keeps the
# majority of the weight and a sparse vocabulary cannot veto the best question.
DISAGREEMENT_FLOOR = 0.6

_ATTRIBUTE_PATTERNS: dict[str, re.Pattern[str]] = {
    "color": intent_module.COLOR_RE,
    "material": intent_module.MATERIAL_RE,
    "style": intent_module.STYLE_RE,
    "use_case": intent_module.USE_CASE_RE,
    "feature": intent_module.FEATURE_RE,
    "size": intent_module.LETTER_SIZE_RE,
}

# Both of the simulator's "no answer" replies, told apart by the single word
# "additional". They mean genuinely different things:
#   "I don't have a preference for X"            -> Boundary burning its one
#                                                   free pass; X may still
#                                                   hold an answer later.
#   "I don't have an additional preference for X" -> X is genuinely empty.
_NO_PREFERENCE_RE = re.compile(
    r"do(?:n'?t| not) have an?\s+(additional\s+)?preference for\s+([a-z_]+)",
    re.IGNORECASE,
)


def attribute_coverage(
    products: list[ProductRecord], attribute: str
) -> tuple[float, float]:
    """How much of the pool states a value for `attribute`, and how much
    those stated values disagree.

    Returns (coverage, disagreement), both 0-1. Disagreement is the share of
    the covered candidates that do *not* hold the modal value, so a pool that
    agrees scores 0 and an evenly split pool approaches 1.

    Coverage below MIN_COVERAGE forces disagreement to 0. Missing data is
    silence, not variety -- without this an attribute nothing mentions would
    look maximally informative and the agent would chase it.
    """
    pattern = _ATTRIBUTE_PATTERNS.get(attribute)
    if pattern is None or not products:
        return 0.0, 0.0

    values: Counter[str] = Counter()
    for product in products:
        match = pattern.search(product.searchable_text)
        if match:
            values[match.group(0).lower()] += 1

    covered = sum(values.values())
    coverage = covered / len(products)
    if coverage < MIN_COVERAGE:
        return coverage, 0.0
    return coverage, 1.0 - (values.most_common(1)[0][1] / covered)


def question_value(
    attribute: str, products: list[ProductRecord]
) -> tuple[float, float, float]:
    """Score one attribute, returning (value, coverage, disagreement) so the
    choice is inspectable rather than a bare number."""
    coverage, disagreement = attribute_coverage(products, attribute)
    prior = ATTRIBUTE_PRIOR_YIELD.get(attribute, 0.0)
    modifier = DISAGREEMENT_FLOOR + (1.0 - DISAGREEMENT_FLOOR) * disagreement
    return prior * modifier, coverage, disagreement


def choose_attribute(
    state: SessionState,
    products: list[ProductRecord],
    allow_wildcard: bool = False,
    use_disagreement: bool = True,
    block_soft_slots: bool = True,
) -> str | None:
    """The one attribute to ask about this turn, or None to stop asking.

    Never a dead attribute, never one already asked, never one already fixed
    by an extracted constraint, never one the customer has said is empty, and
    never once the question budget is spent.

    `block_soft_slots` decides what counts as fixed. Slot extraction is a
    regex guess, and it fires on the category phrase itself: "I'm looking for
    Athletic Walking" sets style=athletic and use_case=walking as *soft*
    values, which then look like answered questions and silently retire two
    of the six attributes. Treating only hard constraints as fixed keeps the
    spec's rule where it means something -- a stated requirement -- without
    letting a guess spend the question budget.
    """
    if state.clarification_turns >= MAX_CLARIFICATIONS:
        return None

    fixed = {
        attribute
        for attribute, constraint in state.constraints.items()
        if block_soft_slots or constraint.strength == "hard"
    }
    unavailable = (
        set(state.asked_attributes)
        | set(state.rejected_attributes)
        | fixed
        | DEAD_ATTRIBUTES
    )
    available = [a for a in ATTRIBUTE_PRIOR_YIELD if a not in unavailable]

    if not available:
        if allow_wildcard and WILDCARD_ATTRIBUTE not in unavailable:
            return WILDCARD_ATTRIBUTE
        return None

    if use_disagreement:
        def rank(attribute: str) -> tuple[float, float]:
            value, _coverage, _disagreement = question_value(attribute, products)
            # Ties fall back to the measured prior, which keeps the ordering
            # deterministic when the pool carries no signal at all.
            return value, ATTRIBUTE_PRIOR_YIELD[attribute]
    else:
        def rank(attribute: str) -> tuple[float, float]:
            prior = ATTRIBUTE_PRIOR_YIELD[attribute]
            return prior, prior

    return max(available, key=rank)


def interpret_reply(message: str) -> tuple[str | None, bool]:
    """Read the customer's answer to our last question.

    Returns (attribute_to_reject, is_boundary_pass). A boundary pass rejects
    nothing: the Boundary scenario declines the first question whatever it
    was about, so that attribute is still worth asking again later.
    """
    match = _NO_PREFERENCE_RE.search(message or "")
    if match is None:
        return None, False
    is_boundary = match.group(1) is None
    attribute = match.group(2).lower()
    if is_boundary:
        return None, True
    if attribute not in ATTRIBUTE_PRIOR_YIELD and attribute != WILDCARD_ATTRIBUTE:
        return None, False
    return attribute, False

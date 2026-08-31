"""Measure how much of the score depends on the customer quoting verbatim.

The public-set simulator never paraphrases: `intent_card()` takes whole values
out of the target's own `features`/`details`, and `customer_reply()` hands those
strings back untouched. So every substantive customer utterance is a literal
substring of the one product we are looking for.

That is a property of the benchmark, not of shopping. It flatters lexical
retrieval (BM25 is handed the answer key) and it is the reason the dense route
measured as a loss in E5. This script replaces the quoting customer with a
paraphrasing one and re-measures, so the dependence is a number rather than an
argument -- the open item named in README's "What is left".

Only the customer's OUTGOING TEXT changes. The hidden card, the disclosure
bookkeeping, `classify_constraint` routing, the catalog and the target are all
untouched, so the task is identical and only the surface form moves.

TWO PARAPHRASERS, ON PURPOSE (`--paraphraser`)
----------------------------------------------
A single paraphraser is a single hypothesis about how a customer might depart
from the script, and tuning ranking changes against one of them measures
robustness to *that* paraphraser rather than robustness to paraphrase. The two
here are built to fail in different directions, so agreement between them is
evidence and disagreement localises what a change actually bought.

  * `synonym` (default, P7's original) -- **changes vocabulary, preserves
    structure.** It swaps content words out of a hand-built lexicon
    ("cotton" -> "natural plant fibre") and leaves the sentence shape alone.
    It defeats every token-level feature at once: BM25, the evidence token
    coverage, and slot ownership all lose the word itself.

  * `structural` (P7-T2, this file's second half) -- **changes structure,
    preserves vocabulary.** It reframes a bare spec-sheet value as speech
    ("Material: 100% Cotton" -> "the material is 100 percent cotton"),
    inverts head nouns ("Rubber sole" -> "the sole is rubber"), rotates
    clause order and spells out units, while keeping every content token the
    card value carried. It leaves BM25 and token coverage almost untouched
    and goes after the two features that assume *exactness*: whole-value slot
    ownership (`slots.normalize_disclosure(said) in product.card_values`) and
    contiguous phrase containment (`evidence.phrase_coverage_from`).

Neither is "the" paraphrase distribution. Together they bracket it: a ranking
change that survives both is not surviving one lexicon.

The structural mode is deliberately LOSSLESS at every level, including level 3.
The synonym mode's level 3 drops 40% of the remaining content words, which
makes it an information-loss probe as well as a paraphrase probe; those are
different questions and this file now keeps one of each rather than conflating
them. `--audit` prints the token bookkeeping that backs that claim.

MEASURED (P7-T2, seed 20260831, BM25 route, this catalogue):

  run                                        score     overlap
  --level 0                    (control)     0.945497   0.4387
  --paraphraser structural --level 0         0.945497   0.4387
  --paraphraser structural --level 1         0.824792   0.3193
  --paraphraser structural --level 2         0.811102   0.3090
  --paraphraser structural --level 3         0.817477   0.2890
  --level 2                    (synonym)     0.696015   0.1741

Level 0 agrees to six decimals in both modes, which is the point of it. The
structural mode costs 0.134 against the synonym mode's 0.249, and that gap is
the finding rather than a shortfall: structural leaves the token bag intact,
so BM25 still retrieves and only the ranking degrades, while synonym takes the
words away and damages retrieval and ranking together. Two mechanisms, two
sizes of hole, which is what a held-out check is for.

Structural level 3 scores ABOVE level 2 (+0.006). The levels are ordered by
how much surface form moves, not by how much they hurt, and level 3's hedges
add common words that happen to help retrieval slightly. Reported rather than
tidied away: treating level 3 as "harder" would be an assumption, and it is
the reason the headline comparison in this file is level 2.

ISOLATING THE VALUE REWRITE from the wrapper (`--keep-carrier --keep-opener`,
which restores the simulator's own parseable framing and changes only the
constraint text):

  --paraphraser structural --level 2 --keep-carrier --keep-opener   0.853733
  --level 2                          --keep-carrier --keep-opener   0.729474

So roughly a third of each mode's damage at level 2 is the customer no longer
using a wrapper the orchestrator can parse, and the rest is the value text
itself. That split is shared plumbing and is common to both modes, which is
exactly why it has a flag: without it the two paraphrasers would look more
alike than their value rewrites actually are.

WHAT `--audit --level 2` SAYS about whether they are independent:

  mode         token retention   tokens added   whole value   contiguity
  structural            0.9971         0.0402        0.0000       0.4799
  synonym               0.3576         0.6836        0.0887       0.0877

The two axes move in opposite directions, which is the claim. Note that
synonym leaves 8.9% of values EXACTLY intact -- the words in them are simply
not in the lexicon -- while structural leaves 0%. That residue is the concrete
form of the overfitting risk this second mode exists to cover.

Usage:
    python3 -m scripts.paraphrase_eval --level 2 --paraphrase-category
    python3 -m scripts.paraphrase_eval --paraphraser structural --level 2
    python3 -m scripts.paraphrase_eval --paraphraser structural --level 2 --keep-carrier
    python3 -m scripts.paraphrase_eval --audit --level 2
    SHOPPING_AGENT_DENSE=1 python3 -m scripts.paraphrase_eval --level 2
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from dataclasses import dataclass
from typing import Callable

from evaluator import local_evaluator as ev
from shopping_agent import evidence as _evidence
from shopping_agent import slots as _slots
from shopping_agent.slots import stated_category as _stated_category

# Synonyms chosen to share NO token with the word they replace -- the FTS5 index
# is `unicode61` with no stemmer, and shopping_agent/text.py tokenizes on bare
# lowercased alphanumerics, so a substitution defeats BM25 and the evidence
# features alike. Meaning is preserved; only the surface form moves.
#
# The list is the top content words measured across all 800 constraint strings
# in the public set, so it covers the bulk of what customers actually say
# rather than words chosen to make a point.
SYNONYMS: dict[str, str] = {
    # materials -- the single densest cluster in the card values
    "polyester": "man made synthetic fibre",
    "cotton": "natural plant fibre",
    "leather": "tanned animal hide",
    "spandex": "elastane stretch yarn",
    "rayon": "viscose",
    "nylon": "polyamide",
    "rubber": "vulcanised latex",
    "alloy": "blended metal",
    "acrylic": "synthetic wool substitute",
    "wool": "sheep fleece",
    "denim": "twill weave jean cloth",
    "suede": "napped hide",
    "silk": "filament from silkworms",
    "linen": "flax cloth",
    "velvet": "pile weave cloth",
    "fleece": "brushed pile",
    "mesh": "open weave netting",
    "satin": "glossy weave",
    # construction and hardware
    "closure": "fastening",
    "zipper": "zip fastening",
    "zip": "sliding fastening",
    "button": "stud fastening",
    "buckle": "clasp",
    "elastic": "stretchy",
    "sole": "underside",
    "heel": "raised rear section",
    "lining": "inner layer",
    "pocket": "pouch",
    "collar": "neckband",
    "sleeve": "arm covering",
    "hood": "head covering",
    "strap": "band",
    # care
    "wash": "launder",
    "washable": "launderable",
    "machine": "appliance",
    "bleach": "whitening agent",
    "dry": "moisture free",
    "iron": "press flat",
    "tumble": "rotary drying",
    # descriptors
    "imported": "brought in from abroad",
    "lightweight": "low in mass",
    "durable": "long lasting",
    "soft": "gentle to the touch",
    "breathable": "air permeable",
    "waterproof": "impervious to rain",
    "adjustable": "able to be resized",
    "comfortable": "easy to wear",
    "stretch": "give",
    "casual": "informal",
    "fabric": "cloth",
    "material": "substance",
    "color": "shade",
    "colour": "shade",
    "design": "styling",
    "pattern": "motif",
    "style": "look",
    "fit": "cut",
    # people / colours (kept semantically faithful)
    "women": "ladies",
    "womens": "ladies",
    "men": "gentlemen",
    "mens": "gentlemen",
    "girls": "young ladies",
    "boys": "young gentlemen",
    "black": "jet coloured",
    "white": "ivory coloured",
    "blue": "azure coloured",
    "red": "crimson coloured",
    "green": "emerald coloured",
    "grey": "ash coloured",
    "gray": "ash coloured",
    "brown": "chestnut coloured",
    "pink": "rose coloured",
    "navy": "deep marine coloured",
}

_WORD = re.compile(r"[A-Za-z]+")

# Carrier phrases. The simulator's own wrappers are fixed strings; a real
# customer would not repeat them verbatim either.
CARRIERS = [
    "For that, what matters is: {}.",           # L0, the simulator's own
    "What I care about is that it is {}.",
    "It needs to be {}, that is the main thing.",
    "I would say {} is what I am after.",
]

OPENERS = [
    "I'm looking for {cat}. A key requirement is: {c}.",   # L0, the simulator's own
    "I want {cat}. It has to be {c}.",
    "I'm after {cat} — the thing that matters is {c}.",
    "Looking for {cat}, and it should be {c}.",
]


def substitute(text: str, level: int, rng: random.Random) -> str:
    """Reword `text` while preserving its meaning.

    level 0 -- identity (control)
    level 1 -- synonym substitution on known words only
    level 2 -- level 1, plus drop the structural punctuation a spec sheet has
               and a customer does not ("Material:alloy" -> "the substance is
               blended metal")
    level 3 -- level 2, plus drop 40% of the remaining unmapped content words,
               modelling a customer who summarises instead of reciting
    """
    if level <= 0:
        return text

    def repl(match: re.Match[str]) -> str:
        word = match.group(0)
        key = word.lower()
        if key not in SYNONYMS:
            return word
        return SYNONYMS[key]

    out = _WORD.sub(repl, text)

    if level >= 2:
        out = out.replace(":", " is ").replace(";", " and ").replace("/", " or ")
        out = re.sub(r"[【】♥■●★]+", " ", out)
        out = re.sub(r"\s+", " ", out).strip()

    if level >= 3:
        tokens = out.split()
        kept = [t for t in tokens if rng.random() > 0.4 or not _WORD.fullmatch(t)]
        out = " ".join(kept) if kept else out

    return out


def reword_category(category: str) -> str:
    """Say the same category in a different word order.

    Deliberately token-preserving: the bag of words is unchanged, so BM25 sees
    exactly the same evidence it always did. What breaks is *exact string*
    matching -- `category_exact` (weight 8.0) and E9's category retrieval
    filter, both of which compare the opener's category to a string the target
    reproduces character for character. This isolates the cost of that
    exactness assumption from every other effect in this script.
    """
    words = [w for w in re.split(r"[\s&]+", category) if w]
    if len(words) < 2:
        return category.lower()
    return " ".join(reversed([w.lower() for w in words]))


# ---------------------------------------------------------------------------
# Paraphraser 2 (P7-T2): structure moves, vocabulary stays.
#
# WHY A SECOND ONE AT ALL. The synonym paraphraser above is a ~130-entry
# hand-built lexicon. Four ranking changes tuned against it would be tuned
# against that lexicon -- a word it happens to miss is a word the agent keeps
# scoring on, and there would be no way to tell "robust to paraphrase" from
# "robust to this list". This one is the held-out check: it shares no table
# with the one above and it attacks a different mechanism, so a change that
# moves both is not moving a lexicon.
#
# WHAT IT ATTACKS. The simulator hands back whole card values, and two of the
# reranker's features are built on exactly that:
#
#   * slot ownership (weight 16.0, the table's dominant feature) asks whether
#     `slots.normalize_disclosure(said)` is an exact member of the candidate's
#     `card_values` set -- WHOLE-VALUE EQUALITY;
#   * phrase evidence (weight 6.0) asks whether the normalized disclosure
#     appears CONTIGUOUSLY in the candidate's token stream.
#
# Both survive any amount of rewording that leaves the value standing alone
# and in order, and both die the moment it is spoken rather than recited.
# Meanwhile BM25, the dense route and the token-coverage feature are bags of
# tokens and do not care about order at all. So this paraphraser is a clean
# separation: it should cost the two exactness features nearly everything and
# the bag-of-token features nearly nothing.
#
# MEASURED over the 800 disclosable constraints of the public set, which is
# what the transformation tables below were sized against rather than chosen
# to make a point:
#
#   shape                                    count  handled by
#   single bare word ("cotton", "Imported")    275  the speech frame only
#   2-4 words ("Rubber sole")                  286  frame + head-noun inversion
#   "label: value" ("color: black")            117  label reframe
#   >= 2 comma/period clauses                  191  clause rotation
#   contains a percentage                      121  "%" -> " percent"
#   contains a number+unit (mm/cm/g/oz/")       65  unit spelled out
#
# The frame is the workhorse because 61% of the constraints are one or two
# words long and have no internal structure to move; for those, turning the
# card value into a sentence is the only structural move available.
#
# LOSSLESS BY CONSTRUCTION, and checked. Every content token of the input
# appears in the output, except for the abbreviations in SURFACE_VARIANTS,
# which are replaced by their spelled-out form. That is the whole invariant,
# and tests/test_phase7_structural.py asserts it over all 800 constraints.
# It matters because an information-loss probe and a paraphrase probe answer
# different questions, and the synonym mode's level 3 already covers the
# first one.

# Abbreviations the structural customer spells out, and the tokens each one
# becomes. Declared as a table rather than done inline so the losslessness
# test has something to check against: a content token may leave the output
# only by appearing here, and only with its replacement present.
#
# Chosen from the units that actually occur after a number in the public
# set's constraints (mm 29, " 19, g 9, cm 3, pcs 2, oz 1); the rest are free
# and a private set may differ.
SURFACE_VARIANTS: dict[str, tuple[str, ...]] = {
    "mm": ("millimeters",),
    "cm": ("centimeters",),
    "oz": ("ounces",),
    "lb": ("pounds",),
    "lbs": ("pounds",),
    "pcs": ("pieces",),
    "pc": ("piece",),
    "g": ("grams",),
    "ct": ("carats",),
}

# Anchored on a preceding digit or space so the "mm" inside "swimming" and the
# "g" inside "grade" are untouched, with \b on the right for the same reason.
_UNIT_RE = re.compile(
    r"(?<=[\d\s])(" + "|".join(sorted(SURFACE_VARIANTS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
# "95%" -> "95 percent" and '0.5"' -> '0.5 inches'. Both are pure additions:
# neither "%" nor '"' is a token under shopping_agent/text.py's [a-z0-9]+, so
# the number survives and a word is gained. Deliberately NOT the fuller
# "one hundred percent" the brief suggests -- that would delete the token
# "100", which is exactly the information loss this probe must not commit.
#
# BOTH SIDES ARE PADDED WITH A SPACE, and that is load-bearing rather than
# cosmetic. 9 of the public set's 121 percentages are written glued to the
# next word ("70%Cotton"), and 40 of its 44 number+unit pairs are glued too
# ("18mm", "14g"). Substituting in place produced "70 percentCotton", whose
# only token is "percentcotton" -- the word "cotton" silently deleted, which
# would have made this an information-loss probe by accident. Caught by the
# losslessness check in `surface_variant`; the padding is the fix.
_PERCENT_RE = re.compile(r"\s*%\s*")
_INCH_MARK_RE = re.compile(r'(\d)\s*"')

# Trailing nouns that make a 2-4 word card value a <modifier> <head> phrase,
# so it can be inverted into "the <head> is <modifier>". Measured over the
# public set's 2-4 word constraints: closure 91, sole 43, lining 8. The rest
# cost nothing and cover the same construction elsewhere in the catalogue.
HEAD_NOUNS = frozenset({
    "closure", "sole", "outsole", "insole", "upper", "lining", "footbed",
    "material", "fabric", "strap", "heel", "toe", "collar", "cuff", "hem",
})

# A spec-sheet label: short, alphabetic, and followed by a colon. Capped at
# three words so "Stainless Steel Bike Bracelet: Handcrafted with ..." is
# treated as prose rather than as a label, which is what it is.
_LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z /&-]{0,28})\s*:\s*(\S.*)$", re.DOTALL)

# Sentence frames. Every word they add is either a stopword under
# shopping_agent/evidence.py's list or a single letter that tokenize() drops,
# so framing changes the SHAPE of the utterance and not its content-token
# multiset. That is the entire point of the mode, and it is why the frames
# are this dull: "I would want", "something like" and "ideally" all add
# content tokens, so they are held back to level 3 (see HEDGES).
#
# REJECTED: contractions ("it's cotton"). They read better and they cheat --
# `evidence._normalize_phrase` tokenizes on [a-z0-9]+, so "it's" becomes
# "it s" and the stray "s" defeats phrase containment through a tokenizer
# artifact rather than through structure. Every frame here is apostrophe-free
# on purpose.
#
# REJECTED: reframing INTO a label ("the material is: 100% Cotton"), the
# inverse transformation the brief lists. The orchestrator's _LEAD_IN_RE
# strips "<=120 chars ending in a copula> :" off the front of a disclosure,
# so that frame is undone by the agent before it is ever scored and
# whole-value ownership comes straight back. `_tidy` drops colons for the
# same reason.
LABELLED_FRAMES = (
    "the {label} is {body}",
    "the {label} has to be {body}",
    "for the {label} it is {body}",
)
BARE_FRAMES = (
    "it is {body}",
    "it has to be {body}",
)

# Level 3 only. These DO add content tokens ("think", "something", "ideally",
# "also"), which dilutes the token-coverage feature as well as the exactness
# features -- so level 3 is not a clean single-mechanism probe, and that is
# why the headline comparison in this file is level 2.
HEDGES = ("I think ", "something like ", "ideally ", "I would say ")

_DECORATION_RE = re.compile(r"[【】♥■●★]+")
_CLAUSE_RE = re.compile(r"[,.]\s+")


def _split_label(text: str) -> tuple[str | None, str]:
    """Separate a spec-sheet label from the value it labels.

    Two shapes, both measured on the public set: an explicit "Label: value"
    (117 constraints, 40 of them the generator's synthetic "color: x"), and a
    bare "<modifier> <head noun>" phrase such as "Rubber sole" (142). Either
    yields a label the frame can speak, which is what turns the card value
    into a sentence without inventing or dropping a word.
    """
    match = _LABEL_RE.match(text)
    if match and len(match.group(1).split()) <= 3:
        return match.group(1).strip().lower(), match.group(2).strip()
    words = text.split()
    if 2 <= len(words) <= 4 and words[-1].lower().strip('.,"') in HEAD_NOUNS:
        return words[-1].lower().strip('.,"'), " ".join(words[:-1])
    return None, text


_GLUED_UNIT_RE = re.compile(
    r"^(\d[\d.]*)(" + "|".join(sorted(SURFACE_VARIANTS, key=len, reverse=True)) + r")$"
)


def surface_variant(token: str) -> tuple[str, ...] | None:
    """The tokens `token` is allowed to become, or None if it may not change.

    This is the definition of "lossless" for the structural mode, in one
    place, used by both `audit` and tests/test_phase7_structural.py rather
    than restated in each.

    Two shapes. A bare abbreviation ("mm") becomes its spelled-out form. A
    glued number+unit ("18mm") is a SINGLE token under [a-z0-9]+, so spelling
    it out splits it in two -- the token "18mm" genuinely disappears, and what
    stands in for it is "18" plus "millimeters". Nothing is lost, but the
    bookkeeping has to say so explicitly instead of the check quietly passing.
    """
    lowered = token.lower()
    replacement = SURFACE_VARIANTS.get(lowered)
    if replacement is not None:
        return replacement
    match = _GLUED_UNIT_RE.match(lowered)
    if match:
        number = [part for part in match.group(1).split(".") if len(part) > 1]
        return tuple(number) + SURFACE_VARIANTS[match.group(2)]
    return None


def _reformat_units(text: str) -> str:
    """Spell out the units and the percent sign a spec sheet abbreviates."""
    out = _INCH_MARK_RE.sub(r"\1 inches ", text)
    out = _PERCENT_RE.sub(" percent ", out)
    return _UNIT_RE.sub(
        lambda m: " " + " ".join(SURFACE_VARIANTS[m.group(1).lower()]) + " ", out
    )


def _clauses(text: str) -> list[str]:
    return [part.strip() for part in _CLAUSE_RE.split(text) if part.strip()]


def _rotate_clauses(text: str, rng: random.Random) -> str:
    """Rotate a multi-clause value so a different clause leads.

    Rotation rather than a shuffle: it is the smallest change that guarantees
    a different leading clause, it preserves the relative order of everything
    else (so the result still reads like a person listing things), and it
    cannot drop or duplicate a clause the way a resample could. The token
    multiset is identical; only contiguity across the seam moves, which is
    exactly what phrase containment measures.
    """
    parts = _clauses(text)
    if len(parts) < 2:
        return text
    cut = 1 + rng.randrange(len(parts) - 1)
    return ", ".join(parts[cut:] + parts[:cut])


def _split_sentences(text: str, rng: random.Random) -> str:
    """Say one long value as two sentences instead of one."""
    parts = _clauses(text)
    if len(parts) < 2:
        return text
    cut = 1 + rng.randrange(len(parts) - 1)
    return ", ".join(parts[:cut]) + ". Also " + ", ".join(parts[cut:])


def _tidy(text: str) -> str:
    """Normalize the punctuation a customer would not pronounce.

    Colons go because leaving one in hands the agent's _LEAD_IN_RE a framing
    clause to strip, which would undo the reframe. Semicolons become full
    stops because `evidence.split_disclosures` splits the reply on ";" and a
    surviving one would fragment the disclosure -- a different effect from the
    one being measured, so it is removed rather than exploited.
    """
    out = _DECORATION_RE.sub(" ", text)
    out = out.replace(";", ".").replace(":", " ")
    return re.sub(r"\s+", " ", out).strip()


def restructure(text: str, level: int, rng: random.Random) -> str:
    """Say `text` as a person would, keeping every content token it carries.

    level 0 -- identity (control), and it draws nothing from `rng`
    level 1 -- speak the value: split off any spec-sheet label or head noun
               and put the remainder in a sentence frame. Defeats whole-value
               ownership.
    level 2 -- level 1, plus spell out units and percentages and rotate the
               clause order. Defeats contiguous phrase containment too.
    level 3 -- level 2, plus hedging and splitting one value across two
               sentences. Still lossless; unlike the synonym mode's level 3
               it deletes nothing.
    """
    if level <= 0:
        return text
    label, body = _split_label(text)
    if level >= 2:
        body = _reformat_units(body)
        body = _rotate_clauses(body, rng)
    if level >= 3:
        body = _split_sentences(body, rng)
    frames = LABELLED_FRAMES if label else BARE_FRAMES
    out = frames[rng.randrange(len(frames))].format(label=label or "", body=body)
    if level >= 3 and rng.random() < 0.5:
        out = HEDGES[rng.randrange(len(HEDGES))] + out
    return _tidy(out)


def restructure_phrase(text: str, level: int, rng: random.Random) -> str:
    """The same treatment for a noun phrase that must stay a noun phrase.

    `--paraphrase-category` feeds its result into "I'm looking for {}", where
    a sentence frame would produce "I'm looking for it is sandals". So the
    category gets the label, unit and punctuation handling and skips the
    frame; the synonym mode needs no such split because a word swapped for a
    word is still a noun phrase.
    """
    if level <= 0:
        return text
    label, body = _split_label(text)
    if level >= 2:
        body = _reformat_units(body)
        body = _rotate_clauses(body, rng)
    return _tidy(f"{label} {body}" if label else body)


@dataclass(frozen=True)
class Paraphraser:
    """How one mode rewrites a constraint value, and a bare noun phrase.

    Two callables rather than one because the category is spoken inside a
    template that already supplies the grammar; see restructure_phrase.
    """

    value: Callable[[str, int, random.Random], str]
    phrase: Callable[[str, int, random.Random], str]


PARAPHRASERS: dict[str, Paraphraser] = {
    "synonym": Paraphraser(substitute, substitute),
    "structural": Paraphraser(restructure, restructure_phrase),
}


def install_paraphrasing_customer(
    level: int,
    paraphrase_category: bool,
    seed: int,
    reword_cat: bool = False,
    keep_opener: bool = False,
    *,
    paraphraser: str = "synonym",
    keep_carrier: bool = False,
) -> dict:
    """Monkeypatch the simulator's two text-producing functions.

    `evaluate()` looks both up as module globals, so rebinding them here changes
    what the agent sees without touching disclosure bookkeeping or routing.

    `paraphraser` and `keep_carrier` are keyword-only and default to the
    behaviour that existed before them, so scripts/paraphrase_headroom.py and
    scripts/paraphrase_weight_probe.py -- which call this positionally -- are
    byte-for-byte unaffected.
    """
    para = PARAPHRASERS[paraphraser]
    real_reply = ev.customer_reply
    real_initial = ev.initial_message
    rng = random.Random(seed)
    stats = {"utterances": 0, "overlap": [], "openers": 0, "openers_parsed": 0}

    def measure(said: str, source_terms: set[str]) -> None:
        said_terms = {w.lower() for w in _WORD.findall(said) if len(w) > 1}
        if not said_terms:
            return
        stats["utterances"] += 1
        shared = said_terms & source_terms
        stats["overlap"].append(len(shared) / len(said_terms))

    def patched_reply(sample, ask_attribute, disclosed, boundary_used):
        before = set(disclosed)
        text, used = real_reply(sample, ask_attribute, disclosed, boundary_used)
        newly = disclosed - before
        if newly and text.startswith("For that, what matters is:"):
            values = [para.value(v, level, rng) for v in sorted(newly)]
            # Drawn whether or not the carrier is kept, and after the values
            # exactly as before this flag existed, so --keep-carrier moves the
            # wrapper and leaves every other paraphrase decision in the run
            # bit-for-bit unchanged. Same discipline as --keep-opener below.
            carrier_choice = rng.randrange(1, len(CARRIERS)) if level else 0
            keep = level == 0 or keep_carrier
            carrier = CARRIERS[0] if keep else CARRIERS[carrier_choice]
            text = carrier.format("; ".join(values))
        source = {w.lower() for v in newly for w in _WORD.findall(v) if len(w) > 1}
        if source:
            measure(text, source)
        return text, used

    def patched_initial(sample, category, disclosed):
        text = real_initial(sample, category, disclosed)
        if level == 0 and not reword_cat:
            stats["openers"] += 1
            if _stated_category(text) is not None:
                stats["openers_parsed"] += 1
            return text
        cat = para.phrase(category, level, rng) if paraphrase_category else category
        if reword_cat:
            cat = reword_category(cat)
        if level == 0:
            # Category-only probe: keep the simulator's own wording elsewhere.
            scenario = sample["scenario_type"]
            if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
                raw = str(sample["intent_card"]["hard_constraints"][0])
                text = f"I'm looking for {cat}. A key requirement is: {raw}."
                stats["openers"] += 1
                if _stated_category(text) is not None:
                    stats["openers_parsed"] += 1
                return text
            if scenario == "intent_override":
                old = str(sample["behavior"]["override"]["old_value"])
                text = f"I'm looking for {cat}. {old}"
            else:
                text = f"I'm looking for {cat}, but I'm still exploring."
            stats["openers"] += 1
            if _stated_category(text) is not None:
                stats["openers_parsed"] += 1
            return text
        scenario = sample["scenario_type"]
        if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
            raw = str(sample["intent_card"]["hard_constraints"][0])
            # Drawn whether or not the opener is kept, and only on the branch
            # that drew before this flag existed, so --keep-opener moves the
            # opener's wording and leaves every other paraphrase decision in
            # the run bit-for-bit unchanged.
            opener_choice = rng.randrange(1, len(OPENERS))
            opener = OPENERS[0] if keep_opener else OPENERS[opener_choice]
            text = opener.format(cat=cat, c=para.value(raw, level, rng))
        elif scenario == "intent_override":
            old = str(sample["behavior"]["override"]["old_value"])
            lead = "I'm looking for" if keep_opener else "I want"
            text = f"{lead} {cat}. {para.value(old, level, rng)}"
        elif keep_opener:
            text = f"I'm looking for {cat}, but I'm still exploring."
        else:
            text = f"I want {cat}, though I am still browsing."
        stats["openers"] += 1
        if _stated_category(text) is not None:
            stats["openers_parsed"] += 1
        return text

    ev.customer_reply = patched_reply
    ev.initial_message = patched_initial
    return stats


def constraint_values(dataset: str, catalog: str) -> list[str]:
    """Every value the simulator can ever disclose, in dataset order.

    Materialized the way `evaluate()` does it, so this is the same 800 strings
    the run itself would paraphrase and not a re-derivation that could drift.
    """
    samples = ev.load_jsonl(dataset)
    _, _, products = ev.catalog_index(catalog)
    values: list[str] = []
    for sample in samples:
        card, _ = ev.materialize_hidden_fields(sample, products)
        values.extend(str(v) for v in card.get("hard_constraints", []))
        values.extend(str(v) for v in card.get("soft_preferences", []))
    return values


def _content(text: str) -> frozenset[str]:
    sets = _evidence.disclosure_token_sets([text])
    return sets[0] if sets else frozenset()


def _phrase(text: str) -> str:
    found = _evidence.disclosure_phrases([text])
    return found[0][0] if found else ""


def audit(dataset: str, catalog: str, level: int, seed: int) -> dict:
    """What each paraphraser actually does to the values, side by side.

    The point of running two paraphrasers is that they attack different
    things, and that claim has to be checkable rather than asserted. Four
    numbers per mode say it:

      * `content_token_retention` -- share of the value's content tokens still
        present in the rewritten text. This is the VOCABULARY axis; the
        synonym mode is built to drive it down and the structural mode is
        built to leave it at 1.0.
      * `content_tokens_added` -- share of the OUTPUT's content tokens that
        were not in the input, i.e. how much noise the rewrite injects.
      * `whole_value_survives` -- share of values whose rewrite still
        normalizes to the card value, so `slot_evidence` (weight 16.0) can
        still fire. This is the EXACTNESS axis.
      * `contiguity_survives` -- share of values that still appear as an
        unbroken run of tokens somewhere inside the rewritten utterance. Read
        it as "how much of the value is still there in order", not as
        "phrase_evidence still fires": the agent scores the whole utterance it
        was handed, frame and all, against the product text, so the feature
        dies well before this number does. The end-to-end run is what measures
        that; this says how recoverable the value remains in principle.

    A mode that is a genuine held-out check for the other shows a different
    pair of axes moving.

    `token_preserving` is the structural mode's hard invariant: every content
    token that left the value is a `surface_variant` whose replacement is
    present. The synonym mode reports False here and that is not a fault --
    replacing tokens IS its mechanism, and `replaced_values` counts how many
    values it reaches. The flag exists so the structural mode cannot quietly
    start deleting information and be mistaken for a harder paraphraser.
    """
    values = constraint_values(dataset, catalog)
    report: dict = {"level": level, "seed": seed, "values": len(values), "modes": {}}
    for name, para in sorted(PARAPHRASERS.items()):
        rng = random.Random(seed)
        retained = added = 0.0
        with_src = with_dst = with_phrase = whole = contiguous = 0
        lossy: list[str] = []
        for value in values:
            out = para.value(value, level, rng)
            src, dst = _content(value), _content(out)
            # Each rate gets the denominator it is actually defined over. A
            # value with no content tokens at all ("100%", say) has no
            # retention to report, and averaging it in as a zero understated
            # the structural mode by a quarter of a point of retention -- a
            # metric artifact that looked exactly like real information loss.
            if src:
                with_src += 1
                retained += len(src & dst) / len(src)
            if dst:
                with_dst += 1
                added += len(dst - src) / len(dst)
            if _slots.normalize_disclosure(out) == _slots.normalize_disclosure(value):
                whole += 1
            phrase = _phrase(value)
            if phrase:
                with_phrase += 1
                if f" {phrase} " in f" {_phrase(out)} ":
                    contiguous += 1
            for token in src - dst:
                replacement = surface_variant(token)
                if replacement is None or not set(replacement) <= dst:
                    lossy.append(f"{value!r} lost {token!r} -> {out!r}")
        total = len(values) or 1
        report["modes"][name] = {
            "content_token_retention": round(retained / (with_src or 1), 4),
            "content_tokens_added": round(added / (with_dst or 1), 4),
            "whole_value_survives": round(whole / total, 4),
            "contiguity_survives": round(contiguous / (with_phrase or 1), 4),
            "token_preserving": not lossy,
            "replaced_values": len({item.split(" lost ")[0] for item in lossy}),
            "replacement_examples": lossy[:5],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--level", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--paraphrase-category", action="store_true")
    parser.add_argument("--reword-category", action="store_true")
    parser.add_argument(
        "--keep-opener",
        action="store_true",
        help="Paraphrase the disclosures but leave the opening line in the "
             "simulator's own wording, so slots.stated_category still parses "
             "it. Isolates the cost of paraphrased constraints from the cost "
             "of an unparseable opener.",
    )
    parser.add_argument(
        "--paraphraser",
        default="synonym",
        choices=sorted(PARAPHRASERS),
        help="Which paraphrase mechanism to apply. `synonym` moves vocabulary "
             "and keeps structure; `structural` moves structure and keeps "
             "vocabulary. Default `synonym`, so every command that predates "
             "this flag reproduces its old number exactly.",
    )
    parser.add_argument(
        "--keep-carrier",
        action="store_true",
        help="Paraphrase the disclosed values but wrap them in the "
             "simulator's own \"For that, what matters is:\" carrier. That "
             "carrier is the only one with a colon, so it is the only one the "
             "orchestrator strips back to a bare constraint -- which makes "
             "this the flag that isolates what the value rewriting itself "
             "costs, separately from what losing a parseable wrapper costs.",
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Do not run the agent. Print what each paraphraser does to the "
             "dataset's constraint values -- token retention, whole-value "
             "survival, contiguity survival -- which is what says whether the "
             "two modes are attacking different mechanisms or the same one.",
    )
    args = parser.parse_args()

    if args.audit:
        print(json.dumps(audit(args.dataset, args.catalog, args.level, args.seed), indent=2))
        return

    stats = install_paraphrasing_customer(
        args.level,
        args.paraphrase_category,
        args.seed,
        args.reword_category,
        args.keep_opener,
        paraphraser=args.paraphraser,
        keep_carrier=args.keep_carrier,
    )

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    result = ev.evaluate(ev.Agent(args.catalog), samples, catalog_ids, categories, products)

    overlap = statistics.fmean(stats["overlap"]) if stats["overlap"] else float("nan")
    # The default mode keeps the bare "L2" label it has always printed, so a
    # command that predates --paraphraser produces the identical line; only a
    # non-default mode prefixes itself. The `paraphraser` field below is
    # purely additive and disambiguates either way.
    prefix = "" if args.paraphraser == "synonym" else args.paraphraser + "-"
    summary = {
        "label": args.label or ("%sL%d%s" % (
            prefix, args.level, "+cat" if args.paraphrase_category else ""
        )),
        "paraphraser": args.paraphraser,
        "keep_carrier": args.keep_carrier,
        "level": args.level,
        "paraphrase_category": args.paraphrase_category,
        "utterances_measured": stats["utterances"],
        "mean_verbatim_overlap": round(overlap, 4),
        "keep_opener": args.keep_opener,
        "openers": stats["openers"],
        "openers_parsed": stats["openers_parsed"],
        "score": result["recommended_technical_score"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

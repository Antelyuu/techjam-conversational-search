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

Usage:
    python3 -m scripts.paraphrase_eval --level 2 --paraphrase-category
    SHOPPING_AGENT_DENSE=1 python3 -m scripts.paraphrase_eval --level 2
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics

from evaluator import local_evaluator as ev

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


def install_paraphrasing_customer(
    level: int, paraphrase_category: bool, seed: int, reword_cat: bool = False
) -> dict:
    """Monkeypatch the simulator's two text-producing functions.

    `evaluate()` looks both up as module globals, so rebinding them here changes
    what the agent sees without touching disclosure bookkeeping or routing.
    """
    real_reply = ev.customer_reply
    real_initial = ev.initial_message
    rng = random.Random(seed)
    stats = {"utterances": 0, "overlap": []}

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
            values = [substitute(v, level, rng) for v in sorted(newly)]
            carrier = CARRIERS[0] if level == 0 else CARRIERS[rng.randrange(1, len(CARRIERS))]
            text = carrier.format("; ".join(values))
        source = {w.lower() for v in newly for w in _WORD.findall(v) if len(w) > 1}
        if source:
            measure(text, source)
        return text, used

    def patched_initial(sample, category, disclosed):
        text = real_initial(sample, category, disclosed)
        if level == 0 and not reword_cat:
            return text
        cat = substitute(category, level, rng) if paraphrase_category else category
        if reword_cat:
            cat = reword_category(cat)
        if level == 0:
            # Category-only probe: keep the simulator's own wording elsewhere.
            scenario = sample["scenario_type"]
            if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
                raw = str(sample["intent_card"]["hard_constraints"][0])
                return f"I'm looking for {cat}. A key requirement is: {raw}."
            if scenario == "intent_override":
                old = str(sample["behavior"]["override"]["old_value"])
                return f"I'm looking for {cat}. {old}"
            return f"I'm looking for {cat}, but I'm still exploring."
        scenario = sample["scenario_type"]
        if scenario == "buying" and sample["intent_card"].get("hard_constraints"):
            raw = str(sample["intent_card"]["hard_constraints"][0])
            opener = OPENERS[rng.randrange(1, len(OPENERS))]
            text = opener.format(cat=cat, c=substitute(raw, level, rng))
        elif scenario == "intent_override":
            old = str(sample["behavior"]["override"]["old_value"])
            text = f"I want {cat}. {substitute(old, level, rng)}"
        else:
            text = f"I want {cat}, though I am still browsing."
        return text

    ev.customer_reply = patched_reply
    ev.initial_message = patched_initial
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--level", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--paraphrase-category", action="store_true")
    parser.add_argument("--reword-category", action="store_true")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    stats = install_paraphrasing_customer(
        args.level, args.paraphrase_category, args.seed, args.reword_category
    )

    samples = ev.load_jsonl(args.dataset)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    result = ev.evaluate(ev.Agent(args.catalog), samples, catalog_ids, categories, products)

    overlap = statistics.fmean(stats["overlap"]) if stats["overlap"] else float("nan")
    summary = {
        "label": args.label or ("L%d%s" % (args.level, "+cat" if args.paraphrase_category else "")),
        "level": args.level,
        "paraphrase_category": args.paraphrase_category,
        "utterances_measured": stats["utterances"],
        "mean_verbatim_overlap": round(overlap, 4),
        "score": result["recommended_technical_score"],
        "hit_rate_at_10": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

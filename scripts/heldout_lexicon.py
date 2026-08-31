"""A second paraphraser, built to disagree with the first one's vocabulary.

The house rule is that a gain measured against `scripts/paraphrase_eval.py`'s
hand-built synonym list has to be confirmed somewhere that list did not reach,
because tuning against one word list overfits to *it* rather than to
paraphrasing. E10 used `--paraphrase-category` for that, and it is the right
probe for the category fix -- but it substitutes the *category's* words while
leaving the disclosure vocabulary exactly as the original lexicon left it, so
it is not a held-out test of a feature that scores disclosures.

This is. It is built by the same method as the original -- the measured top
content words across all 800 public constraint strings -- but restricted to
the words the original does **not** contain, so the two vocabularies are
disjoint by construction (asserted below, not assumed). A feature that only
recovers the first list's words scores nothing here.

It is a weaker paraphraser than the original, and deliberately reported as
one: it rewrites 10.8% of content-word token mass against the original's
29.4%, so scores under it sit much closer to the verbatim control. What it
measures is not how much damage a paraphrase does but whether a gain
*transfers* to words nobody tuned against.

Usage: passed as `--heldout` to scripts/semantic_evidence_probe.py.
"""

from __future__ import annotations

# Each replacement shares no token with the word it replaces -- the same rule
# scripts/paraphrase_eval.SYNONYMS follows, so a substitution defeats the
# non-stemming FTS5 index and the exact evidence features alike.
HELDOUT_SYNONYMS: dict[str, str] = {
    # construction and form
    "pull": "tug",
    "hand": "manual",
    "made": "constructed",
    "measures": "spans",
    "approximately": "roughly",
    "size": "dimension",
    "arch": "instep curve",
    "shaft": "upper tube",
    "drawstring": "tie cord",
    "weight": "mass",
    # jewellery and accessories, the categories the original list barely touches
    "earrings": "ear ornaments",
    "hoop": "circular loop",
    "bracelet": "wrist ornament",
    "watch": "timepiece",
    "nose": "nasal",
    "socks": "foot coverings",
    # materials the original list does not carry
    "synthetic": "artificial",
    "gold": "yellow precious metal",
    "stainless": "rustproof",
    "steel": "hardened metal",
    "water": "liquid",
    # qualities
    "quality": "grade",
    "high": "elevated",
    "comfort": "ease",
    "perfect": "ideal",
    "easy": "simple",
    "quick": "rapid",
    "long": "lengthy",
    "light": "not heavy",
    "warm": "toasty",
    "cool": "chilly",
    "smooth": "sleek",
    "thick": "chunky",
    "thin": "slender",
    "stretchy": "pliable",
    "slip": "slide",
    "wear": "don",
    "keep": "retain",
    "colors": "hues",
    "featuring": "showcasing",
}


def install() -> None:
    """Swap the lexicon `paraphrase_eval.substitute()` reads.

    `substitute` looks SYNONYMS up as a module global on every call, so
    replacing the dict's contents redirects it without touching the customer
    installation, the carrier phrases, or the rng draw order -- the run
    differs from a normal one only in which words get replaced.
    """
    from scripts import paraphrase_eval

    overlap = set(HELDOUT_SYNONYMS) & set(paraphrase_eval.SYNONYMS)
    if overlap:
        raise AssertionError(
            f"held-out lexicon must be disjoint from the original; shares {sorted(overlap)}"
        )
    paraphrase_eval.SYNONYMS.clear()
    paraphrase_eval.SYNONYMS.update(HELDOUT_SYNONYMS)

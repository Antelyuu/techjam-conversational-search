from __future__ import annotations

import re

from .contracts import Constraint

Candidate = tuple[str, object, str]  # (attribute, value, strength)

CATEGORY_GROUPS: dict[str, tuple[str, ...]] = {
    "footwear": ("sneakers", "shoes", "boots", "sandals", "heels", "flats", "slippers"),
    "jewelry": ("necklace", "bracelet", "ring", "earrings", "earring", "watch", "pendant", "jewelry"),
    "bags": ("backpack", "handbag", "purse", "bag", "wallet", "tote"),
    "apparel": ("t-shirt", "shirt", "dress", "jacket", "coat", "sweater", "hoodie", "pants", "jeans", "shorts", "skirt", "socks"),
    "accessories": ("belt", "hat", "scarf", "gloves", "sunglasses"),
}

CATEGORY_TERMS: tuple[str, ...] = tuple(
    sorted((term for terms in CATEGORY_GROUPS.values() for term in terms), key=len, reverse=True)
)

USE_CASE_WORDS = (
    "running", "hiking", "walking", "gym", "workout", "training",
    "outdoor", "winter", "summer", "everyday", "travel", "yoga", "swimming", "work",
)

STYLE_WORDS = (
    "casual", "formal", "athletic", "classic", "vintage", "modern",
    "minimalist", "chunky", "slim", "oversized", "wedding", "party",
)

FEATURE_WORDS = (
    "waterproof", "breathable", "lightweight", "slip-resistant", "non-slip",
    "water-resistant", "adjustable", "stretch", "quick-dry", "insulated", "cushioned",
)

COLOR_WORDS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "tan", "beige", "silver", "gold", "navy",
    "teal", "maroon", "cream", "khaki",
)

MATERIAL_WORDS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "suede", "canvas", "denim", "mesh", "rubber",
)

BRANDS = (
    "Nike", "Adidas", "Puma", "Reebok", "New Balance", "Skechers", "Vans",
    "Converse", "Fila", "Asics", "Timberland", "Clarks", "Columbia", "Champion",
    "Under Armour", "Ralph Lauren", "Tommy Hilfiger", "Calvin Klein", "Levi's",
    "Guess", "Coach", "Michael Kors", "Pandora", "Fossil",
)

# Sizes that are words in their own right and safe to match bare.
LETTER_SIZES = ("xxl", "xl", "xs", "small", "medium", "large", "wide", "narrow")

# Single-letter sizes only count next to an explicit size cue. Matched bare
# they are a trap: \b treats an apostrophe as a word boundary, so "s", "m" and
# "l" match inside "it's", "I'm" and "I'll". Every evaluator session opens with
# "I'm looking for ...", which silently set size="m" on all 200 of them --
# polluting the query text with a junk term and, once P4 arrived, permanently
# blocking the size question because the slot looked fixed.
# "xs" is absent because LETTER_SIZES already matches it bare, and the letter
# branch is checked first -- listing it here was dead (P4 review).
SIZE_ABBREVIATIONS = ("s", "m", "l")


def _word_re(words: tuple[str, ...]) -> re.Pattern[str]:
    escaped = sorted((re.escape(word) for word in words), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


CATEGORY_RE = _word_re(CATEGORY_TERMS)
USE_CASE_RE = _word_re(USE_CASE_WORDS)
STYLE_RE = _word_re(STYLE_WORDS)
FEATURE_RE = _word_re(FEATURE_WORDS)
COLOR_RE = _word_re(COLOR_WORDS)
MATERIAL_RE = _word_re(MATERIAL_WORDS)
BRAND_RE = _word_re(BRANDS)
LETTER_SIZE_RE = _word_re(LETTER_SIZES)
NUMERIC_SIZE_RE = re.compile(r"\bsize\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)
SIZE_ABBREVIATION_RE = re.compile(
    r"\bsizes?\s*[:\-]?\s*(" + "|".join(SIZE_ABBREVIATIONS) + r")\b", re.IGNORECASE
)

BUDGET_CEILING_RE = re.compile(
    r"(?:under|below|less than)\s*\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars)?"
    r"|\$?\s*(\d+(?:\.\d+)?)\s*(?:dollars)?\s*or less"
    r"|budget(?:\s*(?:of|is))?\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
BUDGET_TARGET_RE = re.compile(r"(?:around|about|near)\s*\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

OVERRIDE_CUE_RE = re.compile(
    r"\bactually\b|\binstead\b|\bignore (?:my|that|the|earlier)\b|\bchange of mind\b|"
    r"\bsomething different\b|\bnevermind\b|\bscratch that\b|\bno longer\b",
    re.IGNORECASE,
)

BROWSING_CUE_RE = re.compile(
    r"\bjust (?:looking|browsing)\b|\bstill (?:exploring|looking|browsing)\b|"
    r"\bnot sure\b|\bexploring\b|\bany suggestions\b|\bbrowsing\b",
    re.IGNORECASE,
)

HARD_CUE_RE = re.compile(r"\b(?:need|must|require|has to|only|definitely)\b", re.IGNORECASE)
SOFT_CUE_RE = re.compile(r"\b(?:prefer|ideally|would like|maybe|possibly|if possible|nice to have)\b", re.IGNORECASE)

DEFAULT_HARD_ATTRIBUTES = {"category"}


def _strength(attribute: str, text: str) -> str:
    if HARD_CUE_RE.search(text):
        return "hard"
    if SOFT_CUE_RE.search(text):
        return "soft"
    return "hard" if attribute in DEFAULT_HARD_ATTRIBUTES else "soft"


def _canonical_brand(matched_text: str) -> str:
    lowered = matched_text.lower()
    for brand in BRANDS:
        if brand.lower() == lowered:
            return brand
    return matched_text


def extract_candidate_slots(text: str) -> list[Candidate]:
    """Deterministic, regex-based slot extraction. Never raises; returns
    an empty list for empty, irrelevant, or unsupported input. Each
    attribute yields at most one candidate per message."""
    if not text:
        return []
    candidates: list[Candidate] = []

    budget_match = BUDGET_CEILING_RE.search(text)
    if budget_match:
        amount = next(group for group in budget_match.groups() if group is not None)
        candidates.append(("budget", float(amount), "hard"))
    else:
        target_match = BUDGET_TARGET_RE.search(text)
        if target_match:
            candidates.append(("budget", float(target_match.group(1)), "soft"))

    category_match = CATEGORY_RE.search(text)
    if category_match:
        value = category_match.group(0).lower()
        candidates.append(("category", value, _strength("category", text)))

    use_case_match = USE_CASE_RE.search(text)
    if use_case_match:
        candidates.append(("use_case", use_case_match.group(0).lower(), _strength("use_case", text)))

    style_match = STYLE_RE.search(text)
    if style_match:
        candidates.append(("style", style_match.group(0).lower(), _strength("style", text)))

    color_match = COLOR_RE.search(text)
    if color_match:
        candidates.append(("color", color_match.group(0).lower(), _strength("color", text)))

    material_match = MATERIAL_RE.search(text)
    if material_match:
        candidates.append(("material", material_match.group(0).lower(), _strength("material", text)))

    brand_match = BRAND_RE.search(text)
    if brand_match:
        candidates.append(("brand", _canonical_brand(brand_match.group(0)), _strength("brand", text)))

    numeric_size_match = NUMERIC_SIZE_RE.search(text)
    abbreviation_match = SIZE_ABBREVIATION_RE.search(text)
    letter_size_match = LETTER_SIZE_RE.search(text)
    if numeric_size_match:
        candidates.append(("size", numeric_size_match.group(1), _strength("size", text)))
    elif letter_size_match:
        candidates.append(("size", letter_size_match.group(0).lower(), _strength("size", text)))
    elif abbreviation_match:
        candidates.append(("size", abbreviation_match.group(1).lower(), _strength("size", text)))

    feature_match = FEATURE_RE.search(text)
    if feature_match:
        candidates.append(("feature", feature_match.group(0).lower(), _strength("feature", text)))

    return candidates


def detect_override_cue(text: str) -> bool:
    return bool(text) and OVERRIDE_CUE_RE.search(text) is not None


def classify_intent(text: str, candidates: list[Candidate], override_triggered: bool) -> str:
    """Transparent rule-based routing. Override takes precedence; a hard
    constraint implies Buying; explicit vague phrasing implies Browsing;
    no signal at all is Unknown."""
    if override_triggered:
        return "override"
    if any(strength == "hard" for _, _, strength in candidates):
        return "buying"
    if BROWSING_CUE_RE.search(text or ""):
        return "browsing"
    if not candidates:
        return "unknown"
    return "browsing"


def to_constraints(candidates: list[Candidate], turn: int) -> list[Constraint]:
    return [Constraint(attribute=a, value=v, strength=s, source_turn=turn) for a, v, s in candidates]

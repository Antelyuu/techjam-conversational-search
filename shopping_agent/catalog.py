from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)")
LOWER_BOUND_PRICE_RE = re.compile(r"\b(?:from|starting at|as low as)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ProductRecord:
    """Normalized view of one catalog row. Never raises on missing or
    mixed-type source fields (dict, list, scalar, or absent)."""

    parent_asin: str
    title: str
    categories: tuple[str, ...]
    price: float | None
    # True when `price` is a lower bound (e.g. a "from $X" listing across
    # variants), not the exact price of the specific product.
    price_is_lower_bound: bool
    searchable_text: str


def _flatten(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _normalize_categories(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = value.split(",")
    elif value is None:
        raw = []
    else:
        raw = [value]
    return tuple(str(item).strip().lower() for item in raw if str(item).strip())


def _normalize_price(value: object) -> tuple[float | None, bool]:
    if value is None or value == "":
        return None, False
    if isinstance(value, (int, float)):
        return float(value), False
    text = str(value)
    match = PRICE_RE.search(text)
    if not match:
        return None, False
    return float(match.group(1)), LOWER_BOUND_PRICE_RE.search(text) is not None


def normalize_product(raw: dict) -> ProductRecord:
    searchable_text = " ".join(
        _flatten(raw.get(field))
        for field in ("title", "categories", "features", "details", "store", "description")
    )
    price, price_is_lower_bound = _normalize_price(raw.get("price"))
    return ProductRecord(
        parent_asin=str(raw["parent_asin"]),
        title=str(raw.get("title") or ""),
        categories=_normalize_categories(raw.get("categories")),
        price=price,
        price_is_lower_bound=price_is_lower_bound,
        searchable_text=searchable_text,
    )


def load_catalog(catalog_path: str | Path) -> dict[str, ProductRecord]:
    products: dict[str, ProductRecord] = {}
    with Path(catalog_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = normalize_product(json.loads(line))
            products[record.parent_asin] = record
    return products

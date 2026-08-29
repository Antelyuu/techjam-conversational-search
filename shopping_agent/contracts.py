from __future__ import annotations

from dataclasses import dataclass, field

ALLOWED_ATTRIBUTES = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)

INTENT_LABELS = ("buying", "browsing", "override", "unknown")

STRENGTHS = ("hard", "soft")


@dataclass(frozen=True)
class Constraint:
    """One extracted customer requirement.

    strength "hard" means a firm requirement (explicit budget ceiling,
    "need"/"must" language, or category); "soft" is a ranking preference
    that should not eliminate candidates by default.
    """

    attribute: str
    value: object
    strength: str
    source_turn: int


@dataclass
class SessionState:
    """All per-session memory. One instance per session_id; never shared."""

    session_id: str
    user_profile: dict
    constraints: dict[str, Constraint] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    intent: str = "unknown"
    asked_attributes: set[str] = field(default_factory=set)
    rejected_attributes: set[str] = field(default_factory=set)
    clarification_turns: int = 0

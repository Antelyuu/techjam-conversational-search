from __future__ import annotations

from .contracts import Constraint, SessionState
from .intent import CATEGORY_GROUPS, Candidate

# Slots that only make sense within one coarse category group (e.g. shoe
# size or a running/hiking use case do not carry over from footwear to
# jewelry). Budget, color, and material are treated as surviving a
# category change by default; only an explicit new value or rejection
# replaces them.
CATEGORY_DEPENDENT_ATTRIBUTES = {"use_case", "style", "size"}


def _category_group(value: str) -> str | None:
    for group, members in CATEGORY_GROUPS.items():
        if value in members:
            return group
    return None


class SessionStore:
    """Owns one isolated SessionState per session_id. Identical anonymized
    profiles never share history or constraints across session_ids."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create(self, session_id: str, user_profile: dict) -> SessionState:
        state = SessionState(session_id=session_id, user_profile=user_profile)
        self._sessions[session_id] = state
        return state

    def get(self, session_id: str) -> SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise RuntimeError("reset must be called before respond") from exc


def apply_candidates(state: SessionState, candidates: list[Candidate], turn: int) -> None:
    """Accumulate compatible slots; same-slot values replace only that
    slot. A category change to a different coarse group clears
    category-dependent slots the new message did not resupply."""
    new_category = next((value for attribute, value, _ in candidates if attribute == "category"), None)
    if new_category is not None:
        old_category = state.constraints.get("category")
        old_group = _category_group(str(old_category.value)) if old_category else None
        new_group = _category_group(str(new_category))
        if old_category is not None and new_group is not None and new_group != old_group:
            supplied_this_turn = {attribute for attribute, _, _ in candidates}
            for attribute in list(state.constraints):
                if attribute in CATEGORY_DEPENDENT_ATTRIBUTES and attribute not in supplied_this_turn:
                    del state.constraints[attribute]

    for attribute, value, strength in candidates:
        state.constraints[attribute] = Constraint(
            attribute=attribute, value=value, strength=strength, source_turn=turn,
        )


def build_query_text(state: SessionState, latest_message: str) -> str:
    """Cumulative search text: normalized slot values from the whole
    conversation, then everything the customer has disclosed in answer to a
    question, then the latest raw message, so single-turn recall is preserved
    alongside accumulated context.

    Disclosed text is carried explicitly because slot extraction only
    recognizes known vocabulary. An answer like "Machine wash cold" yields no
    slot, so without this it would reach exactly one query and then vanish --
    which is most of the value of having asked."""
    parts = [str(constraint.value) for constraint in state.constraints.values()]
    parts.extend(state.disclosed_text)
    parts.append(latest_message)
    return " ".join(parts)

from __future__ import annotations

from . import intent as intent_module
from . import state as state_module
from .contracts import SessionState


class ConversationOrchestrator:
    """Owns session state across turns. Retrieval and response formatting
    stay in starter/agent.py, the official Agent entry point; this class
    only produces the updated SessionState and a cumulative query text."""

    def __init__(self) -> None:
        self.store = state_module.SessionStore()

    def reset(self, session_id: str, user_profile: dict) -> SessionState:
        return self.store.create(session_id, user_profile)

    def process_turn(self, session_id: str, user_message: str, turn: int) -> tuple[SessionState, str]:
        state = self.store.get(session_id)
        state.history.append(user_message)

        candidates = intent_module.extract_candidate_slots(user_message)
        override_triggered = intent_module.detect_override_cue(user_message)
        state_module.apply_candidates(state, candidates, turn)
        state.intent = intent_module.classify_intent(user_message, candidates, override_triggered)

        query_text = state_module.build_query_text(state, user_message)
        return state, query_text

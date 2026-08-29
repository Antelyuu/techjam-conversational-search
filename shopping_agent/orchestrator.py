from __future__ import annotations

import re

from . import clarification
from . import intent as intent_module
from . import state as state_module
from .contracts import SearchRequest, SessionState

# Customer turns that carry no information about the product wanted. Keeping
# their wording out of the query matters: the nudge below would otherwise
# contribute "options", "specific" and "attribute" as search terms every time
# the agent failed to ask a question.
_EMPTY_REPLY_RE = re.compile(
    r"not quite right yet|do(?:n'?t| not) have an?\s+(?:additional\s+)?preference",
    re.IGNORECASE,
)

# Answers arrive as "For that, what matters is: X; Y." -- a lead-in clause
# followed by the content. Dropping everything up to the first colon keeps the
# constraint text and discards the framing, and leaves a message without a
# colon untouched.
_LEAD_IN_RE = re.compile(r"^[^:]{0,60}:\s*")

# A reversal stated up front, which is what actually displaces the answer to
# our question -- as opposed to the word "instead" appearing somewhere inside
# a product description the customer has just quoted at us.
_ANSWER_REPLACED_RE = re.compile(
    r"^\s*(?:actually|never\s*mind|nevermind|scratch that)\b"
    r"|\bignore (?:my|that|the) (?:earlier|previous|last)\b",
    re.IGNORECASE,
)


class ConversationOrchestrator:
    """Owns session state across turns. Retrieval and response formatting
    stay in starter/agent.py, the official Agent entry point; this class
    only produces the updated SessionState and the SearchRequest for the
    current turn."""

    def __init__(self) -> None:
        self.store = state_module.SessionStore()

    def reset(self, session_id: str, user_profile: dict) -> SessionState:
        return self.store.create(session_id, user_profile)

    def process_turn(self, session_id: str, user_message: str, turn: int, top_k: int) -> SearchRequest:
        state = self.store.get(session_id)
        state.history.append(user_message)

        candidates = intent_module.extract_candidate_slots(user_message)
        override_triggered = intent_module.detect_override_cue(user_message)
        self._absorb_answer(state, user_message, override_triggered)
        state_module.apply_candidates(state, candidates, turn)
        state.intent = intent_module.classify_intent(user_message, candidates, override_triggered)

        query_text = state_module.build_query_text(state, user_message)
        return SearchRequest(query_text=query_text, state=state, top_k=top_k)

    @staticmethod
    def _absorb_answer(state: SessionState, user_message: str, override_triggered: bool) -> None:
        """Fold the reply to our previous question back into session state.

        Three outcomes matter, and they are not interchangeable:

        - The customer named content. Keep it; it is the whole point of asking.
        - The customer has nothing for that attribute. Never ask it again.
        - The question was not actually answered -- a Boundary session spending
          its one free decline, or an Intent Override message arriving in place
          of the reply. The attribute is still unanswered, so put it back.
        """
        asked = state.pending_attribute
        state.pending_attribute = None

        rejected, boundary_pass = clarification.interpret_reply(user_message)
        if rejected is not None:
            state.rejected_attributes.add(rejected)
            return
        # An override only displaces the answer when the customer leads with
        # the reversal. detect_override_cue() is deliberately broad for intent
        # classification -- it fires on "instead" or "no longer" anywhere in
        # the text -- and disclosures are raw product copy, so using it here
        # would un-ask attributes that were in fact answered and let the
        # policy repeat a question.
        replaced_answer = override_triggered and _ANSWER_REPLACED_RE.search(user_message or "")
        if boundary_pass or replaced_answer:
            # The question went unanswered rather than answered emptily, so it
            # is still worth asking. An override message does carry content of
            # its own, so it falls through; a boundary decline does not.
            if asked is not None:
                state.asked_attributes.discard(asked)
            if boundary_pass:
                return

        if not user_message or _EMPTY_REPLY_RE.search(user_message):
            return
        # Keep the part after a lead-in clause ("A key requirement is: ...",
        # "For that, what matters is: ...") wherever the customer used one --
        # that is the constraint text. Absent a lead-in, only keep the message
        # when it is an answer to a question we actually asked, so an opening
        # like "I'm looking for boots, but I'm still exploring" does not
        # persist its filler into every later query.
        content, substitutions = _LEAD_IN_RE.subn("", user_message.strip())
        if content and (substitutions or asked is not None):
            state.disclosed_text.append(content)

    @staticmethod
    def record_question(state: SessionState, attribute: str | None) -> None:
        """Mark a question as asked, so the next turn knows what it is
        waiting on and the policy never repeats itself."""
        if attribute is None:
            return
        state.pending_attribute = attribute
        state.asked_attributes.add(attribute)
        state.clarification_turns += 1

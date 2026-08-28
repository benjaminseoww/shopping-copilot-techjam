from __future__ import annotations

from .models import AttributeName, IntentUpdate, SessionState, UserProfile


class MemoryStore:
    """In-process, per-session conversational state."""

    MAX_MESSAGES = 20
    MAX_EVENTS = 40

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def reset(self, session_id: str, user_profile: dict) -> SessionState:
        if not session_id:
            raise ValueError("session_id must not be empty")
        state = SessionState(
            session_id=session_id,
            profile=UserProfile.from_dict(user_profile),
        )
        # The supplied evaluator is sequential and has no end-session callback.
        # Keeping only the newly reset session bounds memory and prevents leakage.
        self._sessions = {session_id: state}
        return state

    def get(self, session_id: str) -> SessionState:
        try:
            return self._sessions[session_id]
        except KeyError as error:
            raise RuntimeError("reset must be called before respond") from error

    def apply(
        self,
        session_id: str,
        update: IntentUpdate,
        user_message: str,
        turn: int,
    ) -> SessionState:
        state = self.get(session_id)
        state.last_turn = turn
        state.messages.append((turn, user_message))
        del state.messages[:-self.MAX_MESSAGES]

        if update.category:
            state.category = update.category

        if update.supersede_preferences:
            state.superseded_constraints.extend(state.active_constraints)
            state.active_constraints.clear()

        existing = {
            (constraint.attribute, self._normalise(constraint.text))
            for constraint in state.active_constraints
        }
        for constraint in update.constraints:
            key = (constraint.attribute, self._normalise(constraint.text))
            if key not in existing and constraint.text.strip():
                state.active_constraints.append(constraint)
                existing.add(key)

        state.no_preference.update(update.no_preference)
        state.exhausted_attributes.update(update.exhausted)
        state.events.append(
            f"turn={turn} kind={update.interaction_kind} parser={update.parser}"
        )
        del state.events[:-self.MAX_EVENTS]
        return state

    def record_agent_action(
        self,
        session_id: str,
        ask_attribute: AttributeName | None,
        recommendations: list[str],
        turn: int,
    ) -> None:
        state = self.get(session_id)
        state.last_turn = turn
        state.last_ask = ask_attribute
        if ask_attribute is not None:
            state.asked_attributes.append(ask_attribute)
        state.previous_recommendations = list(recommendations[:10])
        state.events.append(
            f"turn={turn} ask={ask_attribute or 'none'} recommendations={len(recommendations)}"
        )
        del state.events[:-self.MAX_EVENTS]

    @staticmethod
    def _normalise(value: str) -> str:
        return " ".join(value.lower().split())

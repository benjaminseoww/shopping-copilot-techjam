from __future__ import annotations

from pathlib import Path

from .intent import IntentUnderstander
from .memory import MemoryStore
from .questions import QuestionsEngine
from .recommendation import RecommendationEngine



class Agent:
    """Stateful shopping agent with deterministic offline components."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.memory = MemoryStore()
        self.intent = IntentUnderstander()
        self.recommendation = RecommendationEngine(self.catalog_path)
        self.questions = QuestionsEngine()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self.memory.reset(session_id, user_profile)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self.memory.get(session_id)
        update = self.intent.parse(user_message, turn, state.last_ask)
        state = self.memory.apply(session_id, update, user_message, turn)
        pool_k = max(top_k, getattr(self.recommendation, "RETRIEVE_K", top_k))
        pool = self.recommendation.recommend(state, pool_k)
        question = self.questions.decide(state, turn, pool)
        parent_asins = [candidate.parent_asin for candidate in pool[:top_k]]
        self.memory.record_agent_action(
            session_id,
            question.ask_attribute,
            parent_asins,
            turn,
        )
        return {
            "message": question.message,
            "ask_attribute": question.ask_attribute,
            "recommendations": [
                {"parent_asin": parent_asin}
                for parent_asin in parent_asins
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

from __future__ import annotations

from pathlib import Path

from .act_classifier import try_build_act_classifier
from .embedder import default_model_dir, try_load_minilm
from .intent import IntentUnderstander
from .memory import MemoryStore
from .models import ScoredProduct, SessionState
from .nli import default_nli_dir, try_load_nli
from .questions import QuestionsEngine
from .recommendation import RecommendationEngine
from .semantic_match import SemanticMatcher


class Agent:
    """Stateful shopping agent with deterministic offline components."""

    SHORTLIST_UNTIL_TURN = 8
    SHORTLIST_ONE = 1

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.memory = MemoryStore()
        embedder = try_load_minilm(default_model_dir(self.catalog_path))
        self.semantic = SemanticMatcher(embedder)
        self.intent = IntentUnderstander(
            act_classifier=try_build_act_classifier(
                embedder=embedder,
                nli=try_load_nli(default_nli_dir(self.catalog_path)),
            ),
            semantic=self.semantic,
        )
        self.recommendation = RecommendationEngine(self.catalog_path, embedder=embedder)
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
        pool_k = max(
            top_k,
            self.questions.candidate_pool,
            getattr(self.recommendation, "RETRIEVE_K", top_k),
        )
        pool = self.recommendation.recommend(state, pool_k)
        question = self.questions.decide(
            state,
            turn,
            pool,
            catalog_text=self.recommendation.catalog_text,
        )
        shown_k = self._shown_k(state, turn, top_k)
        parent_asins = self._unseen_asins(state, pool, shown_k)
        retrieve_k = getattr(self.recommendation, "RETRIEVE_K", pool_k)
        self.memory.record_agent_action(
            session_id,
            question.ask_attribute,
            parent_asins,
            turn,
            retrieve_pool=[candidate.parent_asin for candidate in pool[:retrieve_k]],
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

    @classmethod
    def _shown_k(cls, state: SessionState, turn: int, top_k: int) -> int:
        """Show one best guess until two constraints exist; never return an empty list."""
        if top_k <= 0:
            return 0
        if turn >= cls.SHORTLIST_UNTIL_TURN:
            return top_k
        if len(state.active_constraints) < 2:
            return min(cls.SHORTLIST_ONE, top_k)
        return top_k

    @staticmethod
    def _unseen_asins(
        state: SessionState,
        pool: list[ScoredProduct],
        shown_k: int,
    ) -> list[str]:
        """Skip already-shown products so a shortlist walks after an implicit reject."""
        if shown_k <= 0:
            return []
        shown = set(state.previous_recommendations)
        unseen = [
            candidate.parent_asin
            for candidate in pool
            if candidate.parent_asin not in shown
        ]
        if len(unseen) >= shown_k:
            return unseen[:shown_k]
        leftover = [
            candidate.parent_asin
            for candidate in pool
            if candidate.parent_asin in shown
        ]
        return (unseen + leftover)[:shown_k]

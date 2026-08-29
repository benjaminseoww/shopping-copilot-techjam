from __future__ import annotations

from collections.abc import Sequence

from .models import QuestionDecision, ScoredProduct, SessionState


class QuestionsEngine:
    """Deterministic clarification policy for the evaluator contract."""

    def decide(
        self,
        state: SessionState,
        turn: int,
        candidates: Sequence[ScoredProduct] = (),
    ) -> QuestionDecision:
        # Ranked Top 50-100 pool is reserved for the candidate-aware follow-up.
        del state, candidates
        if turn >= 10:
            return QuestionDecision(
                message="Here are the closest matches based on your preferences.",
                ask_attribute=None,
            )
        return QuestionDecision(
            message="What other requirement or priority matters most to you?",
            ask_attribute="other",
        )

    # TODO(questions): Estimate eligible attribute distributions across an
    # over-fetched candidate set and select the highest expected information
    # gain after excluding answered, declined, exhausted, or constant fields.
    # Candidate diversity is not the same as evaluator answerability, so the
    # adaptive policy must include answer probability, deterministic ties, an
    # "other" fallback, and scenario-level ablation before becoming default.

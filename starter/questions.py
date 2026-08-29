from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

from .models import AttributeName, QuestionDecision, ScoredProduct, SessionState


TYPED_ATTRIBUTES: tuple[AttributeName, ...] = ("material", "color", "style", "size")
NEVER_ASK: frozenset[str] = frozenset({"brand", "budget", "category", "use_case"})
TIE_BREAK: tuple[AttributeName, ...] = ("other", "material", "color", "style", "size")
FAMILY_ANSWER_PRIOR: dict[str, dict[str, float]] = {
    "clothing": {"material": 0.95, "color": 0.11, "style": 0.06, "size": 0.01},
    "shoes": {"material": 0.53, "color": 0.18, "style": 0.11, "size": 0.00},
    "jewelry": {"material": 0.08, "color": 0.85, "style": 0.31, "size": 0.31},
    "watches": {"material": 0.50, "color": 0.50, "style": 0.17, "size": 0.33},
    "bags": {"material": 0.85, "color": 0.62, "style": 0.00, "size": 0.08},
    "accessories": {"material": 0.73, "color": 0.60, "style": 0.13, "size": 0.07},
    "other": {"material": 0.70, "color": 0.25, "style": 0.08, "size": 0.04},
}
OTHER_SCORE = 0.28
MIN_OCCUPANCY = 0.20
MIN_DIVERSITY = 0.12
CANDIDATE_POOL = 80

MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
COLORS = (
    "black",
    "white",
    "blue",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
)
SIZE_VALUES = ("wide", "narrow", "small", "medium", "large", "xl")
STYLE_VALUES = ("women", "men", "unisex", "sleeve", "neck", "fit")

MESSAGES: dict[AttributeName | None, str] = {
    None: "Here are the closest matches based on your preferences.",
    "other": "What other requirement or priority matters most to you?",
    "material": "Do you have a material preference?",
    "color": "Do you have a color preference?",
    "style": "Do you have a style or fit preference?",
    "size": "Do you have a size or width preference?",
}

_ROOT_FRAGMENTS = (
    "clothing, shoes & jewelry",
    "clothing shoes & jewelry",
    "shoes & jewelry",
)
_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("watches", re.compile(r"\b(?:watches?|wristwatch(?:es)?)\b", re.I)),
    ("jewelry", re.compile(r"\b(?:jewel(?:ry|lery)|necklaces?|earrings?|bracelets?|rings?|pendants?)\b", re.I)),
    ("shoes", re.compile(r"\b(?:shoes?|boots?|sneakers?|slippers?|sandals?|loafers?)\b", re.I)),
    ("bags", re.compile(r"\b(?:handbags?|wallets?|totes?|purses?|backpacks?|bags?)\b", re.I)),
    ("accessories", re.compile(r"\b(?:accessories|accessory|hats?|belts?|scar(?:f|ves)|gloves?|sunglasses)\b", re.I)),
    ("clothing", re.compile(r"\b(?:clothing|dresses?|shirts?|pants?|jeans?|skirts?|sweaters?|hoodies?|blouses?|jackets?|coats?|apparel)\b", re.I)),
)
_EXTRACTORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("material", MATERIALS),
    ("color", COLORS),
    ("style", STYLE_VALUES),
    ("size", SIZE_VALUES),
)
_EXTRACT_PATTERNS = {
    field: re.compile(r"\b(" + "|".join(re.escape(value) for value in values) + r")\b", re.I)
    for field, values in _EXTRACTORS
}


def _strip_root(text: str) -> str:
    lowered = text.lower()
    for fragment in _ROOT_FRAGMENTS:
        lowered = lowered.replace(fragment, " ")
    return lowered


def _family_from_text(text: str) -> str:
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(text):
            return family
    return "other"


def family_from_category(
    category: str | None,
    candidates: Sequence[ScoredProduct] = (),
) -> str:
    """Map a coarse category string to a product family, with candidate majority vote."""
    mapped = _family_from_text(_strip_root(category or ""))
    if mapped != "other":
        return mapped
    votes: Counter[str] = Counter()
    for candidate in candidates:
        family = _family_from_text(candidate.text)
        if family != "other":
            votes[family] += 1
    if not votes:
        return "other"
    ranked = votes.most_common()
    if len(ranked) >= 2 and ranked[0][1] == ranked[1][1]:
        return "other"
    return ranked[0][0]


def extract_values(text: str) -> dict[str, str | None]:
    """Return the first closed-vocab hit for each typed attribute."""
    extracted: dict[str, str | None] = {}
    for field, _values in _EXTRACTORS:
        match = _EXTRACT_PATTERNS[field].search(text or "")
        extracted[field] = match.group(1).lower() if match else None
    return extracted


def _diversity(values: Sequence[str]) -> float:
    if not values:
        return 0.0
    counts = Counter(values)
    total = len(values)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


class QuestionsEngine:
    """Score typed questions from live candidate occupancy and family answerability."""

    candidate_pool = CANDIDATE_POOL

    def decide(
        self,
        state: SessionState,
        turn: int,
        candidates: Sequence[ScoredProduct] = (),
    ) -> QuestionDecision:
        if turn >= 10:
            return QuestionDecision(message=MESSAGES[None], ask_attribute=None)

        blocked = self._blocked(state)
        family = family_from_category(state.category, candidates)
        scores: dict[AttributeName, float] = {}
        if "other" not in blocked:
            scores["other"] = OTHER_SCORE

        if candidates:
            n_candidates = len(candidates)
            observed: dict[str, list[str]] = {attribute: [] for attribute in TYPED_ATTRIBUTES}
            for candidate in candidates:
                extracted = extract_values(candidate.text)
                for attribute in TYPED_ATTRIBUTES:
                    value = extracted.get(attribute)
                    if value:
                        observed[attribute].append(value)
            priors = FAMILY_ANSWER_PRIOR.get(family, FAMILY_ANSWER_PRIOR["other"])
            for attribute in TYPED_ATTRIBUTES:
                if attribute in blocked or attribute in NEVER_ASK:
                    continue
                if family == "jewelry" and attribute == "material":
                    continue
                values = observed[attribute]
                occupancy = len(values) / n_candidates
                if occupancy < MIN_OCCUPANCY:
                    continue
                diversity = _diversity(values)
                if diversity < MIN_DIVERSITY:
                    continue
                scores[attribute] = priors[attribute] * occupancy * diversity

        chosen = self._select(scores)
        return QuestionDecision(message=MESSAGES[chosen], ask_attribute=chosen)

    @staticmethod
    def _blocked(state: SessionState) -> set[str]:
        answered = {constraint.attribute for constraint in state.active_constraints}
        return set(answered) | set(state.no_preference) | set(state.exhausted_attributes)

    @staticmethod
    def _select(scores: dict[AttributeName, float]) -> AttributeName:
        if not scores:
            return "other"
        tie_order = {attribute: index for index, attribute in enumerate(TIE_BREAK)}
        return min(
            scores,
            key=lambda attribute: (-scores[attribute], tie_order.get(attribute, len(TIE_BREAK))),
        )

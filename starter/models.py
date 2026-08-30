from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias


AttributeName: TypeAlias = Literal[
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
]

ALLOWED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "category",
        "material",
        "color",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use_case",
        "other",
    }
)


@dataclass(frozen=True)
class UserProfile:
    purchase_frequency: str = ""
    average_prior_rating: float | None = None
    rating_style: str = ""
    preference_tags: tuple[str, ...] = ()
    summary: str = ""

    @classmethod
    def from_dict(cls, value: dict) -> "UserProfile":
        rating = value.get("average_prior_rating")
        return cls(
            purchase_frequency=str(value.get("purchase_frequency") or ""),
            average_prior_rating=float(rating) if isinstance(rating, (int, float)) else None,
            rating_style=str(value.get("rating_style") or ""),
            preference_tags=tuple(str(item) for item in value.get("preference_tags") or ()),
            summary=str(value.get("summary") or ""),
        )


@dataclass(frozen=True)
class Constraint:
    text: str
    attribute: AttributeName
    turn: int
    source: str


@dataclass
class IntentUpdate:
    interaction_kind: str = "unknown"
    category: str | None = None
    constraints: list[Constraint] = field(default_factory=list)
    no_preference: set[AttributeName] = field(default_factory=set)
    exhausted: set[AttributeName] = field(default_factory=set)
    supersede_preferences: bool = False
    fallback_terms: list[str] = field(default_factory=list)
    parser: str = "phrase"


@dataclass
class SessionState:
    session_id: str
    profile: UserProfile
    category: str | None = None
    active_constraints: list[Constraint] = field(default_factory=list)
    superseded_constraints: list[Constraint] = field(default_factory=list)
    no_preference: set[AttributeName] = field(default_factory=set)
    exhausted_attributes: set[AttributeName] = field(default_factory=set)
    asked_attributes: list[AttributeName] = field(default_factory=list)
    messages: list[tuple[int, str]] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    previous_recommendations: list[str] = field(default_factory=list)
    previous_pool: list[str] = field(default_factory=list)
    last_ask: AttributeName | None = None
    last_turn: int = 0


@dataclass(frozen=True)
class ScoredProduct:
    parent_asin: str
    score: float = 0.0


@dataclass(frozen=True)
class QuestionDecision:
    message: str
    ask_attribute: AttributeName | None

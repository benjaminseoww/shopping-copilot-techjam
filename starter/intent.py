from __future__ import annotations

import re
from typing import cast

from .models import ALLOWED_ATTRIBUTES, AttributeName, Constraint, IntentUpdate


BUYING_RE = re.compile(
    r"^I'm looking for (?P<category>.+?)\. A key requirement is:\s*(?P<constraint>.+?)\.?$",
    re.IGNORECASE,
)
BROWSING_RE = re.compile(
    r"^I'm looking for (?P<category>.+?), but I'm still exploring\.?$",
    re.IGNORECASE,
)
INITIAL_PREFERENCE_RE = re.compile(
    r"^I'm looking for (?P<category>.+?)\.\s+(?P<constraint>.+?)\.?$",
    re.IGNORECASE,
)
CLARIFICATION_RE = re.compile(
    r"^For that, what matters is:\s*(?P<constraints>.+?)\.?$",
    re.IGNORECASE,
)
NO_PREFERENCE_RE = re.compile(
    r"^I don't have a preference for (?P<attribute>[a-z_]+); please use your judgment\.?$",
    re.IGNORECASE,
)
EXHAUSTED_RE = re.compile(
    r"^I don't have an additional preference for (?P<attribute>[a-z_]+)\.?$",
    re.IGNORECASE,
)
OVERRIDE_RE = re.compile(
    r"^Actually,\s*ignore my earlier preference\.\s*What I need is:\s*(?P<constraint>.+?)\.?$",
    re.IGNORECASE,
)
CLARIFICATION_REQUEST_RE = re.compile(
    r"^Those options are not quite right yet\. Ask me about one specific attribute\.?$",
    re.IGNORECASE,
)

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


class IntentUnderstander:
    """Phrase-based intent parser with a stable structured output."""

    def parse(
        self,
        user_message: str,
        turn: int,
        last_ask: AttributeName | None = None,
    ) -> IntentUpdate:
        message = " ".join(str(user_message).split())

        match = OVERRIDE_RE.match(message)
        if match:
            text = self._clean(match.group("constraint"))
            return IntentUpdate(
                interaction_kind="override",
                constraints=[self._constraint(text, turn, "override")],
                supersede_preferences=True,
            )

        match = BUYING_RE.match(message)
        if match:
            text = self._clean(match.group("constraint"))
            return IntentUpdate(
                interaction_kind="buying",
                category=self._clean(match.group("category")),
                constraints=[self._constraint(text, turn, "initial")],
            )

        match = BROWSING_RE.match(message)
        if match:
            return IntentUpdate(
                interaction_kind="browsing",
                category=self._clean(match.group("category")),
            )

        match = CLARIFICATION_RE.match(message)
        if match:
            values = [
                self._clean(value)
                for value in re.split(r";\s+", match.group("constraints"))
                if self._clean(value)
            ]
            return IntentUpdate(
                interaction_kind="clarification",
                constraints=[
                    self._constraint(value, turn, "clarification")
                    for value in values
                ],
            )

        match = NO_PREFERENCE_RE.match(message)
        if match:
            attribute = self._attribute(match.group("attribute"), last_ask)
            return IntentUpdate(
                interaction_kind="no_preference",
                no_preference={attribute},
            )

        match = EXHAUSTED_RE.match(message)
        if match:
            attribute = self._attribute(match.group("attribute"), last_ask)
            return IntentUpdate(
                interaction_kind="exhausted",
                exhausted={attribute},
            )

        if CLARIFICATION_REQUEST_RE.match(message):
            return IntentUpdate(interaction_kind="clarification_request")

        match = INITIAL_PREFERENCE_RE.match(message)
        if match:
            text = self._clean(match.group("constraint"))
            return IntentUpdate(
                interaction_kind="initial_preference",
                category=self._clean(match.group("category")),
                constraints=[self._constraint(text, turn, "initial_provisional")],
            )

        fallback = self._fallback_text(message)
        update = IntentUpdate(
            interaction_kind="unknown",
            parser="fallback",
            fallback_terms=fallback.split() if fallback else [],
        )
        if fallback:
            update.constraints.append(self._constraint(fallback, turn, "fallback"))
        return update

    def _constraint(self, text: str, turn: int, source: str) -> Constraint:
        return Constraint(
            text=text,
            attribute=self.classify_constraint(text),
            turn=turn,
            source=source,
        )

    @staticmethod
    def classify_constraint(value: str) -> AttributeName:
        lowered = value.lower()
        if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
            return "budget"
        if "brand" in lowered or re.search(r"\bby\s+[a-z0-9]", lowered):
            return "brand"
        if any(material in lowered for material in MATERIALS):
            return "material"
        if "color" in lowered or any(color in lowered for color in COLORS):
            return "color"
        if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
            return "size"
        if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
            return "style"
        if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
            return "use_case"
        return "feature"

    @staticmethod
    def _attribute(value: str, last_ask: AttributeName | None) -> AttributeName:
        lowered = value.lower()
        if lowered in ALLOWED_ATTRIBUTES:
            return cast(AttributeName, lowered)
        return last_ask or "other"

    @staticmethod
    def _clean(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip(" \t\n.;")

    @staticmethod
    def _fallback_text(message: str) -> str:
        value = re.sub(
            r"\b(?:i'm|i am|looking for|please|those options|not quite right|"
            r"what matters is|a key requirement is|actually|what i need is)\b",
            " ",
            message,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", value).strip(" \t\n.,;:")

    # TODO(intent): Add a schema-constrained LLM or local-model parser that
    # emits IntentUpdate, validates grounded evidence, and reports token usage.
    # Phrase parsing must remain the offline fallback for unavailable models,
    # invalid schemas, low confidence, timeouts, or disabled network access.

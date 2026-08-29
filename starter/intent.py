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

EXHAUSTED_CUE_RE = re.compile(
    r"\b(?:no additional preference|nothing (?:else|more) (?:on|about|for))\b",
    re.IGNORECASE,
)
NO_PREFERENCE_CUE_RE = re.compile(
    r"\b(?:no preference|does not matter|doesn't matter|do not care|don't care|"
    r"you (?:can )?pick|up to you|use your judgment|any(?:thing)? is fine)\b",
    re.IGNORECASE,
)
CLARIFICATION_REQUEST_CUE_RE = re.compile(
    r"\b(?:ask me (?:about|for)|ask (?:about|for))\b.*\b"
    r"(?:attribute|detail|preference|requirement)\b",
    re.IGNORECASE,
)
OVERRIDE_CUE_RE = re.compile(
    r"\b(?:ignore|forget|never mind|instead|change[sd]? my mind|switch(?:ing)? to|"
    r"rather (?:have|use|wear)|actually)\b",
    re.IGNORECASE,
)
BROWSING_CUE_RE = re.compile(
    r"\b(?:just (?:looking|browsing|exploring)|still (?:looking|browsing|exploring)|"
    r"not sure(?: yet)?|exploring for now)\b",
    re.IGNORECASE,
)
CLARIFICATION_CUE_RE = re.compile(
    r"\b(?:what matters is|important part is|priority is|prioritize)\b",
    re.IGNORECASE,
)
BUYING_CUE_RE = re.compile(
    r"\b(?:i need|i want|must be|needs? to be|required|requirement)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(?:do not|don't|does not|doesn't|not|without|avoid|exclude|anything but)\b",
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

MATERIAL_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(value) for value in MATERIALS)})\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(value) for value in COLORS)})\b",
    re.IGNORECASE,
)
SIZE_RE = re.compile(r"\b(?:size|sizing|width|wide|narrow)\b", re.IGNORECASE)
STYLE_RE = re.compile(
    r"\b(?:department|style|fit|sleeve|neck)\b",
    re.IGNORECASE,
)
USE_CASE_RE = re.compile(
    r"\b(?:hiking|running|gym|winter|outdoor|work)\b",
    re.IGNORECASE,
)


class IntentUnderstander:
    """Offline intent parser using exact templates, cues, and turn context."""

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

        rule_update = self._parse_rule_cues(message, turn, last_ask)
        if rule_update is not None:
            return rule_update

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

    def _parse_rule_cues(
        self,
        message: str,
        turn: int,
        last_ask: AttributeName | None,
    ) -> IntentUpdate | None:
        """Handle conservative paraphrases after exact evaluator templates."""
        if EXHAUSTED_CUE_RE.search(message):
            return IntentUpdate(
                interaction_kind="exhausted",
                exhausted={self._mentioned_attribute(message, last_ask)},
                parser="rules",
            )

        if NO_PREFERENCE_CUE_RE.search(message):
            return IntentUpdate(
                interaction_kind="no_preference",
                no_preference={self._mentioned_attribute(message, last_ask)},
                parser="rules",
            )

        if CLARIFICATION_REQUEST_CUE_RE.search(message):
            return IntentUpdate(
                interaction_kind="clarification_request",
                parser="rules",
            )

        if OVERRIDE_CUE_RE.search(message):
            replacement = self._replacement_span(message)
            if replacement and not NEGATION_RE.search(replacement):
                return IntentUpdate(
                    interaction_kind="override",
                    constraints=[self._constraint(replacement, turn, "override_rule")],
                    supersede_preferences=True,
                    parser="rules",
                )

        if BROWSING_CUE_RE.search(message):
            return IntentUpdate(
                interaction_kind="browsing",
                category=self._item_span(message),
                parser="rules",
            )

        if CLARIFICATION_CUE_RE.search(message):
            detail = self._detail_span(message)
            if detail and not NEGATION_RE.search(detail):
                return IntentUpdate(
                    interaction_kind="clarification",
                    constraints=[
                        self._constraint(value, turn, "clarification_rule")
                        for value in self._split_constraints(detail)
                    ],
                    parser="rules",
                )

        if BUYING_CUE_RE.search(message) and not NEGATION_RE.search(message):
            category = self._item_span(message)
            requirement = self._requirement_span(message)
            constraints = (
                [self._constraint(requirement, turn, "initial_rule")]
                if requirement
                else []
            )
            if category or constraints:
                return IntentUpdate(
                    interaction_kind="buying",
                    category=category,
                    constraints=constraints,
                    parser="rules",
                )

        # Unsupported negative constraints are deliberately not converted into
        # positive BM25 terms. Preserve an initial category when one is clear.
        if NEGATION_RE.search(message):
            return IntentUpdate(
                interaction_kind="unknown",
                category=self._item_span(message),
                parser="fallback",
            )
        return None

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
        if MATERIAL_RE.search(lowered):
            return "material"
        if re.search(r"\bcolor\b", lowered) or COLOR_RE.search(lowered):
            return "color"
        if SIZE_RE.search(lowered):
            return "size"
        if STYLE_RE.search(lowered):
            return "style"
        if USE_CASE_RE.search(lowered):
            return "use_case"
        return "feature"

    @staticmethod
    def _mentioned_attribute(
        message: str,
        last_ask: AttributeName | None,
    ) -> AttributeName:
        lowered = message.lower()
        for value in sorted(ALLOWED_ATTRIBUTES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(value)}\b", lowered):
                return cast(AttributeName, value)
        return last_ask or "other"

    @classmethod
    def _replacement_span(cls, message: str) -> str:
        patterns = (
            r"(?:what i (?:need|want) is|make it|go with|switch(?:ing)? to)\s*:?\s*(.+)$",
            r"(?:i(?:'d)? rather (?:have|use|wear)|i (?:need|want|prefer))\s+(.+)$",
            r"(?:instead|change[sd]? my mind)\s*[:;,—-]\s*(.+)$",
            r"[:;—-]\s*(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return cls._trim_discourse(match.group(1))
        return ""

    @classmethod
    def _item_span(cls, message: str) -> str | None:
        patterns = (
            r"\b(?:looking|shopping|browsing)\s+for\s+(.+?)(?:[,.;—-]|$)",
            r"\bexploring\s+(.+?)(?:\s+for now|[,.;—-]|$)",
            r"^\s*(?:i\s+)?(?:need|want)\s+(.+?)(?:[,.;—-]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                value = cls._trim_discourse(match.group(1))
                if value:
                    return value
        return None

    @classmethod
    def _detail_span(cls, message: str) -> str:
        match = re.search(
            r"\b(?:what matters is|important part is|priority is|prioritize)\s*:?\s*(.+)$",
            message,
            flags=re.IGNORECASE,
        )
        return cls._trim_discourse(match.group(1)) if match else ""

    @classmethod
    def _requirement_span(cls, message: str) -> str:
        # Prefer the clause after a dash/semicolon, e.g.
        # "Need running shoes — cotton is required."
        clauses = re.split(r"\s*[—;]\s*", message, maxsplit=1)
        detail = clauses[1] if len(clauses) == 2 else message
        patterns = (
            r"(.+?)\s+is\s+(?:required|a must)\b",
            r"\b(?:must be|needs? to be|requirement is)\s*:?\s*(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, detail, flags=re.IGNORECASE)
            if match:
                return cls._trim_discourse(match.group(1))
        return ""

    @classmethod
    def _split_constraints(cls, value: str) -> list[str]:
        return [
            cleaned
            for item in re.split(r";\s+|,\s+and also\s+", value)
            if (cleaned := cls._clean(item))
        ]

    @classmethod
    def _trim_discourse(cls, value: str) -> str:
        value = re.sub(
            r"\s+(?:instead|for now|please)\s*[.!?]*$",
            "",
            value,
            flags=re.IGNORECASE,
        )
        return cls._clean(value)

    @staticmethod
    def _attribute(value: str, last_ask: AttributeName | None) -> AttributeName:
        lowered = value.lower()
        if lowered in ALLOWED_ATTRIBUTES:
            return cast(AttributeName, lowered)
        return last_ask or "other"

    @staticmethod
    def _clean(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip(" \t\n.,;:!?—-")

    @staticmethod
    def _fallback_text(message: str) -> str:
        if NEGATION_RE.search(message):
            return ""
        value = re.sub(
            r"\b(?:i'm|i am|i|looking for|please|those options|not quite right|"
            r"what matters is|a key requirement is|actually|what i need is|"
            r"judgment|preference|ignore|forget|never mind|instead)\b",
            " ",
            message,
            flags=re.IGNORECASE,
        )
        return re.sub(r"\s+", " ", value).strip(" \t\n.,;:")

    # TODO(intent): Add a schema-constrained LLM or local-model parser that
    # emits IntentUpdate, validates grounded evidence, and reports token usage.
    # Phrase parsing must remain the offline fallback for unavailable models,
    # invalid schemas, low confidence, timeouts, or disabled network access.

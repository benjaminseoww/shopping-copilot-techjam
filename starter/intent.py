from __future__ import annotations

import re

from .attributes import (
    ATTRIBUTE_NAMES,
    COLOR_RE,
    MATERIAL_RE,
    SIZE_RE,
    STYLE_RE,
    USE_CASE_RE,
)
from .models import AttributeName, Constraint, IntentUpdate


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
    r"\b(?:must be|needs? to be|is required|key requirement|a must)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"\b(?:do not|don't|does not|doesn't|not|without|avoid|exclude|anything but)\b",
    re.IGNORECASE,
)


class IntentUnderstander:
    """Offline intent parser using exact templates, then a paraphrase policy."""

    def parse(
        self,
        user_message: str,
        turn: int,
        last_ask: AttributeName | None = None,
    ) -> IntentUpdate:
        message = " ".join(str(user_message).split())

        for pattern, handler in (
            (OVERRIDE_RE, self._handle_override_template),
            (BUYING_RE, self._handle_buying_template),
            (BROWSING_RE, self._handle_browsing_template),
            (CLARIFICATION_RE, self._handle_clarification_template),
            (NO_PREFERENCE_RE, self._handle_no_preference_template),
            (EXHAUSTED_RE, self._handle_exhausted_template),
            (CLARIFICATION_REQUEST_RE, self._handle_clarification_request_template),
        ):
            match = pattern.match(message)
            if match:
                return handler(match, turn, last_ask)

        paraphrase = self._parse_paraphrase(message, turn, last_ask)
        if paraphrase is not None:
            return paraphrase

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

    def _handle_override_template(
        self,
        match: re.Match[str],
        turn: int,
        last_ask: AttributeName | None,
    ) -> IntentUpdate:
        text = self._clean(match.group("constraint"))
        return IntentUpdate(
            interaction_kind="override",
            constraints=[self._constraint(text, turn, "override")],
            supersede_preferences=True,
        )

    def _handle_buying_template(
        self,
        match: re.Match[str],
        turn: int,
        last_ask: AttributeName | None,
    ) -> IntentUpdate:
        text = self._clean(match.group("constraint"))
        return IntentUpdate(
            interaction_kind="buying",
            category=self._clean(match.group("category")),
            constraints=[self._constraint(text, turn, "initial")],
        )

    def _handle_browsing_template(
        self,
        match: re.Match[str],
        turn: int,
        last_ask: AttributeName | None,
    ) -> IntentUpdate:
        return IntentUpdate(
            interaction_kind="browsing",
            category=self._clean(match.group("category")),
        )

    def _handle_clarification_template(
        self,
        match: re.Match[str],
        turn: int,
        last_ask: AttributeName | None,
    ) -> IntentUpdate:
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

    def _handle_no_preference_template(
        self,
        match: re.Match[str],
        turn: int,
        last_ask: AttributeName | None,
    ) -> IntentUpdate:
        attribute = self._attribute(match.group("attribute"), last_ask)
        return IntentUpdate(
            interaction_kind="no_preference",
            no_preference={attribute},
        )

    def _handle_exhausted_template(
        self,
        match: re.Match[str],
        turn: int,
        last_ask: AttributeName | None,
    ) -> IntentUpdate:
        attribute = self._attribute(match.group("attribute"), last_ask)
        return IntentUpdate(
            interaction_kind="exhausted",
            exhausted={attribute},
        )

    def _handle_clarification_request_template(
        self,
        match: re.Match[str],
        turn: int,
        last_ask: AttributeName | None,
    ) -> IntentUpdate:
        return IntentUpdate(interaction_kind="clarification_request")

    def _parse_paraphrase(
        self,
        message: str,
        turn: int,
        last_ask: AttributeName | None,
    ) -> IntentUpdate | None:
        """Classify conservative paraphrases after exact evaluator templates."""
        if OVERRIDE_CUE_RE.search(message):
            replacement = self._replacement_span(message)
            if replacement and not NEGATION_RE.search(replacement):
                return IntentUpdate(
                    interaction_kind="override",
                    constraints=[self._constraint(replacement, turn, "override_rule")],
                    supersede_preferences=True,
                )

        if EXHAUSTED_CUE_RE.search(message):
            return IntentUpdate(
                interaction_kind="exhausted",
                exhausted={self._mentioned_attribute(message, last_ask)},
            )

        if NO_PREFERENCE_CUE_RE.search(message):
            return IntentUpdate(
                interaction_kind="no_preference",
                no_preference={self._mentioned_attribute(message, last_ask)},
            )

        if CLARIFICATION_REQUEST_CUE_RE.search(message):
            return IntentUpdate(interaction_kind="clarification_request")

        if BROWSING_CUE_RE.search(message):
            return IntentUpdate(
                interaction_kind="browsing",
                category=self._item_span(message),
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
        for name in sorted(ATTRIBUTE_NAMES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                return name
        return last_ask or "other"

    @classmethod
    def _replacement_span(cls, message: str) -> str:
        patterns = (
            r"(?:what i (?:need|want) is|make it|go with|switch(?:ing)? to)\s*:?\s*(.+)$",
            r"(?:i(?:'d)? rather (?:have|use|wear)|i (?:need|want|prefer))\s+(.+)$",
            r"(?:instead|change[sd]? my mind)\s*[:;,—-]\s*(.+)$",
            r"(?:ignore|forget|never mind)[^;:—-]*[;:—-]\s*(.+)$",
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
        for name in ATTRIBUTE_NAMES:
            if name == lowered:
                return name
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

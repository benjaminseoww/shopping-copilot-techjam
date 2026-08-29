from __future__ import annotations

import re
from dataclasses import dataclass

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
NOOP_CUE_RE = re.compile(
    r"\b(?:need to think|have to think|let me think|keep looking|"
    r"still thinking|give me a (?:moment|second|minute)|not ready(?: yet)?)\b",
    re.IGNORECASE,
)
VALUE_LEADIN_RE = re.compile(
    r"^(?:(?:it(?:'s| is)|i(?:'d)? (?:prefer|want|need|like)|maybe|perhaps|how about)\s+)+",
    re.IGNORECASE,
)

# Public IntentUpdate.interaction_kind values stay stable for events/tests.
ACT_KIND = {
    "replace": "override",
    "exhaust_ask": "exhausted",
    "decline_ask": "no_preference",
    "ask_me": "clarification_request",
    "open_browse": "browsing",
    "answer_ask": "clarification",
    "open_buy": "buying",
    "reject": "unknown",
    "noop": "noop",
    "fallback": "unknown",
}

# Exact evaluator templates, matched first in this order.
_TEMPLATES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("replace", OVERRIDE_RE),
    ("open_buy", BUYING_RE),
    ("open_browse", BROWSING_RE),
    ("answer_ask", CLARIFICATION_RE),
    ("decline_ask", NO_PREFERENCE_RE),
    ("exhaust_ask", EXHAUSTED_RE),
    ("ask_me", CLARIFICATION_REQUEST_RE),
)

# Residual paraphrase cues. Precedence is the classifier; extractors may reject.
_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("replace", OVERRIDE_CUE_RE),
    ("exhaust_ask", EXHAUSTED_CUE_RE),
    ("decline_ask", NO_PREFERENCE_CUE_RE),
    ("ask_me", CLARIFICATION_REQUEST_CUE_RE),
    ("open_browse", BROWSING_CUE_RE),
    ("answer_ask", CLARIFICATION_CUE_RE),
    ("open_buy", BUYING_CUE_RE),
)


@dataclass(frozen=True)
class _Decision:
    act: str
    match: re.Match[str] | None = None
    origin: str = "cue"


class IntentUnderstander:
    """Classify a turn act, then extract only the slots that act needs."""

    def parse(
        self,
        user_message: str,
        turn: int,
        last_ask: AttributeName | None = None,
    ) -> IntentUpdate:
        message = " ".join(str(user_message).split())
        decision = self._classify(message, last_ask)
        return self._extract(decision, message, turn, last_ask)

    def _classify(self, message: str, last_ask: AttributeName | None) -> _Decision:
        """Return an act. Embeddings should replace this method later, not extractors."""
        for act, pattern in _TEMPLATES:
            match = pattern.match(message)
            if match:
                return _Decision(act, match, "template")

        for act, cue in _CUES:
            if cue.search(message) and self._act_confirmed(act, message):
                return _Decision(act, None, "cue")

        if NEGATION_RE.search(message):
            return _Decision("reject", None, "cue")

        match = INITIAL_PREFERENCE_RE.match(message)
        if match:
            return _Decision("open_buy", match, "initial_preference")

        if self._is_value_reply(message, last_ask):
            return _Decision("answer_ask", None, "value_reply")

        if NOOP_CUE_RE.search(message):
            return _Decision("noop", None, "cue")

        return _Decision("fallback", None, "fallback")

    def _act_confirmed(self, act: str, message: str) -> bool:
        """Cue hits are not enough when the act needs a span."""
        if act == "replace":
            replacement = self._replacement_span(message)
            return bool(replacement) and not NEGATION_RE.search(replacement)
        if act == "answer_ask":
            detail = self._detail_span(message)
            return bool(detail) and not NEGATION_RE.search(detail)
        if act == "open_buy":
            if NEGATION_RE.search(message):
                return False
            return bool(self._item_span(message) or self._requirement_span(message))
        return True

    def _extract(
        self,
        decision: _Decision,
        message: str,
        turn: int,
        last_ask: AttributeName | None,
    ) -> IntentUpdate:
        act = decision.act
        match = decision.match
        parser = "fallback" if act in {"reject", "fallback"} else "phrase"

        if act == "replace":
            text = (
                self._clean(match.group("constraint"))
                if match is not None
                else self._replacement_span(message)
            )
            source = "override" if decision.origin == "template" else "override_rule"
            return IntentUpdate(
                interaction_kind=ACT_KIND[act],
                constraints=[self._constraint(text, turn, source)],
                supersede_preferences=True,
                parser=parser,
            )

        if act == "exhaust_ask":
            attribute = (
                self._attribute(match.group("attribute"), last_ask)
                if match is not None
                else self._mentioned_attribute(message, last_ask)
            )
            return IntentUpdate(
                interaction_kind=ACT_KIND[act],
                exhausted={attribute},
                parser=parser,
            )

        if act == "decline_ask":
            attribute = (
                self._attribute(match.group("attribute"), last_ask)
                if match is not None
                else self._mentioned_attribute(message, last_ask)
            )
            return IntentUpdate(
                interaction_kind=ACT_KIND[act],
                no_preference={attribute},
                parser=parser,
            )

        if act == "ask_me":
            return IntentUpdate(
                interaction_kind=ACT_KIND[act],
                parser=parser,
            )

        if act == "open_browse":
            category = (
                self._clean(match.group("category"))
                if match is not None
                else self._item_span(message)
            )
            return IntentUpdate(
                interaction_kind=ACT_KIND[act],
                category=category,
                parser=parser,
            )

        if act == "answer_ask":
            return self._extract_answer_ask(decision, message, turn, last_ask, parser)

        if act == "open_buy":
            return self._extract_open_buy(decision, message, turn, parser)

        if act == "reject":
            return IntentUpdate(
                interaction_kind=ACT_KIND[act],
                category=self._item_span(message),
                parser="fallback",
            )

        if act == "noop":
            return IntentUpdate(interaction_kind=ACT_KIND[act], parser="phrase")

        fallback = self._fallback_text(message)
        update = IntentUpdate(
            interaction_kind=ACT_KIND["fallback"],
            parser="fallback",
            fallback_terms=fallback.split() if fallback else [],
        )
        if fallback:
            update.constraints.append(self._constraint(fallback, turn, "fallback"))
        return update

    def _extract_open_buy(
        self,
        decision: _Decision,
        message: str,
        turn: int,
        parser: str,
    ) -> IntentUpdate:
        match = decision.match
        if match is not None:
            text = self._clean(match.group("constraint"))
            kind = (
                "initial_preference"
                if decision.origin == "initial_preference"
                else ACT_KIND["open_buy"]
            )
            source = (
                "initial_provisional"
                if decision.origin == "initial_preference"
                else "initial"
            )
            return IntentUpdate(
                interaction_kind=kind,
                category=self._clean(match.group("category")),
                constraints=[self._constraint(text, turn, source)],
                parser=parser,
            )

        category = self._item_span(message)
        requirement = self._requirement_span(message)
        constraints = (
            [self._constraint(requirement, turn, "initial_rule")]
            if requirement
            else []
        )
        return IntentUpdate(
            interaction_kind=ACT_KIND["open_buy"],
            category=category,
            constraints=constraints,
            parser=parser,
        )

    def _extract_answer_ask(
        self,
        decision: _Decision,
        message: str,
        turn: int,
        last_ask: AttributeName | None,
        parser: str,
    ) -> IntentUpdate:
        match = decision.match
        if match is not None:
            values = [
                self._clean(value)
                for value in re.split(r";\s+", match.group("constraints"))
                if self._clean(value)
            ]
            return IntentUpdate(
                interaction_kind=ACT_KIND["answer_ask"],
                constraints=[
                    self._constraint(value, turn, "clarification")
                    for value in values
                ],
                parser=parser,
            )

        if decision.origin == "cue":
            detail = self._detail_span(message)
            return IntentUpdate(
                interaction_kind=ACT_KIND["answer_ask"],
                constraints=[
                    self._constraint(value, turn, "clarification_rule")
                    for value in self._split_constraints(detail)
                ],
                parser=parser,
            )

        text = self._value_text(message)
        return IntentUpdate(
            interaction_kind="answer_ask",
            constraints=[
                self._constraint(
                    text,
                    turn,
                    "answer_ask",
                    attribute=self._bind_attribute(text, last_ask),
                )
            ],
            parser=parser,
        )

    def _is_value_reply(
        self,
        message: str,
        last_ask: AttributeName | None,
    ) -> bool:
        if last_ask is None:
            return False
        if message.endswith("?"):
            return False
        if NEGATION_RE.search(message) or NOOP_CUE_RE.search(message):
            return False
        text = self._value_text(message)
        if not text:
            return False
        tokens = text.split()
        if re.match(r"^[a-z_]+\s*:", text, re.I):
            return True
        if len(tokens) <= 4 and self.classify_constraint(text) != "feature":
            return True
        return len(tokens) <= 3

    def _value_text(self, message: str) -> str:
        return self._clean(VALUE_LEADIN_RE.sub("", message))

    def _bind_attribute(
        self,
        text: str,
        last_ask: AttributeName | None,
    ) -> AttributeName:
        classified = self.classify_constraint(text)
        if classified != "feature":
            return classified
        return last_ask or "feature"

    def _constraint(
        self,
        text: str,
        turn: int,
        source: str,
        attribute: AttributeName | None = None,
    ) -> Constraint:
        return Constraint(
            text=text,
            attribute=attribute or self.classify_constraint(text),
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

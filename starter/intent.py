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


EXHAUSTED_CUE_RE = re.compile(
    r"(?:do\s+not|don'?t)\s+have\s+an\s+additional\s+preference|"
    r"\bno additional preference\b|"
    r"\bnothing (?:else|more) (?:on|about|for)\b|"
    r"\bno more (?:preference|preferences)\b|"
    r"\bthat'?s all\b",
    re.IGNORECASE,
)
NO_PREFERENCE_CUE_RE = re.compile(
    r"(?:do\s+not|don'?t)\s+have\s+a\s+preference|"
    r"\bno preference\b|"
    r"\bdoes not matter\b|\bdoesn't matter\b|"
    r"\bdo not care\b|\bdon't care\b|"
    r"\byou (?:can )?pick\b|\bup to you\b|"
    r"\buse your judgment\b|\bany(?:thing)? is fine\b",
    re.IGNORECASE,
)
CLARIFICATION_REQUEST_CUE_RE = re.compile(
    r"\b(?:ask me (?:about|for)|ask (?:about|for))\b.*\b"
    r"(?:attribute|detail|preference|requirement)\b",
    re.IGNORECASE,
)
OVERRIDE_CUE_RE = re.compile(
    r"\b(?:scratch that|ignore|forget|never mind|instead|change[sd]? my mind|"
    r"switch(?:ing)? to|rather (?:have|use|wear)|actually)\b",
    re.IGNORECASE,
)
BROWSING_CUE_RE = re.compile(
    r"\b(?:just (?:looking|browsing|exploring)|still (?:looking|browsing|exploring)|"
    r"not sure(?: yet)?|exploring for now|nothing specific)\b",
    re.IGNORECASE,
)
CLARIFICATION_CUE_RE = re.compile(
    r"\b(?:what matters is|important (?:part|bit) is|priority is|prioritize|"
    r"care about is)\b",
    re.IGNORECASE,
)
BUYING_CUE_RE = re.compile(
    r"\b(?:must be|needs? to be|has to be|have to be|is required|"
    r"key requirement|a must|main thing)\b",
    re.IGNORECASE,
)
LOOKING_RE = re.compile(
    r"\b(?:looking|shopping)\s+for\b",
    re.IGNORECASE,
)
SHOW_ME_RE = re.compile(
    r"\b(?:show|find|get)\s+me\b",
    re.IGNORECASE,
)
WANT_ITEM_RE = re.compile(
    r"\b(?:i\s+)?(?:need|want)\s+(?!to\b)",
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

# Act precedence. Extractors, not full-sentence templates, decide the slots.
_CUES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("replace", OVERRIDE_CUE_RE),
    ("exhaust_ask", EXHAUSTED_CUE_RE),
    ("decline_ask", NO_PREFERENCE_CUE_RE),
    ("ask_me", CLARIFICATION_REQUEST_CUE_RE),
    ("open_browse", BROWSING_CUE_RE),
    ("open_buy", BUYING_CUE_RE),
    ("answer_ask", CLARIFICATION_CUE_RE),
)


@dataclass(frozen=True)
class _Decision:
    act: str
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
        if NOOP_CUE_RE.search(message):
            return _Decision("noop", "cue")

        for act, cue in _CUES:
            if cue.search(message) and self._act_confirmed(act, message):
                return _Decision(act, "cue")

        if self._is_open_buy(message) and self._act_confirmed("open_buy", message):
            return _Decision("open_buy", "cue")

        if NEGATION_RE.search(message):
            return _Decision("reject", "cue")

        if self._is_value_reply(message, last_ask):
            return _Decision("answer_ask", "value_reply")

        return _Decision("fallback", "fallback")

    def _is_open_buy(self, message: str) -> bool:
        """Buying is looking-for / show-me / want-item, not one evaluator sentence."""
        if LOOKING_RE.search(message) or SHOW_ME_RE.search(message):
            return True
        return bool(WANT_ITEM_RE.search(message) and self._item_span(message))

    def _act_confirmed(self, act: str, message: str) -> bool:
        """Cue hits are not enough when the act needs a span."""
        if act == "replace":
            replacement = self._replacement_span(message)
            return bool(replacement) and not NEGATION_RE.search(replacement)
        if act == "answer_ask":
            detail = self._detail_span(message)
            return bool(detail) and not NEGATION_RE.search(detail)
        if act == "open_buy":
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
        parser = "fallback" if act in {"reject", "fallback"} else "phrase"

        if act == "replace":
            text = self._replacement_span(message)
            return IntentUpdate(
                interaction_kind=ACT_KIND[act],
                constraints=[self._constraint(text, turn, "override")],
                supersede_preferences=True,
                parser=parser,
            )

        if act == "exhaust_ask":
            return IntentUpdate(
                interaction_kind=ACT_KIND[act],
                exhausted={self._mentioned_attribute(message, last_ask)},
                parser=parser,
            )

        if act == "decline_ask":
            return IntentUpdate(
                interaction_kind=ACT_KIND[act],
                no_preference={self._mentioned_attribute(message, last_ask)},
                parser=parser,
            )

        if act == "ask_me":
            return IntentUpdate(
                interaction_kind=ACT_KIND[act],
                parser=parser,
            )

        if act == "open_browse":
            return IntentUpdate(
                interaction_kind=ACT_KIND[act],
                category=self._item_span(message),
                parser=parser,
            )

        if act == "answer_ask":
            return self._extract_answer_ask(decision, message, turn, last_ask, parser)

        if act == "open_buy":
            return self._extract_open_buy(message, turn, parser)

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

    def _extract_open_buy(self, message: str, turn: int, parser: str) -> IntentUpdate:
        category = self._item_span(message)
        requirement = self._requirement_span(message)
        if requirement and not NEGATION_RE.search(requirement):
            looking = bool(LOOKING_RE.search(message))
            return IntentUpdate(
                interaction_kind=ACT_KIND["open_buy"],
                category=category,
                constraints=[
                    self._constraint(
                        requirement,
                        turn,
                        "initial" if looking else "initial_rule",
                    )
                ],
                parser=parser,
            )

        remainder = self._trailing_statement(message)
        if remainder and not NEGATION_RE.search(remainder):
            return IntentUpdate(
                interaction_kind="initial_preference",
                category=category,
                constraints=[self._constraint(remainder, turn, "initial_provisional")],
                parser=parser,
            )

        return IntentUpdate(
            interaction_kind=ACT_KIND["open_buy"],
            category=category,
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
        if decision.origin != "value_reply":
            detail = self._detail_span(message)
            if detail:
                return IntentUpdate(
                    interaction_kind=ACT_KIND["answer_ask"],
                    constraints=[
                        self._constraint(value, turn, "clarification")
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
            r"(?:ignore|forget|never mind|scratch that)[^;:—-]*[;:—-]\s*(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                return cls._trim_discourse(match.group(1))
        return ""

    @classmethod
    def _item_span(cls, message: str) -> str | None:
        patterns = (
            r"\b(?:looking|shopping|browsing)\s+for\s+(.+?)(?:\s+but\b|[,.;—-]|$)",
            r"\b(?:just )?(?:looking at|browsing|exploring)\s+(?:some\s+)?(.+?)"
            r"(?:\s+for now|[,.;—-]|$)",
            r"\b(?:show|find|get)\s+me\s+(?:an?\s+)?(.+?)(?:[,.;—-]|$)",
            r"^\s*(?:i\s+)?(?:need|want)\s+(.+?)(?:[,.;—-]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                value = cls._trim_discourse(match.group(1))
                value = re.sub(r"^(?:an?|the)\s+", "", value, flags=re.IGNORECASE)
                if value:
                    return value
        return None

    @classmethod
    def _detail_span(cls, message: str) -> str:
        match = re.search(
            r"\b(?:what matters is|important (?:part|bit) is|priority is|"
            r"prioritize|care about is)\s*:?\s*(.+)$",
            message,
            flags=re.IGNORECASE,
        )
        return cls._trim_discourse(match.group(1)) if match else ""

    @classmethod
    def _requirement_span(cls, message: str) -> str:
        parts = [part for part in re.split(r"\s*[—;]\s*|(?<=[.!?])\s+", message) if part.strip()]
        candidates = [parts[-1], message] if len(parts) > 1 else [message]
        patterns = (
            r"(.+?)\s+is\s+(?:required|a must)\b",
            r"\b(?:a )?key requirement is\s*:?\s*(.+)$",
            r"\b(?:must be|needs? to be|has to be|have to be|requirement is)\s*:?\s*(.+)$",
            r"(?:that'?s |it'?s )?(?:the )?main thing(?: is|:)\s*(.+)$",
        )
        for detail in candidates:
            for pattern in patterns:
                match = re.search(pattern, detail, flags=re.IGNORECASE)
                if match:
                    return cls._trim_discourse(match.group(1))
        return ""

    @classmethod
    def _trailing_statement(cls, message: str) -> str:
        match = re.match(r"^(.+?[.!?])\s+(.+)$", message)
        if match is None:
            return ""
        rest = match.group(2).strip()
        if not rest or rest.endswith("?"):
            return ""
        return cls._clean(rest)

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
            r"\s*(?:,\s*)?(?:but )?(?:i(?:'m| am) )?still (?:exploring|looking|browsing)"
            r"(?:\s+for now)?\s*$",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(
            r"\s+(?:instead|for now|please)\s*[.!?]*$",
            "",
            value,
            flags=re.IGNORECASE,
        )
        return cls._clean(value)

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

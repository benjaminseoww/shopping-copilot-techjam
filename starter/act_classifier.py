from __future__ import annotations

from typing import Protocol

# Acts the embedding classifier may propose. Extractors still have to fill them.
# `reject` is never predicted: catalog copy often contains "not" / "without".
CLASSIFIABLE_ACTS: tuple[str, ...] = (
    "replace",
    "exhaust_ask",
    "decline_ask",
    "ask_me",
    "open_browse",
    "open_buy",
    "answer_ask",
    "noop",
)

# Short speech-act paraphrases, not evaluator sentence templates.
PROTOTYPES: dict[str, tuple[str, ...]] = {
    "replace": (
        "Ignore what I said before and use this requirement instead.",
        "Scratch my earlier preference and switch to a new one.",
        "Change my mind. The replacement I need is this.",
        "Forget the old constraint. Go with this from now on.",
    ),
    "exhaust_ask": (
        "I have no additional preference for that attribute.",
        "That's all I have to say about this field.",
        "Nothing more to add on that attribute.",
        "No more preferences on this point.",
    ),
    "decline_ask": (
        "I don't have a preference for that. You can pick.",
        "That attribute does not matter to me.",
        "No preference, use your judgment.",
        "Anything is fine for that field. Up to you.",
    ),
    "ask_me": (
        "Ask me about one specific attribute.",
        "Question me on a single detail instead.",
        "Ask about one preference at a time.",
        "Those options are not quite right. Ask me a specific question.",
    ),
    "open_browse": (
        "I'm just browsing and still exploring options.",
        "Not sure yet, just looking around.",
        "Still exploring, nothing specific.",
        "Just window shopping for now.",
    ),
    "open_buy": (
        "I'm looking for an item and a key requirement must be met.",
        "Show me this product. It has to meet a hard constraint.",
        "I need this item and that feature is required.",
        "Find me this. The main thing is a specific requirement.",
    ),
    "answer_ask": (
        "What matters is this detail.",
        "The important part is this constraint.",
        "Priority is this value.",
        "The thing I care about is this.",
    ),
    "noop": (
        "I need to think for a moment.",
        "I want to keep looking before deciding.",
        "Give me a second, I'm not ready yet.",
        "Let me think. Still thinking.",
    ),
}

DEFAULT_MIN_SCORE = 0.42
DEFAULT_MARGIN = 0.04
DEFAULT_NLI_MIN_SCORE = 0.40
DEFAULT_NLI_MARGIN = 0.08

# Extra paraphrases for NLI only. MiniLM cosine keeps the shorter set above.
NLI_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "replace": PROTOTYPES["replace"]
    + (
        "Ignore what I said before. What I need is this.",
        "Scratch that, go with this instead.",
        "Please ignore my earlier preference.",
        "Forget that. Use this instead.",
    ),
    "exhaust_ask": PROTOTYPES["exhaust_ask"]
    + ("I do not have an additional preference for that.",),
    "decline_ask": PROTOTYPES["decline_ask"]
    + (
        "Color is not a factor for me.",
        "That does not matter. You can choose.",
    ),
    "ask_me": PROTOTYPES["ask_me"] + ("Ask me a specific question instead.",),
    "open_browse": PROTOTYPES["open_browse"]
    + ("I'm looking around but I am not sure yet.",),
    "open_buy": PROTOTYPES["open_buy"]
    + (
        "I need a product that must have this feature.",
        "Show me this. It has to be this.",
    ),
    "answer_ask": PROTOTYPES["answer_ask"]
    + (
        "For that, what matters is this.",
        "The important part is this.",
    ),
    "noop": PROTOTYPES["noop"]
    + (
        "Keep looking. I am not ready yet.",
        "I need more time to think.",
    ),
}


class ActClassifier(Protocol):
    def predict(self, message: str) -> str | None:
        """Return an act name, or None when confidence is too low."""


class PrototypeActClassifier:
    """Nearest prototype in MiniLM space. Does not extract spans."""

    def __init__(
        self,
        embedder: object,
        min_score: float = DEFAULT_MIN_SCORE,
        margin: float = DEFAULT_MARGIN,
    ) -> None:
        self._embedder = embedder
        self._min_score = min_score
        self._margin = margin
        self._acts: list[str] = []
        texts: list[str] = []
        for act, examples in PROTOTYPES.items():
            for example in examples:
                self._acts.append(act)
                texts.append(example)
        matrix = _as_normalized_matrix(embedder.encode(texts))
        self._prototypes = matrix

    def predict(self, message: str) -> str | None:
        if not message.strip():
            return None
        try:
            import numpy as np
        except ImportError:
            return None
        query = _as_normalized_matrix(self._embedder.encode([message]))
        if query.size == 0 or self._prototypes.size == 0:
            return None
        scores = self._prototypes @ query[0]
        best_index = int(np.argmax(scores))
        best_act = self._acts[best_index]
        best_score = float(scores[best_index])
        if best_score < self._min_score:
            return None
        other = [
            float(score)
            for act, score in zip(self._acts, scores)
            if act != best_act
        ]
        second = max(other) if other else 0.0
        if best_score - second < self._margin:
            return None
        if best_act not in CLASSIFIABLE_ACTS:
            return None
        return best_act


class NliActClassifier:
    """NLI entailment against speech-act paraphrases. Does not extract spans."""

    def __init__(
        self,
        nli: object,
        min_score: float = DEFAULT_NLI_MIN_SCORE,
        margin: float = DEFAULT_NLI_MARGIN,
        prototypes: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        self._nli = nli
        self._min_score = min_score
        self._margin = margin
        source = prototypes or NLI_PROTOTYPES
        self._acts: list[str] = []
        self._hypotheses: list[str] = []
        for act, examples in source.items():
            if act not in CLASSIFIABLE_ACTS:
                continue
            for example in examples:
                self._acts.append(act)
                self._hypotheses.append(example)

    def predict(self, message: str) -> str | None:
        if not message.strip() or not self._hypotheses:
            return None
        try:
            import numpy as np
        except ImportError:
            return None
        try:
            scores = np.asarray(
                self._nli.entailment_probs(message, self._hypotheses),
                dtype=np.float32,
            )
        except Exception:
            return None
        if scores.ndim != 1 or scores.size != len(self._acts):
            return None
        best: dict[str, float] = {}
        for act, score in zip(self._acts, scores):
            value = float(score)
            if act not in best or value > best[act]:
                best[act] = value
        if not best:
            return None
        ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
        best_act, best_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < self._min_score:
            return None
        if best_score - second < self._margin:
            return None
        if best_act not in CLASSIFIABLE_ACTS:
            return None
        return best_act


def try_build_act_classifier(
    embedder: object | None = None,
    nli: object | None = None,
) -> ActClassifier | None:
    if nli is not None:
        try:
            return NliActClassifier(nli)
        except Exception:
            pass
    if embedder is None:
        return None
    try:
        return PrototypeActClassifier(embedder)
    except Exception:
        return None


def _as_normalized_matrix(encoded: object):
    import numpy as np

    matrix = np.asarray(encoded, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)

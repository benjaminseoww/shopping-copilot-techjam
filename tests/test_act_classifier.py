from __future__ import annotations

import unittest

from pathlib import Path

from starter.act_classifier import (
    CLASSIFIABLE_ACTS,
    NliActClassifier,
    PrototypeActClassifier,
    try_build_act_classifier,
)
from starter.intent import IntentUnderstander
from starter.nli import try_load_nli


class _KeywordEmbedder:
    """Tiny bag-of-keywords encoder so prototype tests do not need MiniLM."""

    def encode(self, texts):
        import numpy as np

        rows = []
        for text in texts:
            lowered = text.lower()
            vector = np.zeros(len(CLASSIFIABLE_ACTS), dtype=np.float32)
            for index, act in enumerate(CLASSIFIABLE_ACTS):
                if any(token in lowered for token in _ACT_TOKENS[act]):
                    vector[index] = 1.0
            rows.append(vector)
        return np.stack(rows)


_ACT_TOKENS = {
    "replace": ("scratch", "ignore what i said", "change my mind", "forget the old", "replacement"),
    "exhaust_ask": ("no additional", "that's all", "nothing more", "no more preference"),
    "decline_ask": ("don't have a preference", "does not matter", "no preference", "anything is fine"),
    "ask_me": ("ask me", "question me", "ask about one"),
    "open_browse": ("just browsing", "still exploring", "window shopping", "not sure yet"),
    "open_buy": ("key requirement", "hard constraint", "has to meet", "main thing"),
    "answer_ask": ("what matters", "important part", "priority is", "care about"),
    "noop": ("need to think", "keep looking", "give me a second", "still thinking"),
}


def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def _live_nli_available() -> bool:
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        from tokenizers import Tokenizer  # noqa: F401
    except ImportError:
        return False
    path = Path("data") / "nli" / "nli-deberta-v3-xsmall"
    return (path / "model.onnx").is_file() and (path / "tokenizer.json").is_file()


class ActClassifierTest(unittest.TestCase):
    def test_missing_embedder_builds_nothing(self) -> None:
        self.assertIsNone(try_build_act_classifier(None))

    def test_broken_embedder_builds_nothing(self) -> None:
        class Broken:
            def encode(self, texts):
                raise RuntimeError("no weights")

        self.assertIsNone(try_build_act_classifier(Broken()))

    @unittest.skipUnless(_has_numpy(), "numpy is required for prototype cosine")
    def test_predicts_the_matching_prototype_family(self) -> None:
        classifier = PrototypeActClassifier(_KeywordEmbedder(), min_score=0.2, margin=0.01)
        self.assertEqual(classifier.predict("That's all I wanted to add."), "exhaust_ask")
        self.assertEqual(classifier.predict("I need to think for a bit."), "noop")
        self.assertIsNone(classifier.predict("asdf qwer zxcv"))

    @unittest.skipUnless(_has_numpy(), "numpy is required for prototype cosine")
    def test_intent_uses_prototype_label_without_rewriting_spans(self) -> None:
        classifier = PrototypeActClassifier(_KeywordEmbedder(), min_score=0.2, margin=0.01)
        intent = IntentUnderstander(act_classifier=classifier)
        update = intent.parse("Show me running shoes. They have to be cotton.", 1)
        self.assertEqual(update.interaction_kind, "buying")
        self.assertEqual(update.constraints[0].text, "cotton")

    @unittest.skipUnless(_has_numpy(), "numpy is required for NLI scores")
    def test_nli_classifier_uses_entailment_not_embedder(self) -> None:
        class Boom:
            def encode(self, texts):
                raise AssertionError("MiniLM prototypes should not run")

        classifier = try_build_act_classifier(
            embedder=Boom(),
            nli=_KeywordNli(),
        )
        self.assertIsInstance(classifier, NliActClassifier)
        self.assertEqual(classifier.predict("That's all I wanted to add."), "exhaust_ask")
        self.assertEqual(classifier.predict("I need to think for a bit."), "noop")
        self.assertIsNone(classifier.predict("asdf qwer zxcv"))

    @unittest.skipUnless(_has_numpy(), "numpy is required for NLI scores")
    def test_nli_abstains_when_two_acts_are_tied(self) -> None:
        classifier = NliActClassifier(_KeywordNli(), min_score=0.2, margin=0.20)
        self.assertIsNone(classifier.predict("No preference and nothing more."))

    @unittest.skipUnless(_live_nli_available(), "NLI ONNX weights are required")
    def test_live_nli_labels_paraphrase_decline_and_stall(self) -> None:
        model = try_load_nli(Path("data") / "nli" / "nli-deberta-v3-xsmall")
        self.assertIsNotNone(model)
        classifier = NliActClassifier(model)
        self.assertEqual(classifier.predict("The color does not matter to me."), "decline_ask")
        self.assertEqual(classifier.predict("I just need a minute to think."), "noop")
        self.assertEqual(classifier.predict("Could you ask me one specific question?"), "ask_me")
        self.assertIsNone(classifier.predict("asdf qwer zxcv"))


class _KeywordNli:
    """Map prototype keywords onto entailment scores without ONNX."""

    def entailment_probs(self, message, hypotheses):
        import numpy as np

        lowered = message.lower()
        rows = []
        for hypothesis in hypotheses:
            score = 0.05
            for tokens in _ACT_TOKENS.values():
                if any(token in hypothesis.lower() and token in lowered for token in tokens):
                    score = 0.90
                    break
            rows.append(score)
        return np.asarray(rows, dtype=np.float32)


if __name__ == "__main__":
    unittest.main()

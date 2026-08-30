from __future__ import annotations

import unittest

from starter.act_classifier import CLASSIFIABLE_ACTS, PrototypeActClassifier, try_build_act_classifier
from starter.intent import IntentUnderstander


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


if __name__ == "__main__":
    unittest.main()

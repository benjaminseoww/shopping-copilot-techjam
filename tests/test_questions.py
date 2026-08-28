from __future__ import annotations

import unittest

from starter.models import SessionState, UserProfile
from starter.questions import QuestionsEngine


class QuestionsEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = QuestionsEngine()
        self.state = SessionState("session-1", UserProfile())

    def test_asks_other_on_turns_one_through_nine(self) -> None:
        for turn in range(1, 10):
            with self.subTest(turn=turn):
                decision = self.engine.decide(self.state, turn)
                self.assertEqual(decision.ask_attribute, "other")
                self.assertTrue(decision.message)

    def test_does_not_ask_on_turn_ten(self) -> None:
        decision = self.engine.decide(self.state, 10)
        self.assertIsNone(decision.ask_attribute)
        self.assertTrue(decision.message)


if __name__ == "__main__":
    unittest.main()

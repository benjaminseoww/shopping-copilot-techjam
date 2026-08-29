from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from starter.embedder import default_model_dir, try_load_minilm


class EmbedderTest(unittest.TestCase):
    def test_try_load_returns_none_when_weights_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(try_load_minilm(Path(directory) / "missing"))

    def test_default_model_dir_is_beside_the_catalog(self) -> None:
        previous = os.environ.pop("SHOPPING_MINILM_DIR", None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                catalog = Path(directory) / "catalog.jsonl"
                self.assertEqual(
                    default_model_dir(catalog),
                    Path(directory).resolve() / "minilm" / "all-MiniLM-L6-v2",
                )
        finally:
            if previous is not None:
                os.environ["SHOPPING_MINILM_DIR"] = previous


if __name__ == "__main__":
    unittest.main()

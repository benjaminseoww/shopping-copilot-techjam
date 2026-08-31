from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from starter.nli import default_nli_dir, try_load_nli


def _has_ort_and_tokenizers() -> bool:
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        from tokenizers import Tokenizer  # noqa: F401
    except ImportError:
        return False
    return True


def _nli_dir() -> Path:
    return Path("data") / "nli" / "nli-deberta-v3-xsmall"


def _nli_weights_present() -> bool:
    path = _nli_dir()
    return (path / "model.onnx").is_file() and (path / "tokenizer.json").is_file()


class NliLoaderTest(unittest.TestCase):
    def test_try_load_returns_none_when_weights_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(try_load_nli(Path(directory) / "missing"))

    def test_default_nli_dir_is_beside_the_catalog(self) -> None:
        previous = os.environ.pop("SHOPPING_NLI_DIR", None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                catalog = Path(directory) / "catalog.jsonl"
                self.assertEqual(
                    default_nli_dir(catalog),
                    Path(directory).resolve() / "nli" / "nli-deberta-v3-xsmall",
                )
        finally:
            if previous is not None:
                os.environ["SHOPPING_NLI_DIR"] = previous

    def test_skip_env_disables_loader(self) -> None:
        previous = os.environ.get("SHOPPING_SKIP_NLI")
        os.environ["SHOPPING_SKIP_NLI"] = "1"
        try:
            self.assertIsNone(try_load_nli(_nli_dir()))
        finally:
            if previous is None:
                os.environ.pop("SHOPPING_SKIP_NLI", None)
            else:
                os.environ["SHOPPING_SKIP_NLI"] = previous

    @unittest.skipUnless(
        _has_ort_and_tokenizers() and _nli_weights_present(),
        "NLI ONNX weights and onnxruntime are required",
    )
    def test_live_weights_load(self) -> None:
        model = try_load_nli(_nli_dir())
        self.assertIsNotNone(model)
        scores = model.entailment_probs(
            "I need to think for a moment.",
            [
                "I need to think for a moment.",
                "I'm just browsing and still exploring options.",
            ],
        )
        import numpy as np

        values = np.asarray(scores, dtype=np.float32)
        self.assertEqual(values.shape, (2,))
        self.assertGreater(float(values[0]), float(values[1]))


if __name__ == "__main__":
    unittest.main()

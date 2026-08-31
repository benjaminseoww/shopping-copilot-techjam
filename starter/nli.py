from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from .embedder import download_file, embeddings_disabled


MODEL_ID = "nli-deberta-v3-xsmall"
MODEL_DIRNAME = "nli"
MAX_LENGTH = 128
ENTAILMENT_INDEX = 1

_TOKENIZER_URL = (
    "https://huggingface.co/Xenova/nli-deberta-v3-xsmall/resolve/main/tokenizer.json"
)
_MODEL_URL = (
    "https://huggingface.co/Xenova/nli-deberta-v3-xsmall/resolve/main/"
    "onnx/model_quantized.onnx"
)


def nli_disabled() -> bool:
    if embeddings_disabled():
        return True
    return os.environ.get("SHOPPING_SKIP_NLI", "").strip() in {"1", "true", "True"}


def default_nli_dir(catalog_path: str | Path) -> Path:
    override = os.environ.get("SHOPPING_NLI_DIR", "").strip()
    if override:
        return Path(override)
    return Path(catalog_path).resolve().parent / MODEL_DIRNAME / MODEL_ID


def try_load_nli(model_dir: str | Path) -> NliEntailmentModel | None:
    if nli_disabled():
        return None
    path = Path(model_dir)
    model_path = path / "model.onnx"
    tokenizer_path = path / "tokenizer.json"
    if not model_path.is_file() or not tokenizer_path.is_file():
        return None
    try:
        return NliEntailmentModel(path)
    except Exception:
        return None


class NliEntailmentModel:
    """Local ONNX NLI cross-encoder. Never downloads at inference time."""

    def __init__(self, model_dir: str | Path, max_length: int = MAX_LENGTH) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.max_length = max_length
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        options = ort.SessionOptions()
        # Quantized DeBERTa is not deterministic on the default ORT thread pool.
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {item.name for item in self.session.get_inputs()}
        self._np = np

    def entailment_probs(self, premise: str, hypotheses: Sequence[str]) -> object:
        """Return P(entailment) for each (premise, hypothesis) pair."""
        np = self._np
        if not hypotheses:
            return np.zeros((0,), dtype=np.float32)
        pairs = [
            (premise if str(premise).strip() else " ", hypothesis if hypothesis.strip() else " ")
            for hypothesis in hypotheses
        ]
        encodings = self.tokenizer.encode_batch(pairs)
        max_len = min(self.max_length, max(len(item.ids) for item in encodings))
        input_ids = np.zeros((len(encodings), max_len), dtype=np.int64)
        attention_mask = np.zeros((len(encodings), max_len), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            ids = encoding.ids[:max_len]
            mask = encoding.attention_mask[:max_len]
            input_ids[row, : len(ids)] = ids
            attention_mask[row, : len(mask)] = mask
        feeds = {}
        if "input_ids" in self._input_names:
            feeds["input_ids"] = input_ids
        if "attention_mask" in self._input_names:
            feeds["attention_mask"] = attention_mask
        logits = np.asarray(self.session.run(None, feeds)[0], dtype=np.float32)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(shifted)
        probs /= np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        if logits.shape[1] <= ENTAILMENT_INDEX:
            return np.zeros((len(hypotheses),), dtype=np.float32)
        return probs[:, ENTAILMENT_INDEX]


def download_nli(model_dir: str | Path) -> Path:
    """Fetch tokenizer + quantized NLI ONNX into model_dir. Network only here."""
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    tokenizer_path = path / "tokenizer.json"
    model_path = path / "model.onnx"
    if not tokenizer_path.is_file():
        download_file(_TOKENIZER_URL, tokenizer_path)
    if not model_path.is_file():
        download_file(_MODEL_URL, model_path)
    return path


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data") / MODEL_DIRNAME / MODEL_ID
    print(f"downloading NLI to {target}", file=sys.stderr)
    download_nli(target)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()

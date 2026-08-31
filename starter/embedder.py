from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
from typing import Protocol, Sequence


MODEL_ID = "all-MiniLM-L6-v2"
MODEL_DIRNAME = "minilm"
MAX_LENGTH = 128
BATCH_SIZE = 64

_TOKENIZER_URL = (
    "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
)
_MODEL_URL = (
    "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx"
)


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> object:
        """Return an array-like of shape [len(texts), dim]."""


def embeddings_disabled() -> bool:
    return os.environ.get("SHOPPING_SKIP_EMBEDDINGS", "").strip() in {"1", "true", "True"}


def default_model_dir(catalog_path: str | Path) -> Path:
    override = os.environ.get("SHOPPING_MINILM_DIR", "").strip()
    if override:
        return Path(override)
    return Path(catalog_path).resolve().parent / MODEL_DIRNAME / MODEL_ID


def try_load_minilm(model_dir: str | Path) -> MiniLmEmbedder | None:
    if embeddings_disabled():
        return None
    path = Path(model_dir)
    model_path = path / "model.onnx"
    tokenizer_path = path / "tokenizer.json"
    if not model_path.is_file() or not tokenizer_path.is_file():
        return None
    try:
        return MiniLmEmbedder(path)
    except Exception:
        return None


class MiniLmEmbedder:
    """Local ONNX MiniLM encoder. Never downloads at inference time."""

    def __init__(self, model_dir: str | Path, max_length: int = MAX_LENGTH) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.max_length = max_length
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {item.name for item in self.session.get_inputs()}
        self._np = np

    def encode(self, texts: Sequence[str]) -> object:
        np = self._np
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        batches = []
        for start in range(0, len(texts), BATCH_SIZE):
            batches.append(self._encode_batch(list(texts[start : start + BATCH_SIZE])))
        return np.vstack(batches)

    def _encode_batch(self, texts: list[str]) -> object:
        np = self._np
        encodings = self.tokenizer.encode_batch(
            [text if text.strip() else " " for text in texts]
        )
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
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        hidden = self.session.run(None, feeds)[0]
        if hidden.ndim == 2:
            pooled = hidden
        else:
            mask = attention_mask.astype(np.float32)[:, :, None]
            pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        return _l2_normalize(np.asarray(pooled, dtype=np.float32))


def _l2_normalize(matrix: object) -> object:
    import numpy as np

    vectors = np.asarray(matrix, dtype=np.float32)
    if vectors.size == 0:
        return vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


def download_minilm(model_dir: str | Path) -> Path:
    """Fetch tokenizer + quantized MiniLM ONNX into model_dir. Network only here."""
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    tokenizer_path = path / "tokenizer.json"
    model_path = path / "model.onnx"
    if not tokenizer_path.is_file():
        download_file(_TOKENIZER_URL, tokenizer_path)
    if not model_path.is_file():
        download_file(_MODEL_URL, model_path)
    return path


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "shopping-copilot"})
    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    tmp_path.replace(destination)


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data") / MODEL_DIRNAME / MODEL_ID
    print(f"downloading MiniLM to {target}", file=sys.stderr)
    download_minilm(target)
    print("done", file=sys.stderr)


if __name__ == "__main__":
    main()

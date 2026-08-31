"""Attribute detection that does not treat the evaluator gazetteer as complete.

Exact word lists still catch `cotton` / `blue`. This matcher labels leftover
query text and title spans by nearest definitional prototype, so values such
as `navy` or `suede` are analyzed even when the public simulator never used them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from .models import AttributeName


SPAN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]+")
STOPWORDS = frozenset(
    {
        "and",
        "for",
        "the",
        "with",
        "from",
        "this",
        "that",
        "item",
        "product",
        "dress",
        "dresses",
        "shirt",
        "shoe",
        "shoes",
        "boot",
        "boots",
        "necklace",
        "women",
        "men",
        "made",
        "available",
    }
)
FIELD_WORDS = frozenset(
    {
        "color",
        "colour",
        "material",
        "size",
        "style",
        "brand",
        "budget",
        "feature",
        "use",
        "case",
    }
)

# Definitional prototypes, not a copy of the evaluator's closed vocab.
PROTOTYPES: tuple[tuple[AttributeName, str], ...] = (
    ("material", "What the item is made of: fabric, leather, metal, suede, canvas, or textile."),
    ("color", "The visible color, shade, or hue of the item, such as navy, burgundy, beige, or ivory."),
    ("size", "The size, width, or dimensions of the item."),
    ("style", "The style, fit, cut, sleeve, or neckline of the item."),
    ("use_case", "The activity, season, or occasion the item is intended for."),
    ("budget", "The price, budget, or how much the item costs."),
    ("brand", "The brand name or store that makes the item."),
    ("feature", "A functional feature, construction detail, or extra product capability."),
)

CLASSIFY_MIN = 0.28
CLASSIFY_MARGIN = 0.04
SPAN_MIN = 0.34
MAX_SPANS = 24
CACHE_LIMIT = 4096


class SemanticMatcher:
    """Nearest-prototype attribute labels and open-vocab title spans."""

    def __init__(self, embedder: object | None) -> None:
        self._embedder = embedder
        self._cache: dict[str, list[float]] = {}
        self._centroids: dict[AttributeName, list[float]] = {}
        if embedder is not None:
            self._fit()

    def available(self) -> bool:
        return bool(self._centroids)

    def classify(self, text: str) -> AttributeName | None:
        if not self._centroids:
            return None
        value = " ".join(str(text).split()).strip()
        if not value:
            return None
        vector = self._encode_one(value)
        if vector is None:
            return None
        scored = [
            (field, _cosine(vector, centroid))
            for field, centroid in self._centroids.items()
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        best_field, best = scored[0]
        second = scored[1][1] if len(scored) > 1 else 0.0
        if best < CLASSIFY_MIN or best - second < CLASSIFY_MARGIN:
            return None
        if best_field == "feature":
            return None
        return best_field

    def best_span(self, text: str, field: AttributeName) -> str | None:
        if not self._centroids or field not in self._centroids:
            return None
        spans = candidate_spans(text)
        if not spans:
            return None
        vectors = self._encode_many(spans)
        centroid = self._centroids[field]
        best_span = None
        best_score = SPAN_MIN
        for span, vector in zip(spans, vectors):
            if vector is None:
                continue
            score = _cosine(vector, centroid)
            if score > best_score:
                best_span = span
                best_score = score
        return best_span

    def _fit(self) -> None:
        texts = [text for _, text in PROTOTYPES]
        vectors = self._encode_many(texts)
        grouped: dict[AttributeName, list[list[float]]] = {}
        for (field, _text), vector in zip(PROTOTYPES, vectors):
            if vector is None:
                continue
            grouped.setdefault(field, []).append(vector)
        self._centroids = {
            field: _mean(rows) for field, rows in grouped.items() if rows
        }

    def _encode_one(self, text: str) -> list[float] | None:
        encoded = self._encode_many([text])
        return encoded[0] if encoded else None

    def _encode_many(self, texts: Sequence[str]) -> list[list[float] | None]:
        if self._embedder is None or not texts:
            return [None] * len(texts)
        missing: list[str] = []
        for text in texts:
            if text not in self._cache:
                missing.append(text)
        if missing:
            try:
                matrix = self._embedder.encode(missing)
            except Exception:
                return [self._cache.get(text) for text in texts]
            rows = [_as_unit_row(row) for row in matrix]
            if len(rows) != len(missing):
                return [self._cache.get(text) for text in texts]
            for text, row in zip(missing, rows):
                if len(self._cache) >= CACHE_LIMIT:
                    self._cache.clear()
                self._cache[text] = row
        return [self._cache.get(text) for text in texts]


def candidate_spans(text: str) -> list[str]:
    tokens = [
        token
        for token in SPAN_RE.findall(text or "")
        if token.lower() not in STOPWORDS and token.lower() not in FIELD_WORDS
    ]
    spans: list[str] = []
    seen: set[str] = set()
    for index, token in enumerate(tokens):
        for span in (token, " ".join(tokens[index : index + 2])):
            key = span.lower()
            if len(key) < 3 or key in seen:
                continue
            seen.add(key)
            spans.append(span)
            if len(spans) >= MAX_SPANS:
                return spans
    return spans


def _as_unit_row(row: object) -> list[float]:
    values = [float(item) for item in row]  # type: ignore[arg-type]
    return _normalize(values)


def _mean(rows: list[list[float]]) -> list[float]:
    dim = len(rows[0])
    total = [0.0] * dim
    for row in rows:
        for index, value in enumerate(row):
            total[index] += value
    scale = 1.0 / len(rows)
    return _normalize([value * scale for value in total])


def _normalize(row: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in row)) or 1.0
    return [value / norm for value in row]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))

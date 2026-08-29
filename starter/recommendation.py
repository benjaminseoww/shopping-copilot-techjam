from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .models import ScoredProduct, SessionState


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "some",
    "that",
    "the",
    "this",
    "to",
    "want",
    "with",
    "would",
    "you",
    "looking",
    "exploring",
    "key",
    "requirement",
    "actually",
    "ignore",
    "earlier",
    "preference",
    "need",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _snippet(product: dict) -> str:
    """Compact catalog text for question-time attribute extraction."""
    categories = product.get("categories") or []
    if isinstance(categories, list) and categories:
        root = str(categories[0]).lower()
        if root in {"clothing, shoes & jewelry", "clothing shoes & jewelry"}:
            categories = categories[1:]
    return " ".join(
        [
            _text(product.get("title")),
            _text(categories),
            _text(product.get("features")),
            _text(product.get("details")),
            _text(product.get("store")),
        ]
    )[:4000]


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class RecommendationEngine:
    """Shared catalog index and stateful BM25 retrieval."""

    MAX_QUERY_TERMS = 80

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.catalog_ids: set[str] = set()
        self._snippets: dict[str, str] = {}
        self._fallback_ids: list[str] = []
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        batch: list[tuple[str, str, str, str, str, str, str]] = []
        fallback: list[tuple[float, float, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                parent_asin = str(product["parent_asin"])
                self.catalog_ids.add(parent_asin)
                self._snippets[parent_asin] = _snippet(product)
                rating_count = product.get("rating_number")
                average_rating = product.get("average_rating")
                fallback.append(
                    (
                        float(rating_count) if isinstance(rating_count, (int, float)) else 0.0,
                        float(average_rating) if isinstance(average_rating, (int, float)) else 0.0,
                        parent_asin,
                    )
                )
                batch.append(
                    (
                        parent_asin,
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
                        batch,
                    )
                    batch.clear()

        if batch:
            cursor.executemany(
                "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
        self.connection.commit()
        self._fallback_ids = [
            parent_asin
            for _, _, parent_asin in sorted(
                fallback,
                key=lambda item: (-item[0], -item[1], item[2]),
            )
        ]

    def recommend(self, state: SessionState, top_k: int = 10) -> list[ScoredProduct]:
        if top_k <= 0:
            return []

        query_text = " ".join(
            [
                state.category or "",
                *(constraint.text for constraint in state.active_constraints),
            ]
        )
        ranked_ids = self._search(query_text, top_k)

        if len(ranked_ids) < top_k and state.category:
            self._extend_unique(ranked_ids, self._search(state.category, top_k), top_k)
        if len(ranked_ids) < top_k:
            self._extend_unique(ranked_ids, self._fallback_ids, top_k)

        return [
            ScoredProduct(
                parent_asin=parent_asin,
                score=float(top_k - rank),
                text=self._snippets.get(parent_asin, ""),
            )
            for rank, parent_asin in enumerate(ranked_ids[:top_k])
        ]

    def _search(self, text: str, limit: int) -> list[str]:
        unique_terms = list(dict.fromkeys(_terms(text)))[: self.MAX_QUERY_TERMS]
        if not unique_terms:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    @staticmethod
    def _extend_unique(target: list[str], values: list[str], limit: int) -> None:
        seen = set(target)
        for value in values:
            if value in seen:
                continue
            target.append(value)
            seen.add(value)
            if len(target) >= limit:
                break

    # TODO(retrieval): Over-fetch and rerank by active-constraint coverage and
    # exact title/category phrase matches.
    # TODO(retrieval): Evaluate reciprocal-rank fusion across category,
    # feature, title, and brand query variants.
    # TODO(retrieval): Add structured attribute boosts and use profile tags
    # only as weak reranking priors, never hard filters.
    # TODO(retrieval): Evaluate optional local embedding reranking only after
    # lexical improvements plateau; retain this offline FTS fallback.

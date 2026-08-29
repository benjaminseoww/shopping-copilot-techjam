from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .models import Constraint, QUESTION_POOL_SIZE, RankedResults, ScoredProduct, SessionState


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

SOURCE_WEIGHTS = {
    "initial": 1.2,
    "override": 1.2,
    "initial_provisional": 1.1,
    "clarification": 1.0,
    "fallback": 0.85,
}

PHRASE_FIELD_WEIGHTS = (
    ("title", 4.0),
    ("features", 3.2),
    ("details", 3.2),
    ("categories", 2.4),
    ("store", 2.2),
    ("description", 1.2),
)


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


@dataclass(frozen=True)
class ProductRecord:
    parent_asin: str
    title: str
    categories: str
    features: str
    details: str
    store: str
    description: str
    blob: str
    terms: frozenset[str]
    rating_number: float
    average_rating: float


class RecommendationEngine:
    """Shared catalog index with retrieve-then-rerank over stored attributes."""

    MAX_QUERY_TERMS = 80
    RETRIEVE_K = 200
    QUESTION_POOL_SIZE = QUESTION_POOL_SIZE
    TIEBREAK = 0.05

    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.catalog_ids: set[str] = set()
        self._products: dict[str, ProductRecord] = {}
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
                title = _text(product.get("title"))
                categories = _text(product.get("categories"))
                features = _text(product.get("features"))
                details = _text(product.get("details"))
                store = _text(product.get("store"))
                description = _text(product.get("description"))
                blob = " ".join(
                    [title, categories, features, details, store, description]
                ).lower()
                rating_count = product.get("rating_number")
                average_rating = product.get("average_rating")
                rating_number = (
                    float(rating_count) if isinstance(rating_count, (int, float)) else 0.0
                )
                avg = (
                    float(average_rating)
                    if isinstance(average_rating, (int, float))
                    else 0.0
                )
                self.catalog_ids.add(parent_asin)
                self._products[parent_asin] = ProductRecord(
                    parent_asin=parent_asin,
                    title=title,
                    categories=categories,
                    features=features,
                    details=details,
                    store=store,
                    description=description,
                    blob=blob,
                    terms=frozenset(_terms(blob)),
                    rating_number=rating_number,
                    average_rating=avg,
                )
                fallback.append((rating_number, avg, parent_asin))
                batch.append(
                    (
                        parent_asin,
                        title,
                        categories,
                        features,
                        details,
                        store,
                        description,
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

    def recommend(self, state: SessionState, top_k: int = 10) -> RankedResults:
        fill_to = max(self.RETRIEVE_K, self.QUESTION_POOL_SIZE, top_k, 0)
        query_text = " ".join(
            [
                state.category or "",
                *(constraint.text for constraint in state.active_constraints),
            ]
        )
        retrieved = self._search(query_text, fill_to)
        if len(retrieved) < fill_to and state.category:
            self._extend_unique(retrieved, self._search(state.category, fill_to), fill_to)
        if len(retrieved) < fill_to:
            self._extend_unique(retrieved, self._fallback_ids, fill_to)

        scored: list[ScoredProduct] = []
        for rank, parent_asin in enumerate(retrieved):
            record = self._products.get(parent_asin)
            if record is None:
                continue
            scored.append(
                ScoredProduct(
                    parent_asin=parent_asin,
                    score=self._score_record(state, record, rank),
                )
            )
        scored.sort(key=lambda item: (-item.score, item.parent_asin))
        return RankedResults(items=tuple(scored))

    def _search(self, text: str, limit: int) -> list[str]:
        unique_terms = list(dict.fromkeys(_terms(text)))[: self.MAX_QUERY_TERMS]
        if not unique_terms or limit <= 0:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
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

    def _score_record(
        self,
        state: SessionState,
        record: ProductRecord,
        retrieve_rank: int,
    ) -> float:
        score = 0.0
        if state.category:
            score += self._text_similarity(state.category, record, category=True)
        for constraint in state.active_constraints:
            score += self._constraint_similarity(constraint, record)
        score += self.TIEBREAK / (1.0 + retrieve_rank)
        if not state.category and not state.active_constraints:
            score += 0.001 * record.rating_number + 0.0001 * record.average_rating
        return score

    def _constraint_similarity(
        self,
        constraint: Constraint,
        record: ProductRecord,
    ) -> float:
        source_weight = SOURCE_WEIGHTS.get(constraint.source, 1.0)
        phrase = self._phrase_score(constraint.text, record)
        coverage = self._term_coverage(constraint.text, record)
        typed = self._typed_bonus(constraint, record)
        similarity = phrase + 1.6 * coverage + typed
        return source_weight * similarity

    def _text_similarity(
        self,
        text: str,
        record: ProductRecord,
        category: bool = False,
    ) -> float:
        phrase = self._phrase_score(text, record)
        coverage = self._term_coverage(text, record)
        if category and text.lower().strip() in record.categories.lower():
            return 3.0 + phrase + coverage
        return phrase + 1.2 * coverage

    def _phrase_score(self, text: str, record: ProductRecord) -> float:
        needle = re.sub(r"\s+", " ", text).strip().lower()
        if len(needle) < 2:
            return 0.0
        fields = {
            "title": record.title,
            "features": record.features,
            "details": record.details,
            "categories": record.categories,
            "store": record.store,
            "description": record.description,
        }
        best = 0.0
        for field_name, weight in PHRASE_FIELD_WEIGHTS:
            haystack = fields[field_name].lower()
            if needle in haystack:
                best = max(best, weight)
        return best

    def _term_coverage(self, text: str, record: ProductRecord) -> float:
        terms = _terms(text)
        if not terms:
            return 0.0
        hits = sum(1 for term in terms if term in record.terms)
        return hits / len(terms)

    @staticmethod
    def _typed_bonus(constraint: Constraint, record: ProductRecord) -> float:
        terms = _terms(constraint.text)
        if not terms:
            return 0.0
        store_terms = set(_terms(record.store))
        title_terms = set(_terms(record.title))
        if constraint.attribute == "brand" and any(term in store_terms for term in terms):
            return 1.5
        if constraint.attribute == "color" and any(term in title_terms for term in terms):
            return 0.4
        if constraint.attribute == "material" and any(term in record.terms for term in terms):
            return 0.35
        return 0.0

    # TODO(retrieval): Evaluate reciprocal-rank fusion across category,
    # feature, title, and brand query variants.
    # TODO(retrieval): Add structured attribute boosts and use profile tags
    # only as weak reranking priors, never hard filters.
    # TODO(retrieval): Evaluate optional local embedding reranking only after
    # lexical improvements plateau; retain this offline FTS fallback.

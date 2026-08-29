from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .embedder import MiniLmEmbedder, default_model_dir, try_load_minilm
from .models import Constraint, ScoredProduct, SessionState

_AUTO_EMBEDDER = object()


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
BUDGET_RE = re.compile(r"(?:\$|under|around|about|budget)?\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
ROOT_FRAGMENT_RE = re.compile(
    r"clothing,\s*shoes\s*&\s*jewelry|clothing\s+shoes\s*&\s*jewelry",
    re.I,
)
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


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _price(value: object) -> float | None:
    if isinstance(value, bool) or value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
    if match:
        return float(match.group())
    return None


def _strip_root_fragments(text: str) -> str:
    return re.sub(r"\s+", " ", ROOT_FRAGMENT_RE.sub(" ", text)).strip()


@dataclass(frozen=True)
class ProductRecord:
    parent_asin: str
    title: str
    categories: str
    features: str
    details: str
    store: str
    description: str
    terms: frozenset[str]
    rating_number: float
    average_rating: float
    price: float | None


class RecommendationEngine:
    """Retrieve a candidate pool with FTS, then rerank with lexical scores plus MiniLM cosine."""

    MAX_QUERY_TERMS = 80
    RETRIEVE_K = 200
    RRF_K = 60
    MAX_CONSTRAINT_ROUTES = 8
    TIEBREAK = 0.05
    PHRASE_TITLE = 3.0
    PHRASE_OTHER = 1.8
    COVERAGE_WEIGHT = 1.2
    STORE_BONUS = 1.6
    CATALOG_TEXT_LIMIT = 4000
    EMBED_TEXT_LIMIT = 400
    EMBED_WEIGHT = 2.5
    EMBED_MODEL_ID = "all-MiniLM-L6-v2"

    def __init__(
        self,
        catalog_path: str | Path,
        embedder: object | None = _AUTO_EMBEDDER,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self.catalog_ids: set[str] = set()
        self._products: dict[str, ProductRecord] = {}
        self._fallback_ids: list[str] = []
        self._store_names: frozenset[str] = frozenset()
        if embedder is _AUTO_EMBEDDER:
            self._embedder = try_load_minilm(default_model_dir(self.catalog_path))
        else:
            self._embedder = embedder
        self._persist_embeddings = isinstance(self._embedder, MiniLmEmbedder)
        self._embed_matrix = None
        self._embed_index: dict[str, int] = {}
        self._build_index()
        self._prepare_embeddings()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )

        batch: list[tuple[str, str, str, str, str, str, str]] = []
        fallback: list[tuple[float, float, str]] = []
        stores: set[str] = set()
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
                field_text = " ".join(
                    [title, categories, features, details, store, description]
                )
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
                if store.strip():
                    stores.add(store.strip().lower())
                self._products[parent_asin] = ProductRecord(
                    parent_asin=parent_asin,
                    title=title,
                    categories=categories,
                    features=features,
                    details=details,
                    store=store,
                    description=description,
                    terms=frozenset(_terms(field_text)),
                    rating_number=rating_number,
                    average_rating=avg,
                    price=_price(product.get("price")),
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
        self._store_names = frozenset(stores)
        self._fallback_ids = [
            parent_asin
            for _, _, parent_asin in sorted(
                fallback,
                key=lambda item: (-item[0], -item[1], item[2]),
            )
        ]

    def recommend(self, state: SessionState, pool_k: int = 10) -> list[ScoredProduct]:
        fill_to = max(self.RETRIEVE_K, pool_k, 0)
        retrieved = self._retrieve(state, fill_to)
        if len(retrieved) < fill_to:
            self._extend_unique(retrieved, self._fallback_ids, fill_to)

        similarities = self._candidate_similarities(state, retrieved)
        scored: list[ScoredProduct] = []
        for rank, parent_asin in enumerate(retrieved):
            record = self._products.get(parent_asin)
            if record is None:
                continue
            score = self._score_record(state, record, rank)
            score += self.EMBED_WEIGHT * similarities.get(parent_asin, 0.0)
            scored.append(ScoredProduct(parent_asin=parent_asin, score=score))
        scored.sort(key=lambda item: (-item.score, item.parent_asin))
        return scored

    def catalog_text(self, parent_asin: str) -> str:
        record = self._products.get(parent_asin)
        if record is None:
            return ""
        return " ".join(
            [
                record.title,
                _strip_root_fragments(record.categories),
                record.features,
                record.details,
                record.store,
            ]
        )[: self.CATALOG_TEXT_LIMIT]

    def _retrieve(self, state: SessionState, fill_to: int) -> list[str]:
        routes: list[list[str]] = []
        combined = " ".join(
            [
                state.category or "",
                *(constraint.text for constraint in state.active_constraints),
            ]
        )
        routes.append(self._search(combined, fill_to))
        if state.category:
            routes.append(self._search(state.category, fill_to))
        for constraint in state.active_constraints[: self.MAX_CONSTRAINT_ROUTES]:
            routes.append(self._search(constraint.text, fill_to))
            if len(_terms(constraint.text)) >= 2:
                routes.append(self._search_phrase(constraint.text, fill_to))
            if self._matches_store_name(constraint.text):
                routes.append(self._search_field("store", constraint.text, fill_to))
        fused = self._rrf(routes, fill_to)
        if len(fused) < fill_to and state.category:
            self._extend_unique(fused, self._search(state.category, fill_to), fill_to)
        return fused

    def _search(self, text: str, limit: int) -> list[str]:
        unique_terms = list(dict.fromkeys(_terms(text)))[: self.MAX_QUERY_TERMS]
        if not unique_terms or limit <= 0:
            return []
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        return self._match(expression, limit)

    def _search_phrase(self, text: str, limit: int) -> list[str]:
        terms = _terms(text)
        if len(terms) < 2 or limit <= 0:
            return []
        return self._match('"' + " ".join(terms) + '"', limit)

    def _search_field(self, field: str, text: str, limit: int) -> list[str]:
        unique_terms = list(dict.fromkeys(_terms(text)))[: self.MAX_QUERY_TERMS]
        if not unique_terms or limit <= 0:
            return []
        expression = " OR ".join(f'{field}:"{term}"' for term in unique_terms)
        return self._match(expression, limit)

    def _match(self, expression: str, limit: int) -> list[str]:
        try:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [str(row[0]) for row in rows]

    def _rrf(self, routes: list[list[str]], limit: int) -> list[str]:
        scores: dict[str, float] = {}
        for ranked in routes:
            for rank, parent_asin in enumerate(ranked, start=1):
                scores[parent_asin] = scores.get(parent_asin, 0.0) + 1.0 / (
                    self.RRF_K + rank
                )
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [parent_asin for parent_asin, _ in ordered[:limit]]

    def _matches_store_name(self, text: str) -> bool:
        needle = re.sub(r"\s+", " ", text).strip().lower()
        if len(needle) < 3:
            return False
        return any(needle in store or store in needle for store in self._store_names)

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
            score += self._lexical_score(state.category, record)
        for constraint in state.active_constraints:
            score += self._constraint_score(constraint, record)
        score += self._profile_prior(state, record)
        score += self.TIEBREAK / (1.0 + retrieve_rank)
        if not state.category and not state.active_constraints:
            score += 0.001 * record.rating_number + 0.0001 * record.average_rating
        return score

    def _constraint_score(self, constraint: Constraint, record: ProductRecord) -> float:
        return (
            self._lexical_score(constraint.text, record)
            + self._store_bonus(constraint.text, record)
            + self._budget_bonus(constraint, record)
        )

    def _lexical_score(self, text: str, record: ProductRecord) -> float:
        return self._phrase_score(text, record) + self.COVERAGE_WEIGHT * self._term_coverage(
            text, record
        )

    def _phrase_score(self, text: str, record: ProductRecord) -> float:
        needle = re.sub(r"\s+", " ", text).strip().lower()
        if len(needle) < 2:
            return 0.0
        if needle in record.title.lower():
            return self.PHRASE_TITLE
        haystack = " ".join(
            (
                record.features,
                record.details,
                record.categories,
                record.store,
                record.description,
            )
        ).lower()
        if needle in haystack:
            return self.PHRASE_OTHER
        return 0.0

    def _term_coverage(self, text: str, record: ProductRecord) -> float:
        terms = _terms(text)
        if not terms:
            return 0.0
        hits = sum(1 for term in terms if term in record.terms)
        return hits / len(terms)

    def _store_bonus(self, text: str, record: ProductRecord) -> float:
        store = record.store.strip()
        if not store:
            return 0.0
        needle = re.sub(r"\s+", " ", text).strip().lower()
        haystack = store.lower()
        if len(needle) >= 2 and (needle in haystack or haystack in needle):
            return self.STORE_BONUS
        query_terms = set(_terms(text))
        store_terms = set(_terms(store))
        if query_terms and query_terms & store_terms:
            return self.STORE_BONUS
        return 0.0

    @staticmethod
    def _budget_bonus(constraint: Constraint, record: ProductRecord) -> float:
        looks_like_budget = constraint.attribute == "budget" or bool(
            re.search(r"(?:budget|\$|under|around)\s*\$?\s*\d", constraint.text, re.I)
        )
        if not looks_like_budget or record.price is None:
            return 0.0
        match = BUDGET_RE.search(constraint.text)
        if not match:
            return 0.0
        amount = float(match.group(1))
        if amount <= 0:
            return 0.0
        distance = abs(record.price - amount) / amount
        return max(0.0, 1.1 - distance)

    @staticmethod
    def _profile_prior(state: SessionState, record: ProductRecord) -> float:
        tags = state.profile.preference_tags
        if not tags:
            return 0.0
        field_text = " ".join(
            (
                record.title,
                record.categories,
                record.features,
                record.details,
                record.store,
                record.description,
            )
        ).lower()
        hits = 0
        for tag in tags:
            token = tag.lower().strip()
            if len(token) < 2:
                continue
            if token in record.terms or token in field_text:
                hits += 1
        if not hits:
            return 0.0
        scale = 0.15 if len(state.active_constraints) <= 1 else 0.04
        return scale * hits

    def _prepare_embeddings(self) -> None:
        self._embed_matrix = None
        self._embed_index = {}
        if self._embedder is None or not self._products:
            return
        asins = list(self._products)
        cache_path = self._embedding_cache_path()
        if self._persist_embeddings and self._load_embedding_cache(cache_path, asins):
            return
        texts = [self._product_embed_text(self._products[asin]) for asin in asins]
        try:
            matrix = _as_normalized_matrix(self._embedder.encode(texts))
        except Exception:
            return
        self._embed_matrix = matrix
        self._embed_index = {asin: index for index, asin in enumerate(asins)}
        if self._persist_embeddings:
            self._save_embedding_cache(cache_path, asins)

    def _embedding_cache_path(self) -> Path:
        return self.catalog_path.with_suffix(".minilm.npz")

    def _load_embedding_cache(self, cache_path: Path, asins: list[str]) -> bool:
        if not cache_path.is_file() or not self.catalog_path.is_file():
            return False
        try:
            import numpy as np

            payload = np.load(cache_path, allow_pickle=False)
            stat = self.catalog_path.stat()
            if int(payload["catalog_size"]) != stat.st_size:
                return False
            if abs(float(payload["catalog_mtime"]) - stat.st_mtime) > 1e-6:
                return False
            if str(payload["model_id"]) != self.EMBED_MODEL_ID:
                return False
            cached_asins = [str(item) for item in payload["asins"].tolist()]
            if cached_asins != asins:
                return False
            matrix = np.asarray(payload["vectors"], dtype=np.float32)
            if matrix.shape[0] != len(asins):
                return False
            self._embed_matrix = matrix
            self._embed_index = {asin: index for index, asin in enumerate(asins)}
            return True
        except Exception:
            return False

    def _save_embedding_cache(self, cache_path: Path, asins: list[str]) -> None:
        if self._embed_matrix is None:
            return
        try:
            import numpy as np

            stat = self.catalog_path.stat()
            np.savez_compressed(
                cache_path,
                asins=np.array(asins),
                vectors=self._embed_matrix,
                catalog_size=np.int64(stat.st_size),
                catalog_mtime=np.float64(stat.st_mtime),
                model_id=np.array(self.EMBED_MODEL_ID),
            )
        except Exception:
            return

    def _candidate_similarities(
        self,
        state: SessionState,
        retrieved: list[str],
    ) -> dict[str, float]:
        if self._embed_matrix is None or not retrieved:
            return {}
        query = self._query_embed_text(state)
        if not query:
            return {}
        try:
            query_vec = _as_normalized_matrix(self._embedder.encode([query]))[0]
        except Exception:
            return {}
        rows = [self._embed_index[asin] for asin in retrieved if asin in self._embed_index]
        if not rows:
            return {}
        scores = _row_dots(self._embed_matrix, rows, query_vec)
        ranked = [asin for asin in retrieved if asin in self._embed_index]
        return {asin: float(score) for asin, score in zip(ranked, scores)}

    def _query_embed_text(self, state: SessionState) -> str:
        return re.sub(
            r"\s+",
            " ",
            " ".join(
                [
                    state.category or "",
                    *(constraint.text for constraint in state.active_constraints),
                ]
            ),
        ).strip()[: self.EMBED_TEXT_LIMIT]

    def _product_embed_text(self, record: ProductRecord) -> str:
        return re.sub(
            r"\s+",
            " ",
            " ".join(
                [
                    record.title,
                    _strip_root_fragments(record.categories),
                    record.features,
                    record.details,
                    record.store,
                ]
            ),
        ).strip()[: self.EMBED_TEXT_LIMIT]


def _as_normalized_matrix(vectors: object):
    try:
        import numpy as np
    except ImportError:
        rows = [list(map(float, row)) for row in vectors]  # type: ignore[arg-type]
        return [_normalize_row(row) for row in rows]

    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


def _normalize_row(row: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in row)) or 1.0
    return [value / norm for value in row]


def _row_dots(matrix: object, row_indices: list[int], query: object) -> list[float]:
    try:
        scores = matrix[row_indices] @ query  # type: ignore[index, operator]
        return [float(score) for score in scores]
    except TypeError:
        query_row = list(query)  # type: ignore[arg-type]
        return [
            sum(left * right for left, right in zip(matrix[index], query_row))  # type: ignore[index]
            for index in row_indices
        ]

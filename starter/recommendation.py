from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .attributes import MATERIALS, first_color, first_material, strip_constraint_label
from .embedder import MiniLmEmbedder, default_model_dir, try_load_minilm
from .models import Constraint, ScoredProduct, SessionState

_AUTO_EMBEDDER = object()


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
BUDGET_RE = re.compile(r"(?:\$|under|around|about|budget)?\s*\$?\s*(\d+(?:\.\d+)?)", re.I)
COMPOSITION_RE = re.compile(
    rf"(\d+(?:\.\d+)?)\s*%\s*({'|'.join(re.escape(value) for value in MATERIALS)})",
    re.IGNORECASE,
)
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
    material: str | None
    color: str | None


class RecommendationEngine:
    """Retrieve a candidate pool with FTS, then rerank with lexical scores plus MiniLM cosine."""

    MAX_QUERY_TERMS = 80
    RETRIEVE_K = 200
    RRF_K = 60
    MAX_CONSTRAINT_ROUTES = 8
    TIEBREAK = 0.05
    PHRASE_TITLE = 4.0
    PHRASE_OTHER = 3.2
    PHRASE_TERM_BONUS = 0.35
    COVERAGE_WEIGHT = 1.2
    STORE_BONUS = 1.6
    TYPED_MATCH = 2.8
    TYPED_MISMATCH = -2.0
    SUPERSEDED_SCALE = 0.45
    LEAF_CATEGORY = 2.2
    PATH_CATEGORY = 0.9
    COMPOSITION_WEIGHT = 1.4
    WEAK_LEAVES = frozenset(
        {"men", "women", "boys", "girls", "kids", "baby", "unisex", "clothing"}
    )
    CATALOG_TEXT_LIMIT = 4000
    EMBED_TEXT_LIMIT = 400
    EMBED_WEIGHT = 1.0
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
        self._idf: dict[str, float] = {}
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
                    material=first_material(field_text),
                    color=first_color(field_text),
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
        catalog_size = max(len(self._products), 1)
        document_frequency: dict[str, int] = {}
        for record in self._products.values():
            for term in record.terms:
                document_frequency[term] = document_frequency.get(term, 0) + 1
        self._idf = {
            term: math.log((catalog_size + 1) / (count + 1)) + 1.0
            for term, count in document_frequency.items()
        }

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
                *(self._constraint_query_text(constraint) for constraint in state.active_constraints),
            ]
        )
        routes.append(self._search(combined, fill_to))
        if state.category:
            routes.append(self._search(state.category, fill_to))
        for constraint in state.active_constraints[: self.MAX_CONSTRAINT_ROUTES]:
            query_text = self._constraint_query_text(constraint)
            routes.append(self._search(query_text, fill_to))
            if len(_terms(query_text)) >= 2:
                routes.append(self._search_phrase(query_text, fill_to))
            if self._matches_store_name(query_text):
                routes.append(self._search_field("store", query_text, fill_to))
        precise: list[str] = []
        for expression in self._precision_and_queries(state):
            hits = self._match(expression, fill_to)
            routes.append(hits)
            self._extend_unique(precise, hits, fill_to)
        fused = self._rrf(routes, fill_to)
        ordered: list[str] = []
        self._extend_unique(ordered, precise, fill_to)
        self._extend_unique(ordered, fused, fill_to)
        if len(ordered) < fill_to and state.category:
            self._extend_unique(ordered, self._search(state.category, fill_to), fill_to)
        return ordered

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

    def _precision_and_queries(self, state: SessionState) -> list[str]:
        leaves = self._specific_category_tokens(state.category or "")
        if not leaves:
            return []
        needles: list[str] = []
        for constraint in state.active_constraints[:4]:
            text = self._constraint_query_text(constraint)
            for value in (first_material(text), first_color(text)):
                if value:
                    needles.append(value)
            terms = _terms(text)
            if terms:
                needles.append(terms[0])
        queries: list[str] = []
        seen: set[str] = set()
        for token in reversed(leaves):
            for needle in needles:
                if needle == token:
                    continue
                key = (token, needle)
                if key in seen:
                    continue
                seen.add(key)
                queries.append(f'("{token}" AND "{needle}")')
                if len(queries) >= 6:
                    return queries
        return queries

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
            score += self._leaf_category_bonus(state.category, record)
        for constraint in state.active_constraints:
            score += self._constraint_score(constraint, record)
        score += self._superseded_score(state, record)
        score += self._profile_prior(state, record)
        score += self.TIEBREAK / (1.0 + retrieve_rank)
        if not state.category and not state.active_constraints:
            score += 0.001 * record.rating_number + 0.0001 * record.average_rating
        return score

    def _superseded_score(self, state: SessionState, record: ProductRecord) -> float:
        if not state.superseded_constraints:
            return 0.0
        replaced = {
            constraint.attribute
            for constraint in state.active_constraints
            if constraint.attribute in {"material", "color", "size", "style", "brand", "budget"}
        }
        active_colors = {
            color
            for constraint in state.active_constraints
            if (color := first_color(self._constraint_query_text(constraint)))
        }
        active_materials = {
            material
            for constraint in state.active_constraints
            if (material := first_material(self._constraint_query_text(constraint)))
        }
        total = 0.0
        for constraint in state.superseded_constraints:
            if constraint.attribute in replaced:
                continue
            query_text = self._constraint_query_text(constraint)
            old_color = first_color(query_text)
            old_material = first_material(query_text)
            if old_color and active_colors and old_color not in active_colors:
                continue
            if old_material and active_materials and old_material not in active_materials:
                continue
            total += self.SUPERSEDED_SCALE * self._constraint_score(constraint, record)
        return total

    def _leaf_category_bonus(self, category: str, record: ProductRecord) -> float:
        tokens = self._specific_category_tokens(category)
        if not tokens:
            return 0.0
        title_terms = set(_terms(record.title))
        category_terms = set(_terms(record.categories))
        leaf = tokens[-1]
        score = 0.0
        for token in tokens:
            variants = self._token_variants(token)
            if not (variants & title_terms or variants & category_terms):
                continue
            weight = self._idf.get(token, 1.0)
            base = self.LEAF_CATEGORY if token == leaf else self.PATH_CATEGORY
            score += base * min(weight, 4.0) / 2.0
        return score

    @classmethod
    def _specific_category_tokens(cls, category: str) -> list[str]:
        tokens: list[str] = []
        for term in _terms(category):
            if len(term) < 4 or term in cls.WEAK_LEAVES or term in tokens:
                continue
            tokens.append(term)
        return tokens

    @staticmethod
    def _token_variants(token: str) -> set[str]:
        variants = {token}
        if token.endswith("s") and len(token) > 4:
            variants.add(token[:-1])
        else:
            variants.add(token + "s")
        return variants

    def _composition_bonus(self, constraint: Constraint, record: ProductRecord) -> float:
        wanted = first_material(self._constraint_query_text(constraint))
        if not wanted:
            return 0.0
        haystack = " ".join(
            (record.title, record.features, record.details, record.description)
        )
        best = 0.0
        for match in COMPOSITION_RE.finditer(haystack):
            if match.group(2).lower() != wanted:
                continue
            percent = min(float(match.group(1)), 100.0) / 100.0
            best = max(best, self.COMPOSITION_WEIGHT * percent)
        return best

    def _constraint_score(self, constraint: Constraint, record: ProductRecord) -> float:
        query_text = self._constraint_query_text(constraint)
        return (
            self._lexical_score(query_text, record)
            + self._store_bonus(query_text, record)
            + self._budget_bonus(constraint, record)
            + self._typed_bonus(constraint, record)
            + self._composition_bonus(constraint, record)
        )

    @staticmethod
    def _constraint_query_text(constraint: Constraint) -> str:
        return strip_constraint_label(constraint.text) or constraint.text

    def _lexical_score(self, text: str, record: ProductRecord) -> float:
        return self._phrase_score(text, record) + self.COVERAGE_WEIGHT * self._term_coverage(
            text, record
        )

    def _phrase_score(self, text: str, record: ProductRecord) -> float:
        needle = re.sub(r"\s+", " ", strip_constraint_label(text)).strip().lower()
        if len(needle) < 2:
            return 0.0
        extra = self.PHRASE_TERM_BONUS * max(0, len(_terms(needle)) - 1)
        if needle in record.title.lower():
            return self.PHRASE_TITLE + extra
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
            return self.PHRASE_OTHER + extra
        return 0.0

    def _term_coverage(self, text: str, record: ProductRecord) -> float:
        terms = _terms(strip_constraint_label(text) or text)
        if not terms:
            return 0.0
        weights = [self._idf.get(term, 1.0) for term in terms]
        denom = sum(weights) or 1.0
        hits = sum(
            weight for term, weight in zip(terms, weights) if term in record.terms
        )
        return hits / denom

    def _typed_bonus(self, constraint: Constraint, record: ProductRecord) -> float:
        query_text = self._constraint_query_text(constraint)
        bonus = 0.0
        wanted_color = first_color(query_text)
        if wanted_color and (
            constraint.attribute == "color" or wanted_color in _terms(query_text)
        ):
            bonus += self._typed_presence(wanted_color, record.color, record)
        wanted_material = first_material(query_text)
        if wanted_material and (
            constraint.attribute == "material" or wanted_material in _terms(query_text)
        ):
            bonus += self._typed_presence(wanted_material, record.material, record)
        return bonus

    def _typed_presence(
        self,
        wanted: str,
        extracted: str | None,
        record: ProductRecord,
    ) -> float:
        aliases = {wanted}
        if wanted == "gray":
            aliases.add("grey")
        elif wanted == "grey":
            aliases.add("gray")
        if extracted in aliases or record.terms & aliases:
            return self.TYPED_MATCH
        if extracted:
            return self.TYPED_MISMATCH
        return 0.0

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
                    *(
                        self._constraint_query_text(constraint)
                        for constraint in state.active_constraints
                    ),
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

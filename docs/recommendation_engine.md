# Recommendation Engine

## Purpose

`RecommendationEngine` converts accumulated session intent into an ordered list of catalog `parent_asin` values. It owns retrieval and ranking, while language parsing, memory mutation, question selection, and response wording remain separate.

It must use only the participant-visible catalog and current session state, run offline, and never inspect ground truth or hidden evaluator fields.

## MVP

Build one in-memory SQLite FTS5 catalog index when the Agent is initialized and reuse it across all sessions and turns.

Index the existing searchable fields and initially retain the starter BM25 weights:

- `title`: 6.0
- `categories`: 4.0
- `features`: 2.5
- `details`: 2.5
- `store`: 1.5
- `description`: 1.0

The important MVP change is stateful query construction:

- Search with the active category and every accumulated active constraint.
- Do not search only the latest message.
- Exclude superseded constraints, no-preference replies, and simulator boilerplate.
- Return unique catalog-valid results in rank order, up to `top_k`.
- If the primary query is empty or has no matches, retry a reduced category-only query and then use a deterministic valid-catalog fallback.
- Produce a Top 10 on every turn, including turn 1 and turn 10.

## Proposed interface

The engine is constructed with the catalog path and builds or acquires the shared FTS5 index. It also retains the catalog identifiers required for output validation.

Its recommendation operation receives:

- current `SessionState`; and
- requested `top_k`.

It returns ordered product records containing at least a valid `parent_asin`. Internal scores may support diagnostics, but the evaluator ignores them.

The Agent converts these records into response objects. The Recommendation Engine does not choose `ask_attribute`, generate message text, or report model usage.

## Query construction

1. Add meaningful category terms.
2. Add terms from every active raw constraint.
3. Normalize case, remove known boilerplate and stopwords, and deduplicate terms.
4. Escape FTS syntax and use bound SQL parameters.
5. Use a recall-oriented OR expression for the first MVP.
6. Apply a conservative term cap.

Raw constraint text is important because evaluator disclosures are derived from target catalog metadata and often contain strong lexical matches.

Override handling is state-driven. Once memory marks earlier preferences superseded, rebuild the query only from the retained category and active replacement constraints. Never concatenate the complete transcript.

## Future-code comments

The implementation should identify these extension points without implementing them in the MVP:

- `TODO(retrieval): Over-fetch candidates and rerank by active-constraint coverage and exact title/category phrase matches.`
- `TODO(retrieval): Evaluate reciprocal-rank fusion across category, feature, title, and brand query variants.`
- `TODO(retrieval): Add structured material, color, size, and budget boosts; avoid hard budget filters while prices are sparse.`
- `TODO(retrieval): Use profile tags only as weak reranking priors, never as hard filters.`
- `TODO(retrieval): Evaluate optional local embedding reranking only after lexical improvements plateau; retain the offline FTS fallback.`

## Trade-offs

- FTS5 is deterministic, offline, inexpensive per turn, and already available through Python's standard library, but it has limited semantic matching.
- Accumulated state preserves useful disclosures across turns, but depends on correct parsing and override handling.
- A recall-oriented OR query protects HitRate but common terms can reduce ranking precision and MRR.
- Retaining the existing weights gives a reproducible baseline before tuning, though the weights may not be optimal.
- One shared index pays startup cost once and prevents per-session catalog duplication.
- A simple fallback prevents empty responses but may return weakly relevant products.

The first priority is collecting and retaining useful constraints. More complex reranking should be added only after measuring this stateful lexical baseline.

## Scoring and resource contribution

- **HitRate@10:** retrieval directly determines whether the exact target appears in the scored Top 10.
- **MRR:** result ordering directly determines reciprocal rank on the first successful turn.
- **MTTC/Efficiency:** generating recommendations every turn and immediately using new constraints enables earlier hits.
- **Latency:** index construction is a one-time cost; each turn should require only local query construction and SQLite search.
- **Memory:** one index covers all 50,000 products; never create catalog copies per session.
- **Tokens:** the MVP uses no model calls and reports zero prompt and completion tokens.

## Failure cases

- Synonyms, paraphrases, spelling variants, or implicit requirements have little lexical overlap.
- Broad categories create many near-equivalent candidates.
- Common or noisy terms dominate an OR query.
- Sparse catalog fields remove useful evidence.
- Upstream parsing retains a superseded constraint or drops an active one.
- No-preference text accidentally becomes search evidence.
- Unescaped punctuation breaks the FTS expression.
- Invalid or duplicate identifiers reduce the effective Top 10.
- Catalog loading, FTS5 availability, or index construction fails.
- The fallback returns valid but irrelevant products.
- Public-set-specific weight tuning overfits the private evaluation.

## Follow-up options

1. Over-fetch BM25 candidates and rerank by active-constraint coverage and exact phrase matches.
2. Fuse multiple lexical routes with reciprocal-rank fusion.
3. Add structured material, color, size, and soft budget boosts.
4. Apply anonymized profile tags as weak priors after current-intent evidence.
5. Add optional local embedding reranking if lexical approaches plateau.

Hosted retrieval and remote vector databases should not be mandatory because official scoring may disable network access.

## Planned tests

Future tests in `tests/test_recommendation.py` should cover:

- Catalog index construction once and reuse across sessions and turns.
- Existing field weighting with controlled catalog fixtures.
- Retrieval from accumulated active constraints rather than only the newest message.
- Superseded constraints disappearing after an override.
- No-preference and boilerplate text being excluded.
- Safe handling of punctuation and FTS operators.
- Ordered, unique, catalog-valid output limited to `top_k`.
- Category-only and deterministic fallback behavior.
- Session isolation while sharing one catalog index.
- Missing catalog fields and prices.
- Safe degradation after query failure.
- Public-evaluator comparisons for overall and scenario-level metrics.
- Ablations for latest-message versus accumulated-state retrieval and later ranking stages.

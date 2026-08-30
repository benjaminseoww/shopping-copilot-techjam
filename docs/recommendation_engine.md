# Recommendation Engine

## Purpose

`RecommendationEngine` converts accumulated session intent into an ordered list of catalog `parent_asin` values. It owns retrieval and ranking. Language parsing, memory mutation, question selection, and response wording stay elsewhere.

It uses only the participant-visible catalog and current session state, runs offline, and never inspects ground truth.

## Current behavior

One in-memory SQLite FTS5 index is built at Agent startup and reused across sessions.

Field weights for BM25:

- `title`: 6.0
- `categories`: 4.0
- `features`: 2.5
- `details`: 2.5
- `store`: 1.5
- `description`: 1.0

Each turn:

1. Over-fetch about 200 candidates with reciprocal-rank fusion of FTS routes.
2. Rerank that pool with lexical, typed, profile, and optional MiniLM signals.
3. Return the ordered pool. The Agent slices the customer-facing list: full `top_k` once two constraints are known or the turn is 8+, otherwise a single best guess. Question scoring still sees the over-fetched pile.

`catalog_text(parent_asin)` is a compact snippet (title, non-root categories, features, details, store) for question-time attribute extraction.

## Retrieval

Search uses the active category plus every **active** constraint. Query text strips labels such as `color:` so the word `color` does not become a retrieval term.

Routes fused with RRF (`k=60`):

- combined category + constraints
- category-only
- each constraint as a bag-of-words query
- phrase query when a constraint has two or more terms
- store-field query when the constraint looks like a brand/store name

If the fused list is short, category search and a rating-ordered catalog fallback fill unique ids.

Punctuation is stripped before FTS. MATCH failures return no rows for that route rather than crashing.

## Ranking

Each retrieved product gets a score from:

| Signal | Role |
| --- | --- |
| Phrase match | Full constraint string in title (stronger) or other fields. Longer phrases get a small extra boost because feature sentences are identifying. |
| IDF-weighted term coverage | Fraction of constraint terms present, weighted by catalog rarity so `color` does not equal `spandex`. |
| Typed color/material | Presence of the requested value is a bonus; a different extracted value without the requested one is a penalty. Missing extractions are not penalized. |
| Store/brand | Substring or term overlap with `store`. |
| Soft budget | Distance to a parsed price when the product has a price. Missing prices are never filtered out. |
| Superseded constraints | Kept at 0.45× if they do not contradict an active typed color/material. A replacement like "leather" should not erase an earlier identifying feature. Conflicting colors/materials are ignored. |
| Profile tags | Weak prior only, never a hard filter. Stronger when little session evidence exists. |
| MiniLM cosine | Optional. Added at weight 1.0 so it cannot drown a unique lexical phrase. Missing weights fall back to lexical ranking. |
| Retrieve-rank tie-break | Small bonus for earlier FTS rank. |

Grey/gray are treated as the same color.

## Override handling

Memory still moves replaced preferences into `superseded_constraints` and retrieval queries only active text. Ranking may still use a superseded **non-conflicting** feature at reduced weight. The customer changed their mind; the product often still satisfies the earlier detail.

## What it does not do

- Hard-filter on budget, color, or material (sparse catalog fields would drop the target).
- Search the raw transcript or superseded text.
- Choose `ask_attribute`, generate `message`, or report token usage.
- Require embeddings. `SHOPPING_SKIP_EMBEDDINGS=1` or missing ONNX files keep the lexical path.

## Scoring contribution

- **HitRate@10:** retrieval must put the target in the scored Top 10.
- **MRR:** typed matches and unique phrases should move the target up on the first successful turn.
- **MTTC:** recommendations every turn, including turn 1, so a strong first constraint can convert immediately.
- **Latency / tokens:** local FTS and CPU MiniLM; zero model tokens.

## Failure cases

- Synonyms with no lexical overlap (MiniLM can help; it can also blur distinctive phrases).
- First-extracted color/material on a product is the wrong mention (lining vs shell). Presence of the requested value still counts as a match.
- Broad categories with a common material (`cotton`) leave many near-duplicates.
- Upstream parsing drops an active constraint or keeps a conflicting one.
- Profile tags applied too strongly can bury the target; they remain a weak prior.

## Tests

`tests/test_recommendation.py` covers accumulated-state retrieval, phrase vs tokens, joint coverage, over-fetch rerank, store/brand, budget without price, profile non-exclusion, catalog snippet stripping, MiniLM fallback/paraphrase, labeled `color:` queries, material mismatch, and compatible superseded features.

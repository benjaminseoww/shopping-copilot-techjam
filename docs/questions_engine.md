# Questions Engine

## Purpose

`QuestionsEngine` chooses the next clarification message and structured `ask_attribute` from the current session state and an over-fetched retrieval pile.

It does not parse intent, mutate memory, retrieve products, rank recommendations, or inspect ground truth. Recommendations must be generated independently on every turn; asking a question must never suppress the current Top 10. The Agent must over-fetch for question scoring, then slice the returned ASINs to `top_k` before handing them to the evaluator.

## Current policy

Retrieve first, then ask. The Agent requests about 80 candidates (`max(top_k, QuestionsEngine.candidate_pool)`), scores questions against that pile using `catalog_text(parent_asin)` lookups, and returns only the first `top_k` products.

On turns 1–9 the engine always asks something (`ask_attribute` is never `None`). On turn 10 it returns `ask_attribute=None` because the evaluator generates no subsequent reply. The turn-10 message is the closest-matches template.

Typed questions are limited to `material`, `color`, `style`, and `size`. The engine never asks `brand`, `budget`, `category`, or `use_case`. `use_case` is never a typed question even when candidate text looks activity-like.

## Scoring

Each eligible typed attribute is scored as:

`P(answer | family) * occupancy * diversity`

- **Family prior** `P(answer | family)` is the probability that a hidden constraint in that coarse family is classified under the attribute. Jewelry `material` prior is `0.0`, so jewelry never wins on material.
- **Occupancy** is the share of the live pile with a closed-vocab extraction for that field, read from `catalog_text(parent_asin)` (empty string if the lookup is missing). Attributes below `MIN_OCCUPANCY` (0.20) are skipped.
- **Diversity** is `1 - sum p^2` over extracted value shares. Constant piles (all cotton, all black) have diversity 0 and are not worth asking. Attributes below `MIN_DIVERSITY` (0.12) are skipped.

Only typed attributes that pass occupancy and diversity floors are scored. `_select` picks the highest typed score, with ties broken by `material`, `color`, `style`, `size`. `other` is not scored. If there is no typed winner (empty pile, constant pile, all blocked, or all below floors), the engine asks `other`. If `other` is blocked and no typed attribute qualifies, the empty-select path still returns `other`.

Answered, `no_preference`, and exhausted attributes are not re-asked. Messages always match `ask_attribute`.

## Family mapping

Coarse category strings are mapped to `watches`, `jewelry`, `shoes`, `bags`, `accessories`, `clothing`, or `other`. Root Amazon fragments (`Clothing, Shoes & Jewelry`, `Clothing Shoes & Jewelry`, `Shoes & Jewelry`) are stripped first so they do not dominate the keyword map.

If the remaining string is unrecognized or otherwise ambiguous, the engine majority-votes family labels from candidate catalog snippets. The `other` family still has a universal prior used when the pile does not resolve a more specific vertical.

Closed-vocab extraction uses the first regex hit per field on each candidate snippet (title, non-root categories, features, details, store).

## Scenario handling

### Buying

Recommend immediately using the initial hard constraint, then score the live pile. A split material or color field is asked; a constant pile falls back to `other`.

### Browsing

Recommend from the coarse category and ask the highest-scoring eligible question so target-derived constraints can surface.

### Intent override

After the replacement message arrives, questions use only active state and must not repeat or reinforce superseded preferences.

### Boundary

The first non-null question may receive a one-time no-preference response. Record it without adding search evidence and continue asking; it does not prove that no hidden constraints remain.

## Trade-offs

- Live occupancy and diversity keep questions aligned with the current retrieval pile instead of a fixed typed cycle.
- Family answerability priors reduce asking attributes the evaluator is unlikely to disclose (for example material on jewelry).
- `other` remains the disclosure fallback when the pile is empty, constant, or weakly split, which is evaluator-specific but protects turns 1–9.
- Candidate diversity is still not the same as hidden-card answerability; the prior is only a coarse correction.
- Deterministic templates report zero tokens.

## Scoring contribution

- **HitRate@10:** questions indirectly expose target-derived evidence that improves retrieval.
- **MRR:** better evidence can improve rank, though ranking remains the Recommendation Engine's responsibility.
- **MTTC/Efficiency:** asking an informative question early can lead to an earlier hit on the following turn.
- **Tokens:** deterministic templates report zero tokens. Model-generated wording must report actual usage.

Questions receive no direct score.

## Failure cases

- Returning `None` before turn 10 prevents further disclosure.
- Asking on turn 10 has no effect.
- A message names one attribute while `ask_attribute` contains another.
- Re-asking answered, declined, or exhausted typed fields wastes a turn.
- Returning the over-fetched pile to the evaluator instead of slicing to `top_k` changes scored HitRate.
- Stale state causes questions about superseded intent.
- Invalid attributes, malformed responses, or exceptions produce missed opportunities.

## Tests

`tests/test_questions.py` covers turn 10 `None`, empty-pile `other`, clothing material splits versus constant piles, jewelry skipping material and asking color, skipped answered/no-preference/exhausted fields, never asking `use_case`/`brand`/`budget`/`category`, family mapping, and closed-vocab extraction.

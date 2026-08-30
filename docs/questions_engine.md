# Questions Engine

## Purpose

`QuestionsEngine` chooses the next clarification message and structured `ask_attribute` from the current session state and an over-fetched retrieval pile.

It does not parse intent, mutate memory, retrieve products, or inspect ground truth. Recommendations are generated independently on every turn; asking never suppresses the current list. The Agent over-fetches for question scoring, then slices ASINs: full `top_k` once two constraints are known or the turn is 8+, otherwise a single best guess.

## Current policy

Retrieve first, then ask. The Agent requests about 80 candidates, scores typed questions against that pile using `catalog_text(parent_asin)`, and returns only the first `top_k` products.

On turns 1–9 the engine always asks (`ask_attribute` is never `None`). On turn 10 it returns `ask_attribute=None` because no later reply is generated. The turn-10 message is the closest-matches template.

Typed questions are limited to `material`, `color`, `style`, and `size`. The engine never asks `brand`, `budget`, `category`, or `use_case`.

**Open follow-up:** once the session has any active constraint, or the customer already declined a typed field, the next question is `other` ("What other requirement or priority matters most to you?"). A second typed attribute (often color after material) is weaker identifying evidence than an unconstrained remaining requirement. Before any evidence exists, the engine still asks the best typed split so browsing turn 1 is not an empty prompt.

If `other` is blocked (answered, declined, or exhausted), it falls back to the highest typed score.

## Scoring

Each eligible typed attribute is scored as:

`P(answer | family) * occupancy * diversity`

- **Family prior** `P(answer | family)` is the probability that a hidden constraint in that coarse family is classified under the attribute. Jewelry `material` prior is `0.0`, so jewelry never wins on material.
- **Occupancy** is the share of the live pile with a closed-vocab extraction for that field. Attributes below `MIN_OCCUPANCY` (0.20) are skipped.
- **Diversity** is `1 - sum p^2` over extracted value shares. Constant piles have diversity 0. Attributes below `MIN_DIVERSITY` (0.12) are skipped.

`_select` uses that typed ranking only when there is not yet session evidence, or when `other` is blocked. Ties break by `material`, `color`, `style`, `size`. If there is no typed winner, the engine asks `other`.

Answered, `no_preference`, and exhausted attributes are not re-asked. Messages always match `ask_attribute`.

## Family mapping

Coarse category strings map to `watches`, `jewelry`, `shoes`, `bags`, `accessories`, `clothing`, or `other`. Root Amazon fragments are stripped first.

If the remaining string is unrecognized, the engine majority-votes family labels from candidate catalog snippets.

Closed-vocab extraction uses the first regex hit per field on each candidate snippet.

## Scenario handling

### Buying

Recommend immediately using the initial hard constraint, then ask `other` so remaining feature-like requirements can surface on the next turn.

### Browsing

With no constraints yet, ask the highest-scoring typed split (often material). After the first answer or a decline, switch to `other`.

### Intent override

After the replacement message, active state has one new constraint, so the next question is `other` rather than re-asking a typed field that the pile happens to split.

### Boundary

The first non-null question may receive a one-time no-preference response. That counts as evidence that typed probing failed, so the next question is `other` instead of cycling through more typed fields.

## Trade-offs

- Live occupancy and diversity keep the first question aligned with the retrieval pile.
- Switching to an open follow-up after any evidence prefers identifying leftover requirements over a second coarse attribute.
- `other` is still a fallback when the pile is empty, constant, or weakly split.
- Candidate diversity is not the same as hidden-card answerability; the family prior is only a coarse correction.
- Deterministic templates report zero tokens.

## Scoring contribution

- **HitRate@10:** questions expose target-derived evidence that retrieval can match.
- **MRR:** better evidence can improve rank; ranking stays the Recommendation Engine's job.
- **MTTC:** an informative follow-up on the first evidence-bearing turn can convert on the next turn.
- **Tokens:** deterministic templates report zero tokens.

## Failure cases

- Returning `None` before turn 10 prevents further disclosure.
- Asking on turn 10 has no effect.
- A message names one attribute while `ask_attribute` contains another.
- Re-asking answered, declined, or exhausted fields wastes a turn.
- Returning the over-fetched pile to the evaluator instead of slicing to `top_k` changes scored HitRate.
- Stale state causes questions about superseded intent.

## Tests

`tests/test_questions.py` covers turn 10 `None`, empty-pile `other`, clothing material splits versus constant piles, jewelry skipping material, skipped answered/no-preference/exhausted fields, never asking `use_case`/`brand`/`budget`/`category`, family mapping, closed-vocab extraction, open follow-up after a known constraint, after a typed decline, and after a preference replacement.

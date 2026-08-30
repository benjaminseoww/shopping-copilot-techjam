# Questions Engine

## Purpose

`QuestionsEngine` chooses the next clarification message and structured `ask_attribute` from the current session state and an over-fetched retrieval pile.

It does not parse intent, mutate memory, retrieve products, or inspect ground truth. Recommendations are generated independently on every turn; asking never suppresses the current list. The Agent over-fetches for question scoring, then slices ASINs: full `top_k` once two constraints are known or the turn is 8+, otherwise a single best guess.

## Current policy

Retrieve first, then ask. The Agent requests about 80 candidates, scores typed questions against that pile using `catalog_text(parent_asin)`, and returns only the first `top_k` products.

On turns 1–9 the engine always asks (`ask_attribute` is never `None`). On turn 10 it returns `ask_attribute=None` because no later reply is generated. The turn-10 message is the closest-matches template.

Typed questions are limited to `material`, `color`, `style`, and `size`. The engine never asks `brand`, `budget`, `category`, or `use_case`.

**Open follow-up:** once the session has identifying evidence, the next question is `other`. Identifying means a distinctive constraint (rare catalog terms, or a longer feature line), two typed attributes (for example material and color), or a declined typed field. A lone common fiber like `cotton` is not enough; the engine keeps asking the best typed split so the pile can actually be partitioned. The provisional `I'm looking for X. {text}` opener still does not count.

If `other` is blocked (answered, declined, or exhausted), it falls back to the highest typed score.

## Scoring

Each eligible typed attribute is scored as:

`P(answer | family) * occupancy * diversity`

- **Family prior** `P(answer | family)` is the probability that a hidden constraint in that coarse family is classified under the attribute. Jewelry `material` prior is `0.0`, so jewelry never wins on material.
- **Occupancy** is the share of the live pile with a closed-vocab extraction for that field. Attributes below `MIN_OCCUPANCY` (0.20) are skipped.
- **Diversity** is `1 - sum p^2` over extracted value shares. Constant piles have diversity 0. Attributes below `MIN_DIVERSITY` (0.12) are skipped.

`_select` uses that typed ranking while evidence is still weak (a single common material, or only the provisional opener), or when `other` is blocked. Ties break by `material`, `color`, `style`, `size`. If there is no typed winner, the engine asks `other`.

Answered, `no_preference`, and exhausted attributes are not re-asked. Messages always match `ask_attribute`.

## Family mapping

Coarse category strings map to `watches`, `jewelry`, `shoes`, `bags`, `accessories`, `clothing`, or `other`. Root Amazon fragments are stripped first.

If the remaining string is unrecognized, the engine majority-votes family labels from candidate catalog snippets.

Closed-vocab extraction uses the first regex hit per field on each candidate snippet.

## Scenario handling

### Buying

Recommend immediately using the initial hard constraint. If that line is only a common material, ask the next typed split (often color). If it is already a distinctive feature, ask `other`.

### Browsing

With no constraints yet, ask the highest-scoring typed split (often material). After a common-material answer, keep asking typed splits that still divide the pile. After a distinctive feature, two typed attributes, or a decline, switch to `other`.

### Intent override

After the opening sentence, the stored constraint is provisional, so the engine still asks a typed split until a confirmed answer or the replacement arrives. After a replacement that is only a common material, it keeps a typed split; a distinctive replacement goes to `other`.

### Boundary

The first non-null question may receive a one-time no-preference response. That counts as evidence that typed probing failed, so the next question is `other` instead of cycling through more typed fields.

## Trade-offs

- Live occupancy and diversity keep the first question aligned with the retrieval pile.
- Switching to an open follow-up after identifying evidence prefers leftover requirements over a second coarse attribute. A lone `cotton` still gets a typed split.
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

`tests/test_questions.py` covers turn 10 `None`, empty-pile `other`, clothing material splits versus constant piles, jewelry skipping material, skipped answered/no-preference/exhausted fields, never asking `use_case`/`brand`/`budget`/`category`, family mapping, closed-vocab extraction, keeping a typed split after a weak material, open follow-up after a distinctive feature or two typed attributes or a typed decline, keeping a typed split after a common-material replacement, and keeping a typed split on a provisional opener.

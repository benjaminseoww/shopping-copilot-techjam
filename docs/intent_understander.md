# Intent Understander

## Purpose

`IntentUnderstander` converts the newest customer message into a structured update for session memory and retrieval. It identifies categories, constraints, preference exhaustion, and intent replacement while preserving the customer’s original wording.

It does not store session state, choose questions, rank products, inspect hidden intent cards or ground truth, or import evaluator helpers.

## MVP

The first version uses phrase and template matching for the messages produced by the supplied evaluator:

- **Buying:** extract the category and text after `A key requirement is:`.
- **Browsing:** extract the category from `I'm looking for ...` without inventing constraints.
- **Positive clarification:** extract one or more constraints after `For that, what matters is:`.
- **No additional preference:** mark the requested attribute exhausted and add no positive search terms.
- **Boundary:** record the requested attribute as no-preference and add no positive search terms.
- **Clarification request:** treat `Ask me about one specific attribute` as interaction feedback, not product evidence.
- **Intent override:** treat `Actually, ignore my earlier preference. What I need is: ...` as a replacement operation. Preserve the category, deactivate earlier conversational preferences, and add the new raw requirement.
- **Unknown phrasing:** retain only conservative product-like terms after removing conversational boilerplate.

Each extracted constraint keeps its raw text for lexical retrieval and may also be classified as one API attribute: `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, or `other`.

## Proposed interface

The parser receives the newest message, turn number, and previous `ask_attribute`. It returns an intent update containing:

- optional category;
- interaction kind;
- constraints to add;
- attribute classifications and raw text;
- attributes marked no-preference or exhausted;
- whether existing preferences should be superseded;
- conservative fallback terms; and
- parser provenance, such as phrase, semantic, or fallback.

The parser describes a state change but does not mutate memory. This lets a future semantic parser emit the same update type without changing downstream components.

How to get there — approaches, trade-offs, and a phased rollout — is in `docs/intent_semantic_plan.md`. Phrase matching remains the default until a paraphrase eval justifies enabling a semantic path.

## Future-code comments

The implementation should include focused comments at the parser boundary:

- `TODO(intent): Add a schema-constrained LLM or local-model parser that emits the same IntentUpdate structure.`
- `TODO(intent): Validate model output and reject invalid attributes, unsupported fields, and malformed replacement operations.`
- `TODO(intent): Require extracted constraints to be grounded in spans from the current message.`
- `TODO(intent): Fall back to phrase parsing on timeout, invalid output, low confidence, missing credentials, or unavailable network.`
- `TODO(intent): Preserve category during an override unless the customer explicitly replaces it.`
- `TODO(intent): Never convert negated or no-preference wording into positive query terms.`
- `TODO(intent): Report actual model token usage when semantic parsing is enabled.`

## Trade-offs

- Phrase matching is fast, deterministic, offline, and uses zero model tokens, but it is not robust semantic understanding.
- Preserving raw text helps lexical search because evaluator disclosures originate from catalog metadata.
- Typed attributes help memory and question selection but can be ambiguous.
- Conservative extraction avoids false hard constraints, at the risk of missing implicit preferences.
- A schema-constrained LLM or local model should improve paraphrase and semantic handling, but adds latency, cost, nondeterminism, validation work, and possible network risk.

## Scoring contribution

- **HitRate@10:** clean accumulated constraints increase target recall; override and no-preference handling prevent misleading query terms.
- **MRR:** specific active constraints can move the target higher in the Top 10.
- **MTTC/Efficiency:** extracting newly disclosed evidence immediately can produce earlier hits.
- **Tokens:** the phrase MVP reports zero tokens. Model parsing would add reported usage but does not directly change the core score.

The contribution is indirect: the Recommendation Engine still produces the scored ASIN list.

## Failure cases

- Paraphrased or reformatted messages do not match a known phrase.
- Punctuation changes break category or constraint boundaries.
- Semicolons inside product text are split incorrectly.
- Negation or conditional language is interpreted as a positive requirement.
- Ambiguous terms such as `fit` are assigned to the wrong attribute.
- An override removes too many or too few previous constraints.
- Boilerplate survives fallback filtering and dilutes retrieval.
- A semantic parser hallucinates constraints, emits invalid structure, times out, or requires unavailable network access.

## Follow-up options

The full approach comparison, architecture, and phased rollout is in `docs/intent_semantic_plan.md`. Short version:

1. Keep phrase parsing as the high-precision path and the mandatory offline fallback.
2. Treat dialogue-act understanding (override, no-preference, exhaustion) as higher value than generative rewriting of constraint text.
3. Add a schema-constrained semantic parser behind the same `IntentUpdate` interface, with span grounding, validation, and conservative arbitration.
4. Improve paraphrase, negation, delimiter, spelling, and domain-attribute handling in the offline layer first.
5. Add a catalog-derived vocabulary for classifying categories, brands, materials, and colors — not for inventing constraints.
6. Compare local and hosted models on score gain, latency, tokens, reproducibility, and offline behavior before changing the submission default.

## Planned tests

Future tests in `tests/test_intent.py` should cover:

- Buying and Browsing initial messages.
- Positive clarification with one and two constraints.
- Raw constraint preservation and ordering.
- No-preference, no-additional-preference, and clarification-request messages.
- Boundary handling.
- Intent Override replacement while preserving category.
- Unknown and paraphrased-message fallback.
- Negation, punctuation, whitespace, empty input, and malformed messages.
- Deterministic phrase-parser output.
- Semantic-parser schema rejection and phrase fallback using mocked model results.
- Zero token usage for phrase parsing and usage propagation for modeled parsing.

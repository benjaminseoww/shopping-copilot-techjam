# Questions Engine

## Purpose

`QuestionsEngine` chooses the next clarification message and structured `ask_attribute` from the current session state and ranked candidates.

It does not parse intent, mutate memory, retrieve products, rank recommendations, or inspect ground truth. Recommendations must be generated independently on every turn; asking a question must never suppress the current Top 10.

## MVP

- Return recommendations on every turn.
- On turns 1–9, return `ask_attribute="other"`.
- On turn 10, return `ask_attribute=None` because the evaluator generates no subsequent reply.
- Return a non-empty natural message consistent with the structured field.
- For `other`, ask broadly for another requirement, priority, or correction rather than naming a typed attribute.
- Track asked, exhausted, and no-preference attributes through session state.
- Never interpret a no-preference reply as a positive retrieval constraint.

The strict MVP continues using `other` through turn 9. This is an evaluator-oriented baseline and may become repetitive after all hidden constraints have been disclosed.

## Proposed interface

The engine receives:

- current `SessionState`;
- one-based turn number; and
- current ranked or over-fetched candidates.

It returns a `QuestionDecision` containing:

- customer-facing message; and
- one allowed attribute or `None`.

The orchestrator should rank products first, call the Questions Engine, and combine both results into one response.

## Why `other` first

In the supplied evaluator, `other` reveals up to two undisclosed constraints regardless of how they are classified. A typed question only reveals constraints assigned to that exact bucket and otherwise receives a no-additional-preference response.

A fixed typed cycle can waste scarce turns on attributes such as brand or budget, ignore conversation state, and delay the useful question. The catch-all policy establishes a simple scoring baseline before adding candidate-aware selection.

## Future-code comments

The implementation should include focused comments at the attribute-selection boundary:

- `TODO(questions): Estimate each eligible attribute's value distribution across the over-fetched candidates.`
- `TODO(questions): Score expected uncertainty reduction or information gain after a possible answer.`
- `TODO(questions): Exclude answered, declined, exhausted, unsupported, or nearly constant attributes.`
- `TODO(questions): Combine information gain with answerability, current ranking confidence, and the cost of another turn.`
- `TODO(questions): Use deterministic tie-breaking and fall back to "other" when no typed attribute has clearly positive value.`
- `TODO(questions): Validate any best-attribute policy against the strict "other" baseline before enabling it.`

Candidate diversity is not the same as simulator answerability. A product field may vary among candidates while the target's hidden intent card has no constraint classified under that attribute. A future policy therefore needs an answer-probability estimate, not information gain alone.

## Scenario handling

### Buying

Recommend immediately using the initial hard constraint, then ask `other` for remaining preferences.

### Browsing

Recommend from the coarse category and ask `other` early to expose target-derived constraints.

### Intent override

After the replacement message arrives, questions must use only active state and must not repeat or reinforce superseded preferences.

### Boundary

The first non-null question may receive a one-time no-preference response. Record it without adding search evidence and continue asking; it does not prove that no hidden constraints remain.

## Trade-offs

- `other` maximizes deterministic information disclosure in the supplied evaluator and uses no tokens.
- It may feel repetitive or evaluator-specific after all constraints are exhausted.
- Fixed typed cycles are easy to implement but often request attributes with no available answer.
- Candidate information gain is more natural and potentially more efficient, but can be misaligned with hidden-card answerability.
- LLM-generated question wording adds cost and latency without changing how the simulator selects its response.

## Scoring contribution

- **HitRate@10:** questions indirectly expose target-derived evidence that improves retrieval.
- **MRR:** better evidence can improve rank, though ranking remains the Recommendation Engine's responsibility.
- **MTTC/Efficiency:** asking an informative question early can lead to an earlier hit on the following turn.
- **Tokens:** deterministic templates report zero tokens. Model-generated wording must report actual usage.

Questions receive no direct score.

## Failure cases

- Repeating `other` after all constraints are exhausted creates an unnatural conversation.
- Treating the first Boundary reply as global exhaustion suppresses later useful questions.
- A message names one attribute while `ask_attribute` contains another.
- A fixed cycle repeatedly selects attributes with no hidden answer.
- Candidate information gain chooses an attribute unavailable in the hidden intent card.
- Stale state causes questions about superseded intent.
- Returning `None` before turn 10 prevents further disclosure.
- Asking on turn 10 has no effect.
- Invalid attributes, malformed responses, or exceptions produce missed opportunities.

## Follow-up options

1. Add candidate uncertainty and information-gain scoring with answerability filtering and an `other` fallback.
2. Add adaptive stopping or switching after confirmed catch-all exhaustion.
3. Calibrate question value with scenario-level public-set ablations.
4. Add retrieval-confidence signals so questions are favored when the Top 10 is ambiguous.
5. Add deterministic message variation while keeping wording aligned with `ask_attribute`.
6. Consider a model for phrasing only if measured quality gains justify its token and latency cost.

## Planned tests

Future tests in `tests/test_questions.py` should cover:

- `other` on turns 1–9 and `None` on turn 10.
- Recommendations remaining present on every turn.
- Non-empty messages matching their structured attribute.
- Contract-valid attributes only.
- No-preference and exhausted attributes not being requested as typed questions.
- Buying and Browsing continuing to recommend while collecting evidence.
- Override questions ignoring superseded preferences.
- The first Boundary response not exhausting all future clarification.
- Turn 10 never requesting another customer response.
- Future candidate selectors with diverse, constant, missing, and previously answered attributes.
- Ablations comparing `None`, fixed cycles, strict `other`, and future best-attribute policies.

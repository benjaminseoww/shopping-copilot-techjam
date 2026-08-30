# Memory Store

## Purpose

`MemoryStore` maintains structured conversational state for the active shopping session. It lets later turns reuse accumulated requirements without searching the complete raw transcript.

It does not parse language, retrieve products, choose questions, score recommendations, or inspect ground truth and hidden evaluator state.

## MVP

Use a small in-memory state keyed by `session_id`. Store the anonymized historical `user_profile` separately from the current shopping intent.

The session state should contain:

- immutable historical profile;
- current category;
- active constraints with raw text, attribute, source turn, and provenance;
- superseded constraints retained for debugging but excluded from retrieval;
- attributes marked no-preference;
- attributes marked exhausted;
- previously requested attributes and the latest `ask_attribute`;
- a bounded list of customer messages and state-change events;
- accumulated already-shown recommendation ids, forgotten on preference override; and
- the latest processed turn and agent action.

Already-shown products are operational implicit rejects of the current query, not evidence of customer preference. A later turn skips them when slicing the customer-facing list. Preference override clears that history because the query changed and an earlier guess may now be valid. Profile tags may later become weak ranking priors, but they must not become hard current-session requirements.

## Proposed interface

- `reset` receives a session identifier and profile, then creates completely fresh state.
- `apply` receives a structured update from `IntentUnderstander` and mutates only that session.
- `record_agent_action` stores the selected attribute and accumulates unique shown recommendation ids, capped at 40.
- `get` returns a structured session snapshot for recommendation and question selection.
- Access for an unknown session fails clearly because the public contract requires `reset` first.

## Lifetime and isolation

Session state begins at `Agent.reset()` and ends when the evaluator moves to the next session or the process exits. The supplied evaluator has no explicit end-session callback.

- Reset must never restore old conversational state.
- Different session IDs must not share preferences.
- Similar or identical profiles must not be treated as the same person.
- The MVP may retain only the newly reset active session because the evaluator processes sessions sequentially.
- The catalog index is separate: it is global, immutable, and shared by all sessions.
- Process restart may discard conversation state; persistent storage is unnecessary for this protocol.

## State transitions

### Positive update

Accumulate new constraints and deduplicate repeated evidence. Preserve the strongest raw wording and source turn.

### No preference

Mark the attribute as no-preference. Do not add words from the reply as positive retrieval terms. Do not erase an earlier positive constraint unless the customer explicitly replaces it.

### Exhaustion

`I don't have an additional preference for X` means no new evidence was supplied. Keep existing constraints active and mark `X` exhausted so it is not requested repeatedly.

### Override

Treat `Actually, ignore my earlier preference...` as replacement, not addition. Deactivate prior conversational preferences in scope and activate the new requirement. Preserve historical profile and category unless explicitly replaced. Clear already-shown recommendation ids so the replacement query can surface a product that was shown under the old intent.

## Trade-offs

- In-memory state is fast, deterministic, offline, and simpler than JSON files or session SQLite, but disappears on restart.
- Structured state prevents stale overrides and negative language from polluting retrieval, but depends on parser accuracy.
- Raw text supports lexical matching while normalized attributes support question selection.
- A bounded event log improves debugging without repeatedly processing an unbounded transcript.
- Per-session isolation prevents leakage but deliberately avoids unsupported cross-session personalization.

## Scoring contribution

- **HitRate@10:** retaining active constraints gives retrieval more target-derived evidence.
- **MRR:** excluding stale and negated evidence can move the target higher.
- **MTTC/Efficiency:** preserving early evidence and avoiding repeated dead questions can produce earlier hits.
- **Tokens:** the deterministic MVP uses none. Structured state would keep future model prompts smaller than full-transcript prompts.

Memory affects scoring indirectly through retrieval and question selection.

## Failure cases

- `respond` is called before `reset`.
- Reset fails to clear prior intent.
- Similar profiles accidentally share conversation state.
- Negative or boilerplate text becomes a positive constraint.
- A paraphrased override is merged instead of replacing stale intent.
- No-preference wording incorrectly deletes a valid earlier constraint.
- Contradictory constraints accumulate without replacement handling.
- Logs and recommendation histories grow without bounds.
- Profile tags become hard filters and suppress the target.
- A single-active-session implementation is used with an unexpected concurrent harness.

## Follow-up options

1. Add confidence, provenance, conflict groups, and narrower replacement scopes.
2. Measure memory-on versus memory-off effects by scenario.
3. Expose candidate and exhaustion signals to a best-attribute question policy.
4. Support a schema-constrained semantic parser that emits the same state updates.
5. Add compact state summaries and stricter retention limits if conversations expand.
6. Add a concurrent session map with locking and eviction only if the harness becomes parallel.

## Planned tests

Future tests in `tests/test_memory.py` should cover:

- Profile and current-intent separation.
- Constraint persistence and accumulation across turns.
- Idempotent duplicate updates.
- Raw text, attribute, source turn, and provenance retention.
- No-preference and exhaustion updates adding no positive terms.
- Existing constraints surviving a no-additional-preference reply.
- Override superseding stale constraints while preserving profile and category.
- Superseded constraints remaining inspectable but absent from retrieval state.
- Fresh state when resetting the same or a different session ID.
- Identical profiles never sharing conversational state.
- Unknown-session access failure.
- Latest question and accumulating shown-id recording without treating recommendations as preferences.
- Override clearing shown ids so a replacement query can re-offer an earlier guess.
- Bounded message, event, and recommendation histories.
- Shared catalog lifetime remaining separate from session memory.

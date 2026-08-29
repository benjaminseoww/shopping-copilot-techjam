# Intent Understander: From Phrase Scanning to Semantic Context

This plan describes how to evolve `IntentUnderstander` from a high-precision evaluator-template scanner into a conservative semantic context parser, without changing the downstream `IntentUpdate` → `MemoryStore` → retrieval contract.

It is a design document, not an implementation. The phrase MVP in `starter/intent.py` stays the default until a measured paraphrase eval says otherwise.

The two parsers to build are specified in [Two options](#two-options-build-these):

1. **Option 1 — Non-LLM:** templates, cue lists, gazetteers, `last_ask`. Offline default.
2. **Option 2 — LLM API:** schema-constrained hosted call that names the act and returns grounded spans, falling back to Option 1.

Do not build nearest-neighbor matching over hand-written phrases per intent.

## Why this is not “just add an LLM”

The current parser does two different jobs in one regex ladder:

1. **Dialogue-act classification.** Is this buying, browsing, clarification, no-preference, exhaustion, override, clarification-request, or unknown?
2. **Slot extraction.** What category, constraint spans, and attribute labels should memory apply *this turn*?

Retrieval quality depends more on (1) being correct than on fancy paraphrasing of (2). A missed override leaves stale constraints in the BM25 query. A rewritten constraint string can *lose* the lexical overlap that the evaluator planted from catalog metadata. Those two failure modes matter more than “the model understood the vibe.”

Semantic work therefore has to be **turn-delta parsing with grounded spans**, not free-form restatement of the whole conversation.

## Current system

```text
user_message + turn + last_ask
        │
        ▼
IntentUnderstander.parse()     # regex templates, then conservative fallback
        │ IntentUpdate
        ▼
MemoryStore.apply()            # accumulate / supersede / mark no-pref
        │ SessionState
        ├──► RecommendationEngine.recommend()   # category + active constraint text
        └──► QuestionsEngine.decide()           # last_ask, exhausted, no_pref
```

### What the phrase layer already covers

Exact templates from `evaluator/local_evaluator.py`:

| Kind | Template | Memory effect |
|---|---|---|
| buying | `I'm looking for {category}. A key requirement is: {constraint}.` | set category + one hard constraint |
| browsing | `I'm looking for {category}, but I'm still exploring.` | category only |
| initial_preference | `I'm looking for {category}. {old_value}` | category + provisional constraint (override sessions, turn 1) |
| clarification | `For that, what matters is: {c1}; {c2}.` | add raw constraints |
| no_preference | `I don't have a preference for {attr}; please use your judgment.` | mark attribute, add no search terms |
| exhausted | `I don't have an additional preference for {attr}.` | mark exhausted, keep prior constraints |
| override | `Actually, ignore my earlier preference. What I need is: {constraint}.` | supersede + new constraint, keep category |
| clarification_request | `Those options are not quite right yet. Ask me about one specific attribute.` | interaction feedback, not evidence |
| unknown | anything else | strip boilerplate, keep leftover terms |

Attribute labels are keyword rules (`cotton` → material, `$`/`under` → budget, `hiking` → use_case, else feature). Retrieval mostly ignores those labels and searches the **raw constraint text**, which is why preserving wording is part of scoring, not just cleanliness.

### What it cannot do

- Paraphrase. The spec allows the organizer to reword customer messages; correctness is still exact ASIN match.
- Punctuation / extra clauses around a known template.
- Semicolons inside product-derived constraint text (clarification splitter).
- Negation and conditionals (`not leather`, `unless it's cotton`).
- Context beyond `last_ask` (`that`, `the blue one`, replies that omit the attribute name).
- Reliable override detection when the replacement sentence does not match `OVERRIDE_RE`.
- Distinguishing “no preference for material” from “I don't want that material.”

On the **current public evaluator**, these gaps barely matter: messages are generated from the templates above. The phrase scanner is nearly optimal there. Semantic understanding is insurance for private paraphrasing, messy catalog strings, and a future typed question policy.

## The two semantic problems (do not mix them)

| Problem | Lives in | Helps | Does not help |
|---|---|---|---|
| Dialogue-act / context understanding | Intent Understander | override, no-pref, exhaustion, “what kind of reply is this?” | synonym matching of `cotton` to products |
| Product-language understanding | Recommendation Engine | synonyms, embedding rerank, query rewrite | noticing that a paraphrase was an override |

If a paraphrased override is ingested as a positive add, hybrid retrieval cannot recover. If constraint wording is preserved, BM25 already has a strong channel because disclosures are sliced from target metadata.

**Rule:** put conversation semantics in intent; put catalog semantics in retrieval.

## Failure-mode ranking

Work the list in this order. Later stages are wasted if earlier ones are wrong.

1. **Missed override** → stale query terms. Catastrophic for Intent Override sessions (15% of the mix). An override session also cannot score a hit until the new intent is sent, so a late or missed replacement burns MTTC.
2. **No-preference / negation ingested as positive terms** → query pollution. Boundary is only 5%, but the same bug hits any declined attribute.
3. **Dropped constraint span** → weaker evidence, later or missed hits. Fallback already recovers some tokens, so this is less deadly than (1) or (2).
4. **LLM rewrites the constraint** → *worse* than leaving it raw. `color: blue` and long feature bullets are retrieval gold; “the customer prefers a blue item” is not.
5. **Wrong attribute label** → little BM25 effect today; it will matter once `QuestionsEngine` stops asking `other` every turn.
6. **Hallucinated constraint** → false hard filter / noisy OR terms. Treat as a parser failure and fall back.

## Two options (build these)

Do not build a nearest-neighbor intent classifier over hand-written paraphrase banks. The two parsers below share `IntentUpdate` and `MemoryStore`. Option 1 is the offline default. Option 2 is a gated API that must fall back to Option 1.

### Option 1 — Non-LLM (rules + context)

No model, no network, zero tokens. Improve the three jobs with code the agent already has: templates, cue lists, gazetteers, and `last_ask`.

**Buying / browsing / change of mind**

Keep the current regexes as exact hits. Add small **cue lexicons** that fire only when no exact template matches:

| Act | Cues (examples) | Extra rule |
|---|---|---|
| browsing | `still exploring`, `just looking`, `just browsing`, `not sure yet` | category span if present; **no** constraints |
| buying | `key requirement`, `must be`, `I need` + a product-like span | category + one constraint span |
| change_preference | `ignore`, `forget`, `never mind`, `instead`, `actually` **and** a replacement clause | `supersede_preferences=True` only with a leftover product-like span |
| no_preference | `don't have a preference`, `doesn't matter`, `you pick`, `use your judgment` | mark attribute; **empty** constraints |
| exhausted | `no additional preference`, `nothing more on` | mark attribute; keep prior constraints |
| clarification | `what matters is`, `the important part is` | split values; add constraints |
| clarification_request | `not quite right`, `ask me about one` | no product evidence |

Cues label the **act**. They do not invent slot text. A lone `actually` without a replacement span is not an override. False override is worse than a missed add.

**Item (category)**

- Exact templates still copy `I'm looking for (?P<category>.+?)`.
- Otherwise take a conservative span after `looking for` / `need` / `want`, or keep `ParseContext.category`.
- On override emit `category=None` unless the message clearly names a new class. Memory keeps the old item.

**Attributes**

- Values stay **raw spans** of the message. Do not rewrite `color: blue`.
- Split clarifications on `"; "` as today; if the blob looks like one catalog bullet, keep it whole.
- Classify buckets with **word-boundary** gazetteers (`MATERIALS`, `COLORS`, plus optional catalog `store` / category n-grams). These lists are not enums of legal values. Unknown text (`navy`) is still stored; the bucket may be `feature`.
- Negation and no-pref/exhausted replies never become `Constraint.text`.
- Short replies (`I don't care`) use `last_ask` when the message does not name an `AttributeName`.

**Fallback**

If nothing fires: strip boilerplate (`looking for`, `please`, `judgment`, `ignore`, `preference`, …) and keep leftover product-like terms with `interaction_kind="unknown"`. Never set `supersede_preferences` from fallback.

**Interface**

Same `parse(message, turn, last_ask)` as today, optionally with `ParseContext`. Always `parser="phrase"` or `"fallback"`, `usage = {0, 0}`.

| For | Against |
|---|---|
| Official scoring can disable the network; this still runs | Novel paraphrases still miss |
| Deterministic, testable, no credentials | Cue lists are a maintenance surface |
| Enough for the current templated evaluator | Weak at implicit “change preference” without cues |

**When to ship:** always. This is the MVP hardening and the fallback for Option 2.

---

### Option 2 — LLM API (schema-constrained hosted call)

One hosted model call that **names the act immediately** (including “the person wants to change preference”) and returns grounded spans. No prototype bank. Phrase/Option 1 remains the fallback because official scoring may disable network access.

**When it runs**

Default: Option 1 first. Call the API only if Option 1 returns `unknown` (or a low-confidence cue hit). Env, for example:

- `INTENT_PARSER=phrase` — Option 1 only (default)
- `INTENT_PARSER=api` — Option 1 miss → API → Option 1/fallback on failure
- Credentials via environment (never committed), e.g. `INTENT_LLM_API_KEY`, `INTENT_LLM_BASE_URL`, `INTENT_LLM_MODEL`

Do not make `Agent.respond` depend on a live key.

**Request**

The client sends only what disambiguates this turn. Do not send intent cards, ASINs, catalog rows, or `scenario_type`.

```json
{
  "task": "shopping_intent_delta",
  "message": "Forget the earlier preference; make it leather.",
  "turn": 4,
  "last_ask": "other",
  "category": "Shoes Running",
  "active_constraints": ["cotton"],
  "recent_user_messages": ["I'm looking for Shoes Running. cotton"]
}
```

System instructions (conceptual): classify this turn’s act; copy category and constraint **substrings from `message`**; do not restate session state; do not paraphrase product wording.

**Response schema (must validate)**

```json
{
  "act": "change_preference",
  "category_span": null,
  "constraints": [
    {"text": "leather", "attribute": "material"}
  ],
  "no_preference": [],
  "exhausted": [],
  "supersede": true,
  "confidence": 0.86
}
```

`act` is one of: `buying`, `browsing`, `clarification`, `no_preference`, `exhausted`, `change_preference`, `clarification_request`, `initial_preference`, `unknown`.

Map into today’s `IntentUpdate`:

| API field | `IntentUpdate` |
|---|---|
| `act` | `interaction_kind` (`change_preference` → `override`) |
| `category_span` | `category` (or `None` to keep memory) |
| `constraints[].text` | `Constraint.text` |
| `constraints[].attribute` | `Constraint.attribute` if in `ALLOWED_ATTRIBUTES`, else `classify_constraint(text)` |
| `no_preference` / `exhausted` | same sets |
| `supersede` | `supersede_preferences` |
| token counts from the HTTP response | `usage` |

**Code-side guards (reject the whole payload, then Option 1)**

- JSON/schema invalid, unknown keys, missing `act`.
- `text` / `category_span` not a whitespace-normalized substring of `message`.
- `supersede` true unless `act` is `change_preference` **and** there is at least one grounded constraint.
- `act` in `{no_preference, exhausted, clarification_request}` with non-empty `constraints`.
- Attribute not in `ALLOWED_ATTRIBUTES`.
- Timeout, HTTP error, missing credentials, or no network.
- Optional: `confidence` below a threshold.

Override may **not** replace category unless `category_span` is present and grounded. Default `category_span` is `null`.

**Usage and failures**

Report real `prompt_tokens` and `completion_tokens` on `Agent.respond`. Option 1 reports zeros. Timeouts and exceptions must not crash the session; miss/timeout handling is on the evaluator.

**LLM strengths here**

- Direct act classification (“change preference”) without a phrase bank.
- Paraphrase, negation, short answers given `last_ask`.
- Weak at copying spans unless grounding is enforced. Ungrounded rewrites (`"prefers a blue item"`) are a retrieval regression.

| For | Against |
|---|---|
| Immediate act label, including preference change | Latency, cost, nondeterminism |
| Handles wording the cue list will never cover | Network may be off for official scoring |
| Same downstream `IntentUpdate` if validation holds | Bad output is worse than Option 1 unless rejected |

**When to ship:** behind `INTENT_PARSER=api`, mocked tests in CI, no live calls in CI. Enable for submission only if a paraphrase fixture shows fewer override/no-pref failures **and** templated public-set HitRate does not drop.

---

## Other approaches considered (not the build path)

The following stay in the design notes for contrast. They are not the two options above.

### A. Harden the phrase layer (no model) — folded into Option 1

Relax anchors, optional punctuation, synonym templates (`never mind` / `forget what I said` → override), catalog gazetteers for material, color, brand (`store`), and category tokens. Add a small negation lexicon so `don't`/`no preference` never become fallback query terms.

| For | Against |
|---|---|
| Zero tokens, offline, deterministic, already matches the supplied evaluator | Still fails on true paraphrase |
| Cheap to ship; no credentials | Template list becomes a maintenance surface |
| Gazetteers improve attribute labels for questions | Over-matching common words (`fit`, `running`, `work`) |

**Use for:** always-on baseline and official scoring if the network is disabled.

### B. Embedding dialogue-act router + rule slots — skipped

Nearest-neighbor over paraphrase prototypes is **out of scope**. It needs a hand-maintained utterance bank per act; Option 2’s LLM classifies the act directly instead.

| For | Against |
|---|---|
| Offline paraphrase routing | Requires example phrases per intent |
| | Confuses near-acts; extra model asset |
| | Slot fills still need rules |

Do not implement this unless Option 1 and Option 2 both fail offline.

### C. Sequence tagger / span extractor (local NLU)

Train a tiny classifier + BIO tagger on (a) evaluator templates and (b) generated paraphrases of those templates, with constraint spans labeled as substrings of the message.

| For | Against |
|---|---|
| Directly outputs grounded spans | Needs a synthetic corpus; risk of overfitting template family |
| Offline, stable, no generation hallucinations | Weak on out-of-family private paraphrases |
| Attribute classification can share the encoder | Engineering cost is high relative to 8 dialogue acts |

**Use for:** only if B mis-routes kinds *and* a hosted LLM is disallowed. Unlikely as the first semantic bet.

### D. Schema-constrained generative parser (local or hosted)

A model emits JSON that validates against the `IntentUpdate` shape: kind, optional category, constraints with text + attribute + span, no-preference / exhausted sets, `supersede_preferences`, confidence.

Required guards (from the existing TODOs, now as acceptance rules):

- Constraint `text` must be a span of the current message (whitespace-normalized substring).
- Attributes ∈ `ALLOWED_ATTRIBUTES`.
- `supersede_preferences` only with `interaction_kind="override"`.
- No-preference / exhausted replies add **no** positive constraints.
- Invalid JSON, timeout, low confidence, missing credentials, or no network → phrase or fallback parser.
- Report real `prompt_tokens` / `completion_tokens`.

| For | Against |
|---|---|
| Best paraphrase and implicit-language coverage | Latency, cost, nondeterminism |
| Can use `last_ask`, category, and active constraints as context | Official scoring may disable network |
| Same downstream type if validation is strict | Ungrounded output is worse than phrase matching |
| Local instruct models avoid the network issue | Local models still add install size, warmup, and variance |

**Use for:** this is Option 2 when the backend is a hosted API. A local instruct model can implement the same schema later without changing `IntentUpdate`; it is not required for the first API path.

### E. Phrase-first hybrid (recommended default)

```text
phrase.parse(message)
if high-precision template hit:
    return phrase update          # parser="phrase"
else:
    semantic.parse(message, context)
    if valid and grounded and confidence >= T:
        return semantic update    # parser="semantic"
    return conservative fallback  # parser="fallback"
```

| For | Against |
|---|---|
| Current public set stays zero-token and deterministic | A *partial* template match can hide a paraphrase (false phrase hit) |
| Semantic cost only on the messages that need it | Must keep phrase patterns strict (precision over recall) |
| Matches submission rules: document network + fallback | Two implementations to keep consistent |

**Critical:** phrase patterns must stay **high precision**. Do not loosen `BUYING_RE` until it matches “I'm looking for shoes, maybe something cotton-ish” and then skip the semantic path.

### F. Semantic-first with phrase fallback

Always call the model; phrase is disaster recovery.

| For | Against |
|---|---|
| Maximum paraphrase robustness | Pays tokens and latency on every templated turn |
| | Higher chance of rewriting catalog-derived strings |
| | Unnecessary on the current evaluator |

**Use only if** a paraphrase fixture shows phrase-first missing overrides because a relaxed template swallowed them.

### G. Conservative arbitration (phrase ∥ semantic)

Run both. Merge with rules, not averaging:

- Kind: if phrase is a strict template hit, keep it. Else take semantic kind if grounded.
- Constraints: union of spans that appear in the message; drop any semantic span phrase would have treated as boilerplate.
- Override: require *either* a strict phrase override *or* (semantic override ∧ override cues ∧ grounded new constraint). Never override from semantics alone on a vague “actually…”.
- Conflict on kind → safer act: prefer no-pref/exhausted over add; prefer add over override unless cues exist.

| For | Against |
|---|---|
| Best robustness if both sources can fail | Complexity and disagreement tests |
| Caps hallucination | Double latency if the model always runs |

**Use for:** after D exists, as the production merger. Cheap version: only run semantic when phrase returns `unknown`.

### H. Do nothing in intent; put semantics in retrieval

Query rewrite, dense rerank, synonym expansion on whatever memory already stored.

| For | Against |
|---|---|
| Helps when the constraint text is preserved but catalog wording differs | Does not fix dialogue-act errors |
| | Does not fix no-pref pollution |

**Use for:** a parallel retrieval track, not a substitute for this plan.

## Recommended architecture

Keep `parse(...)` as a pure function that returns a **delta**. Memory remains the only mutator.

```text
                    ┌──────────────────────────┐
 user_message       │  ParseContext            │
 turn               │  last_ask                │
 last_ask  ────────►│  category                │
                    │  active constraint texts │
                    │  last few user messages  │
                    └────────────┬─────────────┘
                                 ▼
                      normalize whitespace
                                 ▼
                 Option 1 PhraseParser (templates + cues)
                    │ hit                │ miss
                    ▼                    ▼
            IntentUpdate           Option 2 LLM API
            parser=phrase          (schema-constrained JSON)
                                         │
                                   Validator
                                   • schema
                                   • allowed attributes
                                   • span grounding
                                   • override / negation rules
                                   • confidence threshold
                                      │ valid     │ invalid
                                      ▼           ▼
                               IntentUpdate   FallbackParser
                               parser=semantic parser=fallback
                                         │
                                         ▼
                              IntentUpdate (+ usage)
```

### Interface (compatible extension)

Today:

```python
def parse(self, user_message: str, turn: int, last_ask: AttributeName | None = None) -> IntentUpdate
```

Target:

```python
@dataclass(frozen=True)
class ParseContext:
    last_ask: AttributeName | None = None
    category: str | None = None
    active_constraint_texts: tuple[str, ...] = ()
    recent_user_messages: tuple[str, ...] = ()  # bounded, current session only

@dataclass
class IntentUpdate:
    # existing fields...
    parser: str = "phrase"                 # phrase | semantic | fallback | hybrid
    confidence: float = 1.0
    evidence_spans: list[tuple[int, int, str]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 0, "completion_tokens": 0
    })
```

`Agent.respond` already returns `usage`; it should sum intent-parser tokens with any later model calls. Phrase and fallback keep zeros.

Do **not** pass `scenario_type`, intent cards, ASINs, or catalog rows into the parser. Infer override from language, not from the evaluator.

### Semantic JSON schema (conceptual)

```json
{
  "interaction_kind": "override",
  "category": null,
  "constraints": [
    {"text": "leather", "attribute": "material", "start": 52, "end": 59}
  ],
  "no_preference": [],
  "exhausted": [],
  "supersede_preferences": true,
  "confidence": 0.84
}
```

Validation that rejects the whole payload (then fallback):

- Unknown keys / missing `interaction_kind`.
- `text` not found in the current message.
- `supersede_preferences` without override kind, or override kind without a new grounded constraint.
- Positive constraints on no-preference / exhausted / clarification-request.
- Category replaced during override unless the message clearly names a new category (default: keep memory’s category by emitting `category=None`).

### Context policy

Feed the model (or router) only what disambiguates this turn:

- `last_ask` for “I don't care” / short answers.
- Current category so override does not invent a new one.
- Active constraint texts so the model can avoid re-emitting them as new adds.
- At most the last 2–3 user messages, not the agent’s questions (those are recoverable from `last_ask`).

Never ask the model to dump a full session state. `MemoryStore.apply` already accumulates, dedupes, and supersedes. A restated full state would duplicate or fight that logic.

### Override cues (high precision)

Semantic override is allowed only if at least one cue is present, for example:

- `ignore` / `forget` / `never mind` / `instead` / `actually` + a replacement clause
- explicit “not that, I need X”

Cue-less semantics must not set `supersede_preferences`. False override is worse than a missed add.

## What “accurate semantic context” means here

Accuracy is not open-ended NLU. For this agent it means:

1. The **dialogue act** matches what the customer just did, including paraphrases of the eight evaluator acts.
2. Extracted constraints are **spans the customer actually said**, preferably the catalog-derived wording the simulator copied.
3. Negative language never becomes a retrieval term.
4. Replacement is scoped: conversational preferences clear; category and profile stay unless explicitly replaced.
5. The parser can use **session context** (`last_ask`, active constraints) without reading hidden evaluator state.
6. Downstream components do not change: questions and retrieval still consume `IntentUpdate` / `SessionState`.

## Implementation phases

Do not skip measurement gates. Each phase should land behind the same `parse()` entry point.

### Phase 0 — already done

Phrase MVP, structured `IntentUpdate`, memory isolation, zero tokens.

### Phase 1 — semantic-ready contract (no behavior change)

- Add `ParseContext`, `confidence`, `evidence_spans`, `usage` with defaults.
- Thread `usage` from `IntentUnderstander` through `Agent.respond`.
- Split `PhraseParser` as an explicit collaborator so a later semantic parser can share `_constraint` / `classify_constraint`.
- Tests: existing phrase cases still pass; usage stays `{0, 0}`.

**Gate:** unit tests green; public-set metrics unchanged if run.

### Phase 2 — offline robustness (still no generative model)

- Strict phrase stays first.
- Fallback: drop no-pref / override / clarification-request boilerplate more aggressively so unknown paraphrases do not index `judgment`, `ignore`, `preference`.
- High-precision override cue detector for *kind only*; still require a leftover product-like span before `supersede_preferences=True`.
- Optional catalog gazetteer (materials, colors, frequent `store` tokens, category n-grams) **only** for `classify_constraint`, not for inventing constraints.
- Clarification split: prefer `"; "` as today; if a single blob looks like one catalog bullet, keep it whole.

**Gate:** phrase gold still 100%. Add a small paraphrase fixture (below) and report override-miss / no-pref-pollution rates. If those rates are already low, Phase 2 may be enough for v1.

### Phase 3 — Option 1 complete

- Cue lexicons + `last_ask` short-answer handling as specified in Option 1.
- Word-boundary gazetteers for bucket labels.
- Paraphrase fixture used as a **kind/span gold set**, not as nearest-neighbor training data.

**Gate:** phrase gold still 100%. Report override-miss / no-pref-pollution on the fixture.

### Phase 4 — Option 2 LLM API (gated)

- `SemanticParser` protocol: `parse(message, turn, context) -> IntentUpdate | None`.
- Env: `INTENT_PARSER=phrase|api` (default `phrase`); API key and model from the environment.
- JSON schema validation + span grounding + override/negation rules (see Option 2).
- Timeouts, HTTP errors, and credential misses return `None` (caller uses Option 1).
- Mocked tests first: invalid schema, hallucinated constraint, timeout, token accounting.

**Gate:** with mocks, fallback always fires on bad output. No live API in CI.

### Phase 5 — arbitration + agent wiring

- Phrase-first hybrid as default when `INTENT_SEMANTIC != off`.
- Sum token usage.
- Log `parser=` in `MemoryStore` events (already present) and keep it for ablations.

**Gate:** public templated eval ≈ Phase 0 score (regression budget: no HitRate drop). Paraphrase fixture: fewer override misses than phrase-only.

### Phase 6 — measure and choose the submission default

Compare, by scenario:

- Option 1 only (phrase + cues)
- Option 1 miss → Option 2 API
- API-first (diagnostic only)

Metrics: HitRate@10, MRR, MTTC, override-miss count, no-pref pollution count, latency, tokens.

**Submission default:** the cheapest configuration that does not lose templated public-set score and that wins on the paraphrase fixture’s override/no-pref tests. If generative parsing does not move those, leave it off for official scoring (network may be disabled anyway).

## Paraphrase fixture (do not edit the evaluator)

Keep the official simulator untouched. Add a **participant-side** fixture that rewrites the eight templates, for example:

- Buying: “Need running shoes — cotton is required.”
- Browsing: “Just exploring necklaces for now.”
- Clarification: “The important part is lightweight, and also color: blue.”
- No-pref: “Material doesn’t matter, you pick.”
- Exhausted: “Nothing more on color.”
- Override: “Forget the earlier preference; make it leather.”
- Clarification-request: “These aren’t right. Ask about one attribute.”

Each line stores: source kind, paraphrased text, expected `IntentUpdate` (kind, supersede flag, grounded constraint substrings, no-pref/exhausted attributes).

This fixture is the semantic test set. Public `local_evaluator.py` remains the scoring test set.

## Risks and non-goals

- **Do not** train on hidden intent cards or private labels.
- **Do not** let profile tags become constraints (memory already treats them as historical).
- **Do not** make network or a large local model mandatory for `Agent.respond`.
- **Do not** use the generative parser to rewrite queries; that belongs in retrieval, after this delta is stored.
- **Do not** expand `QuestionsEngine` until kind + no-pref parsing is trustworthy; typed questions amplify attribute-classification errors.
- Watch `INITIAL_PREFERENCE_RE`: it is a broad `I'm looking for X. Y` catch-all. Tightening it is good for precision; loosening it will steal paraphrases from the semantic path.

## Test plan additions (`tests/test_intent.py`)

Existing phrase tests stay. Add:

- Deterministic phrase output and zero usage.
- Paraphrase fixture: kind, supersede, grounded substrings, empty constraints on no-pref/exhausted.
- Negation and “please use your judgment” never appear in `fallback_terms` or constraint text.
- Override preserves category (`category is None` in the update).
- Mocked semantic: schema reject, ungrounded span reject, timeout → phrase/fallback.
- Hybrid: strict template still `parser="phrase"` even if a fake semantic parser is injected.
- Token usage propagated on a mocked successful semantic parse.

## Suggested decision

Ship **Option 1** as the default. Add **Option 2 (LLM API)** behind `INTENT_PARSER=api`, always falling back to Option 1. Do not add nearest-neighbor paraphrase routing.

That order matches the scoring surface: keep templated HitRate, stop override/no-pref disasters under paraphrase, preserve raw catalog wording for BM25, and only then spend API tokens.

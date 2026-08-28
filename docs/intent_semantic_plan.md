# Intent Understander: From Phrase Scanning to Semantic Context

This plan describes how to evolve `IntentUnderstander` from a high-precision evaluator-template scanner into a conservative semantic context parser, without changing the downstream `IntentUpdate` → `MemoryStore` → retrieval contract.

It is a design document, not an implementation. The phrase MVP in `starter/intent.py` stays the default until a measured paraphrase eval says otherwise.

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

## Approaches and trade-offs

### A. Harden the phrase layer (no model)

Relax anchors, optional punctuation, synonym templates (`never mind` / `forget what I said` → override), catalog gazetteers for material, color, brand (`store`), and category tokens. Add a small negation lexicon so `don't`/`no preference` never become fallback query terms.

| For | Against |
|---|---|
| Zero tokens, offline, deterministic, already matches the supplied evaluator | Still fails on true paraphrase |
| Cheap to ship; no credentials | Template list becomes a maintenance surface |
| Gazetteers improve attribute labels for questions | Over-matching common words (`fit`, `running`, `work`) |

**Use for:** always-on baseline and official scoring if the network is disabled.

### B. Embedding dialogue-act router + rule slots

Embed the message (and optionally `last_ask`) with a small local sentence model. Nearest-neighbor against paraphrased prototypes of the eight kinds. Then extract slots with the existing rules / gazetteers, not with generation.

| For | Against |
|---|---|
| Offline, low latency, good at “this is an override even though the wording changed” | Slot fills stay brittle |
| No API; can ship a quantized MiniLM-class model | Adds a binary asset and a dependency |
| Easy to unit-test with frozen vectors | Confuses near-intents (`no_preference` vs `exhausted`) without extra features |

**Use for:** kind detection when phrase confidence is low. Do not use embeddings to invent constraint strings.

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

**Use for:** unknown / paraphrased messages after validation, never as an unguarded default.

Hosted vs local:

- Hosted: higher quality per token, simple client, **cannot be the only path**.
- Local: aligns with “offline fallback required,” but quality of small models on JSON+spans must be measured; a bad local model that fails schema will just hit phrase fallback every time.

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
                 PhraseParser (strict templates)
                    │ hit                │ miss
                    ▼                    ▼
            IntentUpdate           SemanticParser
            parser=phrase          (embed kind and/or
                                   schema-constrained JSON)
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

### Phase 3 — dialogue-act router

- Build prototype phrases: evaluator templates plus a frozen paraphrase list checked into `tests/fixtures/intent_paraphrases.jsonl`.
- Local embedding or a hashed lexical cosine over character n-grams if we refuse extra deps.
- Router proposes `interaction_kind`; slot rules still extract spans.
- If kind ∈ {no_preference, exhausted} and `last_ask` is set, prefer `last_ask` when the message does not name a valid attribute.

**Gate:** kind accuracy on the paraphrase fixture vs phrase-only. Router must not lower phrase-gold accuracy (phrase-first).

### Phase 4 — schema-constrained generative parser (gated)

- `SemanticParser` protocol: `parse(message, turn, context) -> IntentUpdate | None`.
- Env flags, e.g. `INTENT_SEMANTIC=off|local|api` (default `off`).
- JSON schema validation + span grounding + override/negation rules.
- Timeouts and credential misses return `None` (caller falls back).
- Mocked tests first: invalid schema, hallucinated constraint, timeout, token accounting.

**Gate:** with mocks, fallback always fires on bad output. No live API in CI.

### Phase 5 — arbitration + agent wiring

- Phrase-first hybrid as default when `INTENT_SEMANTIC != off`.
- Sum token usage.
- Log `parser=` in `MemoryStore` events (already present) and keep it for ablations.

**Gate:** public templated eval ≈ Phase 0 score (regression budget: no HitRate drop). Paraphrase fixture: fewer override misses than phrase-only.

### Phase 6 — measure and choose the submission default

Compare, by scenario:

- phrase-only
- phrase + Phase 2 rules
- phrase-first + router
- phrase-first + generative parser
- semantic-first (diagnostic only)

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

Ship **phrase-first hybrid**, with Phase 2 rules always on, a local dialogue-act router if embeddings stay small, and a schema-constrained generative parser **off by default** until Phase 6 numbers justify enabling it.

That order matches the scoring surface: keep templated HitRate, stop override/no-pref disasters under paraphrase, preserve raw catalog wording for BM25, and only then spend tokens.

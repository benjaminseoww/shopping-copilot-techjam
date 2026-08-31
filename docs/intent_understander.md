# Intent Understander

`IntentUnderstander` turns each customer message into a small update for session memory. It identifies:

- what the customer is doing: buying, browsing, adding details, declining a preference, or replacing an earlier preference;
- the product category, when the message names one;
- raw requirement text such as `cotton` or `color: blue`; and
- a broad attribute label such as `material`, `color`, or `budget`.

It does not choose products or modify memory directly.

## Techniques used

The parser is fully offline. Official evaluator wording is one realization of an act, not a special first pass. Span extractors always copy text from the message; they do not depend on how the act was chosen.

### 1. Act classification

Each turn is labeled with a single act, in this order:

1. **NLI entailment** (optional, preferred) — a local DeBERTa-v3-xsmall cross-encoder scores the message against speech-act paraphrases and keeps the act with the highest P(entailment) when score and margin clear a threshold *and* the existing extractors can fill that act (replacement span for override, item or requirement span for buying, an attribute name in the message for decline/exhaust, and so on). A stall cue such as `keep looking` cannot be relabeled as a buy. The classifier never emits `reject` and never rewrites constraint text.
2. **Untrained MiniLM prototypes** (optional fallback) — used when NLI weights are missing: embed the message and pick the nearest act paraphrase with the same extractor fill checks.
3. **Cue fallback** — used when both models are missing, confidence is low, or the proposed act cannot be filled: speech-act wrappers (looking-for, still exploring, what-matters), then stall, replace, exhaust, decline, ask-me, negation, a short value reply bound to the last asked attribute, and conservative fallback.

A word such as `actually` is not enough to replace preferences. A replacement value must also be found. Words inside a copied catalog bullet (`without`, `forget`, `instead`) do not change the speech act: `For that, what matters is: …` stays a clarification even when the product text contains those words.

Lone `need` / `want` is not enough to classify a stall as buying: `I need to think` stays a no-op, while `I need a rain jacket` is a buy. Hyphens in category names (`Button-Down`) are part of the item span.

### 2. Span extraction

Category and requirement values are copied from the customer message. They are not rewritten.

The same extractors cover official public-set wording and paraphrases:

- item: after `looking for`, `show me`, `need` / `want`, or `exploring`;
- hard requirement: after `key requirement is`, `must be` / `have to be`, or `is required`;
- a following statement after `looking for X.` with no requirement cue is a provisional opener (Intent Override's first turn);
- clarification values: after `what matters is` / `important part is`, split on `;` or `, and also`;
- replacement: after `what I need is`, `make it`, `go with`, or the clause following `ignore` / `forget` / `scratch that`.

For example:

```text
Need running shoes — cotton is required.
```

produces category `running shoes` and requirement `cotton`, the same slots as `I'm looking for running shoes. A key requirement is: cotton.`

Keeping the original wording helps lexical product search because customer requirements often come directly from catalog text.

### 3. Attribute labels

Exact whole-word lists still catch common values (`cotton`, `blue`, `$25`). They are a fast path, not the definition of the attribute: unknown text is kept as raw search evidence.

When MiniLM is loaded, leftover text is labeled by nearest definitional prototype (`navy` → `color`, `suede` → `material`). Prototypes describe the attribute, not the evaluator's closed vocab. If embeddings are missing, unknown values stay `feature`.

### 4. Turn context

Short replies can use the attribute asked on the previous turn. If the agent asked about `material` and the customer says `It doesn't matter`, the parser records no preference for `material`.

### 5. Negation safety

Negative requirements are not converted into positive search terms. For example, `I don't want leather` does not add `leather` to the retrieval query.

The current retrieval state has no exclusion field, so dropping unsupported negative constraints is safer than searching for the rejected value.

### 6. Conservative fallback

Unknown messages have conversational filler removed, and only the remaining product-like words are retained. Fallback never clears previous preferences. Negated messages still skip fallback terms.

## Output

The parser returns an `IntentUpdate`. Memory then:

- keeps the category across later turns;
- accumulates positive requirements;
- records declined or exhausted attributes; and
- moves old requirements aside when a valid replacement is detected.

A looking-for message whose second sentence is not a requirement cue is marked `source=initial_provisional` so the question engine keeps asking a typed attribute until a confirmed constraint exists.

Updates are marked with `parser="phrase"` or `"fallback"` for debugging.

## Limitations

Rules still cannot understand every paraphrase or subtle sentence. Ambiguous phrases may be missed, and unsupported negative constraints are not represented as exclusions.

MiniLM attribute prototypes can still confuse close families (a metal color vs a metal material). Gazetteer hits stay authoritative when they fire. NLI act labels can still confuse close families (decline vs exhaust); extractors and stall guards reject labels they cannot fill.

A schema-constrained LLM, if added later, should emit the same `IntentUpdate`, copy spans from the message, and fall back to this parser.

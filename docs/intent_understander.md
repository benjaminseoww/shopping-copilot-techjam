# Intent Understander

`IntentUnderstander` turns each customer message into a small update for session memory. It identifies:

- what the customer is doing: buying, browsing, adding details, declining a preference, or replacing an earlier preference;
- the product category, when the message names one;
- raw requirement text such as `cotton` or `color: blue`; and
- a broad attribute label such as `material`, `color`, or `budget`.

It does not choose products or modify memory directly.

## Techniques used

The parser is fully offline and uses no model tokens. Official evaluator wording is one realization of an act, not a special first pass.

### 1. Act classification

Each turn is labeled with a single act. Precedence is:

1. stall (`keep looking`, `need to think`) → no update;
2. replace an earlier preference (`ignore` / `forget` / `scratch that` / `instead`, plus a replacement value);
3. no further preference on an attribute (`additional preference`, `that's all`, `nothing more on`);
4. decline the asked attribute (`no preference`, `doesn't matter`, `you pick`);
5. ask the agent to question a specific field;
6. browse (`still exploring`, `just browsing`, `not sure yet`);
7. open a buy (`looking for`, `show me`, `must be`, `key requirement`, `have to be`);
8. answer a clarification (`what matters is`, `important part is`);
9. negation that is not a buy or replace → drop, do not search for the rejected value;
10. short value reply bound to the last asked attribute;
11. conservative fallback.

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

### 3. Attribute gazetteers

Small word lists label requirement text:

- `cotton`, `leather`, `wool` → `material`;
- `blue`, `red`, `black` → `color`;
- `$25`, `under 50` → `budget`; and
- `hiking`, `running`, `winter` → `use_case`.

Matching uses whole words, so `red` does not accidentally match inside `required`. These lists are classifiers, not enums: unknown values are still retained as raw text.

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

A future schema-constrained LLM should emit the same `IntentUpdate`, copy spans from the message, and fall back to this offline parser.

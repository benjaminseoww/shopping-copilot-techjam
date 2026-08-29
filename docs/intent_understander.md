# Intent Understander

`IntentUnderstander` turns each customer message into a small update for session memory. It identifies:

- what the customer is doing: buying, browsing, adding details, declining a preference, or replacing an earlier preference;
- the product category, when the message names one;
- raw requirement text such as `cotton` or `color: blue`; and
- a broad attribute label such as `material`, `color`, or `budget`.

It does not choose products or modify memory directly.

## Techniques used

The parser is fully offline and uses no model tokens.

### 1. Exact templates

Known evaluator messages are matched first, as a single `(regex, handler)` table. These cover buying, browsing, clarification, no-preference, exhaustion, and preference replacement.

Examples:

- `I'm looking for Shoes. A key requirement is: cotton.`
- `I'm looking for Shoes, but I'm still exploring.`
- `Actually, ignore my earlier preference. What I need is: leather.`

Keeping these matches first preserves the deterministic MVP behavior.

### 2. Paraphrase policy

If no exact template matches, one cue policy runs with a fixed precedence. Override wins mixed messages when a replacement span is present:

- `ignore`, `forget`, `instead`, `change my mind` plus a replacement value → replace the old preference;
- `nothing more on` → no additional preference;
- `doesn't matter`, `you pick`, `no preference` → no preference;
- `ask me about` an attribute → clarification request;
- `just browsing`, `just exploring` → browsing;
- `important part is`, `priority is` → clarification;
- `must be`, `needs to be`, `is required`, `key requirement`, `a must` → buying.

A word such as `actually` is not enough by itself to replace preferences. A replacement value must also be found. Lone `need` / `want` is not enough to classify a turn as buying.

### 3. Raw span extraction

Category and requirement values are copied from the customer message. They are not rewritten.

For example:

```text
Need running shoes — cotton is required.
```

produces category `running shoes` and requirement `cotton`.

Keeping the original wording helps lexical product search because customer requirements often come directly from catalog text.

### 4. Attribute gazetteers

Small word lists label requirement text:

- `cotton`, `leather`, `wool` → `material`;
- `blue`, `red`, `black` → `color`;
- `$25`, `under 50` → `budget`; and
- `hiking`, `running`, `winter` → `use_case`.

Matching uses whole words, so `red` does not accidentally match inside `required`. These lists are classifiers, not enums: unknown values are still retained as raw text.

### 5. Turn context

Short replies can use the attribute asked on the previous turn. If the agent asked about `material` and the customer says `It doesn't matter`, the parser records no preference for `material`.

### 6. Negation safety

Negative requirements are not converted into positive search terms. For example, `I don't want leather` does not add `leather` to the retrieval query.

The current retrieval state has no exclusion field, so dropping unsupported negative constraints is safer than searching for the rejected value.

### 7. Conservative fallback

Unknown messages have conversational filler removed, and only the remaining product-like words are retained. Fallback never clears previous preferences. Negated messages still skip fallback terms.

## Output

The parser returns an `IntentUpdate`. Memory then:

- keeps the category across later turns;
- accumulates positive requirements;
- records declined or exhausted attributes; and
- moves old requirements aside when a valid replacement is detected.

Updates are marked with `parser="phrase"` or `"fallback"` for debugging.

## Limitations

Rules still cannot understand every paraphrase or subtle sentence. Ambiguous phrases may be missed, and unsupported negative constraints are not represented as exclusions.

A future schema-constrained LLM should emit the same `IntentUpdate`, copy spans from the message, and fall back to this offline parser.

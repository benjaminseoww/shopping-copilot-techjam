# TechJam Conversational E-Commerce Search

A multi-turn shopping agent for the TechJam conversational e-commerce search challenge. For each session the agent asks one structured clarification and ranks catalog products until the hidden target appears in the Top 10 or turn 10 ends.

This repository is a **solo** participant submission. It extends the organizer starter: lexical FTS retrieval, a cue-based intent parser, and an optional untrained MiniLM for turn-act prototypes and a light ranking blend.

## Project overview

The agent never sees the hidden intent card or the target `parent_asin`. It only sees an anonymized profile and the simulated customer’s messages.

On each turn it:

1. **Classifies the speech act** (buy, browse, clarify, replace, decline, exhaust, stall). Untrained MiniLM prototypes run first when weights are present; cue rules are the fallback. Requirement text is sliced from the message with regex extractors and is not rewritten.
2. **Updates session memory** — category, accumulated constraints, declined/exhausted attributes, and superseded preferences after an override.
3. **Retrieves** with FTS5 BM25 and RRF (category + constraint routes, rare-term expansion, superseded non-conflicting constraints, sticky previous pool).
4. **Reranks** with phrase, IDF, leaf-category, and typed color/material signals. MiniLM cosine is a light blend when weights load.
5. **Asks** a typed attribute until a confirmed constraint exists, then an open follow-up. While evidence is thin it returns a short list so a speculative Top 10 cannot lock a poor first-hit rank.

Reported local scores on the 200-session public set:

| Path | HitRate@10 | MRR | MTTC | TechnicalScore |
| --- | ---: | ---: | ---: | ---: |
| Lexical (`SHOPPING_SKIP_EMBEDDINGS=1`) | 0.995 | 0.815 | 2.77 | **0.907** |
| MiniLM on (weights in this repo) | 0.995 | 0.824 | 2.76 | **0.910** |

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × clip((11 − MTTC) / 10, 0, 1)
```

## Setup and installation

Python 3.10 or later.

### 1. Catalog

The 50,000-product catalog is not in git. Download `catalog.jsonl.gz` from the GitHub Release, then:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Expect 50,000 lines. Verify against the published `SHA256SUMS` if you have it.

### 2. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-embeddings.txt
```

`numpy`, `onnxruntime`, and `tokenizers` are required for MiniLM. The rest of the agent is the Python standard library.

### 3. MiniLM weights

This submission already includes:

```text
data/minilm/all-MiniLM-L6-v2/tokenizer.json
data/minilm/all-MiniLM-L6-v2/model.onnx
```

No Hugging Face download is needed at scoring time. If those files are missing, the agent stays on the cue parser and lexical ranker. To restore them later:

```bash
python3 -m starter.embedder
```

Optional: `SHOPPING_MINILM_DIR=/path/to/all-MiniLM-L6-v2` to point at a copy elsewhere. `SHOPPING_SKIP_EMBEDDINGS=1` forces the lexical path even when weights exist.

## Steps to reproduce your results

From the repo root, with `data/catalog.jsonl` in place:

```bash
python3 -m unittest discover -s tests -q
python3 -m evaluator.local_evaluator
```

The evaluator writes `results.json`. Do not edit `evaluator/` or `data/public_set.jsonl` when reporting a score.

**Lexical (no MiniLM), public set of 200:**

```bash
SHOPPING_SKIP_EMBEDDINGS=1 python3 -m evaluator.local_evaluator
```

Expected: HitRate@10 `0.995`, MRR `0.815`, MTTC `2.77`, TechnicalScore `0.907`.

**With MiniLM (judge default if weights are present):**

```bash
python3 -m evaluator.local_evaluator
```

Expected: HitRate@10 `0.995`, MRR `0.824`, MTTC `2.76`, TechnicalScore `0.910`.
The first MiniLM run encodes the catalog and caches `data/catalog.minilm.npz` (gitignored). Later runs reuse that cache.

Confirm weights loaded:

```bash
python3 -c "from starter.embedder import default_model_dir, try_load_minilm; e=try_load_minilm(default_model_dir('data/catalog.jsonl')); print('minilm', 'on' if e is not None else 'off')"
```

Prints `minilm on` when `tokenizer.json` and `model.onnx` are in place and embeddings are not skipped.

## Limitations and what we would improve

- **Act vs span.** MiniLM only labels the speech act. Category and requirement strings still come from regex spans. A paraphrase that never hits those extractors cannot add a constraint even if the act is right.
- **Untrained prototypes.** The act model is nearest-neighbor to a handful of paraphrases, not a classifier trained on shopping dialogue. Decline vs exhaust and browse vs buy can still confuse it; extractors and stall guards reject labels they cannot fill.
- **Simulator fit.** Public (and likely private) customer lines are generated from product *features/details*, not titles. Ranking that chases title style or unique title tokens does not match this protocol. A real shopper who names the garment would look different.
- **One leftover miss.** `public_0020` sits in a large novelty-cotton pile; the target ties just outside rank 10. We did not add per-sample hacks.
- **No exclusions.** “I don’t want leather” drops the term rather than encoding a negative. Retrieval has no not-field.
- **Questions are heuristic.** Typed then `other`, shortlist until two constraints or turn 8. There is no information-gain model.
- **MiniLM in ranking** can help paraphrases and can also blur a distinctive lexical phrase. That is why the lexical path remains first-class.

Given more time we would train a small act head on synthetic paraphrases (not the 200 public IDs), add a real exclusion slot in memory/retrieval, and test MiniLM ranking only on the private-like paraphrase regime rather than stacking lexical knobs on the public set.

## Team member contributions

Solo participant: **Benjamin Seow**.

## Agent interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`. Token usage is zero unless a paid API is added; this agent does not call one.

## Data source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md`. MiniLM weights are the public Xenova ONNX conversion of `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0).

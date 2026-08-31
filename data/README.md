# Competition Data

## `public_set.jsonl`

Contains 200 labeled development sessions: 80 Buying, 80 Browsing, 30 Intent Override, and 10 Boundary sessions.

Each session contains a safe aggregate `user_profile` and public labels for local development. Direct user identifiers, timestamps, free-text reviews, raw purchase history, hidden intent cards, and simulator-policy internals are not shipped in this participant file.

## `catalog.jsonl`

Download `catalog.jsonl.gz` from the GitHub Release and decompress it as `catalog.jsonl` in this directory. Expected row count: 50,000.

## `minilm/` and `nli/`

Offline ONNX weights are committed here so a clone can score without Hugging Face:

- `minilm/all-MiniLM-L6-v2/` — ranking and attribute prototypes
- `nli/nli-deberta-v3-xsmall/` — speech-act entailment

Never place API keys, private evaluation data, or participant outputs in this directory.

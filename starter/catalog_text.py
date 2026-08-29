from __future__ import annotations

ROOT_FRAGMENTS = (
    "clothing, shoes & jewelry",
    "clothing shoes & jewelry",
    "shoes & jewelry",
)


def strip_root(text: str) -> str:
    lowered = text.lower()
    for fragment in ROOT_FRAGMENTS:
        lowered = lowered.replace(fragment, " ")
    return lowered


def _field_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def product_snippet(product: dict) -> str:
    """Compact catalog text for question-time attribute extraction."""
    return strip_root(
        " ".join(
            [
                _field_text(product.get("title")),
                _field_text(product.get("categories")),
                _field_text(product.get("features")),
                _field_text(product.get("details")),
                _field_text(product.get("store")),
            ]
        )
    )[:4000]

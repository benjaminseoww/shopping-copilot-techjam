"""Whole-word gazetteers used to label requirement text."""

from __future__ import annotations

import re

from .models import AttributeName

MATERIALS = (
    "cotton",
    "polyester",
    "nylon",
    "leather",
    "wool",
    "spandex",
    "silk",
    "rayon",
    "fabric",
)
COLORS = (
    "black",
    "white",
    "blue",
    "red",
    "pink",
    "green",
    "brown",
    "gray",
    "grey",
    "purple",
    "yellow",
    "orange",
)
SIZE_VALUES = ("wide", "narrow", "small", "medium", "large", "xl")
STYLE_VALUES = ("women", "men", "unisex", "sleeve", "neck", "fit")

MATERIAL_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(value) for value in MATERIALS)})\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(value) for value in COLORS)})\b",
    re.IGNORECASE,
)
LABEL_PREFIX_RE = re.compile(
    r"^(?:color|colour|material|size|style|brand|feature|use[_ ]case)\s*:\s*",
    re.IGNORECASE,
)
BOILERPLATE_CONSTRAINT_RE = re.compile(
    r"^(?:imported|machine wash(?:able)?|hand wash(?: only)?|dry clean(?: only)?|"
    r"(?:button|pull on|zipper|tie|snap|hook[ -]and[ -]loop) closure)$",
    re.IGNORECASE,
)
COLOR_CANON = {"grey": "gray"}
SIZE_RE = re.compile(r"\b(?:size|sizing|width|wide|narrow)\b", re.IGNORECASE)
STYLE_RE = re.compile(
    r"\b(?:department|style|fit|sleeve|neck)\b",
    re.IGNORECASE,
)
USE_CASE_RE = re.compile(
    r"\b(?:hiking|running|gym|winter|outdoor|work)\b",
    re.IGNORECASE,
)

ATTRIBUTE_NAMES: tuple[AttributeName, ...] = (
    "category",
    "material",
    "color",
    "size",
    "style",
    "brand",
    "budget",
    "feature",
    "use_case",
    "other",
)


def strip_constraint_label(text: str) -> str:
    """Drop 'color:' / 'material:' prefixes so search uses the value, not the label."""
    return LABEL_PREFIX_RE.sub("", text or "").strip()


def is_boilerplate_constraint(text: str) -> bool:
    """True for generic catalog care/closure/import lines that do not identify a product."""
    needle = re.sub(r"\s+", " ", strip_constraint_label(text)).strip(" \t\n.,;:!?—-").lower()
    return bool(needle and BOILERPLATE_CONSTRAINT_RE.match(needle))


def first_material(text: str) -> str | None:
    match = MATERIAL_RE.search(text or "")
    return match.group(0).lower() if match else None


def first_color(text: str) -> str | None:
    match = COLOR_RE.search(text or "")
    if not match:
        return None
    value = match.group(0).lower()
    return COLOR_CANON.get(value, value)

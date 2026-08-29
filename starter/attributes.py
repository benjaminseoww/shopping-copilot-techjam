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

MATERIAL_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(value) for value in MATERIALS)})\b",
    re.IGNORECASE,
)
COLOR_RE = re.compile(
    rf"\b(?:{'|'.join(re.escape(value) for value in COLORS)})\b",
    re.IGNORECASE,
)
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

"""Extract structured fields from a raw user description before retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class ParsedInput:
    item_name: str
    quality: str | None          # assessed (possibly corrected) condition
    quality_note: str | None     # explanation if condition was corrected from what seller stated
    additional_info: str | None  # brand, model, size, colour, defects, etc.
    suggested_category_id: str | None
    search_query: str            # optimised query using assessed (not stated) quality
    assessed_price_range: tuple[float, float] | None  # Gemini's price estimate before seeing comparables


_EXTRACTION_SCHEMA = {
    "item_name": "string",
    "quality": "new|like-new|good|fair|poor",
    "quality_note": "string or null — only if you corrected the seller-stated condition based on the image",
    "additional_info": "string or null — brand, model, size, defects",
    "suggested_category_name": "string or null — best match from the available categories",
    "search_query": "2-6 keywords for similarity search using assessed quality",
    "assessed_price_range": ["min (₴) — your estimate before seeing market data", "max (₴)"],
}

_EXTRACTION_PROMPT_TMPL = """\
You are a product assessment assistant for a Ukrainian second-hand marketplace.
Analyze the product image (if provided) AND the seller's description.
If the seller's stated condition does not match what is visible in the image, correct it and explain why.
Available categories: {categories}

Seller description:
{description}

Return ONLY valid JSON matching this schema:
{schema}
"""


def _build_extraction_prompt(description: str, available_categories: dict[str, str]) -> str:
    cat_list = ", ".join(sorted(available_categories.values()))
    return _EXTRACTION_PROMPT_TMPL.format(
        categories=cat_list,
        description=description,
        schema=json.dumps(_EXTRACTION_SCHEMA, ensure_ascii=False, indent=2),
    )


def _category_id_for_name(
    name: str | None, available_categories: dict[str, str]
) -> str | None:
    """Case-insensitive name → id lookup."""
    if not name:
        return None
    name_lower = name.strip().lower()
    for cid, cname in available_categories.items():
        if cname.lower() == name_lower:
            return cid
    return None


class InputParser:
    """Parses raw user input into structured fields using the LLM provider."""

    def __init__(self, provider) -> None:
        self._provider = provider

    def parse(
        self, description: str, available_categories: dict[str, str], image_url: str | None = None
    ) -> ParsedInput:
        prompt = _build_extraction_prompt(description, available_categories)
        result = self._provider.extract(prompt, image_url=image_url)

        category_id = _category_id_for_name(
            result.get("suggested_category_name"), available_categories
        )

        search_query = result.get("search_query") or description
        item_name = result.get("item_name") or description[:60]

        raw_range = result.get("assessed_price_range")
        try:
            assessed_price_range = (float(raw_range[0]), float(raw_range[1])) if raw_range else None
        except (TypeError, ValueError, IndexError):
            assessed_price_range = None

        return ParsedInput(
            item_name=item_name,
            quality=result.get("quality"),
            quality_note=result.get("quality_note"),
            additional_info=result.get("additional_info"),
            suggested_category_id=category_id,
            search_query=search_query,
            assessed_price_range=assessed_price_range,
        )

"""Builds the prompt sent to the LLM."""

import json


SYSTEM_ROLE = (
    "You are a pricing expert for Monobazar, a Ukrainian second-hand marketplace. "
    "Your job is to recommend fair, evidence-based prices for items sellers list. "
    "Analyze the provided image and description carefully. "
    "Always respond with valid JSON only — no markdown, no extra text."
)

OUTPUT_SCHEMA = {
    "_rules": "All price fields must be plain integers or floats — no ₴ symbol, no ranges, no strings.",
    "recommended_price": 1500,
    "price_range": [1200, 1800],
    "condition_assessment": "new|like-new|good|fair|poor",
    "strategies": [
        {
            "name": "Quick Sale",
            "emoji": "⚡",
            "price": 1100,
            "listing_price": 1150,
            "days_to_sell": "1–5 days",
            "sale_probability": 0.85,
            "bargain_advice": "specific advice for this item",
            "explanation": "reference specific comparables or item features",
        },
        {
            "name": "Balanced",
            "emoji": "⚖️",
            "price": 1500,
            "listing_price": 1600,
            "days_to_sell": "7–14 days",
            "sale_probability": 0.65,
            "bargain_advice": "specific advice",
            "explanation": "reference comparables",
        },
        {
            "name": "Max Profit",
            "emoji": "💰",
            "price": 2000,
            "listing_price": 2200,
            "days_to_sell": "21–45 days",
            "sale_probability": 0.40,
            "bargain_advice": "specific advice",
            "explanation": "reference comparables",
        },
    ],
    "evidence": ["each point referencing specific comparables or category stats"],
}


def build_prompt(
    description: str,
    category_name: str | None,
    category_stats: dict | None,
    comparables: list[dict],
    item_name: str | None = None,
    quality: str | None = None,
    additional_info: str | None = None,
    assessed_price_range: tuple[float, float] | None = None,
) -> str:
    parts = [SYSTEM_ROLE, ""]

    if category_name:
        parts.append(f"## Category\n{category_name}\n")

    if item_name or quality or additional_info or assessed_price_range:
        parts.append("## Item Assessment (from image analysis)")
        if item_name:
            parts.append(f"- Item: {item_name}")
        if quality:
            parts.append(f"- Condition: {quality}")
        if additional_info:
            parts.append(f"- Details: {additional_info}")
        if assessed_price_range:
            parts.append(f"- Pre-market price estimate: ₴{assessed_price_range[0]:,.0f} – ₴{assessed_price_range[1]:,.0f}")
        parts.append("")

    if category_stats:
        parts.append("## Category Market Data")
        parts.append(f"- Listings analysed: {category_stats.get('total_listings', '?')}")
        parts.append(f"- Sell rate: {round(category_stats.get('sell_rate', 0) * 100)}%")
        parts.append(
            f"- Price distribution (sold): "
            f"P20=₴{category_stats.get('price_p20')} "
            f"| P50=₴{category_stats.get('price_p50')} "
            f"| P85=₴{category_stats.get('price_p85')}"
        )
        parts.append(f"- Median days to sell: {category_stats.get('median_days_to_sell', '?')}")
        parts.append(f"- Avg bargain discount: {category_stats.get('avg_bargain_discount_pct', 0)}%\n")

    if comparables:
        parts.append("## Comparable Listings")
        for i, item in enumerate(comparables, 1):
            price_info = (
                f"sold for ₴{item['sold_price']}"
                if item.get("sold_price")
                else f"listed at ₴{item['original_price']} ({item['status']})"
            )
            parts.append(f"{i}. {item['title']} — {price_info}")
        parts.append("")

    parts.append("## Item to Price")
    parts.append(description)
    parts.append("")
    parts.append("## Required Output (JSON only)")
    parts.append(json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, indent=2))

    return "\n".join(parts)
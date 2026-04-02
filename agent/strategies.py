"""Compute the three pricing strategies from category stats."""

from dataclasses import dataclass


@dataclass
class Strategy:
    name: str
    emoji: str
    price: float
    listing_price: float
    days_to_sell: str
    sale_probability: float
    bargain_advice: str
    explanation: str


def compute_strategies(
    category_stats: dict,
    recommended_price: float,
) -> list[Strategy]:
    """
    Derives three strategies anchored to the LLM's item-specific recommended_price.
    Category stats provide sell rate, days-to-sell, and bargain metadata only.
    """
    avg_bargain = category_stats.get("avg_bargain_discount_pct", 0)
    sell_rate = category_stats.get("sell_rate", 0)
    median_days = category_stats.get("median_days_to_sell", 14)

    quick_price  = round(recommended_price * 0.80)
    balanced_price = round(recommended_price)
    max_price    = round(recommended_price * 1.25)

    def listing(price: float) -> float:
        return round(price * (1 + avg_bargain / 100))

    return [
        Strategy(
            name="Quick Sale",
            emoji="⚡",
            price=quick_price,
            listing_price=listing(quick_price),
            days_to_sell="1–5 days",
            sale_probability=min(sell_rate * 1.4, 0.95),
            bargain_advice=f"Expect ~{round(avg_bargain)}% negotiation" if avg_bargain else "Little negotiation expected",
            explanation=f"20% below assessed value — moves fast",
        ),
        Strategy(
            name="Balanced",
            emoji="⚖️",
            price=balanced_price,
            listing_price=listing(balanced_price),
            days_to_sell=f"7–{round(median_days or 14)} days",
            sale_probability=sell_rate,
            bargain_advice=f"~{round(avg_bargain)}% bargaining common" if avg_bargain else "Moderate negotiation",
            explanation="At assessed market value — best trade-off for most sellers",
        ),
        Strategy(
            name="Max Profit",
            emoji="💰",
            price=max_price,
            listing_price=listing(max_price),
            days_to_sell="21–45 days",
            sale_probability=max(sell_rate * 0.6, 0.1),
            bargain_advice="Buyers will negotiate — budget room",
            explanation="25% above assessed value — patience required",
        ),
    ]

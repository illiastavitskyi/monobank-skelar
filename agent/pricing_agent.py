"""Orchestrator: ties together RAG, prompt, LLM, and strategy computation."""

from collections import Counter
from dataclasses import dataclass

import config
from retrieval.retrieval_engine import RetrievalEngine
from llm.provider import LLMProvider, PricingResult, get_provider
from llm.prompts import build_prompt
from agent.strategies import Strategy
from agent.input_parser import InputParser, ParsedInput


@dataclass
class RecommendationResponse:
    strategies: list[Strategy]
    llm_result: PricingResult
    category_name: str | None
    category_source: str | None  # "user" | "inferred" | None
    comparables: list[dict]
    parsed_input: ParsedInput


def _infer_category(comparables: list[dict]) -> str | None:
    """Return the most common category_id among search results."""
    ids = [c["category_id"] for c in comparables if c.get("category_id")]
    if not ids:
        return None
    return Counter(ids).most_common(1)[0][0]


class PricingAgent:
    def __init__(self, engine: RetrievalEngine | None = None, provider: LLMProvider | None = None):
        self.engine = engine or RetrievalEngine()
        self.provider = provider or get_provider()
        self._parser = InputParser(self.provider)
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.engine.load()
            self._loaded = True

    def recommend(
        self,
        description: str,
        image_url: str | None,
        category_id: str | None = None,
    ) -> RecommendationResponse:
        import time
        self._ensure_loaded()

        # Step 1: Gemini — extract features + assess condition + price estimate from image
        t0 = time.perf_counter()
        parsed = self._parser.parse(description, self.engine.category_dict, image_url=image_url)
        print(f"[timer] call 1 (extract): {time.perf_counter() - t0:.2f}s")

        effective_category_id = category_id or parsed.suggested_category_id
        category_source = "user" if category_id else ("inferred" if effective_category_id else None)
        category_name = (
            self.engine.category_dict.get(str(effective_category_id)) if effective_category_id else None
        )
        category_stats = (
            self.engine.get_category_stats(str(effective_category_id)) if effective_category_id else None
        )

        # Step 2: Build rich query from all extracted features + category context, single search
        query_parts = []
        if category_name:
            query_parts.append(category_name)
        llm_cat = parsed.suggested_category_id
        if llm_cat and llm_cat != effective_category_id:
            llm_cat_name = self.engine.category_dict.get(str(llm_cat))
            if llm_cat_name:
                query_parts.append(llm_cat_name)
        query_parts.append(parsed.item_name)
        if parsed.quality:
            query_parts.append(parsed.quality)
        if parsed.additional_info:
            query_parts.append(parsed.additional_info)
        rich_query = " ".join(query_parts)

        t1 = time.perf_counter()
        comparables = self.engine.find_similar(rich_query, top_k=config.RAG_TOP_K)
        print(f"[timer] search: {time.perf_counter() - t1:.2f}s")

        # Step 3: Gemini — pricing from extracted features + comparables (text only, no image)
        prompt = build_prompt(
            description=description,
            category_name=category_name,
            category_stats=category_stats,
            comparables=comparables,
            item_name=parsed.item_name,
            quality=parsed.quality,
            additional_info=parsed.additional_info,
            assessed_price_range=parsed.assessed_price_range,
        )

        t2 = time.perf_counter()
        llm_result = self.provider.analyze(image_url=None, prompt=prompt)
        print(f"[timer] call 2 (pricing): {time.perf_counter() - t2:.2f}s")

        strategies = [
            Strategy(
                name=s["name"],
                emoji=s.get("emoji", ""),
                price=float(s["price"]),
                listing_price=float(s["listing_price"]),
                days_to_sell=s["days_to_sell"],
                sale_probability=float(s["sale_probability"]),
                bargain_advice=s["bargain_advice"],
                explanation=s["explanation"],
            )
            for s in llm_result.strategies
        ]

        return RecommendationResponse(
            strategies=strategies,
            llm_result=llm_result,
            category_name=category_name,
            category_source=category_source,
            comparables=comparables,
            parsed_input=parsed,
        )

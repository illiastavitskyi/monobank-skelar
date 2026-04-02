from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PricingResult:
    recommended_price: float
    price_range: tuple[float, float]
    strategies: list[dict]       # list of 3 strategy dicts
    evidence: list[str]
    condition_assessment: str
    raw: dict                    # full LLM response for debugging


class LLMProvider(ABC):
    @abstractmethod
    def analyze(self, image_url: str | None, prompt: str) -> PricingResult:
        ...

    @abstractmethod
    def extract(self, prompt: str, image_url: str | None = None) -> dict:
        """Extract structured fields from description + optional image. Returns parsed dict."""
        ...


class MockProvider(LLMProvider):
    """Returns hardcoded data — no API needed. Used for dev/testing."""

    def extract(self, prompt: str, image_url: str | None = None) -> dict:
        return {
            "item_name": "Sample item",
            "quality": "good",
            "quality_note": None,
            "additional_info": None,
            "suggested_category_name": None,
            "search_query": "sample item good condition",
        }

    def analyze(self, image_url: str | None, prompt: str) -> PricingResult:
        return PricingResult(
            recommended_price=1500.0,
            price_range=(1000.0, 2200.0),
            strategies=[
                {
                    "name": "⚡ Quick Sale",
                    "price": 1100.0,
                    "listing_price": 1150.0,
                    "days_to_sell": "1–5",
                    "sale_probability": 0.85,
                    "bargain_advice": "Expect 5% negotiation",
                    "explanation": "Priced to move — below 70% of similar listings",
                },
                {
                    "name": "⚖️ Balanced",
                    "price": 1500.0,
                    "listing_price": 1600.0,
                    "days_to_sell": "7–14",
                    "sale_probability": 0.65,
                    "bargain_advice": "Some negotiation expected",
                    "explanation": "Near market median — good value signal",
                },
                {
                    "name": "💰 Max Profit",
                    "price": 2000.0,
                    "listing_price": 2200.0,
                    "days_to_sell": "21–45",
                    "sale_probability": 0.40,
                    "bargain_advice": "Buyers will negotiate hard",
                    "explanation": "Top 25% of category — patience required",
                },
            ],
            evidence=["Mock: 7 similar items sold for ₴800–₴2000", "Mock: condition assessed as good"],
            condition_assessment="good (mock)",
            raw={"mock": True},
        )


class GeminiProvider(LLMProvider):
    MODEL = "gemini-2.5-flash-lite"

    def __init__(self, api_key: str):
        from google import genai
        self._client = genai.Client(api_key=api_key)

    @staticmethod
    def _to_float(value) -> float:
        """Convert LLM price value to float, handling '₴4500', '950 - 1300', etc."""
        if isinstance(value, (int, float)):
            return float(value)
        import re
        cleaned = str(value).replace("₴", "").replace(",", "").strip()
        # range string like "950 - 1300" → take midpoint
        parts = re.split(r"\s*[\–\-—]\s*", cleaned)
        if len(parts) == 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2
            except ValueError:
                pass
        return float(cleaned)

    @staticmethod
    def _parse_json_response(text: str) -> dict:
        import json
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)

    def extract(self, prompt: str, image_url: str | None = None) -> dict:
        import httpx
        from PIL import Image
        import io

        parts = []
        if image_url:
            if image_url.startswith("http"):
                raw = httpx.get(image_url).content
            else:
                with open(image_url, "rb") as f:
                    raw = f.read()
            parts.append(Image.open(io.BytesIO(raw)))
        parts.append(prompt)
        response = self._client.models.generate_content(model=self.MODEL, contents=parts)
        return self._parse_json_response(response.text)

    def analyze(self, image_url: str | None, prompt: str) -> PricingResult:
        import httpx
        from PIL import Image
        import io

        parts = []
        if image_url:
            if image_url.startswith("http"):
                raw = httpx.get(image_url).content
            else:
                with open(image_url, "rb") as f:
                    raw = f.read()
            parts.append(Image.open(io.BytesIO(raw)))
        parts.append(prompt)
        response = self._client.models.generate_content(model=self.MODEL, contents=parts)
        data = self._parse_json_response(response.text)
        f = self._to_float
        pr = data["price_range"]
        price_range = (f(pr[0]), f(pr[1])) if isinstance(pr, list) else (f(pr), f(pr))
        return PricingResult(
            recommended_price=f(data["recommended_price"]),
            price_range=price_range,
            strategies=data["strategies"],
            evidence=data["evidence"],
            condition_assessment=data["condition_assessment"],
            raw=data,
        )


def get_provider() -> LLMProvider:
    from config import LLM_PROVIDER, GEMINI_API_KEY

    if LLM_PROVIDER == "gemini":
        return GeminiProvider(api_key=GEMINI_API_KEY)
    return MockProvider()
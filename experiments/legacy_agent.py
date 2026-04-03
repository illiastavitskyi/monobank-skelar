"""
Legacy pipeline methods — superseded by PricingAgent.analyze() in app/agent.py.
predict() used Gemini LLM text generation with build_prompt().
predict_without_category() used Gemini Vision to infer the category first.

"""
import json
import logging

import numpy as np
import google.generativeai as genai
from PIL import Image

from app.agent import PricingAgent
from experiments.legacy_prompts import build_prompt


class LegacyPricingAgent(PricingAgent):

    def predict(self, description: str, image_path: str, category_id: int | None = None) -> dict:
        text_features = self._text_features(description)
        visual_price, comparatives, _ = self._knn_lookup(
            self._clip_embedding([image_path]) or []
        )

        tfidf_vec = (
            self.tfidf.transform([description]).toarray()[0]
            if self.tfidf
            else np.zeros(20)
        )
        prices = self._xgboost_prices(text_features, visual_price, category_id or 4, tfidf_vec)

        is_functional = text_features["func_prob_1"] > text_features["func_prob_0"]
        func_text = "повністю справний" if is_functional else "має дефекти / на запчастини"
        has_box = text_features["comp_prob_1"] > text_features["comp_prob_0"]
        comp_text = "у повній комплектації" if has_box else "без оригінального комплекту"
        cosm_idx = int(np.argmax([text_features[f"cosm_prob_{i}"] for i in range(3)]))
        cosm_text = {0: "зі слідами використання", 1: "у гарному стані", 2: "в ідеальному стані"}.get(cosm_idx, "")
        human_explanation = f"Товар {cosm_text}, {func_text}, {comp_text}."

        prompt = build_prompt(
            description=description,
            category_name=f"Category {category_id}",
            category_stats=None,
            comparables=[{"id": a.id, "price": a.sold_price} for a in comparatives[:10]],
            quality=cosm_text,
            additional_info=f"Functionality: {func_text}. Completeness: {comp_text}.",
            assessed_price_range=(prices.fast, prices.max),
        )

        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(response_mime_type="application/json"),
            )
            return json.loads(response.text)
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            return {
                "recommended_price": round(prices.balanced),
                "price_range": [round(prices.fast), round(prices.max)],
                "condition_assessment": cosm_text,
                "strategies": [
                    {"name": "Quick Sale", "emoji": "⚡", "price": round(prices.fast)},
                    {"name": "Balanced", "emoji": "⚖️", "price": round(prices.balanced)},
                    {"name": "Max Profit", "emoji": "💰", "price": round(prices.max)},
                ],
            }

    def predict_without_category(self, description: str, image_path: str) -> dict:
        logging.info("predict_without_category started...")

        try:
            img = Image.open(image_path)
            response = genai.GenerativeModel("gemini-1.5-flash").generate_content(
                [img, "Analyze this image and concisely describe what the item is, its condition, "
                      "and any notable features. Answer in Ukrainian."]
            )
            vision_desc = response.text
        except Exception as e:
            vision_desc = f"Error calling Gemini Vision: {e}"

        embedding = self._clip_embedding([image_path])
        visual_price, comparatives, _ = self._knn_lookup(embedding) if embedding else (1000.0, [], 4)

        prompt = build_prompt(
            description=description,
            category_name=None,
            category_stats=None,
            comparables=[{"id": a.id, "price": a.sold_price} for a in comparatives],
            additional_info=f"Gemini Vision Output: {vision_desc}",
        )

        try:
            response = genai.GenerativeModel("gemini-1.5-pro").generate_content(
                prompt,
                generation_config=genai.GenerationConfig(response_mime_type="application/json"),
            )
            result = json.loads(response.text)
        except Exception as e:
            logging.error(f"Gemini API Error: {e}")
            raise

        result["explanation"] = {
            "text_analysis_conclusion": f"Визначено через Gemini Vision: {vision_desc}",
            "visual_anchor_price": round(visual_price),
            "similar_items": [{"id": a.id, "title": a.title, "sold_price": a.sold_price} for a in comparatives[:5]],
        }
        return result
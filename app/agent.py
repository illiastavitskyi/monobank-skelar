import json
import logging
import time
from collections import Counter
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
import chromadb
import joblib
import pandas as pd
import google.generativeai as genai
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor

from config import (
    ROBERTA_WEIGHTS, XGB_FAST, XGB_BAL, XGB_MAX, TFIDF_PATH,
    CHROMA_PATH, CHROMA_COLLECTION, CHROMA_HOST, CHROMA_PORT, TITLES_DATASET,
    ROBERTA_MODEL_NAME, CLIP_MODEL_NAME,
    GEMINI_API_KEY, GEMINI_MODEL,
    DEVICE, DEFAULT_VISUAL_PRICE, N_KNN_RESULTS, N_ANALOG_DISPLAY, N_TFIDF_FEATURES,
)
from app.prompts import build_vision_prompt
from app.schemas import AnalogItem, AnalysisResult, PriceEstimate, VisionAssessment

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception:
    pass


class MultiTaskRoBERTa(nn.Module):
    """XLM-RoBERTa with three classification heads: cosmetic condition, functional state, completeness."""

    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(ROBERTA_MODEL_NAME)
        hidden_size = self.roberta.config.hidden_size
        self.head_cosmetic = nn.Linear(hidden_size, 3)
        self.head_functional = nn.Linear(hidden_size, 2)
        self.head_completeness = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        pooled = self.roberta(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        return (
            self.head_cosmetic(pooled),
            self.head_functional(pooled),
            self.head_completeness(pooled),
        )


class PricingAgent:
    def __init__(self):
        logging.info("Initializing PricingAgent...")

        self.tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
        self.text_model = MultiTaskRoBERTa()
        self.text_model.load_state_dict(torch.load(ROBERTA_WEIGHTS, map_location=DEVICE, weights_only=True))
        self.text_model.to(DEVICE).eval()

        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to(DEVICE).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

        chroma = (
            chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            if CHROMA_HOST
            else chromadb.PersistentClient(path=CHROMA_PATH)
        )
        self.collection = chroma.get_collection(name=CHROMA_COLLECTION)

        self.model_fast = xgb.XGBRegressor()
        self.model_fast.load_model(XGB_FAST)
        self.model_bal = xgb.XGBRegressor()
        self.model_bal.load_model(XGB_BAL)
        self.model_max = xgb.XGBRegressor()
        self.model_max.load_model(XGB_MAX)

        self.tfidf = joblib.load(TFIDF_PATH) if TFIDF_PATH.exists() else None
        if not self.tfidf:
            logging.warning(f"TF-IDF vectorizer not found at {TFIDF_PATH}, text features will be zeros.")

        self.titles_map: dict[str, str] = {}
        self.category_map: dict[str, int] = {}
        if TITLES_DATASET.exists():
            try:
                df = pd.read_csv(TITLES_DATASET, usecols=["advertisement_id", "title", "category_id"])
                self.titles_map = dict(zip(df["advertisement_id"].astype(str), df["title"].astype(str)))
                self.category_map = dict(zip(df["advertisement_id"].astype(str), df["category_id"]))
                logging.info(f"Loaded {len(self.titles_map)} titles for analog lookup.")
            except Exception as e:
                logging.warning(f"Could not load titles dataset: {e}")

        logging.info("PricingAgent ready.")

    def _text_features(self, text: str) -> dict:
        encoding = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
        )
        with torch.no_grad():
            logits_cosm, logits_func, logits_comp = self.text_model(
                encoding["input_ids"].to(DEVICE),
                encoding["attention_mask"].to(DEVICE),
            )
            p_cosm = F.softmax(logits_cosm, dim=1).cpu().numpy()[0]
            p_func = F.softmax(logits_func, dim=1).cpu().numpy()[0]
            p_comp = F.softmax(logits_comp, dim=1).cpu().numpy()[0]

        return {
            "cosm_prob_0": p_cosm[0], "cosm_prob_1": p_cosm[1], "cosm_prob_2": p_cosm[2],
            "func_prob_0": p_func[0], "func_prob_1": p_func[1],
            "comp_prob_0": p_comp[0], "comp_prob_1": p_comp[1],
        }

    def _clip_embedding(self, image_paths: List[str]) -> list | None:
        embeddings = []
        for path in image_paths:
            try:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                with torch.no_grad():
                    features = self.clip_model.get_image_features(
                        **self.clip_processor(images=image, return_tensors="pt").to(DEVICE)
                    )
                    if hasattr(features, "pooler_output"):
                        features = features.pooler_output
                    features = features / features.norm(p=2, dim=-1, keepdim=True)
                    embeddings.append(features.cpu().numpy()[0])
            except Exception as e:
                logging.error(f"CLIP encoding failed for {path}: {e}")

        if not embeddings:
            return None
        return np.mean(embeddings, axis=0).tolist()

    def _knn_lookup(self, embedding: list) -> tuple[float, List[AnalogItem], int]:
        results = self.collection.query(query_embeddings=[embedding], n_results=N_KNN_RESULTS)
        prices, analogs, categories = [], [], []

        for meta in results.get("metadatas", [[]])[0]:
            price = meta.get("sold_price", 0)
            if price <= 0:
                continue
            adv_id = str(meta.get("advertisement_id", ""))
            if adv_id in self.category_map:
                categories.append(self.category_map[adv_id])
            title = self.titles_map.get(adv_id) or meta.get("title") or f"Схожий товар ({adv_id[:8]})"
            prices.append(price)
            analogs.append(AnalogItem(id=adv_id, title=title, sold_price=price))

        raw = results.get("metadatas", [[]])[0]
        logging.info(f"k-NN raw: {len(raw)} neighbours, sold_prices={[m.get('sold_price') for m in raw]}")
        logging.info(f"k-NN analogs: {[(a.title, a.sold_price) for a in analogs]}")

        competitor_price = float(np.median(prices[:5])) if prices else DEFAULT_VISUAL_PRICE
        category = int(Counter(categories).most_common(1)[0][0]) if categories else 4
        return competitor_price, analogs[:N_ANALOG_DISPLAY], category

    def _xgboost_prices(self, text_feats: dict, visual_price: float, category_id: int, tfidf_vec) -> PriceEstimate:
        features = {
            **text_feats,
            "visual_competitor_price": visual_price,
            **{f"tfidf_{i}": float(tfidf_vec[i]) for i in range(N_TFIDF_FEATURES)},
            "category_id": category_id,
        }
        cols = (
            ["cosm_prob_0", "cosm_prob_1", "cosm_prob_2",
             "func_prob_0", "func_prob_1",
             "comp_prob_0", "comp_prob_1",
             "visual_competitor_price"]
            + [f"tfidf_{i}" for i in range(N_TFIDF_FEATURES)]
            + ["category_id"]
        )
        df = pd.DataFrame([features])[cols]
        return PriceEstimate(
            fast=float(self.model_fast.predict(df)[0]),
            balanced=float(self.model_bal.predict(df)[0]),
            max=float(self.model_max.predict(df)[0]),
        )

    def _gemini_vision_assessment(self, description: str, image_paths: List[str]) -> VisionAssessment:
        pil_images = []
        for path in image_paths:
            try:
                pil_images.append(Image.open(path))
            except Exception:
                pass

        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = model.generate_content(
                pil_images + [build_vision_prompt(description)],
                generation_config=genai.GenerationConfig(response_mime_type="application/json"),
            )
            data = json.loads(response.text.replace("```json", "").replace("```", "").strip())
            return VisionAssessment(
                coefficient=float(data.get("coefficient", 1.0)),
                reason=str(data.get("reason", "")),
            )
        except Exception as e:
            logging.error(f"Gemini vision assessment failed: {e}")
            return VisionAssessment(coefficient=1.0, reason=f"Помилка аналізу: {e}")

    def analyze(self, description: str, image_paths: List[str]) -> AnalysisResult:
        t_start = time.time()

        t0 = time.time()
        embedding = self._clip_embedding(image_paths)
        if embedding is None:
            visual_price, analogs, category = DEFAULT_VISUAL_PRICE, [], 4
        else:
            visual_price, analogs, category = self._knn_lookup(embedding)
        logging.info(f"[1] CLIP + k-NN:      {time.time() - t0:.2f}s  (category={category}, price={visual_price:.0f})")

        t0 = time.time()
        text_feats = self._text_features(description)
        tfidf_vec = (
            self.tfidf.transform([description]).toarray()[0]
            if self.tfidf
            else np.zeros(N_TFIDF_FEATURES)
        )
        logging.info(f"[2] RoBERTa + TF-IDF: {time.time() - t0:.2f}s")

        t0 = time.time()
        prices = self._xgboost_prices(text_feats, visual_price, category, tfidf_vec)
        logging.info(f"[3] XGBoost:          {time.time() - t0:.2f}s  (fast={prices.fast:.0f} bal={prices.balanced:.0f} max={prices.max:.0f})")

        t0 = time.time()
        vision = self._gemini_vision_assessment(description, image_paths)
        logging.info(f"[4] Gemini Vision:    {time.time() - t0:.2f}s  (coeff={vision.coefficient})")

        logging.info(f"    TOTAL:            {time.time() - t_start:.2f}s")

        vision.reason += f" (Категорія: {category})"
        return AnalysisResult(prices=prices, analogs=analogs, vision=vision)
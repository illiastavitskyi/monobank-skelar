import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor
import chromadb
import pandas as pd
import numpy as np
import xgboost as xgb
from PIL import Image
import joblib
import os
import json
from collections import Counter
import google.generativeai as genai
from dotenv import load_dotenv
from llm_prompts import build_prompt

load_dotenv()

try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY", ""))
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# === НАЛАШТУВАННЯ ШЛЯХІВ ===
ROOT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT_DIR / "data"

ROBERTA_WEIGHTS = ROOT_DIR / "roberta_student_weights.pth"
XGB_FAST = ROOT_DIR / "xgb_fast.json"
XGB_BAL = ROOT_DIR / "xgb_bal.json"
XGB_MAX = ROOT_DIR / "xgb_max.json"
TFIDF_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
CHROMA_PATH = str(ROOT_DIR / "chroma_db")

MODEL_NAME = "xlm-roberta-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_VISUAL_PRICE = 1000.0

# === КЛАС ДЛЯ ТЕКСТОВОЇ МОДЕЛІ ===
class MultiTaskRoBERTa(nn.Module):
    def __init__(self):
        super(MultiTaskRoBERTa, self).__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.roberta.config.hidden_size
        self.head_cosmetic = nn.Linear(hidden_size, 3)
        self.head_functional = nn.Linear(hidden_size, 2)
        self.head_completeness = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return (
            self.head_cosmetic(pooled_output),
            self.head_functional(pooled_output),
            self.head_completeness(pooled_output)
        )

class PricingAgent:
    def __init__(self):
        logging.info("Ініціалізація AI-агента...")
        
        # 1. Текстова модель
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.text_model = MultiTaskRoBERTa()
        self.text_model.load_state_dict(torch.load(ROBERTA_WEIGHTS, map_location=DEVICE, weights_only=True))
        self.text_model.to(DEVICE)
        self.text_model.eval()
        
        # 2. Візуальна модель (CLIP)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        
        # 3. ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_collection(name="monobazar_items")
        
        # 4. XGBoost Моделі
        self.model_fast = xgb.XGBRegressor()
        self.model_fast.load_model(XGB_FAST)
        
        self.model_bal = xgb.XGBRegressor()
        self.model_bal.load_model(XGB_BAL)
        
        self.model_max = xgb.XGBRegressor()
        self.model_max.load_model(XGB_MAX)
        
        # 5. TF-IDF Векторизатор (для розуміння характеристик товару)
        if TFIDF_PATH.exists():
            self.tfidf = joblib.load(TFIDF_PATH)
        else:
            logging.warning(f"TFIDF file missing at {TFIDF_PATH}, text feature vectors will be all zeroes.")
            self.tfidf = None
            
        # 6. Мапа реальних назв та категорій товарів
        self.titles_map = {}
        self.category_map = {}
        try:
            titles_path = ROOT_DIR.parent / "hackaton_advertisements_with_id.csv"
            if titles_path.exists():
                titles_df = pd.read_csv(titles_path, usecols=["advertisement_id", "title", "category_id"])
                self.titles_map = dict(zip(titles_df["advertisement_id"].astype(str), titles_df["title"].astype(str)))
                self.category_map = dict(zip(titles_df["advertisement_id"].astype(str), titles_df["category_id"]))
                logging.info(f"Loaded {len(self.titles_map)} titles & categories for analogs.")
        except Exception as e:
            logging.warning(f"Could not load titles map: {e}")
        
        logging.info("Агент готовий до роботи!")

    def extract_text_features(self, text):
        encoding = self.tokenizer(
            str(text), truncation=True, padding='max_length', max_length=128, return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)

        with torch.no_grad():
            logits_cosm, logits_func, logits_comp = self.text_model(input_ids, attention_mask)
            
            probs_cosm = F.softmax(logits_cosm, dim=1).cpu().numpy()[0]
            probs_func = F.softmax(logits_func, dim=1).cpu().numpy()[0]
            probs_comp = F.softmax(logits_comp, dim=1).cpu().numpy()[0]
            
        return {
            "cosm_prob_0": probs_cosm[0], "cosm_prob_1": probs_cosm[1], "cosm_prob_2": probs_cosm[2],
            "func_prob_0": probs_func[0], "func_prob_1": probs_func[1],
            "comp_prob_0": probs_comp[0], "comp_prob_1": probs_comp[1]
        }

    def get_visual_competitor_price(self, image_path, description_fallback="", n_results=15):
        with Image.open(image_path) as f_img:
            image = f_img.convert("RGB")
        
        with torch.no_grad():
            inputs = self.clip_processor(images=image, return_tensors="pt").to(DEVICE)
            features = self.clip_model.get_image_features(**inputs)
            
            # Якщо features це об'єкт (наприклад BaseModelOutputWithPooling), дістаємо тензор
            if not isinstance(features, torch.Tensor):
                if hasattr(features, 'image_embeds'):
                    features = features.image_embeds
                elif hasattr(features, 'pooler_output'):
                    features = features.pooler_output
                else:
                    features = features[0]
                    
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            emb = features.cpu().numpy().tolist()[0]
            
        results = self.collection.query(query_embeddings=[emb], n_results=n_results)
        neighbor_prices = []
        similar_items_info = []
        found_categories = []
        
        if results and results.get("metadatas") is not None:
            for i, meta in enumerate(results["metadatas"][0]):
                price = meta.get("sold_price", 0)
                if price > 0:
                    adv_id = str(meta.get("advertisement_id", ""))
                    
                    if adv_id in self.category_map:
                        found_categories.append(self.category_map[adv_id])
                        
                    neighbor_prices.append(price)
                    
                    real_title = self.titles_map.get(adv_id) or meta.get("title")
                    
                    if real_title:
                        human_title = f"{real_title}"
                    else:
                        short_id = adv_id[:8]
                        human_title = f"Схожий товар ({short_id})"
                        
                    similar_items_info.append({
                        "id": adv_id, 
                        "title": human_title,
                        "sold_price": price
                    })
                
        avg_price = DEFAULT_VISUAL_PRICE
        if neighbor_prices:
            avg_price = float(np.median(neighbor_prices[:5]))
            
        predicted_category = 4
        if found_categories:
            predicted_category = int(Counter(found_categories).most_common(1)[0][0])
            
        return avg_price, similar_items_info[:5], predicted_category

    def analyze_for_frontend(self, description, image_path):
        logging.info("Analyze for frontend started...")
        
        # 1. Base Prices via XGBoost & ChromaDB (k-NN category)
        text_features = self.extract_text_features(description)
        visual_price, comparatives, guessed_category = self.get_visual_competitor_price(image_path, description_fallback=str(description), n_results=20)
        logging.info(f"k-NN predicted category: {guessed_category}")
        
        if self.tfidf:
            text_tfidf = self.tfidf.transform([str(description)]).toarray()[0]
        else:
            text_tfidf = np.zeros(20)
            
        features_dict = {**text_features, 'visual_competitor_price': visual_price}
        for i in range(20):
            features_dict[f'tfidf_{i}'] = float(text_tfidf[i])
            
        features_dict['category_id'] = guessed_category
        
        feature_cols = [
            'cosm_prob_0', 'cosm_prob_1', 'cosm_prob_2',
            'func_prob_0', 'func_prob_1',
            'comp_prob_0', 'comp_prob_1',
            'visual_competitor_price'
        ]
        feature_cols.extend([f'tfidf_{i}' for i in range(20)])
        feature_cols.append('category_id')
        
        df_input = pd.DataFrame([features_dict])[feature_cols]
        
        price_fast = float(self.model_fast.predict(df_input)[0])
        price_bal = float(self.model_bal.predict(df_input)[0])
        price_max = float(self.model_max.predict(df_input)[0])

        # 2. Vision Assessment via Gemini
        vision_result = {"coefficient": 1.0, "reason": "Помилка аналізу. Дефолт."}
        try:
            available_models = ['gemini-3.1-flash-lite-preview']
            response_vision = None
            
            with Image.open(image_path) as img:
                for model_name in available_models:
                    try:
                        logging.info(f"Trying Gemini model: {model_name}")
                        model_vision = genai.GenerativeModel(model_name)
                        
                        prompt_vision = (
                            f"You are an expert appraiser. Read the description: '{str(description)[:500]}' and look at the item photo. "
                            "Find any defects mentioned in the text (like scratches, battery health, missing kit, etc) and combine them with what you see in the photo. "
                            "Give a SINGLE unified condition coefficient from 0.5 (bad/broken) to 1.0 (perfect). "
                            "Output pure JSON STRICTLY in this format, without codeblocks (reason MUST be in Ukrainian):\n"
                            "{\"coefficient\": 0.85, \"reason\": \"На фото видно знос, а в описі вказано про відсутність коробки.\"}"
                        )
                        
                        kwargs = {}
                        if any(v in model_name for v in ['1.5', '2.0', '2.5', '3.1']):
                            kwargs['generation_config'] = genai.GenerationConfig(response_mime_type="application/json")
                            
                        response_vision = model_vision.generate_content([img, prompt_vision], **kwargs)
                        
                        raw_text = response_vision.text.replace("```json", "").replace("```", "").strip()
                        vision_result = json.loads(raw_text)
                        
                        logging.info(f"Successfully used {model_name}")
                        break
                    except Exception as e:
                        logging.warning(f"Failed with {model_name}: {e}")
                        
            if not response_vision:
                raise Exception("Помилка: всі версії моделей Gemini виявилися недоступними (404/API Error).")
                
        except Exception as e:
            logging.error(f"Gemini API Error: {e}")
            vision_result["reason"] = f"Помилка Gemini API: {str(e)}"
            
        return {
            "prices": {
                "fast": price_fast,
                "balanced": price_bal,
                "max": price_max
            },
            "analogs": comparatives,
            "vision": {
                "coefficient": vision_result.get("coefficient", 1.0),
                "reason": vision_result.get("reason", "Дефолтний результат.") + f" (Категорія: {guessed_category})"
            }
        }

    def predict(self, description, image_path, category_id=None):
        # 1. Аналіз стану з тексту (RoBERTa)
        text_features = self.extract_text_features(description)

        # 2. Аналіз фото (CLIP + ChromaDB)
        visual_price, comparatives = self.get_visual_competitor_price(image_path)

        # 3. Аналіз ключових слів (TF-IDF)
        if self.tfidf:
            text_tfidf = self.tfidf.transform([str(description)]).toarray()[0]
        else:
            text_tfidf = np.zeros(20)

        # 4. Формування словника ознак
        features_dict = {**text_features, 'visual_competitor_price': visual_price}
        for i in range(20):
            features_dict[f'tfidf_{i}'] = float(text_tfidf[i])
            
        if category_id is not None:
            features_dict['category_id'] = category_id
            
        # Гарантуємо ПРАВИЛЬНИЙ порядок колонок, як очікує XGBoost
        feature_cols = [
            'cosm_prob_0', 'cosm_prob_1', 'cosm_prob_2',
            'func_prob_0', 'func_prob_1',
            'comp_prob_0', 'comp_prob_1',
            'visual_competitor_price'
        ]
        feature_cols.extend([f'tfidf_{i}' for i in range(20)])
        if category_id is not None:
            feature_cols.append('category_id')
            
        df_input = pd.DataFrame([features_dict])[feature_cols]
        
        # 5. Прогнозування
        price_fast = float(self.model_fast.predict(df_input)[0])
        price_bal = float(self.model_bal.predict(df_input)[0])
        price_max = float(self.model_max.predict(df_input)[0])
        
        # 6. Генерація людської аргументації
        is_functional = text_features['func_prob_1'] > text_features['func_prob_0']
        func_confidence = max(text_features['func_prob_0'], text_features['func_prob_1'])
        func_text = "повністю справний" if is_functional else "має дефекти / на запчастини"

        has_box = text_features['comp_prob_1'] > text_features['comp_prob_0']
        comp_text = "у повній комплектації" if has_box else "без оригінального комплекту"

        cosm_probs = [text_features['cosm_prob_0'], text_features['cosm_prob_1'], text_features['cosm_prob_2']]
        cosm_idx = int(np.argmax(cosm_probs))
        cosm_labels = {0: "зі слідами використання", 1: "у гарному стані", 2: "в ідеальному стані"}
        cosm_text = cosm_labels.get(cosm_idx, "у невідомому стані")

        # Call Gemini (замість Qwen)
        gemini_prompt = build_prompt(
            description=description,
            category_name=f"Category {category_id}",
            category_stats=None,
            comparables=comparatives[:10],
            quality=cosm_text,
            additional_info=f"Functionality: {func_text}. Completeness: {comp_text}.",
            assessed_price_range=(price_fast, price_max)
        )
        
        try:
            model_text = genai.GenerativeModel('gemini-1.5-pro')
            response = model_text.generate_content(
                gemini_prompt,
                generation_config=genai.GenerationConfig(response_mime_type="application/json")
            )
            result = json.loads(response.text)
            
            # Додаємо наші локальні змінні в результат для Streamlit (як було раніше)
            result["explanation"] = {
                "text_analysis_conclusion": human_explanation,
                "visual_anchor_price": round(visual_price),
                "comparatives_found": len(comparatives),
                "similar_items": comparatives
            }
            return result
        except Exception as e:
            logging.error(f"Gemini API error: {e}")
            # Fallback у випадку помилки API
            return {
                "recommended_price": round(price_bal),
                "price_range": [round(price_fast), round(price_max)],
                "condition_assessment": cosm_text,
                "strategies": [
                    {"name": "Quick Sale", "emoji": "⚡", "price": round(price_fast), "explanation": "Fallback Fast"},
                    {"name": "Balanced", "emoji": "⚖️", "price": round(price_bal), "explanation": "Fallback Balanced"},
                    {"name": "Max Profit", "emoji": "💰", "price": round(price_max), "explanation": "Fallback Max"}
                ],
                "explanation": {
                    "text_analysis_conclusion": human_explanation,
                    "visual_anchor_price": round(visual_price),
                    "comparatives_found": len(comparatives),
                    "similar_items": comparatives
                }
            }

    def predict_without_category(self, description, image_path):
        logging.info("Predict without category started...")
        # 1. Gemini Vision to detect category and features
        try:
            model_vision = genai.GenerativeModel('gemini-1.5-flash')
            img = Image.open(image_path)
            prompt_vision = "Analyze this image and concisely describe what the item is, its condition, and any notable features. Answer in Ukrainian."
            response_vision = model_vision.generate_content([img, prompt_vision])
            vision_desc = response_vision.text
        except Exception as e:
            vision_desc = f"Error calling Gemini Vision: {e}"

        # 2. Get comparables from ChromaDB based on image (n=20)
        visual_price, comparatives = self.get_visual_competitor_price(image_path, n_results=20)
        
        # 3. Build Prompt for Gemini LLM
        prompt = build_prompt(
            description=description,
            category_name=None,
            category_stats=None,
            comparables=comparatives,
            additional_info=f"Gemini Vision Output: {vision_desc}"
        )

        # 4. Call Gemini LLM for price prediction
        try:
            model_text = genai.GenerativeModel('gemini-1.5-pro')
            response = model_text.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(response_mime_type="application/json")
            )
            result = json.loads(response.text)
        except Exception as e:
            logging.error(f"Gemini API Error: {e}")
            raise Exception(f"Gemini API Error: {e}")

        # format output identically for app.py
        result["explanation"] = {
            "text_analysis_conclusion": f"Визначено через Gemini Vision: {vision_desc}",
            "visual_anchor_price": round(visual_price),
            "comparatives_found": len(comparatives),
            "similar_items": comparatives[:5]
        }
        return result



if __name__ == "__main__":
    agent = PricingAgent()
    print("Agent is ready!")

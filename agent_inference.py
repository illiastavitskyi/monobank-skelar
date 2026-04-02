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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# === НАЛАШТУВАННЯ ШЛЯХІВ ===
BASE_DIR = Path(__file__).resolve().parent / "data"
ROBERTA_WEIGHTS = Path(__file__).resolve().parent / "roberta_student_weights.pth"
XGB_FAST = BASE_DIR / "xgb_fast.json"
XGB_BAL = BASE_DIR / "xgb_bal.json"
XGB_MAX = BASE_DIR / "xgb_max.json"
TFIDF_PATH = BASE_DIR / "tfidf_vectorizer.pkl"
CHROMA_PATH = str(BASE_DIR / "chroma_db")

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
        self.tfidf = joblib.load(TFIDF_PATH)
        
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

    def get_visual_competitor_price(self, image_path):
        image = Image.open(image_path).convert("RGB")
        with torch.no_grad():
            inputs = self.clip_processor(images=image, return_tensors="pt").to(DEVICE)
            features = self.clip_model.get_image_features(**inputs)
            features /= features.norm(p=2, dim=-1, keepdim=True)
            emb = features.cpu().numpy().tolist()[0]
            
        results = self.collection.query(query_embeddings=[emb], n_results=15)
        neighbor_prices = []
        similar_items_info = []
        
        if results and results.get("metadatas") is not None:
            for meta in results["metadatas"][0]:
                price = meta.get("sold_price", 0)
                if price > 0:
                    neighbor_prices.append(price)
                    similar_items_info.append({"id": meta.get("advertisement_id"), "price": price})
                
        if neighbor_prices:
            avg_price = float(np.median(neighbor_prices[:5]))
            return avg_price, similar_items_info[:5]
        else:
            return DEFAULT_VISUAL_PRICE, []

    def predict(self, description, image_path, category_id=None):
        # 1. Аналіз стану з тексту (RoBERTa)
        text_features = self.extract_text_features(description)

        # 2. Аналіз фото (CLIP + ChromaDB)
        visual_price, comparatives = self.get_visual_competitor_price(image_path)

        # 3. Аналіз ключових слів (TF-IDF)
        text_tfidf = self.tfidf.transform([str(description)]).toarray()[0]

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

        human_explanation = (
            f"🧠 Аналіз тексту: Нейромережа визначила, що товар {func_text} "
            f"(впевненість {func_confidence:.0%}). Візуально він {cosm_text} та продається {comp_text}."
        )

        result = {
            "recommended_price": round(price_bal),
            "price_range": f"{round(price_fast)} - {round(price_max)} грн",
            "strategies": {
                "fast": round(price_fast),
                "balanced": round(price_bal),
                "max_profit": round(price_max)
            },
            "explanation": {
                "text_analysis_conclusion": human_explanation,
                "visual_anchor_price": round(visual_price),
                "comparatives_found": len(comparatives),
                "similar_items": comparatives
            }
        }
        return result


if __name__ == "__main__":
    agent = PricingAgent()
    print("Agent is ready!")

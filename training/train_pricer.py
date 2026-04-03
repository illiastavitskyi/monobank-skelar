import logging

import chromadb
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from config import (
    MODELS_DIR, ASSETS_DIR, MAIN_DATASET,
    CHROMA_PATH, CHROMA_COLLECTION, CHROMA_HOST, CHROMA_PORT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

logging.info("Connecting to ChromaDB")
chroma_client = (
    chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    if CHROMA_HOST
    else chromadb.PersistentClient(path=CHROMA_PATH)
)
collection = chroma_client.get_collection(name=CHROMA_COLLECTION)


def build_features_and_train_xgb():
    logging.info("Loading data")
    df = pd.read_csv(MAIN_DATASET)

    df["sold_price"] = pd.to_numeric(df.get("sold_price", 0), errors="coerce")
    df = df[(df["sold_price"] > 50) & (df["sold_price"] < 200_000)].copy()
    df.reset_index(drop=True, inplace=True) 


    logging.info("Генеруємо текстові ознаки (TF-IDF)...")
    df['description'] = df['description'].fillna("").astype(str)
    
    tfidf = TfidfVectorizer(max_features=20, lowercase=True)
    text_features = tfidf.fit_transform(df['description']).toarray()
    
    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.pkl")
    logging.info("TF-IDF векторизатор збережено.")

    tfidf_col_names = [f'tfidf_{i}' for i in range(20)]
    for i, col_name in enumerate(tfidf_col_names):
        df[col_name] = text_features[:, i]

    visual_prices = [np.nan] * len(df)

    logging.info("Pre-loading embeddings to RAM in chunks (safe mode)...")
    total_vectors = collection.count()
    chunk_size = 5000
    emb_dict = {}
    
    for offset in tqdm(range(0, total_vectors, chunk_size), desc="Вивантаження векторів з БД"):
        chunk = collection.get(include=["embeddings", "metadatas"], limit=chunk_size, offset=offset)
        if chunk and chunk.get("embeddings") is not None and chunk.get("metadatas") is not None:
            for meta, emb in zip(chunk["metadatas"], chunk["embeddings"]):
                adv_id = str(meta["advertisement_id"])
                if adv_id not in emb_dict:
                    if isinstance(emb, np.ndarray):
                        emb = emb.tolist()
                    emb_dict[adv_id] = emb
                    
    logging.info(f"Loaded {len(emb_dict)} unique item embeddings into memory.")

    logging.info("Batch visual prices generation via ChromaDB")
    BATCH_SIZE = 256
    batch_embs = []
    batch_adv_ids = []
    valid_indices = []

    for idx in tqdm(range(len(df)), desc="Пошук компаративів"):
        adv_id = str(df.at[idx, "advertisement_id"])
        
        if adv_id in emb_dict:
            batch_embs.append(emb_dict[adv_id])
            batch_adv_ids.append(adv_id)
            valid_indices.append(idx)
            
        if len(batch_embs) >= BATCH_SIZE or (idx == len(df) - 1 and len(batch_embs) > 0):
            results = collection.query(query_embeddings=batch_embs, n_results=16)
            if results and results.get("metadatas") is not None:
                for i, metadatas in enumerate(results["metadatas"]):
                    q_adv_id = batch_adv_ids[i]
                    real_idx = valid_indices[i]
                    neighbor_prices = []
                    for meta in metadatas:
                        if meta["advertisement_id"] != q_adv_id and meta.get("sold_price", 0) > 0:
                            neighbor_prices.append(meta["sold_price"])
                    
                    if neighbor_prices:
                        avg_price = float(np.median(neighbor_prices[:5]))
                        visual_prices[real_idx] = avg_price
            
            batch_embs = []
            batch_adv_ids = []
            valid_indices = []

    df["visual_competitor_price"] = visual_prices
    median_visual = df["visual_competitor_price"].median()
    df["visual_competitor_price"] = df["visual_competitor_price"].fillna(median_visual)

    logging.info("Processing data for XGB")

    features = [
        'cosm_prob_0', 'cosm_prob_1', 'cosm_prob_2',
        'func_prob_0', 'func_prob_1',
        'comp_prob_0', 'comp_prob_1',
        'visual_competitor_price'
    ]

    features.extend(tfidf_col_names)

    if "category_id" in df.columns:
        features.append("category_id")

    X = df[features]
    y = df['sold_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    xgb_params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "random_state": 42,
        "n_jobs": -1
    }

    logging.info("fitting [Fast] (20-й percentile)")
    model_fast = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.20, **xgb_params)
    model_fast.fit(X_train, y_train)

    logging.info("fitting [Balanced] (50-й percentile)")
    model_bal = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.50, **xgb_params)
    model_bal.fit(X_train, y_train)

    logging.info("fitting [Max Profit] (80-й percentile)")
    model_max = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.80, **xgb_params)
    model_max.fit(X_train, y_train)

    preds_bal = model_bal.predict(X_test)
    mae = mean_absolute_error(y_test, preds_bal)
    logging.info(f"Середня похибка (MAE) збалансованої ціни: {mae:.2f} грн")

    model_fast.save_model(MODELS_DIR / "xgb_fast.json")
    model_bal.save_model(MODELS_DIR / "xgb_bal.json")
    model_max.save_model(MODELS_DIR / "xgb_max.json")
    logging.info("Saved models in JSON!")


if __name__ == "__main__":
    build_features_and_train_xgb()

import logging
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
import pandas as pd
import requests
import torch
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from config import (
    DATA_DIR, MAIN_DATASET, PHOTOS_CSV,
    CLIP_MODEL_NAME, CHROMA_PATH, CHROMA_COLLECTION, CHROMA_HOST, CHROMA_PORT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

BASE_IMG_URL = "https://resale-media.monobazar.com.ua/"
BATCH_SIZE = 64
MAX_PHOTOS_PER_ITEM = 5

session = requests.Session()
session.mount("https://", HTTPAdapter(
    pool_connections=10,
    pool_maxsize=10,
    max_retries=Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504]),
))


def fetch_image(adv_id: str, idx: int, key: str, row) -> tuple:
    try:
        response = session.get(f"{BASE_IMG_URL}{key}", timeout=5)
        if response.status_code != 200:
            return None, None, None

        img = Image.open(BytesIO(response.content)).convert("RGB")

        def safe_float(val) -> float:
            s = str(val).replace(",", ".").replace(" ", "")
            return float(s) if s.lower() != "nan" else 0.0

        metadata = {
            "advertisement_id": str(adv_id),
            "original_price": safe_float(row.get("original_price", 0)),
            "sold_price": safe_float(row.get("sold_price", 0)),
            "sold_via_bargain": bool(row.get("sold_via_bargain", False)),
        }
        return img, f"{adv_id}_{idx}", metadata
    except Exception:
        return None, None, None


def build_vision_db():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

    chroma = (
        chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        if CHROMA_HOST
        else chromadb.PersistentClient(path=CHROMA_PATH)
    )
    collection = chroma.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    logging.info("Loading datasets...")
    df_main = pd.read_csv(MAIN_DATASET)
    grouped_keys = (
        pd.read_csv(PHOTOS_CSV)
        .groupby("advertisement_id")["s3_key"]
        .apply(list)
        .to_dict()
    )
    logging.info(f"Processing {len(df_main)} items...")

    batch_images, batch_ids, batch_metadatas = [], [], []

    def flush_batch():
        if not batch_images:
            return
        with torch.no_grad():
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            features = model.get_image_features(**inputs)
            features /= features.norm(p=2, dim=-1, keepdim=True)
        collection.upsert(
            ids=batch_ids,
            embeddings=features.cpu().numpy().tolist(),
            metadatas=batch_metadatas,
        )
        batch_images.clear()
        batch_ids.clear()
        batch_metadatas.clear()

    for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="Indexing"):
        adv_id = str(row["advertisement_id"])
        keys = grouped_keys.get(adv_id) or grouped_keys.get(int(float(adv_id)), [])
        if not keys:
            continue

        with ThreadPoolExecutor(max_workers=MAX_PHOTOS_PER_ITEM) as executor:
            futures = [
                executor.submit(fetch_image, adv_id, idx, key, row)
                for idx, key in enumerate(keys[:MAX_PHOTOS_PER_ITEM])
            ]
            for future in as_completed(futures):
                img, entry_id, meta = future.result()
                if img:
                    batch_images.append(img)
                    batch_ids.append(entry_id)
                    batch_metadatas.append(meta)

        if len(batch_images) >= BATCH_SIZE:
            flush_batch()

    flush_batch()
    logging.info("Vector DB built successfully.")


if __name__ == "__main__":
    build_vision_db()
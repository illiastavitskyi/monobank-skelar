import logging
from io import BytesIO
from pathlib import Path
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


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


BASE_DIR = Path(__file__).resolve().parent / "data"
MAIN_DATASET = BASE_DIR / "full_dataset_with_roberta_features.csv"
S3_KEYS_FILE = BASE_DIR / "advertisement_photos.csv"
BASE_IMG_URL = "https://resale-media.monobazar.com.ua/"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using {DEVICE}")


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


chroma_client = chromadb.PersistentClient(path=str(BASE_DIR / "chroma_db"))
collection = chroma_client.get_or_create_collection(
    name="monobazar_items",
    metadata={"hnsw:space": "cosine"},
)

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retries))


def fetch_image(adv_id, idx, key, row_price):
    url = f"{BASE_IMG_URL}{key}"
    try:
        response = session.get(url, timeout=5) 
        
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert("RGB")
            
            raw_orig = str(row_price.get("original_price", "0")).replace(",", ".").replace(" ", "")
            safe_orig = float(raw_orig) if raw_orig.lower() != "nan" else 0.0
            
            raw_sold = str(row_price.get("sold_price", "0")).replace(",", ".").replace(" ", "")
            safe_sold = float(raw_sold) if raw_sold.lower() != "nan" else 0.0
            
            is_bargain = bool(row_price.get("sold_via_bargain", False))

            metadata = {
                "advertisement_id": str(adv_id),
                "original_price": safe_orig,
                "sold_price": safe_sold,
                "sold_via_bargain": is_bargain
            }
            return img, f"{adv_id}_{idx}", metadata
    except Exception as e:
        pass
        
    return None, None, None

def build_vision_db():
    logging.info("Loading and merging data")
    df_main = pd.read_csv(MAIN_DATASET)
    df_keys = pd.read_csv(S3_KEYS_FILE)

    grouped_keys = (
        df_keys.groupby("advertisement_id")["s3_key"]
        .apply(list)
        .to_dict()
    )

    logging.info(f"Starting processing {len(df_main)} objects")

    BATCH_SIZE = 64
    batch_images = []
    batch_ids = []
    batch_metadatas = []

    for _, row in tqdm(df_main.iterrows(), total=len(df_main), desc="Indexation"):
        adv_id = str(row["advertisement_id"])
        
        keys = grouped_keys.get(adv_id, [])
        if not keys:
            try:
                keys = grouped_keys.get(int(float(adv_id)), [])
            except ValueError:
                pass
                
        if not keys:
            continue

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(fetch_image, adv_id, idx, key, row)
                for idx, key in enumerate(keys[:5])
            ]
            for future in as_completed(futures):
                img, entry_id, meta = future.result()
                if img:
                    batch_images.append(img)
                    batch_ids.append(entry_id)
                    batch_metadatas.append(meta)

        if len(batch_images) >= BATCH_SIZE:
            with torch.no_grad():
                inputs = processor(
                    images=batch_images, return_tensors="pt", padding=True
                ).to(DEVICE)
                features = model.get_image_features(**inputs)
                features /= features.norm(p=2, dim=-1, keepdim=True)

            collection.upsert(
                ids=batch_ids,
                embeddings=features.cpu().numpy().tolist(),
                metadatas=batch_metadatas,
            )
            batch_images, batch_ids, batch_metadatas = [], [], []

    if batch_images:
        with torch.no_grad():
            inputs = processor(
                images=batch_images, return_tensors="pt", padding=True
            ).to(DEVICE)
            features = model.get_image_features(**inputs)
            features /= features.norm(p=2, dim=-1, keepdim=True)

        collection.upsert(
            ids=batch_ids,
            embeddings=features.cpu().numpy().tolist(),
            metadatas=batch_metadatas,
        )

    logging.info("Chromadb created successfully!")

if __name__ == "__main__":
    build_vision_db()

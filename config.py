from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "mock")
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "https://mock-images.local/")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")  # "gemini" | "mock"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ADVERTISEMENTS_CSV = os.path.join(DATA_DIR, "hackaton_advertisements_with_id.csv")
PHOTOS_CSV = os.path.join(DATA_DIR, "advertisement_photos.csv")
CATEGORY_DICT = os.path.join(DATA_DIR, "category_dictionary.json")
CATEGORY_STATS_CACHE = os.path.join(DATA_DIR, "category_stats.json")

RAG_TOP_K = 7
RAG_BACKEND = os.getenv("RAG_BACKEND", "tfidf")  # "tfidf" | "qdrant" | "hybrid"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BM25_MODEL = "Qdrant/bm25"

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "listings")


def image_url(s3_key: str) -> str:
    """Build full image URL from an s3_key: BASE_URL/s3_key"""
    return IMAGE_BASE_URL.rstrip("/") + "/" + s3_key

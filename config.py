import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# === Directories ===
ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "data"
MODELS_DIR = DATA_DIR / "models"
ASSETS_DIR = DATA_DIR / "assets"

# === Model weights ===
ROBERTA_WEIGHTS = MODELS_DIR / "roberta_student_weights.pth"
XGB_FAST        = MODELS_DIR / "xgb_fast.json"
XGB_BAL         = MODELS_DIR / "xgb_bal.json"
XGB_MAX         = MODELS_DIR / "xgb_max.json"
TFIDF_PATH      = MODELS_DIR / "tfidf_vectorizer.pkl"

# === ChromaDB ===
CHROMA_PATH       = str(ASSETS_DIR / "chroma_db")
CHROMA_COLLECTION = "monobazar_items"
# When set, connects to a remote ChromaDB HTTP server (e.g. in Docker)
# instead of the local persistent client.
CHROMA_HOST = os.environ.get("CHROMA_HOST")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))

# Frontend
FRONTEND_HTML = ROOT_DIR / "frontend" / "index.html"

# === Reference / dataset files ===
CATEGORY_DICT  = ASSETS_DIR / "category_dictionary.json"
TITLES_DATASET = ASSETS_DIR / "hackaton_advertisements_with_id.csv"
MAIN_DATASET   = ASSETS_DIR / "full_dataset_with_roberta_features.csv"
PHOTOS_CSV     = ASSETS_DIR / "advertisement_photos.csv"

# === HuggingFace model IDs ===
ROBERTA_MODEL_NAME = "xlm-roberta-base"
CLIP_MODEL_NAME    = "openai/clip-vit-base-patch32"

# === Gemini ===
GEMINI_API_KEY   = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL     = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash-lite")

# === Inference hyperparameters ===
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_VISUAL_PRICE = 1000.0
N_KNN_RESULTS        = 15       # neighbours fetched from ChromaDB at inference
N_ANALOG_DISPLAY     = 5        # analogues returned to frontend
N_TFIDF_FEATURES     = 20       # must match the fitted TF-IDF vectorizer

# === Server ===
HOST   = os.environ.get("HOST", "0.0.0.0")
PORT   = int(os.environ.get("PORT", 8000))
RELOAD = os.environ.get("RELOAD", "true").lower() == "true"
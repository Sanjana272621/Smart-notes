from pathlib import Path

# ===============================
# GENERAL SETTINGS
# ===============================
USE_CREW_SDK = False  # set True if integrating with Crew AI orchestration

# ===============================
# DATA PATHS
# ===============================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "../data"
FAISS_INDEX_PATH = BASE_DIR / "../data/faiss_index.bin"
METADATA_DB = BASE_DIR / "../data/faiss_metadata.json"

# ===============================
# CHUNKING SETTINGS
# ===============================
CHUNK_TOKENS = 200       # max tokens per chunk
CHUNK_OVERLAP = 20       # token overlap between consecutive chunks

# ===============================
# EMBEDDING SETTINGS
# ===============================
# Local embedding model for Gemma / SentenceTransformers
EMBEDDING_BACKEND = "local"  # "local" or "openai" (but we use local Gemma)
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model
GEMMA_API_KEY = None  # not used when local embeddings

# ===============================
# SUMMARIZER SETTINGS
# ===============================
# Model name for Hugging Face summarization
SUMMARIZER_MODEL = "t5-small"  # Lightweight model, properly configured

# ===============================
# FLASHCARD SETTINGS
# ===============================
MAX_FLASHCARDS_PER_DOC = 15
MAX_SIMPLE_FLASHCARDS = 10

# ===============================
# QA SETTINGS
# ===============================
TOP_K_RETRIEVAL = 5  # number of chunks to retrieve for QA

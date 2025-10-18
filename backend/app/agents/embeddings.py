from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Use a small, efficient local model for embeddings
# You can switch to a larger model if GPU is available
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

class EmbeddingService:
    """
    Local embedding service using SentenceTransformers (Gemma / SBERT compatible)
    No API keys needed; runs fully locally.
    """

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the local SentenceTransformer model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into vectors.

        Args:
            texts: List of text strings

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings.astype(np.float32)

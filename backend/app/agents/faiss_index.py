import faiss
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from .config import FAISS_INDEX_PATH, METADATA_DB


class FaissIndex:
    """
    FAISS wrapper using HNSWFlat for vector search with parallel metadata store.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Dimension of embedding vectors.
        """
        self.dim: int = dim
        self.index = faiss.IndexHNSWFlat(dim, 32)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 50
        self.metadb: List[Dict[str, Any]] = []

    def add(self, vectors: np.ndarray, metadata_list: List[Dict[str, Any]]):
        """
        Add vectors and metadata to the index.

        Args:
            vectors: np.ndarray of shape (num_vectors, dim)
            metadata_list: List of metadata dicts aligned with vectors
        """
        vectors = np.asarray(vectors).astype('float32')
        self.index.add(vectors)
        self.metadb.extend(metadata_list)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the FAISS index for nearest neighbors.

        Args:
            query_vector: Single query vector of shape (dim,)
            top_k: Number of top hits to return

        Returns:
            List of metadata dicts corresponding to top hits
        """
        q = np.asarray([query_vector]).astype('float32')
        D, I = self.index.search(q, top_k)
        hits: List[Dict[str, Any]] = []
        for idx in I[0]:
            if idx < len(self.metadb):
                hits.append(self.metadb[idx])
        return hits

    def save(self, index_path: Optional[Path] = FAISS_INDEX_PATH, meta_path: Optional[Path] = METADATA_DB):
        """
        Persist FAISS index and metadata to disk.

        Args:
            index_path: Path to save FAISS index
            meta_path: Path to save metadata JSON
        """
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadb, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, dim: int, index_path: Optional[Path] = FAISS_INDEX_PATH, meta_path: Optional[Path] = METADATA_DB):
        """
        Load FAISS index and metadata from disk.

        Args:
            dim: Dimension of embedding vectors
            index_path: Path of saved FAISS index
            meta_path: Path of saved metadata JSON

        Returns:
            FaissIndex instance with loaded index and metadata
        """
        inst = cls(dim)
        index_path = Path(index_path)
        meta_path = Path(meta_path)
        if index_path.exists():
            inst.index = faiss.read_index(str(index_path))
        if meta_path.exists():
            inst.metadb = json.loads(meta_path.read_text(encoding='utf-8'))
        return inst

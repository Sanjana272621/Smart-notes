from pathlib import Path
import re
from typing import List, Dict, Any

# ===============================
# TEXT UTILITIES
# ===============================

def clean_text(text: str) -> str:
    """
    Cleans text by removing excessive newlines, extra spaces, and normalizing whitespace.
    """
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 500) -> str:
    """
    Truncates text to a maximum number of characters, adding ellipsis if needed.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


# ===============================
# FILE UTILITIES
# ===============================

def ensure_dir(path: Path):
    """
    Ensure that a directory exists; if not, create it.
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def list_pdf_files(directory: Path) -> List[Path]:
    """
    Returns a sorted list of all PDF files in a directory.
    """
    return sorted([f for f in directory.glob("*.pdf") if f.is_file()])


# ===============================
# METADATA UTILITIES
# ===============================

def format_chunk_metadata(chunk: Dict[str, Any], pdf_id: str = None) -> Dict[str, Any]:
    """
    Standardizes chunk metadata for FAISS storage.
    """
    meta = {
        "text": chunk.get("text", ""),
        "page": chunk.get("page"),
        "pdf_id": pdf_id or chunk.get("pdf_id")
    }
    return meta

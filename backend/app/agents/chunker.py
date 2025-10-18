import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from typing import List, Dict, Optional
from .config import CHUNK_TOKENS, CHUNK_OVERLAP

# Initialize tokenizer once
tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)

def tokens_of(text: str) -> List[int]:
    """
    Returns token IDs for the given text using the tokenizer.
    """
    return tokenizer.encode(text, truncation=False)

def chunk_by_headings(pages: List[Dict[str, str]]) -> List[Dict[str, Optional[int]]]:
    """
    Chunks text based on headings or uppercase short lines.
    Each chunk keeps track of its originating page.
    """
    chunks = []
    buffer = ""
    cur_page = None
    for p in pages:
        lines = p["text"].splitlines()
        for line in lines:
            if line.strip().lower().startswith(("chapter","section")) or (line.isupper() and len(line.split())<10):
                if buffer.strip():
                    chunks.append({"text": buffer.strip(), "page": cur_page})
                    buffer = ""
                cur_page = p["page"]
            buffer += line + " "
    if buffer.strip():
        chunks.append({"text": buffer.strip(), "page": cur_page})
    return chunks

def chunk_by_slides(pages: List[Dict[str, str]]) -> List[Dict[str, int]]:
    """
    Treats each page as a single chunk (slides mode).
    """
    return [{"text": p["text"], "page": p["page"]} for p in pages]

def chunk_by_fixed_tokens(
    pages: List[Dict[str, str]], 
    max_tokens: int = CHUNK_TOKENS, 
    overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, str]]:
    """
    Chunks text by token count, preserving sentence boundaries with optional overlap.
    """
    joined = "\n".join([f"[page {p['page']}]\n{p['text']}" for p in pages])
    sents = sent_tokenize(joined)
    chunks = []
    cur = ""
    cur_tokens = 0
    for sent in sents:
        tcount = len(tokens_of(sent))
        if cur_tokens + tcount > max_tokens and cur:
            chunks.append({"text": cur.strip()})
            overlap_words = cur.split()[-overlap:] if overlap > 0 else []
            cur = " ".join(overlap_words) + " " + sent
            cur_tokens = len(tokens_of(cur))
        else:
            cur += " " + sent
            cur_tokens += tcount
    if cur.strip():
        chunks.append({"text": cur.strip()})
    return chunks

def adaptive_chunker(pages: List[Dict[str, str]]) -> List[Dict[str, Optional[int]]]:
    """
    Chooses the best chunking strategy based on page structure:
    - Headings-based
    - Slides mode
    - Fixed token budget
    """
    avg_len = sum(len(p["text"]) for p in pages) / max(1, len(pages))
    has_headings = any(('chapter' in p["text"].lower() or 'section' in p["text"].lower()) for p in pages)

    if has_headings and avg_len > 200:
        return chunk_by_headings(pages)
    if avg_len < 200:
        return chunk_by_slides(pages)
    return chunk_by_fixed_tokens(pages)

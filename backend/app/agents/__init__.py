# app/agents/__init__.py

# Expose all agents and key utilities for easy import

from .agents import (
    ReaderAgent,
    ChunkingAgent,
    EmbeddingAgent,
    FAISSAgent,
    SummarizerAgent,
    FlashcardAgent,
    QAAgent,
    AgentResult
)

from .pdf_utils import extract_with_ocr_if_needed
from .flashcards import simple_flashcards_from_text, llm_flashcards_from_text
from .summarizer import chunk_and_summarize_chunks
from .faiss_index import FaissIndex

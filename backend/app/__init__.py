# app/__init__.py

# This makes `app` a Python package
# Optionally, you can import key submodules for easier access

from .crew_orchestrator import CrewOrchestrator
from .agents import (
    ReaderAgent, ChunkingAgent, EmbeddingAgent, FAISSAgent,
    SummarizerAgent, FlashcardAgent, QAAgent, AgentResult
)
from .agents.chunker import adaptive_chunker
from .agents.config import *

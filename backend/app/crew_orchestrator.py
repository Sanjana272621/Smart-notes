import os
from typing import Optional, Dict, Any, List
import numpy as np
from .agents import (
    ReaderAgent, ChunkingAgent, EmbeddingAgent, FAISSAgent,
    SummarizerAgent, FlashcardAgent, QAAgent, AgentResult
)
from .agents.embeddings import EmbeddingService
from .agents.config import USE_CREW_SDK


class CrewAdapter:
    """
    Adapter stub for a hosted CrewAI / orchestration SDK.
    Implement `run_task` to call Crew API and map agent input/outputs.
    """
    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg

    def run_task(self, agent_name: str, payload: Dict[str, Any]) -> AgentResult:
        """
        Stub method. Replace with actual Crew SDK integration.
        """
        raise NotImplementedError("Implement Crew SDK integration here")


class CrewOrchestrator:
    """
    Local orchestrator that sequences agent calls and logs agent-level traces.
    Optionally integrates with Crew SDK if use_crew_sdk=True.
    """

    def __init__(self, use_crew_sdk: bool = False):
        self.use_crew_sdk = use_crew_sdk

        # Initialize agents
        self.reader = ReaderAgent()
        self.chunker = ChunkingAgent()
        self.embedding_agent = EmbeddingAgent()

        # Determine embedding dim from a sample text
        sample_emb = self.embedding_agent.service.embed(["hello"])
        self.dim = sample_emb.shape[1]

        # Try to load existing FAISS index, create new one if it doesn't exist
        try:
            from .agents.faiss_index import FaissIndex
            loaded_index = FaissIndex.load(dim=self.dim)
            self.faiss_agent = FAISSAgent(dim=self.dim)
            self.faiss_agent.idx = loaded_index
            print(f"Loaded existing FAISS index with {len(self.faiss_agent.idx.metadb)} entries")
        except Exception as e:
            print(f"Could not load existing FAISS index: {e}")
            print("Creating new empty FAISS index")
            self.faiss_agent = FAISSAgent(dim=self.dim)
        self.summarizer = SummarizerAgent()
        self.flashcard = FlashcardAgent()
        self.qa_agent = QAAgent(
            embedding_service=self.embedding_agent.service,
            faiss_agent=self.faiss_agent,
            summarizer_agent=self.summarizer
        )

        self.trace: List[Dict[str, Any]] = []  # Logs for explainability
        self.crew_adapter = CrewAdapter() if use_crew_sdk else None

    def run_ingest_pipeline(
        self,
        file_path: str,
        pdf_id: Optional[str] = None,
        pre_summarize: bool = True,
        generate_flashcards: bool = True
    ) -> Dict[str, Any]:
        """
        Full ingest pipeline:
        1) Reads PDF
        2) Chunks pages
        3) Embeds chunks
        4) Adds to FAISS index
        5) Optionally pre-summarizes
        6) Optionally generates flashcards
        """

        # Step 1: Read PDF
        r1 = self.reader.run(file_path)
        self._trace_add(r1)

        # Step 2: Chunk pages
        pages = r1.payload["pages"]
        r2 = self.chunker.run(pages)
        chunks = r2.payload["chunks"]
        # Attach pdf_id to each chunk
        for c in chunks:
            c["pdf_id"] = pdf_id
        self._trace_add(r2)

        # Step 3: Embed chunks
        r3 = self.embedding_agent.run(chunks)
        vectors = r3.payload["vectors"]
        metas = r3.payload["metas"]
        self._trace_add(r3)

        # Step 4: Add embeddings to FAISS
        r4 = self.faiss_agent.add(vectors, metas)
        self._trace_add(r4)

        # Step 5: Optional hierarchical summarization
        summary_pack = None
        if pre_summarize:
            r5 = self.summarizer.run(chunks)
            summary_pack = r5.payload.get("summary_pack")
            self._trace_add(r5)

        # Step 6: Optional flashcard generation
        flashcards: List[Dict[str, Any]] = []
        if generate_flashcards and summary_pack:
            r6 = self.flashcard.run(summary_pack["final_summary"])
            flashcards = r6.payload.get("flashcards", [])
            self._trace_add(r6)

        return {
            "trace": self.trace,
            "summary_pack": summary_pack,
            "flashcards": flashcards,
            "num_chunks": len(chunks)
        }

    def run_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Runs a retrieval-augmented QA query over the indexed corpus.
        Returns answer, sources, and trace logs.
        """
        r = self.qa_agent.run(query, top_k=top_k)
        self._trace_add(r)
        return {
            "answer": r.payload["answer"],
            "sources": r.payload["sources"],
            "trace": self.trace
        }

    def _trace_add(self, agent_result: AgentResult):
        """
        Appends agent result to trace logs.
        """
        trace_entry = {
            "agent": agent_result.agent,
            "logs": agent_result.logs,
            "payload_keys": list(agent_result.payload.keys()) if agent_result.payload else []
        }
        
        # Store the full payload for certain agents that need it later
        if agent_result.agent in ["FlashcardAgent", "SummarizerAgent"]:
            trace_entry["payload"] = agent_result.payload
            
        self.trace.append(trace_entry)

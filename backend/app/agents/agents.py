from .pdf_utils import extract_with_ocr_if_needed
from .chunker import adaptive_chunker
from .embeddings import EmbeddingService
from .faiss_index import FaissIndex
from .summarizer import chunk_and_summarize_chunks
from .flashcards import llm_flashcards_from_text, simple_flashcards_from_text
from .config import EMBEDDING_BACKEND, GEMMA_API_KEY
from typing import Any, List, Dict

class AgentResult:
    """
    Standard result wrapper for agents.
    Stores payload, logs, and agent name for tracing.
    """
    def __init__(self, agent_name: str, payload: Dict[str, Any] = None, logs: List[str] = None):
        self.agent: str = agent_name
        self.payload: Dict[str, Any] = payload or {}
        self.logs: List[str] = logs or []

    def add_log(self, msg: str):
        """Add a log entry for debugging or traceability."""
        self.logs.append(msg)

    def __repr__(self):
        return f"<AgentResult {self.agent}, payload keys={list(self.payload.keys())}, logs={len(self.logs)}>"

class ReaderAgent:
    """
    Reads a PDF file and returns pages with extracted text.
    Uses OCR on low-text pages.
    """
    name = "ReaderAgent"

    def run(self, file_path: str) -> AgentResult:
        pages = extract_with_ocr_if_needed(file_path)
        res = AgentResult(self.name, payload={"pages": pages})
        res.add_log(f"Extracted {len(pages)} pages from {file_path}")
        return res

class ChunkingAgent:
    name = "ChunkingAgent"
    def run(self, pages):
        chunks = adaptive_chunker(pages)
        res = AgentResult(self.name, payload={"chunks": chunks})
        res.add_log(f"produced {len(chunks)} chunks (adaptive strategy)")
        return res

class EmbeddingAgent:
    name = "EmbeddingAgent"

    def __init__(self):
        # Initialize local embedding service
        self.service = EmbeddingService()  # no backend or API key needed

    def run(self, chunks):
        texts = [c["text"] for c in chunks]
        vectors = self.service.embed(texts)
        metas = [
            {
                "text": chunks[i]["text"],
                "page": chunks[i].get("page"),
                "pdf_id": chunks[i].get("pdf_id", None)
            }
            for i in range(len(chunks))
        ]
        res = AgentResult(self.name, payload={"vectors": vectors, "metas": metas})
        res.add_log(f"embedded {len(texts)} chunks; dim={vectors.shape[1]}")
        return res


class FAISSAgent:
    name = "FAISSAgent"
    def __init__(self, dim):
        self.idx = FaissIndex(dim)

    def add(self, vectors, metas):
        self.idx.add(vectors, metas)
        res = AgentResult(self.name, payload={"status": "added", "num_vectors": len(metas)})
        res.add_log(f"added {len(metas)} vectors to FAISS")
        return res

    def search(self, qvec, top_k=5):
        hits = self.idx.search(qvec, top_k=top_k)
        res = AgentResult(self.name, payload={"hits": hits})
        res.add_log(f"found {len(hits)} hits")
        return res

class SummarizerAgent:
    name = "SummarizerAgent"
    def run(self, chunks):
        try:
            pack = chunk_and_summarize_chunks(chunks)
            res = AgentResult(self.name, payload={"summary_pack": pack})
            res.add_log("performed hierarchical summarization")
            return res
        except Exception as e:
            print(f"SummarizerAgent failed, using safe fallback: {e}")
            # Use safe summarizer as fallback
            from .safe_summarizer import safe_chunk_summarize
            pack = safe_chunk_summarize(chunks)
            res = AgentResult(self.name, payload={"summary_pack": pack})
            res.add_log("used safe summarization fallback")
            return res

class FlashcardAgent:
    name = "FlashcardAgent"
    def run(self, text, llm_callable=None):
        cards = llm_flashcards_from_text(text, llm_callable=llm_callable)
        res = AgentResult(self.name, payload={"flashcards": cards})
        res.add_log(f"generated {len(cards)} flashcards")
        return res

class QAAgent:
    name = "QAAgent"
    def __init__(self, embedding_service, faiss_agent, summarizer_agent):
        self.embedding_service = embedding_service
        self.faiss_agent = faiss_agent
        self.summarizer_agent = summarizer_agent

    def run(self, query, top_k=5):
        try:
            print(f"QAAgent processing query: {query}")
            qvecs = self.embedding_service.embed([query])
            print(f"Generated query vector with shape: {qvecs.shape}")
            
            # Check if FAISS index has any data
            if len(self.faiss_agent.idx.metadb) == 0:
                print("FAISS index is empty - no documents have been indexed")
                return AgentResult(
                    self.name,
                    payload={"answer": "No documents have been indexed yet. Please upload and process a PDF first.", "sources": []}
                )
            
            hits_res = self.faiss_agent.search(qvecs[0], top_k=top_k)
            hits = hits_res.payload["hits"]
            print(f"FAISS search returned {len(hits)} hits")

            if not hits:
                # No results found, return a default answer
                print("No search results found")
                return AgentResult(
                    self.name,
                    payload={"answer": "No relevant information found in the indexed documents. Please ensure the PDF has been processed and indexed.", "sources": []}
                )

            combined_text = "\n".join(h["text"] for h in hits)
            print(f"Combined text length: {len(combined_text)}")
            
            # Try to summarize the combined text
            try:
                # For now, use a simple approach to avoid the list index error
                # TODO: Fix the summarization pipeline properly
                if len(combined_text) > 500:
                    final_summary = combined_text[:500] + "..."
                else:
                    final_summary = combined_text
                
                print("Using simple text truncation instead of summarization")
                
                # Try the actual summarization as a fallback
                try:
                    summary_res = self.summarizer_agent.run([{"text": combined_text}])
                    summary_pack = summary_res.payload.get("summary_pack", {})
                    if summary_pack.get("final_summary"):
                        final_summary = summary_pack.get("final_summary")
                        print("Summarization succeeded")
                except Exception as e:
                    print(f"Summarization failed, using simple truncation: {str(e)}")
                    
            except Exception as e:
                # If anything fails, use the first part of the combined text
                final_summary = combined_text[:500] + "..."
                print(f"All summarization failed, using raw text: {str(e)}")
                print(f"Summarization error details: {e}")
                import traceback
                traceback.print_exc()

            res = AgentResult(
                self.name,
                payload={"answer": final_summary, "sources": hits}
            )
            res.add_log("answered query by retrieving and summarizing top chunks")
            return res
            
        except Exception as e:
            # If anything fails, return a basic response
            print(f"QAAgent error: {e}")
            import traceback
            traceback.print_exc()
            return AgentResult(
                self.name,
                payload={"answer": f"Error processing query: {str(e)}", "sources": []}
            )

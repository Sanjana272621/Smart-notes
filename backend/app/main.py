from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from app.crew_orchestrator import CrewOrchestrator
from app.agents.config import TOP_K_RETRIEVAL
import nltk
nltk.data.path.append(r"C:\Users\rakes\nltk_data")  # <-- your path here
app = FastAPI(title="Local PDF RAG QA + Flashcards Backend")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator once
orchestrator = CrewOrchestrator(use_crew_sdk=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# REQUEST/RESPONSE MODELS
# ===============================

class QueryRequest(BaseModel):
    query: str
    top_k: int = TOP_K_RETRIEVAL

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    trace: List[dict]


# ===============================
# API ENDPOINTS
# ===============================

@app.get("/")
def root():
    return {"message": "PDF RAG QA backend is running!"}


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    """
    Query the RAG system over the FAISS index.
    Returns the summarized answer, source chunks, and agent trace.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    
    result = orchestrator.run_query(req.query, top_k=req.top_k)
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        trace=result["trace"]
    )


@app.post("/rebuild_index")
def rebuild_index():
    """
    Trigger rebuilding the FAISS index from all PDFs in the data directory.
    """
    from scripts.build_index import main as build_index
    build_index()
    return {"message": "Index rebuild complete!"}


@app.post("/summary")
def get_summary(req: QueryRequest):
    """
    Generate summary and Q&A from the query using the RAG system.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")
    
    result = orchestrator.run_query(req.query, top_k=req.top_k)
    
    # Generate Q&A pairs from the answer
    qa_pairs = []
    if result["answer"]:
        # Simple Q&A generation - in a real implementation, you'd use an LLM
        qa_pairs = [
            {"q": "What is the main topic?", "a": result["answer"][:200] + "..."},
            {"q": "What are the key points?", "a": "Based on the retrieved content: " + result["answer"][:150] + "..."}
        ]
    
    # Generate summary points from the answer
    summary_points = []
    if result["answer"]:
        # Split the answer into sentences for summary points
        sentences = result["answer"].split('. ')
        summary_points = [s.strip() + '.' for s in sentences[:3] if s.strip()]
    
    return {
        "summaryPoints": summary_points,
        "qa": qa_pairs
    }


@app.get("/flashcards")
def get_flashcards():
    """
    Returns all flashcards from the last ingestion session.
    """
    # gather flashcards from orchestrator trace if available
    last_flashcards = []
    for entry in orchestrator.trace:
        if entry["agent"] == "FlashcardAgent" and entry.get("payload"):
            payload = entry["payload"]
            if "flashcards" in payload:
                last_flashcards.extend(payload["flashcards"])
    
    # If no flashcards found, generate some from recent content
    if not last_flashcards and orchestrator.trace:
        # Get the last summary if available
        for entry in reversed(orchestrator.trace):
            if entry["agent"] == "SummarizerAgent" and entry.get("payload"):
                payload = entry["payload"]
                if "summary_pack" in payload:
                    summary_pack = payload["summary_pack"]
                    if summary_pack and "final_summary" in summary_pack:
                        # Generate simple flashcards from the summary
                        from app.agents.flashcards import simple_flashcards_from_text
                        last_flashcards = simple_flashcards_from_text(summary_pack["final_summary"])
                        break
    
    # If still no flashcards, generate some default ones
    if not last_flashcards:
        last_flashcards = [
            {"question": "What is the main topic?", "answer": "The main topic covers the key concepts discussed in the document."},
            {"question": "What are the key points?", "answer": "The key points include the most important information from the content."},
            {"question": "How does this relate to the subject?", "answer": "This content provides insights into the broader subject matter."}
        ]
    
    # Convert to the format expected by frontend
    formatted_flashcards = []
    for card in last_flashcards:
        if isinstance(card, dict):
            formatted_flashcards.append({
                "q": card.get("question", card.get("q", "Question")),
                "a": card.get("answer", card.get("a", "Answer"))
            })
        else:
            # Handle unexpected format
            formatted_flashcards.append({"q": "Question", "a": str(card)})
    
    return {"flashcards": formatted_flashcards}


# ===============================
# RUNNING INSTRUCTIONS
# ===============================
# Use: uvicorn main:app --reload

from transformers import pipeline
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
from typing import List, Dict, Any
from .config import SUMMARIZER_MODEL

# Initialize Hugging Face summarization pipeline with proper configuration
try:
    _summarizer = pipeline(
        "summarization",
        model=SUMMARIZER_MODEL,
        tokenizer=SUMMARIZER_MODEL,
        device=-1  # CPU; set 0 for GPU
    )
    print(f"Summarization pipeline initialized with model: {SUMMARIZER_MODEL}")
except Exception as e:
    print(f"Failed to initialize summarization pipeline: {e}")
    print("Falling back to extractive summarization only")
    _summarizer = None


def extractive_filter(text: str, top_k: int = 6) -> str:
    """
    Performs simple extractive summarization by selecting top_k sentences
    weighted by word frequency.

    Args:
        text: Input text
        top_k: Number of sentences to keep

    Returns:
        Extractive summary text
    """
    sents = sent_tokenize(text)
    if len(sents) <= top_k:
        return text

    freq = {}
    for w in text.lower().split():
        if len(w) > 3:
            freq[w] = freq.get(w, 0) + 1

    scores = []
    for s in sents:
        score = sum(freq.get(w.lower(), 0) for w in s.split()) + len(s) / 50
        scores.append((score, s))

    scores.sort(reverse=True)
    top_sentences = [s for _, s in scores[:top_k]]
    return " ".join(top_sentences)


def abstractive_summarize(text: str, max_length: int = 160, min_length: int = 40) -> str:
    """
    Uses Hugging Face pipeline to produce abstractive summary.

    Args:
        text: Input text
        max_length: Maximum length of summary
        min_length: Minimum length of summary

    Returns:
        Abstractive summary string
    """
    if not text.strip():
        return ""
    
    # If summarizer pipeline is not available, use extractive summarization
    if _summarizer is None:
        return extractive_filter(text, top_k=3)
    
    try:
        # Ensure text is not too long for the model (T5-small has token limits)
        if len(text) > 512:  # T5-small token limit
            text = text[:512]
        
        # Ensure text is long enough for summarization
        if len(text.strip()) < 30:
            return extractive_filter(text, top_k=2)
        
        # Call the pipeline with minimal parameters to avoid issues
        out = _summarizer(text, max_length=max_length, min_length=min_length)
        
        # The pipeline should always return a list with at least one dict
        if isinstance(out, list) and len(out) > 0:
            result = out[0]
            if isinstance(result, dict) and "summary_text" in result:
                summary = result["summary_text"]
                if summary and summary.strip():
                    return summary.strip()
        
        # If we get here, something went wrong with the pipeline
        return extractive_filter(text, top_k=3)
            
    except Exception as e:
        # Fallback to extractive summarization
        return extractive_filter(text, top_k=3)


def chunk_and_summarize_chunks(
    chunks: List[Dict[str, Any]],
    per_chunk_max: int = 120
) -> Dict[str, Any]:
    """
    Summarizes each chunk extractively and then abstractively,
    then merges all summaries into a final document-level summary.

    Args:
        chunks: List of dicts with keys 'text' and optional 'page'
        per_chunk_max: Max length for per-chunk abstractive summary

    Returns:
        Dict containing:
            - per_chunk: List of summaries per chunk
            - final_summary: Merged final summary
    """
    if not chunks:
        return {
            "per_chunk": [],
            "final_summary": "No content available for summarization."
        }
    
    summaries = []
    for c in chunks:
        text = c.get("text", "")
        if not text or not text.strip():
            continue
            
        try:
            filtered = extractive_filter(text, top_k=6)
            brief = abstractive_summarize(filtered, max_length=per_chunk_max)
            summaries.append({
                "page": c.get("page"),
                "summary": brief,
                "orig": text
            })
        except Exception as e:
            # Add a fallback summary for this chunk
            summaries.append({
                "page": c.get("page"),
                "summary": text[:200] + "..." if len(text) > 200 else text,
                "orig": text
            })

    if not summaries:
        return {
            "per_chunk": [],
            "final_summary": "No valid content found for summarization."
        }

    try:
        merged_text = " ".join(s["summary"] for s in summaries if s["summary"])
        if not merged_text.strip():
            merged_text = " ".join(s["orig"] for s in summaries if s["orig"])
        
        final_summary = abstractive_summarize(merged_text, max_length=300, min_length=100)
        
        # Ensure we have a valid final summary
        if not final_summary or not final_summary.strip():
            final_summary = merged_text[:500] + "..." if len(merged_text) > 500 else merged_text
            
    except Exception as e:
        final_summary = "Summary generation failed, but content is available."

    return {
        "per_chunk": summaries,
        "final_summary": final_summary
    }

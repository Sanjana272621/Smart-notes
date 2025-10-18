"""
Safe summarization module that avoids the list index out of range error.
"""

from typing import List, Dict, Any
import re

def safe_extractive_summarize(text: str, max_sentences: int = 3) -> str:
    """
    Safe extractive summarization that never fails.
    """
    if not text or not text.strip():
        return "No content available for summarization."
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) <= max_sentences:
        return text
    
    # Simple scoring based on length and word frequency
    word_freq = {}
    for word in text.lower().split():
        if len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    scored_sentences = []
    for sentence in sentences:
        score = len(sentence) + sum(word_freq.get(word.lower(), 0) for word in sentence.split())
        scored_sentences.append((score, sentence))
    
    # Sort by score and take top sentences
    scored_sentences.sort(reverse=True)
    top_sentences = [sentence for _, sentence in scored_sentences[:max_sentences]]
    
    return ". ".join(top_sentences) + "."

def safe_chunk_summarize(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Safe chunk summarization that never fails.
    """
    if not chunks:
        return {
            "per_chunk": [],
            "final_summary": "No content available for summarization."
        }
    
    summaries = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text or not text.strip():
            continue
            
        # Use safe extractive summarization
        summary = safe_extractive_summarize(text, max_sentences=2)
        summaries.append({
            "page": chunk.get("page"),
            "summary": summary,
            "orig": text
        })
    
    if not summaries:
        return {
            "per_chunk": [],
            "final_summary": "No valid content found for summarization."
        }
    
    # Combine all summaries
    all_summaries = [s["summary"] for s in summaries if s["summary"]]
    if not all_summaries:
        all_summaries = [s["orig"] for s in summaries if s["orig"]]
    
    combined_text = " ".join(all_summaries)
    final_summary = safe_extractive_summarize(combined_text, max_sentences=5)
    
    return {
        "per_chunk": summaries,
        "final_summary": final_summary
    }

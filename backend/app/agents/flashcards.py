from typing import List, Dict, Callable, Optional
import nltk

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


def simple_flashcards_from_text(text: str, max_cards: int = 10) -> List[Dict[str, str]]:
    """
    Generates simple cloze-style flashcards from input text.
    Uses NLTK POS tagging to blank out a main noun in each sentence.

    Args:
        text: Input text to generate flashcards from.
        max_cards: Maximum number of flashcards to generate.

    Returns:
        List of flashcards as dicts with keys: 'question', 'answer', 'source'.
    """
    from nltk.tokenize import sent_tokenize, word_tokenize

    sents = sent_tokenize(text)
    cards: List[Dict[str, str]] = []

    for s in sents[:max_cards]:
        words = word_tokenize(s)
        tags = nltk.pos_tag(words)
        blank: Optional[str] = None

        for w, t in tags:
            if t.startswith("NN") and len(w) > 4:
                blank = w
                break

        if blank:
            question = s.replace(blank, "_____")
            cards.append({"question": question, "answer": blank, "source": s})
        else:
            # fallback: main idea question
            question = f"What is the main idea of: \"{s[:80]}...\""
            cards.append({"question": question, "answer": s, "source": s})

    return cards


def llm_flashcards_from_text(
    text: str,
    llm_callable: Optional[Callable[[str, int], List[Dict[str, str]]]] = None,
    max_cards: int = 15
) -> List[Dict[str, str]]:
    """
    Generates flashcards using an optional LLM callable.
    Falls back to simple flashcards if no LLM is provided.

    Args:
        text: Input text for flashcard generation.
        llm_callable: Optional function that accepts text and max_cards, returns list of flashcards.
        max_cards: Maximum number of flashcards to generate.

    Returns:
        List of flashcards as dicts with keys: 'question', 'answer', 'source'.
    """
    if llm_callable:
        return llm_callable(text, max_cards=max_cards)
    return simple_flashcards_from_text(text, max_cards=max_cards)

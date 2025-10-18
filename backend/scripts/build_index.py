import sys
from pathlib import Path
from app.crew_orchestrator import CrewOrchestrator
from app.agents.config import DATA_DIR, FAISS_INDEX_PATH, METADATA_DB
from app.utils.helpers import list_pdf_files, ensure_dir

def main(data_dir: Path = DATA_DIR):
    """
    Batch ingest PDFs from data directory, generate chunks, embeddings,
    optionally summaries & flashcards, and save FAISS index & metadata.
    """
    ensure_dir(data_dir)
    pdf_files = list_pdf_files(data_dir)

    if not pdf_files:
        print(f"No PDF files found in {data_dir}. Exiting.")
        sys.exit(0)

    print(f"Found {len(pdf_files)} PDFs. Building index...")

    orchestrator = CrewOrchestrator(use_crew_sdk=False)

    for pdf_path in pdf_files:
        print(f"\nProcessing {pdf_path.name}...")
        result = orchestrator.run_ingest_pipeline(
            file_path=str(pdf_path),
            pdf_id=pdf_path.stem,  # use filename as pdf_id
            pre_summarize=True,
            generate_flashcards=True
        )
        print(f"Chunks: {result['num_chunks']}, "
              f"Flashcards: {len(result['flashcards'])}")

    # Save FAISS index and metadata
    orchestrator.faiss_agent.idx.save(index_path=FAISS_INDEX_PATH, meta_path=METADATA_DB)
    print(f"\nFAISS index saved to {FAISS_INDEX_PATH}")
    print(f"Metadata saved to {METADATA_DB}")
    print("Index building complete!")


if __name__ == "__main__":
    main()

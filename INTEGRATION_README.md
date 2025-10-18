# PDF RAG QA + Flashcards Integration

This document explains how to run the integrated frontend and backend system.

## Prerequisites

- Python 3.8+ with pip
- Node.js 18+ with npm
- The system should have PDF files in `backend/app/data/` directory

## Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK data (if not already done):
   ```bash
   python scripts/download_nltk_data.py
   ```

4. Build the FAISS index from PDFs:
   ```bash
   python scripts/build_index.py
   ```

5. Start the backend server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

The backend will be available at `http://127.0.0.1:8000`

## Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

The frontend will be available at `http://localhost:3000`

## Testing the Integration

1. Run the integration test script:
   ```bash
   python test_integration.py
   ```

2. Open the frontend in your browser and test the following:
   - Upload a PDF file
   - Navigate to the summary page
   - Navigate to the flashcards page
   - Verify that data loads correctly

## API Endpoints

### Backend Endpoints

- `GET /` - Health check
- `POST /query` - Query the RAG system
- `POST /summary` - Generate summary and Q&A
- `GET /flashcards` - Get generated flashcards
- `POST /rebuild_index` - Rebuild the FAISS index

### Frontend Pages

- `/` - Home page with file upload
- `/summary?jobId=<id>` - Summary page
- `/flashcards?jobId=<id>` - Flashcards page

## Troubleshooting

### Common Issues

1. **Backend not starting**: Check if port 8000 is available
2. **Frontend not connecting**: Ensure backend is running on port 8000
3. **No data in summary/flashcards**: Make sure PDFs are indexed and the FAISS index is built
4. **CORS errors**: The backend is configured to allow all origins for development

### Error Handling

The system includes comprehensive error handling:
- Backend APIs return proper HTTP status codes
- Frontend displays error messages to users
- Summarizer failures fall back to raw text
- Flashcard generation has fallback defaults

## Architecture

The system uses:
- **Backend**: FastAPI with agents for PDF processing, summarization, and flashcard generation
- **Frontend**: Next.js with TypeScript
- **Vector Store**: FAISS for similarity search
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Summarization**: Hugging Face T5-small model

## Development Notes

- The system is designed to work with local models to avoid API costs
- All processing happens locally on the server
- The FAISS index persists between server restarts
- Agent traces are stored for debugging and flashcard generation

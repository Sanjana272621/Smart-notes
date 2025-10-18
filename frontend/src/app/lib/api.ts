// lib/api.ts

export type UploadResponse = {
  message: string;
  jobId: string;
};

export async function uploadPdf(file: File): Promise<UploadResponse> {
  await new Promise((r) => setTimeout(r, 800));
  if (!(file instanceof File) || file.size === 0) {
    throw new Error('Invalid file.');
  }
  const jobId = `job_${Math.random().toString(36).slice(2, 10)}`;
  return { message: 'Upload received. Processing will start shortly.', jobId };
}

// ------- NEW: Mock summary + flashcards APIs --------

export type SummaryResponse = {
  summaryPoints: string[];
  qa: { q: string; a: string }[];
};

export type FlashcardResponse = {
  flashcards: { q: string; a: string }[];
};

export async function getSummary(query: string, top_k: number = 3): Promise<SummaryResponse> {
  const res = await fetch('http://127.0.0.1:8000/summary', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k })
  });
  
  if (!res.ok) {
    throw new Error(`Failed to fetch summary: ${res.statusText}`);
  }
  
  return await res.json();
}

export async function getFlashcards(jobId: string): Promise<FlashcardResponse> {
  const res = await fetch('http://127.0.0.1:8000/flashcards', {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' }
  });
  
  if (!res.ok) {
    throw new Error(`Failed to fetch flashcards: ${res.statusText}`);
  }
  
  return await res.json();
}

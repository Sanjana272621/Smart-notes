'use client';

import React, { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { getFlashcards, type FlashcardResponse } from '@/app/lib/api';

export default function FlashcardsPage() {
  const params = useSearchParams();
  const router = useRouter();
  const jobId = params.get('jobId') ?? '';

  const [data, setData] = useState<FlashcardResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!jobId) {
      router.replace('/');
      return;
    }
    (async () => {
      try {
        const res = await getFlashcards(jobId);
        setData(res);
      } catch (error) {
        console.error('Failed to load flashcards:', error);
        alert('Failed to load flashcards');
      } finally {
        setLoading(false);
      }
    })();
  }, [jobId, router]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-slate-50 text-slate-900">
      <header className="sticky top-0 z-50 border-b bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <button
              onClick={() => router.push('/')}
              className="rounded-md border px-2 py-1 text-sm hover:bg-slate-100"
            >
              ← Home
            </button>
            <h1 className="text-lg font-semibold">Flashcards</h1>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-10">
        {loading ? (
          <p className="text-slate-600">Generating flashcards…</p>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {data?.flashcards.map((fc, i) => (
              <div key={i} className="rounded-xl border bg-white p-6 shadow-sm">
                <p className="font-medium">Q: {fc.q}</p>
                <p className="mt-2 text-slate-600">A: {fc.a}</p>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

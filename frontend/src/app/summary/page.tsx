'use client';

import React, { useEffect, useState } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { getSummary, type SummaryResponse } from '@/app/lib/api';
import { exportToPdf } from '@/app/lib/exportToPdf';

export default function SummaryPage() {
  const params = useSearchParams();
  const router = useRouter();
  const jobId = params.get('jobId') ?? '';
  const fileName = params.get('name') ?? 'Your PDF';

  const [data, setData] = useState<SummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!jobId) {
      router.replace('/');
      return;
    }
    (async () => {
      try {
        // Use a generic query to get summary from the indexed content
        const res = await getSummary("Provide a comprehensive summary of the main topics and key concepts");
        setData(res);
      } catch (error) {
        console.error('Failed to load summary:', error);
        alert('Failed to load summary');
      } finally {
        setLoading(false);
      }
    })();
  }, [jobId, router]);

  const handleExport = async () => {
    const el = document.getElementById('summary-root');
    if (el) {
      await exportToPdf(el, 'summary.pdf');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-slate-50 text-slate-900">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <button
              onClick={() => router.push('/')}
              className="rounded-md border px-2 py-1 text-sm hover:bg-slate-100"
            >
              ← Home
            </button>
            <h1 className="text-lg font-semibold">Summary</h1>
          </div>
          <button
            onClick={handleExport}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium hover:bg-slate-100 active:scale-[0.99]"
          >
            Export to PDF
          </button>
        </div>
      </header>

      {/* Main */}
      <main id="summary-root" className="mx-auto max-w-6xl px-4 py-10">
        <h2 className="text-2xl font-bold text-indigo-600 mb-4">{fileName}</h2>

        {loading ? (
          <p className="text-slate-600">Processing your PDF…</p>
        ) : (
          <>
            <div className="rounded-2xl border bg-white p-6 shadow-sm mb-6">
              <h3 className="text-lg font-semibold">Smart Summary</h3>
              <ul className="mt-3 list-disc pl-5 text-slate-700">
                {data?.summaryPoints.map((p, i) => (
                  <li key={i}>{p}</li>
                ))}
              </ul>
            </div>

            <div className="rounded-2xl border bg-white p-6 shadow-sm mb-6">
              <h3 className="text-lg font-semibold">Q&amp;A</h3>
              <div className="mt-3 space-y-3">
                {data?.qa.map((qa, i) => (
                  <div key={i} className="rounded-lg border p-4">
                    <p className="font-medium">Q: {qa.q}</p>
                    <p className="mt-1 text-slate-600">A: {qa.a}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* New button to Flashcards */}
            <div className="text-center mt-8">
              <button
                onClick={() => router.push(`/flashcards?jobId=${encodeURIComponent(jobId)}`)}
                className="rounded-xl bg-indigo-600 px-5 py-2.5 text-sm font-medium text-white hover:bg-indigo-500 active:scale-[0.99]"
              >
                View Flashcards →
              </button>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

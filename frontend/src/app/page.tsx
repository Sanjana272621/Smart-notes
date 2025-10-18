'use client';

import React, { useCallback, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import { exportToPdf } from '@/app/lib/exportToPdf';
import { uploadPdf } from '@/app/lib/api';
import FileDropzone from '@/app/components/FileDropzone';

export default function HomePage() {
  const [isUploading, setIsUploading] = useState(false);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const router = useRouter();

  const handleExport = useCallback(async () => {
    if (!contentRef.current) return;
    try {
      await exportToPdf(contentRef.current, 'summary.pdf');
    } catch {
      alert('Failed to export PDF.');
    }
  }, []);

  const handleFileSelect = useCallback(
    async (file: File) => {
      if (file.type !== 'application/pdf') {
        alert('Please upload a PDF file.');
        return;
      }
      try {
        setIsUploading(true);
        const res = await uploadPdf(file); // fake API
        router.push(
          `/summary?jobId=${encodeURIComponent(res.jobId)}&name=${encodeURIComponent(file.name)}`
        );
      } catch {
        alert('❌ Upload failed. Please try again.');
      } finally {
        setIsUploading(false);
      }
    },
    [router]
  );

  return (
    <div className="min-h-screen bg-gradient-to-b from-white to-slate-50 text-slate-900">
      {/* Sticky Header */}
      <header className="sticky top-0 z-50 border-b bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white">
              E
            </span>
            <h1 className="text-lg font-semibold">Edu-Easy</h1>
          </div>
          <button
            onClick={handleExport}
            className="rounded-lg border border-slate-300 px-3 py-1.5 text-sm font-medium hover:bg-slate-100 active:scale-[0.99]"
          >
            Export to PDF
          </button>
        </div>
      </header>

      <main ref={contentRef}>
        {/* Hero */}
        <section className="mx-auto max-w-6xl px-4 pt-12 pb-6 text-center">
          <h2 className="bg-gradient-to-r from-indigo-600 via-fuchsia-600 to-pink-600 bg-clip-text text-4xl font-extrabold tracking-tight text-transparent sm:text-5xl">
            Edu-Easy: Smarter Learning from Your PDFs
          </h2>
          <p className="mt-4 max-w-2xl mx-auto text-lg text-slate-600">
            Upload any textbook, research article, or class notes. Edu-Easy
            transforms it into summaries, flashcards, and instant Q&amp;A so
            you can learn faster and remember more.
          </p>
        </section>

        {/* Upload */}
        <section className="mx-auto max-w-2xl px-4">
          <FileDropzone
            onFileSelect={handleFileSelect}
            isUploading={isUploading}
          />
        </section>

        {/* Features */}
        <section className="mx-auto max-w-6xl px-4 py-12">
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            <div className="rounded-2xl border bg-white p-6 shadow-sm">
              <h3 className="text-lg font-semibold">Smart Summarization</h3>
              <p className="mt-2 text-slate-600">
                Get concise summaries of long PDFs with key highlights.
              </p>
            </div>
            <div className="rounded-2xl border bg-white p-6 shadow-sm">
              <h3 className="text-lg font-semibold">Flashcards</h3>
              <p className="mt-2 text-slate-600">
                Auto-generate flashcards for revision and self-testing.
              </p>
            </div>
            <div className="rounded-2xl border bg-white p-6 shadow-sm">
              <h3 className="text-lg font-semibold">Q&amp;A Mode</h3>
              <p className="mt-2 text-slate-600">
                Ask your PDF questions and get clear, accurate answers.
              </p>
            </div>
            <div className="rounded-2xl border bg-white p-6 shadow-sm">
              <h3 className="text-lg font-semibold">Accessible Anywhere</h3>
              <p className="mt-2 text-slate-600">
                Works right in your browser — no installs, no hassle.
              </p>
            </div>
            <div className="rounded-2xl border bg-white p-6 shadow-sm">
              <h3 className="text-lg font-semibold">Simple & Free</h3>
              <p className="mt-2 text-slate-600">
                Start learning with zero setup. Upload and go.
              </p>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="mx-auto max-w-6xl px-4 pb-16">
          <div className="rounded-3xl bg-gradient-to-r from-indigo-600 via-fuchsia-600 to-pink-600 p-1">
            <div className="rounded-3xl bg-white p-8 sm:p-10 text-center">
              <h4 className="text-xl font-semibold">Get started with Edu-Easy</h4>
              <p className="mt-2 max-w-2xl mx-auto text-slate-600">
                Upload your first PDF and see how Edu-Easy makes studying
                simpler, faster, and more effective.
              </p>
              <div className="mt-6">
                <button
                  onClick={() =>
                    document
                      .querySelector<HTMLInputElement>(
                        'input[type=file]'
                      )
                      ?.click()
                  }
                  className="rounded-xl bg-slate-900 px-5 py-2.5 text-sm font-medium text-white hover:bg-slate-800 active:scale-[0.99]"
                >
                  Upload a PDF
                </button>
              </div>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t py-6 text-center text-sm text-slate-500">
        Edu-Easy — Learn smarter, not harder.
      </footer>
    </div>
  );
}

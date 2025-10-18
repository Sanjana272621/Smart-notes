// Helper to export a specific DOM node to PDF using html2pdf.js
// Safe for Next.js (dynamic import; no SSR usage here).
// TypeScript-friendly with narrow literals & a final `any` cast to handle community typings.

export async function exportToPdf(el: HTMLElement, filename = 'summary.pdf'): Promise<void> {
  // Dynamic import to avoid SSR issues
  const html2pdf = (await import('html2pdf.js')).default;

  // Options tuned for good quality and A4 pagination
  const opt = {
    margin: 10,
    filename,
    image: { type: 'jpeg' as const, quality: 0.98 },
    html2canvas: { scale: 2, useCORS: true },
    jsPDF: { unit: 'mm' as const, format: 'a4' as const, orientation: 'portrait' as const },
    pagebreak: { mode: ['css', 'legacy'] as const },
  };

  // Some type definitions of html2pdf are incomplete; cast to any to keep TS happy
  await (html2pdf() as any).set(opt as any).from(el).save();
}

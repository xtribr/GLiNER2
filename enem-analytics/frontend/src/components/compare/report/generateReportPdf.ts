import { createRoot } from 'react-dom/client';
import React from 'react';
import type { ReportData } from './types';
import ReportDocument from './ReportDocument';

export interface GeneratedReportFile { filename: string; }

export async function generateReportPdf(data: ReportData): Promise<GeneratedReportFile> {
  const html2pdf = (await import('html2pdf.js')).default;
  const container = document.createElement('div');
  container.style.position = 'fixed';
  container.style.left = '-10000px';
  container.style.top = '0';
  container.style.width = '210mm';
  document.body.appendChild(container);

  const root = createRoot(container);
  await new Promise<void>((resolve) => {
    root.render(React.createElement(ReportDocument, { data, onReady: resolve }));
  });
  // garante layout/charts montados
  await new Promise((r) => setTimeout(r, 400));

  const filename = `XTRI_Relatorio_${slug(data.schoolA.nome_escola)}_vs_${slug(data.schoolB.nome_escola)}.pdf`;
  await html2pdf().set({
    margin: 0,
    filename,
    image: { type: 'jpeg', quality: 0.96 },
    html2canvas: { scale: 2, useCORS: true, logging: false },
    jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
    pagebreak: { mode: ['css', 'legacy'] },
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any).from(container.firstElementChild as HTMLElement).save();

  root.unmount();
  document.body.removeChild(container);
  return { filename };
}

function slug(s: string): string {
  return s.normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[^a-zA-Z0-9]+/g, '_').slice(0, 18);
}

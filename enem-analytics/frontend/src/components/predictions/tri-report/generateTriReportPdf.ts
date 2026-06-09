import React from 'react';
import { renderScopedHtml, printHtml, slug } from '@/components/compare/report/generateReportPdf';
import type { GeneratedReportFile } from '@/components/compare/report/generateReportPdf';
import TriReportDocument from './TriReportDocument';
import type { TriReportData } from './triTypes';

/** Gera o TRI REPORT em PDF via impressão nativa (mesma engine do relatório comparativo). */
export async function generateTriReportPdf(
  data: TriReportData,
  targetWindow?: Window | null,
): Promise<GeneratedReportFile> {
  const filename = `XTRI_TRI_${slug(data.schoolName)}`;
  const { html } = await renderScopedHtml(
    (onReady) => React.createElement(TriReportDocument, { data, onReady }),
    'xtri-tri',
    filename,
  );
  return printHtml(html, filename, targetWindow);
}

export default generateTriReportPdf;

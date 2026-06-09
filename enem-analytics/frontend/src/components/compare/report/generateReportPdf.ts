import { createRoot } from 'react-dom/client';
import React from 'react';
import type { ReportData } from './types';
import ReportDocument from './ReportDocument';

export interface GeneratedReportFile { filename: string; }

/**
 * Coleta as regras de CSS do relatório (escopadas em `.xtri-report`) e a regra
 * `@page` já carregadas no documento (vindas do import de ReportDocument.css),
 * para injetá-las no documento de impressão isolado.
 */
function collectReportCss(): string {
  let css = '';
  for (const sheet of Array.from(document.styleSheets)) {
    let rules: CSSRuleList | null = null;
    try {
      rules = sheet.cssRules;
    } catch {
      continue; // folha cross-origin — ignora
    }
    if (!rules) continue;
    for (const rule of Array.from(rules)) {
      const text = rule.cssText;
      if (text.includes('.xtri-report') || text.startsWith('@page')) {
        css += text + '\n';
      }
    }
  }
  return css;
}

/**
 * Renderiza o ReportDocument fora da tela e devolve um documento HTML autônomo
 * (markup + CSS escopado + @page A4). É a MESMA string usada para imprimir e
 * para a verificação visual — garante fidelidade.
 */
export async function renderReportToHtml(data: ReportData): Promise<{ html: string; filename: string }> {
  const filename = `XTRI_Relatorio_${slug(data.schoolA.nome_escola)}_vs_${slug(data.schoolB.nome_escola)}`;

  const holder = document.createElement('div');
  holder.style.position = 'fixed';
  holder.style.left = '-10000px';
  holder.style.top = '0';
  document.body.appendChild(holder);

  const root = createRoot(holder);
  await new Promise<void>((resolve) => {
    root.render(React.createElement(ReportDocument, { data, onReady: resolve }));
  });
  await new Promise((r) => setTimeout(r, 200)); // garante layout/SVGs montados

  const reportHtml = holder.querySelector('.xtri-report')?.outerHTML ?? holder.innerHTML;
  const css = collectReportCss();

  root.unmount();
  holder.remove();

  const html =
    `<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">` +
    `<title>${filename}</title><style>${css}</style></head>` +
    `<body>${reportHtml}</body></html>`;

  return { html, filename };
}

/**
 * Gera o PDF via IMPRESSÃO NATIVA do navegador (texto vetorial + margens @page A4).
 * Renderiza o relatório num iframe oculto e dispara `print()` — o usuário escolhe
 * "Salvar como PDF". Substitui o antigo fluxo html2pdf/html2canvas (que rasterizava
 * e colapsava os espaços do texto).
 */
export async function generateReportPdf(
  data: ReportData,
  targetWindow?: Window | null,
): Promise<GeneratedReportFile> {
  const { html, filename } = await renderReportToHtml(data);

  // Caminho preferido: aba aberta no gesto do clique (funciona no Safari, onde
  // print via iframe oculto não dispara). A aba mostra o relatório e abre o
  // diálogo de impressão; o usuário escolhe "Salvar como PDF".
  if (targetWindow && !targetWindow.closed) {
    targetWindow.document.open();
    targetWindow.document.write(html);
    targetWindow.document.close();
    try { targetWindow.document.title = filename; } catch { /* noop */ }
    const doPrint = () => { try { targetWindow.focus(); targetWindow.print(); } catch { /* noop */ } };
    if (targetWindow.document.readyState === 'complete') {
      setTimeout(doPrint, 500);
    } else {
      targetWindow.onload = () => setTimeout(doPrint, 500);
    }
    return { filename: `${filename}.pdf` };
  }

  // Fallback (sem janela pré-aberta / popup bloqueado): iframe oculto + print.
  const iframe = document.createElement('iframe');
  iframe.setAttribute('aria-hidden', 'true');
  iframe.style.position = 'fixed';
  iframe.style.left = '-10000px';
  iframe.style.top = '0';
  iframe.style.width = '210mm';
  iframe.style.height = '297mm';
  iframe.style.border = '0';
  document.body.appendChild(iframe);

  const idoc = iframe.contentWindow!.document;
  idoc.open();
  idoc.write(html);
  idoc.close();
  await new Promise((r) => setTimeout(r, 400));
  const win = iframe.contentWindow!;
  win.focus();
  win.onafterprint = () => iframe.remove();
  win.print();
  setTimeout(() => { if (document.body.contains(iframe)) iframe.remove(); }, 60000);

  return { filename: `${filename}.pdf` };
}

function slug(s: string): string {
  return s.normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[^a-zA-Z0-9]+/g, '_').slice(0, 18);
}

export default generateReportPdf;

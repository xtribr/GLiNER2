import { createRoot } from 'react-dom/client';
import React from 'react';
import type { ReportData } from './types';
import ReportDocument from './ReportDocument';

export interface GeneratedReportFile { filename: string; }

/**
 * Coleta as regras de CSS escopadas (que contêm `scopeSelector`) e a regra `@page`
 * já carregadas no documento (vindas do import do .css do relatório), para injetá-las
 * no documento de impressão isolado.
 */
function collectScopedCss(scopeSelector: string): string {
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
      if (text.includes(scopeSelector) || text.startsWith('@page')) {
        css += text + '\n';
      }
    }
  }
  return css;
}

/**
 * Renderiza um documento de relatório fora da tela e devolve um documento HTML
 * autônomo (markup + CSS escopado + @page). Genérico — serve a qualquer relatório
 * print-native. É a MESMA string usada para imprimir e para a verificação visual.
 */
export async function renderScopedHtml(
  renderEl: (onReady: () => void) => React.ReactElement,
  scopeClass: string,
  filename: string,
): Promise<{ html: string; filename: string }> {
  const holder = document.createElement('div');
  holder.style.position = 'fixed';
  holder.style.left = '-10000px';
  holder.style.top = '0';
  document.body.appendChild(holder);

  const root = createRoot(holder);
  await new Promise<void>((resolve) => {
    root.render(renderEl(resolve));
  });
  await new Promise((r) => setTimeout(r, 200)); // garante layout/SVGs montados

  const reportHtml = holder.querySelector(`.${scopeClass}`)?.outerHTML ?? holder.innerHTML;
  const css = collectScopedCss(`.${scopeClass}`);

  root.unmount();
  holder.remove();

  const html =
    `<!DOCTYPE html><html lang="pt-BR"><head><meta charset="utf-8">` +
    `<title>${filename}</title><style>${css}</style></head>` +
    `<body>${reportHtml}</body></html>`;

  return { html, filename };
}

/**
 * Imprime um HTML autônomo via IMPRESSÃO NATIVA do navegador (texto vetorial +
 * margens @page A4). Caminho preferido: aba pré-aberta no gesto do clique (Safari);
 * fallback: iframe oculto + print(). O usuário escolhe "Salvar como PDF".
 */
export async function printHtml(
  html: string,
  filename: string,
  targetWindow?: Window | null,
): Promise<GeneratedReportFile> {
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

export function slug(s: string): string {
  return s.normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/[^a-zA-Z0-9]+/g, '_').slice(0, 18);
}

// ── Relatório comparativo de escolas (API existente, comportamento inalterado) ──

export async function renderReportToHtml(data: ReportData): Promise<{ html: string; filename: string }> {
  const filename = `XTRI_Relatorio_${slug(data.schoolA.nome_escola)}_vs_${slug(data.schoolB.nome_escola)}`;
  return renderScopedHtml((onReady) => React.createElement(ReportDocument, { data, onReady }), 'xtri-report', filename);
}

export async function generateReportPdf(
  data: ReportData,
  targetWindow?: Window | null,
): Promise<GeneratedReportFile> {
  const { html, filename } = await renderReportToHtml(data);
  return printHtml(html, filename, targetWindow);
}

export default generateReportPdf;

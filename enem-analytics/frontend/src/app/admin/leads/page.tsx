'use client';

import { useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import { useAuth } from '@/lib/auth-context';
import { api, type Lead } from '@/lib/api';
import { Users, Search, Download, Phone, Mail, CheckCircle2, Clock, Loader2, ExternalLink } from 'lucide-react';

function onlyDigits(s: string): string {
  return (s || '').replace(/\D/g, '');
}

// Monta link de WhatsApp (assume Brasil quando vem sem DDI).
function whatsappUrl(telefone: string): string {
  const d = onlyDigits(telefone);
  const withDdi = d.length <= 11 ? `55${d}` : d;
  return `https://wa.me/${withDdi}`;
}

function formatDate(iso: string): string {
  if (!iso) return '—';
  const d = new Date(iso.includes('T') ? iso : iso.replace(' ', 'T'));
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString('pt-BR', {
    day: '2-digit', month: '2-digit', year: 'numeric',
    hour: '2-digit', minute: '2-digit', timeZone: 'America/Sao_Paulo',
  });
}

function downloadCsv(leads: Lead[]): void {
  const headers = ['Escola', 'INEP', 'Contato', 'Cargo', 'Telefone', 'E-mail', 'E-mail verificado', 'Cadastro'];
  const esc = (v: string) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const rows = leads.map((l) => [
    l.nome_escola, l.codigo_inep, l.nome_contato, l.cargo, l.telefone,
    l.email, l.email_verified ? 'Sim' : 'Não', formatDate(l.created_at),
  ].map(esc).join(';'));
  // BOM para acentos abrirem corretos no Excel (pt-BR usa ; como separador)
  const csv = '﻿' + [headers.map(esc).join(';'), ...rows].join('\r\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `leads-rankingenem-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export default function LeadsPage() {
  const router = useRouter();
  const { session, user, isLoading: authLoading, isAdmin } = useAuth();
  const [search, setSearch] = useState('');

  useEffect(() => {
    if (!authLoading && user && !isAdmin) router.push('/');
  }, [authLoading, isAdmin, router, user]);

  const {
    data: leads = [],
    isLoading: loading,
    isError,
  } = useQuery({
    queryKey: ['admin-leads'],
    queryFn: () => api.getLeads(),
    enabled: isAdmin,
    retry: false,
  });

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return leads;
    return leads.filter((l) =>
      [l.nome_escola, l.nome_contato, l.cargo, l.telefone, l.email, l.codigo_inep]
        .some((f) => (f || '').toLowerCase().includes(q)),
    );
  }, [leads, search]);

  if (!session || !user || !isAdmin) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <Loader2 className="h-7 w-7 animate-spin text-sky-500" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-black tracking-tight text-slate-900">
            <Users className="h-6 w-6 text-sky-500" /> Leads
          </h1>
          <p className="mt-1 text-sm text-slate-500">
            Escolas que se cadastraram pela vitrine pública · <b>{leads.length}</b> no total
          </p>
        </div>
        <button
          onClick={() => downloadCsv(filtered)}
          disabled={!filtered.length}
          className="inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-sky-500 to-orange-500 px-4 py-2.5 text-sm font-bold text-white shadow-lg shadow-sky-500/20 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Download className="h-4 w-4" /> Exportar CSV{filtered.length ? ` (${filtered.length})` : ''}
        </button>
      </div>

      <div className="relative mt-6">
        <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Buscar por escola, contato, telefone, e-mail…"
          className="w-full rounded-xl border border-slate-200 bg-white py-2.5 pl-10 pr-4 text-sm outline-none transition focus:border-sky-400 focus:ring-2 focus:ring-sky-400/20"
        />
      </div>

      {isError && (
        <div className="mt-6 rounded-xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-700">
          Não foi possível carregar os leads. Tente novamente.
        </div>
      )}

      {loading ? (
        <div className="flex min-h-[40vh] items-center justify-center">
          <Loader2 className="h-7 w-7 animate-spin text-sky-500" />
        </div>
      ) : !filtered.length ? (
        <div className="mt-10 rounded-2xl border border-dashed border-slate-200 bg-slate-50 p-12 text-center">
          <Users className="mx-auto h-10 w-10 text-slate-300" />
          <p className="mt-3 text-sm font-semibold text-slate-600">
            {leads.length ? 'Nenhum lead corresponde à busca.' : 'Ainda não há leads cadastrados.'}
          </p>
        </div>
      ) : (
        <div className="mt-6 overflow-x-auto rounded-2xl border border-slate-200 bg-white shadow-sm">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-left text-[11px] font-bold uppercase tracking-wider text-slate-400">
              <tr>
                <th className="px-4 py-3">Escola</th>
                <th className="px-4 py-3">Contato</th>
                <th className="px-4 py-3">WhatsApp</th>
                <th className="px-4 py-3">E-mail</th>
                <th className="px-4 py-3 whitespace-nowrap">Cadastro</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filtered.map((l) => (
                <tr key={l.id} className="transition hover:bg-sky-50/40">
                  <td className="px-4 py-3">
                    {l.codigo_inep ? (
                      <a
                        href={`/schools/${l.codigo_inep}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        title="Abrir os dados da escola nos microdados 2025"
                        className="group inline-flex items-center gap-1.5 font-semibold text-slate-900 transition hover:text-sky-600 hover:underline"
                      >
                        {l.nome_escola}
                        <ExternalLink className="h-3.5 w-3.5 shrink-0 text-slate-300 transition group-hover:text-sky-500" />
                      </a>
                    ) : (
                      <div className="font-semibold text-slate-900">{l.nome_escola}</div>
                    )}
                    <div className="font-mono text-[11px] text-slate-400">{l.codigo_inep}</div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="font-medium text-slate-800">{l.nome_contato || '—'}</div>
                    <div className="text-[12px] text-slate-500">{l.cargo || '—'}</div>
                  </td>
                  <td className="px-4 py-3">
                    {l.telefone ? (
                      <a href={whatsappUrl(l.telefone)} target="_blank" rel="noopener noreferrer"
                        className="inline-flex items-center gap-1.5 font-medium text-emerald-600 hover:underline">
                        <Phone className="h-3.5 w-3.5" /> {l.telefone}
                      </a>
                    ) : <span className="text-slate-300">—</span>}
                  </td>
                  <td className="px-4 py-3">
                    {l.email ? (
                      <a href={`mailto:${l.email}`} className="inline-flex items-center gap-1.5 text-sky-600 hover:underline">
                        <Mail className="h-3.5 w-3.5" /> {l.email}
                      </a>
                    ) : <span className="text-slate-300">—</span>}
                    {l.email_verified ? (
                      <span className="mt-1 flex items-center gap-1 text-[11px] font-semibold text-emerald-600">
                        <CheckCircle2 className="h-3 w-3" /> verificado
                      </span>
                    ) : (
                      <span className="mt-1 flex items-center gap-1 text-[11px] text-slate-400">
                        <Clock className="h-3 w-3" /> não verificado
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-slate-600">{formatDate(l.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

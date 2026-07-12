'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { ArrowLeft, ArrowRight, Check, Search, ShieldCheck } from 'lucide-react';
import { api } from '@/lib/api';
import { supabase } from '@/lib/supabase';

type Step = 'escola' | 'conta' | 'pronto';

interface SchoolResult {
  codigo_inep: string;
  nome_escola: string;
  uf: string | null;
  ultimo_ano: number;
}

const CARGOS = [
  'Diretor(a)',
  'Coordenador(a)',
  'Professor(a)',
  'Secretaria',
  'Mantenedor(a)',
  'Outro',
];

export default function CadastroPage() {
  const router = useRouter();
  const [step, setStep] = useState<Step>('escola');

  // Passo 1 — escola
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SchoolResult[]>([]);
  const [selected, setSelected] = useState<SchoolResult | null>(null);
  const [searching, setSearching] = useState(false);

  // Passo 2 — conta
  const [nomeContato, setNomeContato] = useState('');
  const [cargo, setCargo] = useState('');
  const [email, setEmail] = useState('');
  const [telefone, setTelefone] = useState('');
  const [senha, setSenha] = useState('');
  const [aceite, setAceite] = useState(false);
  const [website, setWebsite] = useState(''); // honeypot
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Busca typeahead com debounce
  useEffect(() => {
    const q = query.trim();
    if (q.length < 2 || selected) return;

    let cancelled = false;
    const t = setTimeout(async () => {
      try {
        const data = await api.searchSchools(q, 8);
        if (!cancelled) setResults(data as SchoolResult[]);
      } catch {
        if (!cancelled) setResults([]);
      } finally {
        if (!cancelled) setSearching(false);
      }
    }, 300);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [query, selected]);

  function handleSchoolQueryChange(value: string) {
    setQuery(value);
    setResults([]);
    setSearching(value.trim().length >= 2);
  }

  function handleSchoolSelection(school: SchoolResult) {
    setSelected(school);
    setResults([]);
    setSearching(false);
  }

  function handleSchoolChange() {
    setSelected(null);
    setQuery('');
    setResults([]);
    setSearching(false);
  }

  // Acesso imediato → redireciona ao painel após "pronto"
  useEffect(() => {
    if (step !== 'pronto' || !selected) return;
    const t = setTimeout(() => router.push(`/schools/${selected.codigo_inep}`), 1500);
    return () => clearTimeout(t);
  }, [step, selected, router]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!selected) return;
    if (!aceite) {
      setError('É necessário aceitar os Termos de Uso.');
      return;
    }
    if (senha.length < 6) {
      setError('A senha deve ter no mínimo 6 caracteres.');
      return;
    }
    setLoading(true);
    try {
      await api.signup({
        codigo_inep: selected.codigo_inep,
        nome_escola: selected.nome_escola,
        nome_contato: nomeContato,
        cargo,
        email,
        telefone,
        senha,
        aceite_termos: aceite,
        website,
      });

      // Modelo A: acesso na hora — autentica logo após criar a conta.
      const { error: signErr } = await supabase.auth.signInWithPassword({ email, password: senha });
      if (signErr) {
        // Conta criada, mas o login automático falhou — manda para o login.
        router.push('/login');
        return;
      }
      setStep('pronto');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Não foi possível concluir o cadastro.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-[#f5fbff] px-4 py-8">
      <div className="mx-auto w-full max-w-md">
        <Link href="/" className="mb-4 inline-flex items-center gap-2 text-sm font-semibold text-[#139ED3] hover:underline">
          <ArrowLeft className="h-4 w-4" />
          Voltar à vitrine
        </Link>

        <div className="rounded-3xl border border-[#28B7ED]/20 bg-white p-7 shadow-xl shadow-[#28B7ED]/10">
          <h1 className="text-center text-sm font-black uppercase tracking-[0.16em] text-slate-400">Cadastrar</h1>

          {/* Stepper */}
          <div className="my-5 flex items-center justify-center gap-2">
            <Stepper label="Escola" state={step === 'escola' ? 'on' : 'done'} />
            <Bar done={step !== 'escola'} />
            <Stepper label="Conta" state={step === 'conta' ? 'on' : step === 'pronto' ? 'done' : 'idle'} />
            <Bar done={step === 'pronto'} />
            <Stepper label="Pronto" state={step === 'pronto' ? 'on' : 'idle'} />
          </div>

          {step === 'escola' && (
            <div>
              <h2 className="text-center text-2xl font-black tracking-tight text-slate-950">Encontre sua escola</h2>
              <p className="mt-1 text-center text-sm text-slate-500">
                Você está a um passo de ver os dados completos da sua escola no ENEM.
              </p>

              {selected ? (
                <div className="mt-5 flex items-center justify-between rounded-2xl border border-[#28B7ED] bg-[#E9F8FE] p-4">
                  <div>
                    <p className="font-bold text-slate-950">{selected.nome_escola}</p>
                    <p className="text-xs text-slate-500">{selected.uf ? `${selected.uf} · ` : ''}INEP {selected.codigo_inep}</p>
                  </div>
                  <button onClick={handleSchoolChange} className="text-xs font-semibold text-[#139ED3] hover:underline">
                    trocar
                  </button>
                </div>
              ) : (
                <>
                  <div className="mt-5 flex items-center gap-2 rounded-xl border-2 border-[#28B7ED] px-3 py-3">
                    <Search className="h-4 w-4 text-slate-400" />
                    <input
                      autoFocus
                      value={query}
                      onChange={(e) => handleSchoolQueryChange(e.target.value)}
                      placeholder="Digite o nome da sua escola"
                      className="w-full bg-transparent text-sm text-slate-800 outline-none placeholder:text-slate-400"
                    />
                  </div>
                  {searching && <p className="mt-2 text-xs text-slate-400">Buscando…</p>}
                  <div className="mt-2 space-y-2">
                    {results.map((r) => (
                      <button
                        key={r.codigo_inep}
                        onClick={() => handleSchoolSelection(r)}
                        className="flex w-full items-center justify-between rounded-xl border border-slate-200 p-3 text-left transition hover:border-[#28B7ED]"
                      >
                        <div>
                          <p className="text-sm font-bold text-slate-950">{r.nome_escola}</p>
                          <p className="text-xs text-slate-500">{r.uf ? `${r.uf} · ` : ''}INEP {r.codigo_inep}</p>
                        </div>
                        <ArrowRight className="h-4 w-4 text-slate-300" />
                      </button>
                    ))}
                  </div>
                </>
              )}

              <button
                disabled={!selected}
                onClick={() => setStep('conta')}
                className="mt-5 flex w-full items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-[#28B7ED] to-[#FF4B2E] py-3.5 text-sm font-bold text-white shadow-lg shadow-[#28B7ED]/20 transition hover:brightness-105 disabled:cursor-not-allowed disabled:opacity-40"
              >
                Avançar etapa
                <ArrowRight className="h-4 w-4" />
              </button>

              <div className="mt-4 flex items-center justify-between border-t border-slate-100 pt-4 text-[11px] text-slate-500">
                <span className="flex items-center gap-1"><ShieldCheck className="h-3.5 w-3.5" /> Dados históricos</span>
                <span>✨ Insights da IA</span>
                <span>🔒 100% gratuito</span>
              </div>
            </div>
          )}

          {step === 'conta' && selected && (
            <form onSubmit={handleSubmit}>
              <h2 className="text-center text-2xl font-black tracking-tight text-slate-950">Crie sua conta</h2>
              <p className="mt-1 text-center text-sm text-slate-500">
                Falta pouco! Crie sua senha e acesse na hora — sem esperar e-mail.
              </p>

              <div className="mt-4 flex items-center gap-3 rounded-2xl border border-[#28B7ED]/30 bg-[#E9F8FE] p-3">
                <div className="text-xl">🏫</div>
                <div>
                  <p className="text-sm font-bold text-slate-950">{selected.nome_escola}</p>
                  <p className="text-xs text-slate-500">{selected.uf ? `${selected.uf} · ` : ''}INEP {selected.codigo_inep}</p>
                </div>
              </div>

              {/* honeypot — escondido de humanos */}
              <input
                type="text"
                tabIndex={-1}
                autoComplete="off"
                value={website}
                onChange={(e) => setWebsite(e.target.value)}
                style={{ position: 'absolute', left: '-9999px', width: 1, height: 1, opacity: 0 }}
                aria-hidden="true"
              />

              <Field label="Nome completo">
                <input required value={nomeContato} onChange={(e) => setNomeContato(e.target.value)} placeholder="Seu nome completo" className={inputCls} />
              </Field>
              <Field label="Cargo na escola">
                <select required value={cargo} onChange={(e) => setCargo(e.target.value)} className={inputCls}>
                  <option value="" disabled>Selecione seu cargo</option>
                  {CARGOS.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </Field>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                <Field label="E-mail institucional">
                  <input required type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="voce@escola.com.br" className={inputCls} />
                </Field>
                <Field label="Celular (WhatsApp)">
                  <input required value={telefone} onChange={(e) => setTelefone(e.target.value)} placeholder="(11) 99999-9999" className={inputCls} />
                </Field>
              </div>
              <Field label="Crie uma senha">
                <input required type="password" value={senha} onChange={(e) => setSenha(e.target.value)} placeholder="Mínimo 6 caracteres" className={inputCls} />
              </Field>

              <label className="mt-4 flex items-start gap-2 text-xs leading-relaxed text-slate-600">
                <input type="checkbox" checked={aceite} onChange={(e) => setAceite(e.target.checked)} className="mt-0.5 h-4 w-4 accent-[#28B7ED]" />
                <span>
                  Li e aceito os <Link href="/termos-de-uso" className="font-semibold text-[#139ED3] underline">Termos de Uso</Link> e a{' '}
                  <Link href="/politica-de-privacidade" className="font-semibold text-[#139ED3] underline">Política de Privacidade</Link>. Os dados exibidos derivam dos microdados públicos do INEP.
                </span>
              </label>

              {error && <p className="mt-3 rounded-xl bg-[#FFF0EB] px-3 py-2 text-sm font-semibold text-[#E43E24]">{error}</p>}

              <div className="mt-4 flex gap-2">
                <button type="button" onClick={() => setStep('escola')} className="flex items-center justify-center gap-1 rounded-2xl border border-slate-200 px-4 py-3 text-sm font-semibold text-slate-600">
                  <ArrowLeft className="h-4 w-4" /> Voltar
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="flex flex-1 items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-[#28B7ED] to-[#FF4B2E] py-3 text-sm font-bold text-white shadow-lg shadow-[#28B7ED]/20 transition hover:brightness-105 disabled:opacity-60"
                >
                  {loading ? 'Criando…' : 'Criar conta e acessar'}
                  {!loading && <ArrowRight className="h-4 w-4" />}
                </button>
              </div>
              <p className="mt-3 text-center text-xs text-slate-500">
                Já tem conta? <Link href="/login" className="font-bold text-[#139ED3]">Entrar</Link>
              </p>
            </form>
          )}

          {step === 'pronto' && selected && (
            <div className="text-center">
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-green-100 text-green-600">
                <Check className="h-8 w-8" />
              </div>
              <h2 className="mt-4 text-2xl font-black tracking-tight text-slate-950">Conta criada! Acesso liberado.</h2>
              <p className="mt-2 text-sm text-slate-500">
                Você já pode ver a análise completa da <strong>{selected.nome_escola}</strong>. Redirecionando…
              </p>
              <button
                onClick={() => router.push(`/schools/${selected.codigo_inep}`)}
                className="mt-5 inline-flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-[#28B7ED] to-[#FF4B2E] px-5 py-3 text-sm font-bold text-white shadow-lg shadow-[#28B7ED]/20"
              >
                Ir para o painel da minha escola
                <ArrowRight className="h-4 w-4" />
              </button>
            </div>
          )}
        </div>

        <footer className="mt-5 text-center text-xs text-slate-500">
          <Link href="/termos-de-uso" className="font-semibold text-[#139ED3] hover:underline">Termos de Uso</Link> ·{' '}
          <Link href="/politica-de-privacidade" className="font-semibold text-[#139ED3] hover:underline">Política de Privacidade</Link> ·{' '}
          <Link href="/politica-de-compliance-ia" className="font-semibold text-[#139ED3] hover:underline">Compliance IA</Link>
        </footer>
      </div>
    </div>
  );
}

const inputCls =
  'mt-1 w-full rounded-xl border border-slate-200 px-3 py-2.5 text-sm text-slate-800 outline-none focus:border-[#28B7ED] focus:ring-2 focus:ring-[#28B7ED]/20';

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mt-3">
      <label className="text-xs font-semibold text-slate-700">{label}</label>
      {children}
    </div>
  );
}

function Stepper({ label, state }: { label: string; state: 'idle' | 'on' | 'done' }) {
  return (
    <div className="flex w-14 flex-col items-center gap-1 sm:w-16">
      <div
        className={
          'flex h-9 w-9 items-center justify-center rounded-full text-sm font-black ' +
          (state === 'on'
            ? 'bg-gradient-to-br from-[#28B7ED] to-[#139ED3] text-white'
            : state === 'done'
              ? 'border-2 border-green-500 bg-white text-green-600'
              : 'border-2 border-slate-200 bg-slate-50 text-slate-400')
        }
      >
        {state === 'done' ? <Check className="h-4 w-4" /> : label[0]}
      </div>
      <span className={'text-[11px] font-semibold ' + (state === 'idle' ? 'text-slate-400' : 'text-slate-600')}>{label}</span>
    </div>
  );
}

function Bar({ done }: { done: boolean }) {
  return <div className={'mb-4 h-0.5 w-6 sm:w-8 ' + (done ? 'bg-green-500' : 'bg-slate-200')} />;
}

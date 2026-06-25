'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { formatTriScore } from '@/lib/utils';

type Ring = [number, number][];
interface Geometry {
  type: 'Polygon' | 'MultiPolygon';
  coordinates: Ring[] | Ring[][];
}
interface Feature {
  properties: { sigla: string };
  geometry: Geometry;
}
interface UfRec {
  uf: string;
  media: number;
  escolas: number;
  media_cn: number;
  media_ch: number;
  media_lc: number;
  media_mt: number;
  media_redacao: number;
  media_prev: number | null;
  ano: number;
  ano_prev: number | null;
}

const W = 520;
const H = 540;
const PAD = 12;

// Cor: escala da menor (cyan claro e nítido) à maior média (azul profundo).
function lerpColor(t: number): string {
  const a = [144, 206, 240]; // #90cef0
  const b = [7, 64, 105]; // #074069
  const clamped = Math.max(0, Math.min(1, t));
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * clamped));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function eachCoord(g: Geometry, cb: (p: [number, number]) => void) {
  if (g.type === 'Polygon') {
    (g.coordinates as Ring[]).forEach((ring) => ring.forEach(cb));
  } else {
    (g.coordinates as Ring[][]).forEach((poly) => poly.forEach((ring) => ring.forEach(cb)));
  }
}

const fmtDelta = (d: number): string => `${d >= 0 ? '+' : '−'}${Math.abs(d).toFixed(1).replace('.', ',')}`;

export default function BrazilChoropleth() {
  const router = useRouter();
  const wrapRef = useRef<HTMLDivElement>(null);
  const [features, setFeatures] = useState<Feature[] | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [pos, setPos] = useState({ x: 0, y: 0, w: 1, h: 1 });

  useEffect(() => {
    let alive = true;
    fetch('/brazil-uf.geojson')
      .then((r) => r.json())
      .then((d) => alive && setFeatures(d.features))
      .catch(() => alive && setFeatures([]));
    return () => {
      alive = false;
    };
  }, []);

  const { data } = useQuery({ queryKey: ['statsByUf'], queryFn: api.getStatsByUf });

  const byUf = useMemo(
    () => Object.fromEntries((data?.by_uf ?? []).map((r) => [r.uf, r])) as Record<string, UfRec>,
    [data],
  );
  const medias = (data?.by_uf ?? []).map((r) => r.media);
  const hasData = medias.length > 0;
  const min = hasData ? Math.min(...medias) : 0;
  const max = hasData ? Math.max(...medias) : 1;
  const ano = data?.by_uf?.[0]?.ano;

  const paths = useMemo(() => {
    if (!features || features.length === 0) return [];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    features.forEach((f) =>
      eachCoord(f.geometry, ([x, y]) => {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }),
    );
    const s = Math.min((W - 2 * PAD) / (maxX - minX), (H - 2 * PAD) / (maxY - minY));
    const offX = PAD + (W - 2 * PAD - (maxX - minX) * s) / 2;
    const offY = PAD + (H - 2 * PAD - (maxY - minY) * s) / 2;
    const px = (x: number) => offX + (x - minX) * s;
    const py = (y: number) => H - offY - (y - minY) * s; // inverte Y (lat cresce p/ cima)

    const ringToPath = (ring: Ring) =>
      ring.map(([x, y], i) => `${i ? 'L' : 'M'}${px(x).toFixed(1)} ${py(y).toFixed(1)}`).join(' ') + ' Z';

    return features.map((f) => {
      const g = f.geometry;
      const d =
        g.type === 'Polygon'
          ? (g.coordinates as Ring[]).map(ringToPath).join(' ')
          : (g.coordinates as Ring[][]).flatMap((poly) => poly.map(ringToPath)).join(' ');
      return { sigla: f.properties.sigla, d };
    });
  }, [features]);

  if (!features) {
    return <div className="flex h-72 items-center justify-center text-sm text-slate-400">Carregando mapa…</div>;
  }

  const rec = hover ? byUf[hover] : null;
  const delta = rec && rec.media_prev != null ? rec.media - rec.media_prev : null;
  const flipX = pos.x > pos.w * 0.5;
  const flipY = pos.y > pos.h * 0.55;

  const onMove = (e: React.MouseEvent) => {
    const r = wrapRef.current?.getBoundingClientRect();
    if (r) setPos({ x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height });
  };
  const goToUf = (uf: string) => router.push(`/schools?uf=${uf}`);

  const areas: [string, number][] = rec
    ? [
        ['Média Geral', rec.media],
        ['Redação', rec.media_redacao],
        ['Linguagens', rec.media_lc],
        ['Matemática', rec.media_mt],
        ['Natureza', rec.media_cn],
        ['Humanas', rec.media_ch],
      ]
    : [];

  return (
    <div>
      <div ref={wrapRef} className="relative mx-auto max-w-[480px]" onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Mapa de calor da média TRI por estado">
          {paths.map((p) => {
            const r = byUf[p.sigla];
            const fill = r ? lerpColor((r.media - min) / (max - min || 1)) : '#e2e8f0';
            return (
              <path
                key={p.sigla}
                d={p.d}
                fill={fill}
                stroke={hover === p.sigla ? '#FF4B2E' : '#ffffff'}
                strokeWidth={hover === p.sigla ? 1.6 : 0.5}
                className="cursor-pointer transition-[stroke]"
                onMouseEnter={(e) => {
                  setHover(p.sigla);
                  onMove(e);
                }}
                onMouseMove={onMove}
                onClick={() => goToUf(p.sigla)}
              >
                <title>{`${p.sigla}${r ? ` — média TRI ${formatTriScore(r.media)} · ${r.escolas} escolas` : ' — sem dados'}`}</title>
              </path>
            );
          })}
        </svg>

        {rec && (
          <div
            className="pointer-events-none absolute z-20 w-[244px] rounded-2xl border border-slate-200 bg-white p-4 shadow-xl"
            style={{
              left: pos.x,
              top: pos.y,
              transform: `translate(${flipX ? 'calc(-100% - 16px)' : '16px'}, ${flipY ? 'calc(-100% - 16px)' : '16px'})`,
            }}
          >
            <div className="flex items-start justify-between">
              <span className="text-xl font-black leading-none text-slate-900">{rec.uf}</span>
              {delta != null && (
                <span className={`flex items-center gap-1 text-sm font-bold ${delta >= 0 ? 'text-emerald-600' : 'text-rose-600'}`}>
                  {delta >= 0 ? '▲' : '▼'} {fmtDelta(delta)}
                </span>
              )}
            </div>
            <p className="mt-0.5 text-xs text-slate-500">
              {rec.escolas.toLocaleString('pt-BR')} escolas · {rec.ano}
            </p>

            {rec.media_prev != null && (
              <div className="mt-2.5 flex items-center justify-between gap-2 rounded-lg bg-slate-50 px-2.5 py-1.5 text-[11px]">
                <span className="text-slate-500">
                  {rec.ano_prev}: <b className="text-slate-600">{formatTriScore(rec.media_prev)}</b>
                </span>
                <span className="text-slate-300">→</span>
                <span className="text-slate-500">
                  {rec.ano}: <b className="text-slate-900">{formatTriScore(rec.media)}</b>
                </span>
              </div>
            )}

            <div className="mt-2.5 space-y-1">
              {areas.map(([label, val]) => (
                <div key={label} className="flex items-center justify-between text-[12px]">
                  <span className="text-slate-500">{label}</span>
                  <span className="font-bold tabular-nums text-slate-900">{formatTriScore(val)}</span>
                </div>
              ))}
            </div>

            <div className="mt-3 border-t border-slate-100 pt-2.5 text-[12px] font-bold text-[#139ED3]">
              Clique para ver as escolas →
            </div>
          </div>
        )}
      </div>

      {/* Legenda + dica */}
      <div className="mx-auto mt-3 max-w-[360px] px-2">
        <div className="h-2 w-full rounded-full" style={{ background: `linear-gradient(90deg, ${lerpColor(0)}, ${lerpColor(1)})` }} />
        <div className="mt-1 flex justify-between text-[10px] font-semibold text-slate-400">
          <span>{hasData ? formatTriScore(min) : '—'}</span>
          <span className="uppercase tracking-[0.12em]">média TRI por estado{ano ? ` · ${ano}` : ''}</span>
          <span>{hasData ? formatTriScore(max) : '—'}</span>
        </div>
        <p className="mt-1.5 text-center text-[11px] text-slate-400">Passe o mouse num estado · clique para ver as escolas</p>
      </div>
    </div>
  );
}

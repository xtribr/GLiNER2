'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

type Ring = [number, number][];
interface Geometry {
  type: 'Polygon' | 'MultiPolygon';
  coordinates: Ring[] | Ring[][];
}
interface Feature {
  properties: { sigla: string };
  geometry: Geometry;
}
export interface UfDivergencia {
  uf: string;
  divergencia_media: number;
  pct_terceiro: number;
  n: number;
}

const W = 520;
const H = 540;
const PAD = 12;

// Menor divergência (laranja XTRI claro) → maior divergência (vermelho XTRI #FF4B2E).
function lerpColor(t: number): string {
  const a = [255, 224, 219]; // #FFE0DB (tom claro do vermelho XTRI)
  const b = [255, 75, 46]; // #FF4B2E (vermelho XTRI)
  const k = Math.max(0, Math.min(1, t));
  const c = a.map((v, i) => Math.round(v + (b[i] - v) * k));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function eachCoord(g: Geometry, cb: (p: [number, number]) => void) {
  if (g.type === 'Polygon') {
    (g.coordinates as Ring[]).forEach((ring) => ring.forEach(cb));
  } else {
    (g.coordinates as Ring[][]).forEach((poly) => poly.forEach((ring) => ring.forEach(cb)));
  }
}

const fmt = (n: number) => n.toLocaleString('pt-BR', { maximumFractionDigits: 1 });

export default function RedacaoDivergenceMap({ ufs }: { ufs: UfDivergencia[] }) {
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

  const byUf = useMemo(
    () => Object.fromEntries(ufs.map((r) => [r.uf, r])) as Record<string, UfDivergencia>,
    [ufs],
  );
  const vals = ufs.map((r) => r.divergencia_media);
  const hasData = vals.length > 0;
  const min = hasData ? Math.min(...vals) : 0;
  const max = hasData ? Math.max(...vals) : 1;

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
    const py = (y: number) => H - offY - (y - minY) * s;
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
  const flipX = pos.x > pos.w * 0.5;
  const flipY = pos.y > pos.h * 0.55;
  const onMove = (e: React.MouseEvent) => {
    const r = wrapRef.current?.getBoundingClientRect();
    if (r) setPos({ x: e.clientX - r.left, y: e.clientY - r.top, w: r.width, h: r.height });
  };

  return (
    <div>
      <div ref={wrapRef} className="relative mx-auto max-w-[480px]" onMouseLeave={() => setHover(null)}>
        <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label="Mapa da divergência entre avaliadores por estado">
          {paths.map((p) => {
            const r = byUf[p.sigla];
            const fill = r ? lerpColor((r.divergencia_media - min) / (max - min || 1)) : '#e2e8f0';
            return (
              <path
                key={p.sigla}
                d={p.d}
                fill={fill}
                stroke={hover === p.sigla ? '#0f172a' : '#ffffff'}
                strokeWidth={hover === p.sigla ? 1.6 : 0.5}
                className="cursor-default transition-[stroke]"
                onMouseEnter={(e) => {
                  setHover(p.sigla);
                  onMove(e);
                }}
                onMouseMove={onMove}
              >
                <title>{`${p.sigla}${r ? ` — divergência média ${fmt(r.divergencia_media)} pts · ${fmt(r.pct_terceiro)}% 3º avaliador` : ' — sem dados'}`}</title>
              </path>
            );
          })}
        </svg>

        {rec && (
          <div
            className="pointer-events-none absolute z-20 w-[240px] rounded-2xl border border-slate-200 bg-white p-4 shadow-xl"
            style={{
              left: pos.x,
              top: pos.y,
              transform: `translate(${flipX ? 'calc(-100% - 16px)' : '16px'}, ${flipY ? 'calc(-100% - 16px)' : '16px'})`,
            }}
          >
            <span className="text-xl font-black leading-none text-slate-900">{rec.uf}</span>
            <p className="mt-2 flex items-center justify-between text-[13px]">
              <span className="text-slate-500">Divergência média</span>
              <span className="font-bold tabular-nums text-slate-900">{fmt(rec.divergencia_media)} pts</span>
            </p>
            <p className="mt-1 flex items-center justify-between text-[13px]">
              <span className="text-slate-500">3º avaliador</span>
              <span className="font-bold tabular-nums text-amber-600">{fmt(rec.pct_terceiro)}%</span>
            </p>
            <p className="mt-2 border-t border-slate-100 pt-2 text-[11px] text-slate-400">
              {rec.n.toLocaleString('pt-BR')} redações corrigidas
            </p>
          </div>
        )}
      </div>

      <div className="mx-auto mt-3 max-w-[360px] px-2">
        <div className="h-2 w-full rounded-full" style={{ background: `linear-gradient(90deg, ${lerpColor(0)}, ${lerpColor(1)})` }} />
        <div className="mt-1 flex justify-between text-[10px] font-semibold text-slate-400">
          <span>{hasData ? `${fmt(min)} pts` : '—'}</span>
          <span className="uppercase tracking-[0.12em]">divergência Av1×Av2 por estado</span>
          <span>{hasData ? `${fmt(max)} pts` : '—'}</span>
        </div>
        <p className="mt-1.5 text-center text-[11px] text-slate-400">Passe o mouse num estado para ver os detalhes</p>
      </div>
    </div>
  );
}

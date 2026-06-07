'use client';

import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
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

const W = 520;
const H = 540;
const PAD = 12;

// Cor: escala da menor (cyan claro e nítido) à maior média (azul profundo).
// O piso é um cyan claramente visível (não quase-branco) para o mapa "ler" como pintado.
function lerpColor(t: number): string {
  const a = [144, 206, 240]; // #90cef0 — cyan claro visível
  const b = [7, 64, 105]; // #074069 — azul profundo
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

export default function BrazilChoropleth() {
  const [features, setFeatures] = useState<Feature[] | null>(null);
  const [hover, setHover] = useState<string | null>(null);

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
    () => Object.fromEntries((data?.by_uf ?? []).map((r) => [r.uf, r])),
    [data],
  );
  const medias = (data?.by_uf ?? []).map((r) => r.media);
  const hasData = medias.length > 0;
  const min = hasData ? Math.min(...medias) : 0;
  const max = hasData ? Math.max(...medias) : 1;

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

  const hovered = hover ? byUf[hover] : null;

  return (
    <div className="grid items-center gap-4 sm:grid-cols-[1fr_auto]">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full max-w-[460px]" role="img" aria-label="Mapa de calor da média TRI por estado">
        {paths.map((p) => {
          const rec = byUf[p.sigla];
          const fill = rec ? lerpColor((rec.media - min) / (max - min || 1)) : '#e2e8f0';
          return (
            <path
              key={p.sigla}
              d={p.d}
              fill={fill}
              stroke={hover === p.sigla ? '#FF4B2E' : '#ffffff'}
              strokeWidth={hover === p.sigla ? 1.5 : 0.5}
              className="cursor-pointer transition-[stroke]"
              onMouseEnter={() => setHover(p.sigla)}
              onMouseLeave={() => setHover((h) => (h === p.sigla ? null : h))}
              onClick={() => setHover((h) => (h === p.sigla ? null : p.sigla))}
            >
              <title>{`${p.sigla}${rec ? ` — média TRI ${formatTriScore(rec.media)} · ${rec.escolas} escolas` : ' — sem dados'}`}</title>
            </path>
          );
        })}
      </svg>

      <div className="w-full sm:min-w-[150px]">
        <div className="rounded-2xl border border-slate-200 bg-slate-50 p-3 text-center">
          <p className="text-[11px] font-bold uppercase tracking-[0.14em] text-slate-400">
            {hovered ? hovered.uf : 'Toque num estado'}
          </p>
          <p className="mt-1 text-2xl font-black text-slate-950">{hovered ? formatTriScore(hovered.media) : '—'}</p>
          <p className="text-[11px] text-slate-500">{hovered ? `${hovered.escolas} escolas · média TRI` : 'ou passe o mouse'}</p>
        </div>
        <div className="mt-3 px-1">
          <div className="h-2 w-full rounded-full" style={{ background: `linear-gradient(90deg, ${lerpColor(0)}, ${lerpColor(1)})` }} />
          <div className="mt-1 flex justify-between text-[10px] font-semibold text-slate-400">
            <span>{hasData ? formatTriScore(min) : '—'}</span>
            <span>{hasData ? formatTriScore(max) : '—'}</span>
          </div>
          <p className="mt-1 text-center text-[9px] font-semibold uppercase tracking-[0.12em] text-slate-400">
            {hasData ? 'média TRI por estado' : 'carregando médias…'}
          </p>
        </div>
      </div>
    </div>
  );
}

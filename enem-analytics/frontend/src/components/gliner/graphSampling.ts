export interface SampleNode { count: number; area?: string }

/**
 * Amostra até `maxNodes` nós de forma BALANCEADA por área (round-robin), em vez de
 * um corte global por frequência — assim, ao reduzir o limite, nenhuma área/tipo é
 * esvaziado. Dentro de cada área os nós seguem ordem de `count` (maior primeiro).
 */
export function balancedSlice<T extends SampleNode>(nodes: T[], maxNodes: number): T[] {
  const sorted = [...nodes].sort((a, b) => b.count - a.count);
  if (sorted.length <= maxNodes) return sorted;

  // Agrupa por área preservando a ordem por count.
  const byArea = new Map<string, T[]>();
  for (const n of sorted) {
    const key = n.area ?? '∅';
    let q = byArea.get(key);
    if (!q) { q = []; byArea.set(key, q); }
    q.push(n);
  }

  // Round-robin entre as áreas até atingir o limite.
  const queues = [...byArea.values()];
  const result: T[] = [];
  let i = 0;
  while (result.length < maxNodes && queues.some((q) => q.length > 0)) {
    const q = queues[i % queues.length];
    const n = q.shift();
    if (n) result.push(n);
    i++;
  }
  return result;
}

export interface Region { xMin: number; xMax: number; yMin: number; yMax: number }
export interface PlacedNode { id: string; x: number; y: number; rx: number; ry: number; cluster?: string }

/**
 * Relaxação anti-sobreposição (separação AABB): a cada iteração, empurra os pares
 * de nós que se sobrepõem ao longo do eixo de MENOR penetração, mantendo cada nó
 * dentro do seu cluster (ou dos limites globais). Reduz a "nuvem de palavras"
 * amontoada sem mudar o agrupamento por área. Tudo em coordenadas % (0–100).
 */
export function relaxOverlaps(
  nodes: PlacedNode[],
  opts: { iterations?: number; regions?: Record<string, Region>; bounds?: Region } = {},
): Map<string, { x: number; y: number }> {
  const iterations = opts.iterations ?? 60;
  const pts = nodes.map((n) => ({ ...n }));

  const clampOne = (n: PlacedNode) => {
    const r = (n.cluster && opts.regions?.[n.cluster]) || opts.bounds;
    if (!r) return;
    n.x = Math.min(Math.max(n.x, r.xMin + n.rx), r.xMax - n.rx);
    n.y = Math.min(Math.max(n.y, r.yMin + n.ry), r.yMax - n.ry);
  };

  for (let it = 0; it < iterations; it++) {
    let moved = false;
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const a = pts[i], b = pts[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const ox = a.rx + b.rx - Math.abs(dx);
        const oy = a.ry + b.ry - Math.abs(dy);
        if (ox > 0 && oy > 0) {
          moved = true;
          if (ox <= oy) {
            const s = dx === 0 ? (i % 2 ? 1 : -1) : Math.sign(dx);
            const p = (ox / 2) * s;
            a.x -= p; b.x += p;
          } else {
            const s = dy === 0 ? (i % 2 ? 1 : -1) : Math.sign(dy);
            const p = (oy / 2) * s;
            a.y -= p; b.y += p;
          }
        }
      }
    }
    for (const n of pts) clampOne(n);
    if (!moved) break;
  }

  const out = new Map<string, { x: number; y: number }>();
  for (const n of pts) out.set(n.id, { x: n.x, y: n.y });
  return out;
}

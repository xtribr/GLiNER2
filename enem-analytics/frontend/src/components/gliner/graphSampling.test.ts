import { describe, it, expect } from 'vitest';
import { balancedSlice, relaxOverlaps } from './graphSampling';

const mk = (area: string, count: number) => ({ area, count, id: `${area}-${count}` });

describe('balancedSlice', () => {
  it('não corta quando há nós <= maxNodes (mas ordena por count desc)', () => {
    const nodes = [mk('CN', 1), mk('MT', 5), mk('CH', 3)];
    expect(balancedSlice(nodes, 10).map((n) => n.count)).toEqual([5, 3, 1]);
  });

  it('balanceia áreas no round-robin em vez de cortar global por frequência', () => {
    // CN domina em count; um corte global pegaria só CN. Round-robin garante todas.
    const nodes = [
      mk('CN', 100), mk('CN', 90), mk('CN', 80), mk('CN', 70),
      mk('MT', 10), mk('CH', 9), mk('LC', 8),
    ];
    const out = balancedSlice(nodes, 4);
    expect(out).toHaveLength(4);
    const areas = new Set(out.map((n) => n.area));
    // pega o top de cada área antes de aprofundar em CN
    expect(areas).toEqual(new Set(['CN', 'MT', 'CH', 'LC']));
  });

  it('dentro da área respeita ordem por count', () => {
    const nodes = [mk('CN', 100), mk('CN', 50), mk('CN', 90), mk('MT', 5)];
    const out = balancedSlice(nodes, 3);
    const cn = out.filter((n) => n.area === 'CN').map((n) => n.count);
    expect(cn).toEqual([100, 90]); // 50 fica de fora; mantém os maiores
  });

  it('lida com nós sem área', () => {
    const nodes = [{ count: 5 }, { count: 3 }, { count: 1 }];
    expect(balancedSlice(nodes, 2).map((n) => n.count)).toEqual([5, 3]);
  });
});

const overlaps = (a: { x: number; y: number; rx: number; ry: number }, b: { x: number; y: number; rx: number; ry: number }) =>
  (a.rx + b.rx - Math.abs(a.x - b.x)) > 1e-6 && (a.ry + b.ry - Math.abs(a.y - b.y)) > 1e-6;

describe('relaxOverlaps', () => {
  it('separa nós sobrepostos (deixam de colidir)', () => {
    const nodes = [
      { id: 'a', x: 50, y: 50, rx: 3, ry: 2 },
      { id: 'b', x: 51, y: 50, rx: 3, ry: 2 }, // sobrepõe a
    ];
    const out = relaxOverlaps(nodes, { bounds: { xMin: 0, xMax: 100, yMin: 0, yMax: 100 } });
    const a = { ...nodes[0], ...out.get('a')! };
    const b = { ...nodes[1], ...out.get('b')! };
    expect(overlaps(a, b)).toBe(false);
  });

  it('não move nós que já estão separados', () => {
    const nodes = [
      { id: 'a', x: 10, y: 10, rx: 2, ry: 1 },
      { id: 'b', x: 80, y: 80, rx: 2, ry: 1 },
    ];
    const out = relaxOverlaps(nodes, { iterations: 10 });
    expect(out.get('a')).toEqual({ x: 10, y: 10 });
    expect(out.get('b')).toEqual({ x: 80, y: 80 });
  });

  it('mantém os nós dentro dos limites', () => {
    const nodes = [
      { id: 'a', x: 5, y: 5, rx: 3, ry: 3 },
      { id: 'b', x: 5, y: 5, rx: 3, ry: 3 },
    ];
    const out = relaxOverlaps(nodes, { bounds: { xMin: 0, xMax: 100, yMin: 0, yMax: 100 } });
    for (const id of ['a', 'b']) {
      const p = out.get(id)!;
      expect(p.x).toBeGreaterThanOrEqual(3);
      expect(p.y).toBeGreaterThanOrEqual(3);
    }
  });

  it('mantém cada nó dentro do seu cluster (região)', () => {
    const regions = { CN: { xMin: 2, xMax: 44, yMin: 2, yMax: 44 } };
    const nodes = [
      { id: 'a', x: 43, y: 43, rx: 2, ry: 2, cluster: 'CN' },
      { id: 'b', x: 43, y: 43, rx: 2, ry: 2, cluster: 'CN' },
    ];
    const out = relaxOverlaps(nodes, { regions, bounds: { xMin: 0, xMax: 100, yMin: 0, yMax: 100 } });
    for (const id of ['a', 'b']) {
      const p = out.get(id)!;
      expect(p.x).toBeLessThanOrEqual(44 - 2);
      expect(p.y).toBeLessThanOrEqual(44 - 2);
      expect(p.x).toBeGreaterThanOrEqual(2 + 2);
    }
  });
});

'use client';

interface RangeSliderProps {
  min: number;
  max: number;
  step?: number;
  value: [number, number];
  onChange: (value: [number, number]) => void;
  formatValue?: (v: number) => string;
}

export function RangeSlider({ min, max, step = 1, value, onChange, formatValue }: RangeSliderProps) {
  const [lo, hi] = value;
  const span = Math.max(max - min, step);
  const loPct = ((Math.min(Math.max(lo, min), max) - min) / span) * 100;
  const hiPct = ((Math.min(Math.max(hi, min), max) - min) / span) * 100;
  const fmt = formatValue ?? ((v: number) => String(Math.round(v)));

  const setLo = (v: number) => onChange([Math.min(v, hi), hi]);
  const setHi = (v: number) => onChange([lo, Math.max(v, lo)]);

  return (
    <div className="w-full select-none">
      <div className="relative h-10">
        {/* value bubbles */}
        <span
          className="pointer-events-none absolute top-0 z-30 -translate-x-1/2 whitespace-nowrap rounded-md bg-[#1B2A6B] px-2 py-0.5 text-xs font-bold text-white"
          style={{ left: `${loPct}%` }}
        >
          {fmt(lo)}
        </span>
        <span
          className="pointer-events-none absolute top-0 z-30 -translate-x-1/2 whitespace-nowrap rounded-md bg-[#1B2A6B] px-2 py-0.5 text-xs font-bold text-white"
          style={{ left: `${hiPct}%` }}
        >
          {fmt(hi)}
        </span>
        {/* track */}
        <div className="absolute top-7 h-1.5 w-full rounded-full bg-slate-200" />
        <div
          className="absolute top-7 h-1.5 rounded-full bg-[#1B2A6B]"
          style={{ left: `${loPct}%`, width: `${Math.max(hiPct - loPct, 0)}%` }}
        />
        {/* dual range inputs */}
        <input
          type="range" min={min} max={max} step={step} value={lo}
          onChange={(e) => setLo(Number(e.target.value))}
          className="xtri-range absolute top-[1.4rem] w-full"
          style={{ zIndex: lo >= hi ? 25 : 20 }}
          aria-label="Mínimo"
        />
        <input
          type="range" min={min} max={max} step={step} value={hi}
          onChange={(e) => setHi(Number(e.target.value))}
          className="xtri-range absolute top-[1.4rem] w-full"
          style={{ zIndex: 24 }}
          aria-label="Máximo"
        />
      </div>
      <div className="mt-0.5 flex justify-between text-[10px] font-medium text-slate-400">
        <span>{fmt(min)}</span>
        <span>{fmt((min + max) / 2)}</span>
        <span>{fmt(max)}</span>
      </div>
      <style jsx>{`
        .xtri-range {
          -webkit-appearance: none;
          appearance: none;
          height: 1.5rem;
          background: transparent;
          pointer-events: none;
          margin: 0;
        }
        .xtri-range::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          pointer-events: auto;
          height: 18px;
          width: 18px;
          border-radius: 9999px;
          background: #1b2a6b;
          border: 3px solid #ffffff;
          box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
          cursor: pointer;
        }
        .xtri-range::-moz-range-thumb {
          pointer-events: auto;
          height: 18px;
          width: 18px;
          border-radius: 9999px;
          background: #1b2a6b;
          border: 3px solid #ffffff;
          box-shadow: 0 1px 4px rgba(0, 0, 0, 0.35);
          cursor: pointer;
        }
        .xtri-range::-webkit-slider-runnable-track {
          background: transparent;
          height: 1.5rem;
        }
        .xtri-range::-moz-range-track {
          background: transparent;
        }
      `}</style>
    </div>
  );
}

"use client";

import { cn } from "@/lib/utils";
import { LucideIcon } from "lucide-react";

export interface TabDefinition {
  id: string;
  label: string;
  icon?: LucideIcon;
  badge?: string | number;
}

interface TabsProps {
  tabs: TabDefinition[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
}

export function Tabs({ tabs, active, onChange, className }: TabsProps) {
  return (
    <div
      role="tablist"
      aria-label="Seções da escola"
      className={cn(
        // Estilo "aba-pasta de fichário": a barra é a prateleira (border inferior),
        // as abas se apoiam nela alinhadas pela base e a ativa sobe um degrau.
        "flex items-end gap-1.5 overflow-x-auto border-b-2 border-slate-300 px-1",
        className
      )}
    >
      {tabs.map((tab) => {
        const isActive = tab.id === active;
        const Icon = tab.icon;
        return (
          <button
            key={tab.id}
            role="tab"
            aria-selected={isActive}
            aria-controls={`panel-${tab.id}`}
            id={`tab-${tab.id}`}
            onClick={() => onChange(tab.id)}
            className={cn(
              "group relative flex items-center gap-2 whitespace-nowrap rounded-t-[11px] border border-b-0 text-sm transition-all",
              isActive
                // Ativa: branca, sem borda embaixo (funde com o painel), sobe e quebra a prateleira.
                ? "z-10 -mb-[2px] overflow-hidden border-slate-300 bg-white px-[18px] py-3 font-semibold text-[#1B2A6B]"
                // Inativa: recolhida (slate, mais baixa); no hover sobe um degrau e clareia.
                : "border-slate-200 bg-slate-100 px-4 py-[9px] font-medium text-slate-500 hover:-translate-y-px hover:bg-white hover:text-slate-700"
            )}
          >
            {/* Etiqueta de cor no topo da aba ativa (a "tag" da pasta). */}
            {isActive && <span aria-hidden="true" className="absolute inset-x-0 top-0 h-[3px] bg-[#139ED3]" />}
            {Icon && (
              <Icon
                className={cn(
                  "h-4 w-4 transition-colors",
                  isActive ? "text-[#139ED3]" : "text-slate-400 group-hover:text-slate-600"
                )}
              />
            )}
            <span>{tab.label}</span>
            {tab.badge != null && (
              <span
                className={cn(
                  "ml-0.5 inline-flex items-center justify-center rounded-full px-2 py-0.5 text-xs font-semibold",
                  isActive ? "bg-sky-100 text-sky-700" : "bg-slate-200 text-slate-600"
                )}
              >
                {tab.badge}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

interface TabPanelProps {
  id: string;
  active: string;
  children: React.ReactNode;
}

export function TabPanel({ id, active, children }: TabPanelProps) {
  if (id !== active) return null;
  return (
    <div
      role="tabpanel"
      id={`panel-${id}`}
      aria-labelledby={`tab-${id}`}
      className="focus:outline-none"
    >
      {children}
    </div>
  );
}

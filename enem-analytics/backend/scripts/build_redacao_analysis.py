#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ANONYMOUS, aggregate analysis of the ENEM 2025 redação double-correction.

Streams the official INEP microdata (RESULTADOS_2025.csv) once and writes
data/redacao_2025_analysis.json. NO per-student / PII data — only national
aggregates: official status breakdown, score distribution, per-competency
evaluator-grade levels, inter-rater divergence, per-competency means, and
inter-rater divergence by UF (for the choropleth map).
"""
import os, json, glob
from collections import Counter
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
OUT = os.path.join(BASE, "data", "redacao_2025_analysis.json")

def _find_resultados():
    for p in (
        os.path.join(BASE, "..", "..", "microdados-2025", "microdados_enem_2025", "DADOS", "RESULTADOS_2025.csv"),
        os.path.join(BASE, "..", "..", "microdados-2025", "**", "RESULTADOS_2025.csv"),
    ):
        g = glob.glob(p, recursive=True)
        if g:
            return g[0]
    raise SystemExit("RESULTADOS_2025.csv não encontrado")

SRC = _find_resultados()
ANO = 2025
STATUS_LABELS = {
    "1": "Sem problemas", "2": "Anulada", "3": "Cópia do texto motivador",
    "4": "Em branco", "6": "Fuga ao tema", "7": "Não atendimento ao tipo textual",
    "8": "Texto insuficiente", "9": "Parte desconectada", "(ausente)": "Ausente (não fez a redação)",
}
COMP_NOMES = {
    1: "C1 — Norma culta", 2: "C2 — Compreensão da proposta",
    3: "C3 — Argumentação", 4: "C4 — Coesão", 5: "C5 — Proposta de intervenção",
}
UFS = {"AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG","PA","PB","PR",
       "PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"}
LEVELS = [0, 40, 80, 120, 160, 200]
FINAL = [f"NU_NOTA_COMP{i}" for i in range(1, 6)]
AV1 = [f"NU_NOTA_COMP{i}_AV1" for i in range(1, 6)]
AV2 = [f"NU_NOTA_COMP{i}_AV2" for i in range(1, 6)]
USECOLS = ["TP_STATUS_REDACAO", "NU_NOTA_REDACAO", "NU_NOTA_AV3", "SG_UF_PROVA"] + FINAL + AV1 + AV2

status = Counter()
total = 0; graded = 0; terceiro = 0
red_sum = 0.0; red_n = 0; nota_mil = 0; nota_zero = 0
hist = Counter()
diff_cnt = {i: Counter() for i in range(1, 6)}
diff_sum = {i: 0.0 for i in range(1, 6)}; diff_n = {i: 0 for i in range(1, 6)}
lvl_cnt = {i: Counter() for i in range(1, 6)}
av_sum = {i: 0.0 for i in range(1, 6)}; av_n = {i: 0 for i in range(1, 6)}
uf_td = Counter(); uf_n = Counter(); uf_t3 = Counter()   # total-divergence sum / count / 3rd-eval by UF

reader = pd.read_csv(SRC, sep=";", usecols=USECOLS,
                     dtype={"TP_STATUS_REDACAO": "string", "SG_UF_PROVA": "string"},
                     chunksize=400_000, encoding="latin-1", low_memory=False)
for ch in reader:
    total += len(ch)
    status.update(ch["TP_STATUS_REDACAO"].fillna("(ausente)").astype(str).value_counts().to_dict())
    g = ch[ch["NU_NOTA_COMP1_AV1"].notna() & ch["NU_NOTA_COMP1_AV2"].notna()]
    graded += len(g)
    terceiro += int(g["NU_NOTA_AV3"].notna().sum())
    # total divergence per row = sum_i |Av1_i - Av2_i|
    td = sum((g[f"NU_NOTA_COMP{i}_AV1"] - g[f"NU_NOTA_COMP{i}_AV2"]).abs() for i in range(1, 6)).fillna(0)
    t3 = g["NU_NOTA_AV3"].notna().astype(int)
    gb = pd.DataFrame({"uf": g["SG_UF_PROVA"], "td": td, "t3": t3}).dropna(subset=["uf"])
    agg = gb.groupby("uf").agg(td=("td", "sum"), n=("td", "count"), t3=("t3", "sum"))
    for uf, row in agg.iterrows():
        uf_td[uf] += float(row["td"]); uf_n[uf] += int(row["n"]); uf_t3[uf] += int(row["t3"])
    for i in range(1, 6):
        a = g[f"NU_NOTA_COMP{i}_AV1"]; b = g[f"NU_NOTA_COMP{i}_AV2"]
        m = a.notna() & b.notna()
        d = (a[m] - b[m]).abs().round().astype(int)
        diff_cnt[i].update(d.value_counts().to_dict())
        diff_sum[i] += float(d.sum()); diff_n[i] += int(m.sum())
        pooled = pd.concat([a[m], b[m]]).round().astype(int)
        lvl_cnt[i].update(pooled.value_counts().to_dict())
        av_sum[i] += float(pooled.sum()); av_n[i] += int(len(pooled))
    r = ch.loc[ch["NU_NOTA_REDACAO"].notna(), "NU_NOTA_REDACAO"].astype(float)
    red_sum += float(r.sum()); red_n += int(len(r))
    nota_mil += int((r == 1000).sum()); nota_zero += int((r == 0).sum())
    hist.update((r // 100 * 100).clip(0, 1000).astype(int).value_counts().to_dict())

def pct(n, d): return round(100.0 * n / d, 2) if d else 0.0

status_legenda = [
    {"codigo": k, "label": STATUS_LABELS.get(k, k), "count": int(v), "pct": pct(v, total)}
    for k, v in sorted(status.items(), key=lambda kv: -kv[1])
]
problemas = [s for s in status_legenda if s["codigo"] not in ("1", "(ausente)")]

por_comp = []
for i in range(1, 6):
    n = diff_n[i]; c = diff_cnt[i]; ln = av_n[i]; lc = lvl_cnt[i]
    por_comp.append({
        "competencia": f"C{i}", "nome": COMP_NOMES[i],
        "nota_media": round(av_sum[i] / ln, 1) if ln else 0.0,
        "concordancia_exata_pct": pct(c.get(0, 0), n),
        "divergencia_media_pts": round(diff_sum[i] / n, 2) if n else 0.0,
        "dist_niveis_pct": {str(k): pct(lc.get(k, 0), ln) for k in LEVELS},
        "dist_divergencia_pct": {str(k): pct(c.get(k, 0), n) for k in LEVELS},
    })
mais_div = max(por_comp, key=lambda x: x["divergencia_media_pts"])["competencia"] if por_comp else None
mais_dificil = min(por_comp, key=lambda x: x["nota_media"])["competencia"] if por_comp else None

por_uf = sorted(
    [
        {"uf": uf, "divergencia_media": round(uf_td[uf] / uf_n[uf], 2),
         "pct_terceiro": pct(uf_t3[uf], uf_n[uf]), "n": int(uf_n[uf])}
        for uf in uf_n if uf in UFS and uf_n[uf] > 0
    ],
    key=lambda x: -x["divergencia_media"],
)

out = {
    "ano": ANO,
    "fonte": "INEP — Microdados ENEM 2025 (RESULTADOS_2025.csv)",
    "observacao": "Agregados nacionais anônimos. Nenhum dado individual/identificável.",
    "total_participantes": total,
    "redacoes_corrigidas": graded,
    "presentes": total - status.get("(ausente)", 0),
    "ausentes": int(status.get("(ausente)", 0)),
    "nota": {
        "media": round(red_sum / red_n, 1) if red_n else 0.0,
        "n": red_n, "nota_1000": nota_mil, "nota_zero": nota_zero,
        "histograma": [{"faixa": k, "count": int(hist.get(k, 0))} for k in range(0, 1001, 100)],
    },
    "status": {
        "descricao": "Situação oficial da redação (dicionário INEP). % sobre o total de participantes.",
        "legenda": status_legenda,
        "problemas": problemas,
        "total_zeradas_anuladas": sum(s["count"] for s in problemas),
    },
    "divergencia_avaliadores": {
        "descricao": "Notas dos dois avaliadores por competência: níveis 0–200 (passo 40), divergência e médias.",
        "pct_terceiro_avaliador": pct(terceiro, graded),
        "competencia_mais_divergente": mais_div,
        "competencia_mais_dificil": mais_dificil,
        "niveis": LEVELS,
        "por_competencia": por_comp,
    },
    "divergencia_por_uf": {
        "descricao": "Divergência média entre Av1 e Av2 (soma das 5 competências, em pontos) por UF de aplicação da prova, e % que precisou de 3º avaliador.",
        "ufs": por_uf,
    },
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("OK ->", OUT)
print("total:", total, "| corrigidas:", graded, "| média:", out["nota"]["media"],
      "| 3º aval %:", out["divergencia_avaliadores"]["pct_terceiro_avaliador"])
print("UF (top divergência):", [(u["uf"], u["divergencia_media"], u["pct_terceiro"]) for u in por_uf[:5]])
print("UF (menor divergência):", [(u["uf"], u["divergencia_media"]) for u in por_uf[-5:]])

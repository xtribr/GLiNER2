import { readFileSync } from "fs";
import { join } from "path";
import type { Metadata } from "next";
import { MarkdownContent } from "@/components/markdown-content";

export const metadata: Metadata = {
  title: "Política de Compliance e Uso Ético de IA — XTRI EdTech",
  description:
    "Conheça os princípios e compromissos da XTRI EdTech para uso ético e responsável de Inteligência Artificial na educação.",
  alternates: { canonical: "/politica-de-compliance-ia" },
};

export default function PoliticaDeComplianceIAPage() {
  const content = readFileSync(
    join(process.cwd(), "public", "03-politica-de-compliance-ia.md"),
    "utf-8",
  );

  return <MarkdownContent content={content} />;
}
